# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .sirius_flat_config import SiriusFlatCfg

class SiriusJoyFlat(BaseTask):
    def __init__(self, cfg: SiriusFlatCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)
        print(f"DEBUG: cfg.env.num_envs = {cfg.env.num_envs}")
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        print(f"DEBUG: After super().__init__, self.num_envs = {self.num_envs}")

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # refresh rigid body state tensor so foot world positions are up-to-date
        try:
            self.gym.refresh_rigid_body_state_tensor(self.sim)
        except Exception:
            pass

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        # Failure conditions: collision or height too low
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf |= self.root_states[:, 2] < 1.0  # Terminate if height < 1m

        # Check for successful completion (reached end pillar)
        # Do not immediately reset on success; instead start a 1s timer and reset after that
        self.success_buf = self._check_goal_reached()

        # update per-env success elapsed timer: accumulate dt while success is True
        self.success_elapsed = torch.where(self.success_buf, self.success_elapsed + self.dt, torch.zeros_like(self.success_elapsed))

        # reset environments that have been successful for >= 1.0s
        success_reset = self.success_elapsed >= 1.0
        self.reset_buf |= success_reset

        # time-out conditions
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        # reset success elapsed timer used for delayed reset after reaching goal
        if hasattr(self, 'success_elapsed'):
            self.success_elapsed[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        # Get front two pillars' top corners (8 corners total, 16 values: x,z coordinates)
        pillar_corners = self._get_front_pillar_corners()  # shape: (num_envs, 16)
        
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel, # 3dim (xy + z)
                                    self.base_ang_vel  * self.obs_scales.ang_vel, # 3dim
                                    self.projected_gravity, # 3dim
                                    self.commands[:, :3] * self.commands_scale, # 3dim
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 12dim
                                    self.dof_vel * self.obs_scales.dof_vel, # 12dim
                                    self.actions, # 12dim
                                    pillar_corners # 16dim (8 corners * 2 coordinates)
                                    ),dim=-1)
        
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='bridge':
            self._create_ground_bridge()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, bridge, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # Add random offset in xy direction (±0.1m)
        self.root_states[env_ids, :2] += torch_rand_float(-0.1, 0.1, (len(env_ids), 2), device=self.device)
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Update pillar difficulty based on performance """
        if not self.init_done:
            return
        
        # Check if robot made good progress (reached far enough)
        distance = self.root_states[env_ids, 0] - self.env_origins[env_ids, 0]
        
        # Increase difficulty if robot traveled > 2m
        move_up = distance > 3.5
        # Decrease difficulty if robot failed quickly (< 1m)
        move_down = (distance < 0.4) * ~move_up
        
        # Update difficulty (0.0 to 1.0)
        self.pillar_difficulty[env_ids] += 0.1 * move_up.float() - 0.05 * move_down.float()
        self.pillar_difficulty[env_ids] = torch.clip(self.pillar_difficulty[env_ids], 0.0, 1.0)
        
        # Regenerate pillars for these environments
        # Note: This only updates the layout data, actual mesh is static
        for env_id in env_ids:
            difficulty = self.pillar_difficulty[env_id].item()
            self.env_pillar_layouts[env_id] = self._generate_pillar_layout(difficulty)
            
            # Update visualization corners if it's env 0
            if env_id == 0:
                self._update_pillar_corners(0)
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure
            
            NEW observation order (64 dims total):
            0:3     - base_lin_vel (3)
            3:6     - base_ang_vel (3)
            6:9     - projected_gravity (3)
            9:12    - commands (3)
            12:24   - dof_pos (12)
            24:36   - dof_vel (12)
            36:48   - actions (12)
            48:64   - pillar_corners (16)

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0. # previous actions
        noise_vec[48:64] = 0.05 * noise_level  # pillar corners (16 values)
        if self.cfg.terrain.measure_heights:
            noise_vec[61:] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        # acquire rigid body state tensor (used to get per-foot world positions)
        # layout is assumed to be [num_envs * num_bodies, ...], we will view it into (num_envs, num_bodies, dim)
        body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # refresh rigid body state tensor as well
        try:
            self.gym.refresh_rigid_body_state_tensor(self.sim)
        except Exception:
            # some gym builds may not have this explicit refresh call; acquire/wrap will still work
            pass

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # wrap rigid body state tensor and view as (num_envs, num_bodies, dim)
        # common rigid body state layout is 13 values per body (pos(3), quat(4), vel(3), angvel(3))
        try:
            self.rigid_body_state = gymtorch.wrap_tensor(body_state_tensor).view(self.num_envs, self.num_bodies, 13)
        except Exception:
            # Fallback: create a placeholder tensor so code won't crash during static checks; runtime may overwrite
            self.rigid_body_state = torch.zeros(self.num_envs, self.num_bodies, 13, device=self.device)

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        # Initialize with placeholder - will be resized after feet_indices is created
        # Use a temporary size of 4 feet (typical for quadruped)
        self.feet_air_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device, requires_grad=False)
        # Initialize these tensors but don't compute values yet (will be computed in post_physics_step)
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.projected_gravity = torch.zeros_like(self.gravity_vec)
        self.success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # per-env timer to delay reset after success (allow 1s of continued operation)
        self.success_elapsed = torch.zeros(self.num_envs, device=self.device)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
        
    def _create_ground_bridge(self):
        """ Creates bridge environment - only ground plane and pillar data """
        # Create ground plane
        self._create_ground_plane()
        
        # Initialize pillar data (will be used later in _create_envs)
        self._init_pillar_data()
        
    def _init_pillar_data(self):
        """ Initialize pillar positions - will be regenerated per environment """
        self.pillar_top_corners = []
        # Initialize curriculum difficulty levels per environment
        self.pillar_difficulty = torch.zeros(self.num_envs, device=self.device)
        print(f"Initialized pillar curriculum system")
    
    def _generate_pillar_layout(self, difficulty=0.0):
        """ Generate pillar positions based on difficulty level (0.0 to 1.0) """
        pillar_positions = []
        
        # Interpolate gap range based on difficulty
        min_gap, max_gap = self.cfg.terrain.pillar_gap_range
        # If curriculum disabled, use full range
        if not self.cfg.terrain.curriculum:
            gap_range = max_gap
        else:
            gap_range = min_gap + difficulty * (max_gap - min_gap)
        
        # Start pillar
        start_pos = [0.0, 0.0, 0.5]
        pillar_positions.append(('start', start_pos))
        
        current_x = 0.5
        
        # Middle pillars with difficulty-based gaps
        for i in range(10):
            gap = round(np.random.uniform(min_gap, gap_range), 3)
            current_x += gap
            middle_pos = [round(current_x + 0.125, 3), 0.0, 0.5]
            pillar_positions.append(('middle', middle_pos))
            current_x += 0.25
        
        # End pillar
        final_gap = round(np.random.uniform(min_gap, gap_range), 3)
        current_x += final_gap
        end_pos = [round(current_x + 0.5, 3), 0.0, 0.5]
        pillar_positions.append(('end', end_pos))
        
        return pillar_positions
    
    def _generate_fixed_gap_layout(self, gap_size):
        """ Generate pillar layout with fixed gap size for testing """
        pillar_positions = []
        
        # Start pillar
        start_pos = [0.0, 0.0, 0.5]
        pillar_positions.append(('start', start_pos))
        
        current_x = 0.5
        
        # Middle pillars with fixed gaps
        for i in range(10):
            current_x += gap_size
            middle_pos = [round(current_x + 0.125, 3), 0.0, 0.5]
            pillar_positions.append(('middle', middle_pos))
            current_x += 0.25
        
        # End pillar
        current_x += gap_size
        end_pos = [round(current_x + 0.5, 3), 0.0, 0.5]
        pillar_positions.append(('end', end_pos))
        
        return pillar_positions
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        print(f"DEBUG: Creating {self.num_envs} environments...")
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
        
        # Create bridge pillars if bridge terrain is used
        if hasattr(self, 'pillar_difficulty'):
            self._create_bridge_pillars()

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        
        # Initialize feet-related tensors now that feet_indices is created
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
    
    def _create_bridge_pillars(self):
        """ Create bridge pillars for each environment with curriculum """
        print(f"Creating bridge pillars for {self.num_envs} environments...")
        
        vertices = []
        faces = []
        vertex_count = 0
        
        # Store pillar layouts for each environment
        self.env_pillar_layouts = []
        
        # Check if we're in test mode (small num_envs typically means testing)
        test_mode = self.num_envs <= 10
        
        # Create pillars for each environment
        for env_id in range(self.num_envs):
            env_origin = self.env_origins[env_id].cpu().numpy()
            
            if test_mode:
                # Test mode: randomly choose 5cm or 15cm for each environment
                import random
                gap_size = random.choice([0.05, 0.15])
                pillar_positions = self._generate_fixed_gap_layout(gap_size)
                print(f"  Test Env {env_id}: Gap = {gap_size*100:.0f}cm")
            else:
                # Training mode: use curriculum
                difficulty = self.pillar_difficulty[env_id].item()
                pillar_positions = self._generate_pillar_layout(difficulty)
            
            self.env_pillar_layouts.append(pillar_positions)
            
            for pillar_idx, (pillar_type, pos) in enumerate(pillar_positions):
                # Choose dimensions based on type
                if pillar_type == 'start' or pillar_type == 'end':
                    length, width, height = 1.0, 1.0, 1.0
                else:
                    length, width, height = 0.25, 1.0, 1.0
                
                # Create box vertices with environment offset
                x, y, z = pos[0] + env_origin[0], pos[1] + env_origin[1], pos[2]
                box_vertices = [
                    [x - length/2, y - width/2, z - height/2],
                    [x + length/2, y - width/2, z - height/2],
                    [x + length/2, y + width/2, z - height/2],
                    [x - length/2, y + width/2, z - height/2],
                    [x - length/2, y - width/2, z + height/2],
                    [x + length/2, y - width/2, z + height/2],
                    [x + length/2, y + width/2, z + height/2],
                    [x - length/2, y + width/2, z + height/2],
                ]
                vertices.extend(box_vertices)
                
                base = vertex_count
                box_faces = [
                    [base+0, base+2, base+1], [base+0, base+3, base+2],
                    [base+4, base+5, base+6], [base+4, base+6, base+7],
                    [base+0, base+1, base+5], [base+0, base+5, base+4],
                    [base+2, base+7, base+6], [base+2, base+3, base+7],
                    [base+0, base+4, base+7], [base+0, base+7, base+3],
                    [base+1, base+2, base+6], [base+1, base+6, base+5],
                ]
                faces.extend(box_faces)
                vertex_count += 8
        
        # Add all pillars as one triangle mesh
        if vertices and faces:
            import numpy as np
            vertices_np = np.array(vertices, dtype=np.float32)
            faces_np = np.array(faces, dtype=np.uint32)
            
            tm_params = gymapi.TriangleMeshParams()
            tm_params.nb_vertices = len(vertices)
            tm_params.nb_triangles = len(faces)
            tm_params.static_friction = 0.8
            tm_params.dynamic_friction = 0.8
            tm_params.restitution = 0.1
            
            self.gym.add_triangle_mesh(self.sim, vertices_np.flatten(), faces_np.flatten(), tm_params)
        
        # Record corners for first environment (for visualization)
        self._update_pillar_corners(0)
        
        if test_mode:
            print(f"✅ Test bridge created with random 5cm/15cm gaps!")
        else:
            print(f"✅ Bridge created for {self.num_envs} environments with curriculum!")
    
    def _update_pillar_corners(self, env_id):
        """ Update pillar corners for given environment """
        if env_id >= len(self.env_pillar_layouts):
            return
        
        self.pillar_top_corners = []
        env_origin = self.env_origins[env_id].cpu().numpy()
        
        for pillar_type, pos in self.env_pillar_layouts[env_id]:
            if pillar_type == 'start' or pillar_type == 'end':
                length, width, height = 1.0, 1.0, 1.0
            else:
                length, width, height = 0.25, 1.0, 1.0
            
            x, y, z = pos[0] + env_origin[0], pos[1] + env_origin[1], pos[2]
            top_corners = [
                [x - length/2, y - width/2, z + height/2],
                [x + length/2, y - width/2, z + height/2],
                [x + length/2, y + width/2, z + height/2],
                [x - length/2, y + width/2, z + height/2],
            ]
            self.pillar_top_corners.append({
                'type': pillar_type,
                'center': [x, y, z],
                'corners': top_corners
            })

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draw debug visualization (optimized for performance) """
        if not self.viewer or not hasattr(self, 'env_pillar_layouts'):
            return
        
        self.gym.clear_lines(self.viewer)
        
        # Limit visualization to reduce overhead
        max_vis_envs = min(20, self.num_envs)  # Only first 20 envs
        
        for env_id in range(max_vis_envs):
            robot_pos = self.root_states[env_id, :3].cpu().numpy()
            env_origin = self.env_origins[env_id].cpu().numpy()
            
            # Draw robot center marker
            center = gymapi.Vec3(robot_pos[0], robot_pos[1], robot_pos[2] + 0.1)
            size = 0.1
            self.gym.add_lines(self.viewer, self.envs[env_id], 1,
                             [center.x-size, center.y, center.z, center.x+size, center.y, center.z],
                             (1, 1, 0))
            self.gym.add_lines(self.viewer, self.envs[env_id], 1,
                             [center.x, center.y-size, center.z, center.x, center.y+size, center.z],
                             (1, 1, 0))
            
            # Draw visible pillar corners for this environment
            robot_x = robot_pos[0]
            pillar_layout = self.env_pillar_layouts[env_id]
            
            # Use an offset reference point 0.5m ahead of the robot when checking which pillars are "in front"
            forward_offset = 0.5
            visible_count = 0
            for pillar_type, pos in pillar_layout:
                pillar_world_x = pos[0] + env_origin[0]
                
                # Calculate front edge
                if pillar_type in ['start', 'end']:
                    length = 1.0
                else:
                    length = 0.25
                pillar_front_edge = pillar_world_x - length / 2
                
                # Check visibility relative to a point 0.5m in front of the robot
                if (robot_x + forward_offset) < pillar_front_edge and visible_count < 2:
                    x = pillar_world_x
                    y = pos[1] + env_origin[1]
                    z = pos[2]
                    width, height = 1.0, 1.0
                    
                    corners = [
                        [x - length/2, y - width/2, z + height/2],
                        [x + length/2, y - width/2, z + height/2],
                        [x + length/2, y + width/2, z + height/2],
                        [x - length/2, y + width/2, z + height/2],
                    ]
                    
                    color = (0, 1, 0) if visible_count == 0 else (0, 0.5, 1)
                    for corner in corners:
                        corner_pos = gymapi.Vec3(corner[0], corner[1], corner[2])
                        s = 0.05
                        self.gym.add_lines(self.viewer, self.envs[env_id], 1,
                                         [corner_pos.x-s, corner_pos.y, corner_pos.z,
                                          corner_pos.x+s, corner_pos.y, corner_pos.z], color)
                        self.gym.add_lines(self.viewer, self.envs[env_id], 1,
                                         [corner_pos.x, corner_pos.y-s, corner_pos.z,
                                          corner_pos.x, corner_pos.y+s, corner_pos.z], color)
                        self.gym.add_lines(self.viewer, self.envs[env_id], 1,
                                         [corner_pos.x, corner_pos.y, corner_pos.z-s,
                                          corner_pos.x, corner_pos.y, corner_pos.z+s], color)
                    
                    visible_count += 1 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward meaningful foot airtime (encourage stepping)
        # Improvements over previous implementation:
        # - use an explicit rising-edge detection (contact now & ~last_contacts)
        # - only count airtime while foot is not contacting
        # - use a smaller, realistic minimum airtime threshold (e.g. 0.15s)
        # - use a slightly stricter contact threshold to avoid noise
        contact_thresh = 5.0  # [N] vertical force threshold to consider contact (tune to robot)
        min_air_time = 0.15   # [s] minimum airtime to consider a meaningful step

        # detect contact now using z component
        contact_now = self.contact_forces[:, self.feet_indices, 2] > contact_thresh

        # rising edge: foot was not in contact previously, and now it is
        contact_rising = contact_now & (~self.last_contacts)

        # first_contact: foot had some airtime and we see a rising edge
        first_contact = (self.feet_air_time > 0.) & contact_rising

        # accumulate airtime while foot is in the air (not contacting)
        self.feet_air_time += self.dt * (~contact_now).float()

        # reward = sum over feet of (airtime - min_air_time) but only for those that just contacted
        effective_air = (self.feet_air_time - min_air_time).clip(min=0.)
        rew_airTime = torch.sum(effective_air * first_contact.float(), dim=1)

        # only reward when there's a non-zero locomotion command
        cmd_mask = (torch.norm(self.commands[:, :2], dim=1) > 0.1).float()
        rew_airTime = rew_airTime * cmd_mask

        # reset airtime for feet that are contacting (avoid double counting)
        self.feet_air_time = self.feet_air_time * (~contact_now).float()

        # update last_contacts for next step
        self.last_contacts = contact_now

        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize standing still when should be moving forward
        # Check if velocity is too low (< 0.1 m/s)
        is_standing = torch.norm(self.base_lin_vel[:, :2], dim=1) < 0.1
        # Penalize if standing still (should always move forward in bridge task)
        return is_standing.float()

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_foot_in_gap(self):
        """Penalize feet that are below the safe height (e.g., z < 1.0).
        Uses rigid body state to get exact foot world z positions.
        Returns a positive penalty per env (to be multiplied by a negative scale in cfg).
        """
        try:
            # body state layout: [..., pos_x, pos_y, pos_z, quat_w, ...] assumed; pos indices 0:3
            foot_positions_z = self.rigid_body_state[:, self.feet_indices, 2]
        except Exception:
            # If rigid body state not available, no penalty
            return torch.zeros(self.num_envs, device=self.device)

        # penalty is how much below threshold the foot is (sum over feet)
        thresh = 1.0
        below = (thresh - foot_positions_z).clip(min=0.)
        penalty = torch.sum(below, dim=1)
        return penalty
    
    def _reward_posture(self):
        weight = torch.tensor([1.0, 1.0, 0.1] * 4, device=self.device).unsqueeze(0) # shape: (1, num_dof)
        return torch.exp(-torch.sum(torch.square(self.dof_pos - self.default_dof_pos) * weight, dim=1))
    
    def _reward_lateral_deviation(self):
        """ Penalize deviation from centerline (y=0 relative to env origin) """
        # Calculate y position relative to environment origin
        y_relative = self.root_states[:, 1] - self.env_origins[:, 1]
        # Penalize deviation from centerline (y=0)
        return torch.square(y_relative)
    
    def _reward_forward_progress(self):
        """ Reward forward distance with one-time milestone bonuses at 1m, 2m, 3m, etc.
        Uses persistent state to track which milestones have been reached to prevent
        repeated rewards from oscillation around milestone positions.
        """
        # Initialize milestone tracker if not exists
        if not hasattr(self, 'milestone_reached'):
            self.milestone_reached = torch.zeros(self.num_envs, 5, dtype=torch.bool, device=self.device)
        
        # Calculate distance from environment origin (start position)
        distance = self.root_states[:, 0] - self.env_origins[:, 0]
        
        # Continuous progress reward (small baseline)
        continuous_reward = torch.clip(distance / 10.0, 0.0, 1.0)
        
        # Milestone bonuses: one-time trigger when distance first exceeds threshold
        milestones = torch.zeros(self.num_envs, device=self.device)
        milestone_distances = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        for idx, m in enumerate(milestone_distances):
            # Check if milestone is newly reached (distance > threshold AND not yet recorded)
            newly_reached = (distance >= m) & (~self.milestone_reached[:, idx])
            
            # Award bonus and mark as reached
            milestones += newly_reached.float() * 0.2  # One-time 0.2 bonus
            self.milestone_reached[:, idx] |= newly_reached
        
        # Reset milestone tracker for environments that reset
        reset_envs = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(reset_envs) > 0:
            self.milestone_reached[reset_envs] = False
        
        return continuous_reward + milestones
    
    def _reward_heading_alignment(self):
        """ Penalize deviation from forward heading (yaw should be 0) """
        # Get yaw angle from quaternion
        forward = quat_apply(self.base_quat, self.forward_vec)
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        # Penalize deviation from 0 yaw (forward direction)
        return torch.square(yaw)
    
    def _reward_knee_contact(self):
        """ Penalize any non-foot contact with ground """
        # Get all body contact forces
        all_contact_forces = torch.norm(self.contact_forces, dim=-1)  # (num_envs, num_bodies)
        
        # Create mask: True for all bodies except feet
        non_foot_mask = torch.ones(all_contact_forces.shape[1], dtype=torch.bool, device=self.device)
        non_foot_mask[self.feet_indices] = False  # Exclude feet
        
        # Check non-foot contacts > 1.0N
        non_foot_contact = all_contact_forces[:, non_foot_mask] > 1.0
        
        # Sum up all non-foot contacts
        penalty = torch.sum(non_foot_contact.float(), dim=1)
        
        return penalty
    
    def _reward_goal_reached(self):
        """ Reward for reaching end pillar (no velocity penalty at goal).
        机器狗成功到达终点柱子即给奖励，不要求零速度到达（否则会强制减速导致散架）。
        """
        # Simple success reward: reaching the end pillar = fixed bonus
        # 不添加速度相关的惩罚或奖励，让机器狗可以正常速度冲过终点
        return self.success_buf.float() * 1.0  # Fixed 1.0 reward for reaching goal
    
    def _check_goal_reached(self):
        """ Check if robot reached the end pillar.
        条件：机器人的 x 坐标必须超过终点柱子的中心 x 坐标（而不是仅进入柱子范围）。
        这样避免在进入柱子时立刻散架，确保机器狗真正"穿过"了终点。
        """
        if not hasattr(self, 'env_pillar_layouts'):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        for env_id in range(self.num_envs):
            robot_pos = self.root_states[env_id, :3]
            env_origin = self.env_origins[env_id]
            
            # Find end pillar position
            pillar_layout = self.env_pillar_layouts[env_id]
            for pillar_type, pos in pillar_layout:
                if pillar_type == 'end':
                    # End pillar world position (center)
                    end_x = pos[0] + env_origin[0].item()
                    end_y = pos[1] + env_origin[1].item()
                    
                    # Success condition: robot x position MUST EXCEED end pillar center x
                    # This ensures the robot truly "passes through" the goal
                    x_passed = robot_pos[0] > end_x  # Must be beyond center, not just inside
                    y_in_bounds = (robot_pos[1] >= end_y - 0.6) and (robot_pos[1] <= end_y + 0.6)
                    height_ok = robot_pos[2] > 1.0  # Above ground
                    
                    success[env_id] = x_passed and y_in_bounds and height_ok
                    break
        
        return success
    
    def _get_front_pillar_corners(self):
        """ Get the top corners of pillars ahead of robot in robot base frame (optimized)
        Uses a 0.5m forward offset for visibility check (determining which pillars to report),
        but returns corner coordinates relative to robot position (without offset).
        
        Returns:
            torch.Tensor: shape (num_envs, 16) - 8 corners * 2 coordinates (x, z) in robot base frame
        """
        if not hasattr(self, 'env_pillar_layouts'):
            return torch.zeros(self.num_envs, 16, device=self.device)
        
        corners_data = torch.zeros(self.num_envs, 16, device=self.device)
        
        # Batch process to reduce Python loop overhead
        robot_pos = self.root_states[:, :3]
        robot_quat = self.base_quat
        # Offset (m) in front of robot used ONLY for visibility check (which pillars are "in front")
        forward_offset = 0.5
        
        for env_id in range(self.num_envs):
            robot_x = robot_pos[env_id, 0].item()
            env_origin_x = self.env_origins[env_id, 0].item()
            env_origin_y = self.env_origins[env_id, 1].item()
            
            pillar_layout = self.env_pillar_layouts[env_id]
            
            # Find first 2 visible pillars (early exit)
            corner_idx = 0
            for pillar_type, pos in pillar_layout:
                if corner_idx >= 8:
                    break
                    
                pillar_world_x = pos[0] + env_origin_x
                length = 1.0 if pillar_type in ['start', 'end'] else 0.25
                
                # Use forward_offset for visibility check: pillar is "in front" if ahead of robot+0.5m reference
                if (robot_x + forward_offset) < pillar_world_x - length / 2:
                    x = pillar_world_x
                    y = pos[1] + env_origin_y
                    z = pos[2]
                    
                    # Pre-compute corners
                    half_l, half_w, half_h = length/2, 0.5, 0.5
                    corners_world = torch.tensor([
                        [x - half_l, y - half_w, z + half_h],
                        [x + half_l, y - half_w, z + half_h],
                        [x + half_l, y + half_w, z + half_h],
                        [x - half_l, y + half_w, z + half_h],
                    ], device=self.device, dtype=torch.float)
                    
                    # Transform corners to robot frame using robot position (NO offset in coordinate transformation)
                    corners_relative = corners_world - robot_pos[env_id]
                    corners_robot = quat_rotate_inverse(robot_quat[env_id].unsqueeze(0).expand(4, -1), corners_relative)
                    
                    # Store x, z coordinates (relative to robot, no bias)
                    for i in range(4):
                        corners_data[env_id, corner_idx * 2] = corners_robot[i, 0]
                        corners_data[env_id, corner_idx * 2 + 1] = corners_robot[i, 2]
                        corner_idx += 1
        
        return corners_data