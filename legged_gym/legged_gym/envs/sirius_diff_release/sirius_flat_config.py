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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class SiriusFlatCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 100  # Will be overridden by args
        num_actions = 12
        num_observations = 64  # Updated: added base_lin_vel (3 dims)
        env_spacing = 7.0  # 7m spacing to avoid collisions

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'bridge'
        measure_heights = False
        curriculum = False
        pillar_gap_range = [0.05, 0.15]  # Start easy (5cm), max hard (30cm)

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.42] # x,y,z [m] - spawn on top of start pillar
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,

            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.0,

            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        }
            
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 30.}  # [N*m/rad]
        damping = {'joint': 0.6}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf"
        name = "go1"
        foot_name = "calf"  # Go1's feet are the calf links
        penalize_contacts_on = ["thigh"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 #1 to disable, 0 to enable...bitwise filter

    class commands( LeggedRobotCfg.commands ):
        heading_command = False
        resampling_time = 4.
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [0.25, 0.25]  # Slow, controlled forward movement for short gaps
            lin_vel_y = [0.0, 0.0]  # No lateral movement
            ang_vel_yaw = [0.0, 0.0]  # No rotation

    class domain_rand( LeggedRobotCfg.domain_rand):
        # Disable large base-mass randomization during debugging / early training to
        # avoid cases where very heavy robots learn to drag instead of stepping.
        randomize_base_mass = False
        # If re-enabled later, prefer a much smaller added mass range to reduce extremes
        added_mass_range = [-1., 1.]
        friction_range = [0., 1.5] # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
  
    class rewards( LeggedRobotCfg.rewards ):
        base_height_target = 1.35
        max_contact_force = 350
        only_positive_rewards = False  # Allow negative rewards for proper learning
        soft_dof_vel_limit = 0.8
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 5.0  # PRIMARY: track target velocity (reduced to avoid dragging shortcut)
            tracking_ang_vel = 0.5
            orientation = -1.0
            feet_air_time = 0.8
            base_height = -2.0
            posture = 0.3
            lateral_deviation = -1.0
            forward_progress = 0.1  # Disabled: conflicts with velocity tracking
            goal_reached = 20.0  # Reduced: don't rush to goal ignoring velocity
            heading_alignment = -1.0
            knee_contact = -2.0
            foot_in_gap = -0.5  # penalty for foot entering pillar gap (z < 1.0)
            action_rate = -0.01
            dof_vel = -0.001
            stand_still = -0.3  # Penalize standing still
            # dof_vel_limits = -0.1
    
    class noise( LeggedRobotCfg.noise ):
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales( LeggedRobotCfg.noise.noise_scales ):
            dof_pos = 0.03
            dof_vel = 0.4  # Reduced from 1.5 to improve learning signal
            lin_vel = 0.1
            ang_vel = 0.5
            gravity = 0.05
            height_measurements = 0.1

    class sim ( LeggedRobotCfg.sim ):
        dt =  0.004

class SiriusFlatCfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = "sirius_diff_release"
        load_run = -1
        max_iterations = 2000  # Increased from 1200 for better convergence
