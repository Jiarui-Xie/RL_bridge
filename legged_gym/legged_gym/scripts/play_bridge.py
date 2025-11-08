# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, Logger
from legged_gym.utils.helpers import class_to_dict, parse_sim_params
from legged_gym.envs.sirius_diff_release.sirius_joystick import SiriusJoyFlat
from legged_gym.envs.sirius_diff_release.sirius_flat_config import SiriusFlatCfg, SiriusFlatCfgPPO

import numpy as np
import torch
import random


def play(args):
    # Use the same task registry as training to ensure identical configuration
    args.task = 'sirius'
    args.num_envs = 5  # Override for testing
    
    # Create environment exactly like training does
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    # Only disable noise and randomization for testing
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    # Disable curriculum for consistent testing
    env.cfg.terrain.curriculum = False
    
    print("=" * 60)
    print("🌉 Bridge Crossing Policy Test")
    print("=" * 60)
    print(f"Environment config:")
    print(f"  - num_observations: {env_cfg.env.num_observations}")
    print(f"  - terrain type: {env_cfg.terrain.mesh_type}")
    print(f"  - curriculum: DISABLED (test mode)")
    print(f"  - num_envs: {env.num_envs}")
    print(f"  - Using: {type(env).__name__}")
    print(f"  - Gap mode: Random 5cm or 15cm per environment")
    
    obs = env.get_observations()
    print(f"Observation shape: {obs.shape}")
    print(f"Expected: ({env.num_envs}, {env_cfg.env.num_observations})")
    
    # Create runner using task registry (same as training)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, log_root=None)
    
    # Load model if specified
    if hasattr(args, 'load_run') and args.load_run:
        ppo_runner.load(args.load_run)
        print(f"\n✅ Loaded model from: {args.load_run}")
    else:
        print("\n⚠️ No model specified. Use --load_run=path/to/model.pt")
    
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1
    stop_state_log = 100
    stop_rew_log = env.max_episode_length + 1
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)


    
    print("\n" + "=" * 60)
    print("🚀 Starting simulation...")
    print(f"  - Environments: {env.num_envs}")
    print(f"  - Episode length: {env.max_episode_length}")
    print(f"  - Camera follow: {'ON' if MOVE_CAMERA else 'OFF'}")
    print(f"  - Bridge gaps: Random 5cm or 15cm per environment")
    print("=" * 60 + "\n")

    for i in range(10 * int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_height': env.root_states[robot_index, 2].item(),
                }
            )
        elif i == stop_state_log:
            logger.plot_states()
            
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()
        





if __name__ == '__main__':
    MOVE_CAMERA = False
    args = get_args()
    play(args)
