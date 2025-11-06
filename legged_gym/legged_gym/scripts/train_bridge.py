#!/usr/bin/env python3
"""
Bridge gap training with visualization
Shows stepping zones and robot movement in real-time
"""

import os
import sys

# Add legged_gym to path
sys.path.append('/home/lumi/Sirius_RL_Gym/legged_gym')

import numpy as np
from datetime import datetime
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def main():
    os.chdir('/home/lumi/Sirius_RL_Gym/legged_gym')
    
    args = get_args()
    args.task = 'sirius'
    
    # Set num_envs based on headless mode
    if args.headless:
        args.num_envs = 4096
        mode = "HEADLESS"
        env_count = "4096 parallel environments"
    else:
        args.num_envs = 100
        mode = "VISUALIZATION"
        env_count = "100 parallel environments"
    
    print("="*60)
    print(f"Sirius Pillar Training - {mode} MODE")
    print("="*60)
    print("Features:")
    print("  🏗️  立柱环境 = 起始立柱 + 10个中间立柱 + 终点立柱")
    print(f"  🤖 {env_count}")
    print("  📏 随机间距 = 5-15cm")
    print("  🎯 目标 = 从起始立柱跳到终点立柱")
    print("  📐 环境间距 = 7m (避免碰撞)")
    print("="*60)
    if not args.headless:
        print("Press 'V' to toggle viewer sync")
        print("Press 'ESC' to exit")
        print("="*60)
    
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    main()
