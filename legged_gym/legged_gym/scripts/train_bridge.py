#!/usr/bin/env python3
"""
Bridge gap training with visualization
Shows stepping zones and robot movement in real-time
"""

import os
import sys
import glob



import numpy as np
from datetime import datetime
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def find_latest_model():
    """Find the latest model in logs directory"""
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        return None
    
    # Find all model files
    model_pattern = os.path.join(logs_dir, '*/model_*.pt')
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        return None
    
    # Sort by modification time, get the latest
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

def main():
    
    args = get_args()
    args.task = 'sirius'
    
    # Handle load_run parameter (model path)
    if hasattr(args, 'load_run') and args.load_run:
        print(f"🔄 Loading from: {args.load_run}")
    else:
        print("🎆 Starting fresh training")
    
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
    print("🎆 TRAINING")
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
    print("")
    print("📝 Usage:")
    print("  Training: python train_bridge.py --headless")
    print("="*60)
    
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    # Load model if specified
    if hasattr(args, 'load_run') and args.load_run:
        ppo_runner.load(args.load_run)
        print(f"✅ Loaded model from: {args.load_run}")
    
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    main()
