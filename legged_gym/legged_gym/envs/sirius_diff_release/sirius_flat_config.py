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
        pos = [0.0, 0.0, 1.40] # x,y,z [m] - spawn on top of start pillar
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,

            'FL_thigh_joint': 0.9,  # 略微增加以提高重心
            'RL_thigh_joint': 1.1,  # 后腿更弯曲以提供更好的推进力
            'FR_thigh_joint': 0.9,
            'RR_thigh_joint': 1.1,

            'FL_calf_joint': -1.6,  # 略微增加弯曲
            'RL_calf_joint': -1.6,
            'FR_calf_joint': -1.6,
            'RR_calf_joint': -1.6,
        }
            
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters: 使用官方Go1参数
        control_type = 'P'
        stiffness = {
            'hip': 100.0,    # Hip joints: p=100, d=5
            'thigh': 300.0,  # Thigh joints: p=300, d=8  
            'calf': 300.0    # Calf joints: p=300, d=8
        }
        damping = {
            'hip': 5.0,
            'thigh': 8.0,
            'calf': 8.0
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.4
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
            lin_vel_x = [0.3, 0.3]  # Slow, controlled forward movement for short gaps
            lin_vel_y = [0.0, 0.0]  # No lateral movement
            ang_vel_yaw = [0.0, 0.0]  # No rotation

    class domain_rand( LeggedRobotCfg.domain_rand):
        # Disable large base-mass randomization during debugging / early training to
        # avoid cases where very heavy robots learn to drag instead of stepping.
        randomize_base_mass = False
        # If re-enabled later, prefer a much smaller added mass range to reduce extremes
        added_mass_range = [-2., 2.]
        friction_range = [0., 1.5] # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
  
    class rewards( LeggedRobotCfg.rewards ):
        base_height_target = 1.35
        max_contact_force = 350
        only_positive_rewards = False  # Allow negative rewards for proper learning
        soft_dof_vel_limit = 1.0
        # 步态相关参数
        min_air_time = 0.7  # 降低最小腾空时间，允许更自然的步态
        max_air_time = 1.2   # 最大腾空时间
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 6.0  # PRIMARY: track target velocity
            tracking_ang_vel = 0.5
            ang_vel_xy = -0.02  # 降低pitch/roll角速度惩罚，允许自然晃动
            orientation = -1.0  # 大幅加强姿态约束，防止爬行
            feet_air_time = 2.0  # 大幅提高步态奖励，鼓励正常步幅
            base_height = -3.0  # 大幅加强高度约束，防止趴下
            posture = 0.8  # 适度提高姿态约束
            lateral_deviation = -1.0
            forward_progress = 0.1
            goal_reached = 10.0
            heading_alignment = -1.0
            knee_contact = -3.0  # 大幅加强，严禁膝盖接触地面
            foot_in_gap = -2.0  # 加强，防止脚落入缝隙
            action_rate = -0.04  # 大幅提高，惩罚高频动作变化
            dof_vel = -0.0004  # 提高5倍，惩罚快速关节运动
            dof_acc = -2.5e-7  # 提高，平滑动作
            stand_still = -0.5
            feet_contact_forces = -0.01
            stumble = -0.5
            hip_motion = -1.0  # 惩罚hip角度超出[0, 0.2]rad范围
            termination = 0
    
    class noise( LeggedRobotCfg.noise ):
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales( LeggedRobotCfg.noise.noise_scales ):
            dof_pos = 0.03
            dof_vel = 0.7  # Reduced from 1.5 to improve learning signal
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

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        learning_rate = 1.e-3  # 增大学习率从 1e-3 到 3e-3

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = "sirius_diff_release"
        load_run = -1
        max_iterations = 2000  # Increased from 1200 for better convergence
