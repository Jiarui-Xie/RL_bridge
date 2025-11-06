# 🌉 Bridge Crossing Task - Modifications Summary
# 过桥任务 - 修改总结

This document describes the modifications made to the original Sirius RL Gym codebase to implement a bridge crossing task with curriculum learning.

本文档描述了对原始 Sirius RL Gym 代码库的修改，以实现带有课程学习的过桥任务。

## 📋 Overview | 概述

The robot is trained to cross a bridge made of pillars with variable gaps. The task uses curriculum learning to progressively increase difficulty based on performance.

机器人被训练穿越由可变间隙柱子组成的桥梁。任务使用课程学习根据性能逐步增加难度。

## 🔧 Key Changes | 主要修改

### 1. Environment Configuration | 环境配置 (`sirius_flat_config.py`)

#### Observation Space | 观测空间
- **Increased from 45 to 61 dimensions | 从45维增加到61维**
- Added 16 values representing pillar corner coordinates (8 corners × 2 coordinates: x, z)
- 新增16个值表示柱子角点坐标（8个角点 × 2个坐标：x, z）
- Structure | 结构: `[ang_vel(3), gravity(3), commands(3), joint_pos(12), joint_vel(12), actions(12), pillar_corners(16)]`

#### Command Configuration | 指令配置
- **Forward-only movement | 仅前进运动**: `lin_vel_x=[0.5, 0.5]`, `lin_vel_y=[0.0, 0.0]`, `ang_vel_yaw=[0.0, 0.0]`
- Robot only moves forward without lateral or rotational movement
- 机器人仅向前移动，无横向或旋转运动

#### Terrain Configuration | 地形配置
```python
mesh_type = 'bridge'              # 桥梁类型
env_spacing = 7.0                 # 环境间距（米）
pillar_gap_range = [0.05, 0.15]   # 柱子间隙范围 5-15cm
curriculum = True                 # 启用课程学习
```

#### Reward Scales | 奖励权重
- `base_height_target`: 0.445m → **1.445m** (elevated bridge height | 抬高的桥梁高度)
- `tracking_lin_vel`: 1.0 → **0.3** (reduced for stability | 降低以提高稳定性)
- `lateral_deviation`: **-10.0** (new, penalize deviation from centerline | 新增，惩罚偏离中心线)
- `forward_progress`: **+2.0** (new, reward forward movement | 新增，奖励前进运动)

#### Termination | 终止条件
- Added height check: terminate if `robot_height < 1.0m`
- 新增高度检查：当机器人高度 < 1.0m 时终止

### 2. Environment Implementation | 环境实现 (`sirius_joystick.py`)

#### Bridge Terrain Creation | 桥梁地形创建
```python
def _create_bridge_pillars(self):
    """使用三角网格创建桥梁（静态几何体）"""
```
- Uses `gym.add_triangle_mesh()` to create static pillar geometry
- 使用 `gym.add_triangle_mesh()` 创建静态柱子几何体
- Each pillar: 8 vertices + 12 triangles | 每个柱子：8个顶点 + 12个三角形
- Pillars are NOT actors (won't affect `root_states`) | 柱子不是actor（不影响 `root_states`）
- All pillars merged into single mesh for performance | 所有柱子合并为单个网格以提高性能

#### Pillar Layout Generation | 柱子布局生成
```python
def _generate_pillar_layout(self, difficulty=0.0):
    """根据难度生成柱子位置 (0.0-1.0)"""
```
- **Start pillar | 起始柱**: 1.0m × 1.0m × 1.0m
- **Middle pillars | 中间柱** (×10): 0.25m × 1.0m × 1.0m
- **End pillar | 结束柱**: 1.0m × 1.0m × 1.0m
- Gap size interpolated | 间隙大小插值: `min_gap + difficulty × (max_gap - min_gap)`

#### Curriculum Learning | 课程学习
```python
def _update_terrain_curriculum(self, env_ids):
    """根据性能更新柱子难度"""
```
- **Increase difficulty | 增加难度** (+0.1): if robot travels > 2m | 当机器人行进 > 2m
- **Decrease difficulty | 降低难度** (-0.05): if robot fails quickly (< 1m) | 当机器人快速失败 (< 1m)
- Difficulty clamped to [0.0, 1.0] | 难度限制在 [0.0, 1.0]
- Each environment has independent difficulty level | 每个环境有独立的难度等级

#### Pillar Visibility System | 柱子可见性系统
```python
def _get_front_pillar_corners(self):
    """获取机器人前方2个柱子的顶部角点"""
```
- Robot only observes pillars ahead of its position | 机器人仅观察其位置前方的柱子
- Visibility check | 可见性检查: `robot_x < pillar_front_edge`
- Returns 16 values in robot's local frame (x, z coordinates) | 返回机器人局部坐标系中的16个值（x, z坐标）
- Batch processing for GPU efficiency | 批处理以提高GPU效率

#### New Reward Functions | 新增奖励函数
```python
def _reward_lateral_deviation(self):
    """惩罚偏离中心线 (y=0)"""
    return torch.square(self.root_states[:, 1])

def _reward_forward_progress(self):
    """奖励x方向的前进运动"""
    return self.base_lin_vel[:, 0]
```

### 3. Training Script | 训练脚本 (`train_bridge.py`)

```python
# 从 train_bridge_visual.py 重命名
# 自动在可视化和无头模式之间切换
num_envs = 100 if not args.headless else 4096
```

### 4. Multi-Environment Setup | 多环境设置

- **Visualization mode | 可视化模式**: 100 environments | 100个环境
- **Headless training | 无头训练**: 4096 environments | 4096个环境
- **Environment spacing | 环境间距**: 7m (prevents interference | 防止干扰)
- **Grid layout | 网格布局**: `sqrt(num_envs)` rows/columns | 行/列
- **Visualization limit | 可视化限制**: First 20 environments only (performance optimization | 仅前20个环境，性能优化)

## ⚙️ Technical Details | 技术细节

### Observation Calculation Optimization | 观测计算优化
- Batch coordinate transformation (4 corners at once) | 批量坐标转换（一次4个角点）
- Early exit when 2 pillars found | 找到2个柱子时提前退出
- GPU-accelerated quaternion rotation | GPU加速的四元数旋转

### Pillar Geometry | 柱子几何形状
- **Large pillars | 大柱子** (start/end | 起始/结束): 1.0m × 1.0m × 1.0m
- **Small pillars | 小柱子** (middle | 中间): 0.25m × 1.0m × 1.0m
- **Height | 高度**: 1.0m above ground | 离地1.0m
- **Material | 材质**: friction=0.8, restitution=0.1 | 摩擦系数=0.8，恢复系数=0.1

### Performance Metrics | 性能指标
- **Triangle mesh | 三角网格**: ~9,600 triangles for 100 envs | 100个环境约9,600个三角形 (12 pillars × 12 triangles × 100 envs)
- **Memory efficient | 内存高效**: Single mesh vs. individual actors | 单个网格 vs. 独立actor
- **Collision detection | 碰撞检测**: O(1) with spatial hashing | 使用空间哈希的O(1)复杂度

## 📁 File Structure | 文件结构

```
legged_gym/envs/sirius_diff_release/
├── sirius_flat_config.py      # 配置文件 (61维观测, 奖励, 地形)
├── sirius_joystick.py          # 环境实现
└── __init__.py

train_bridge.py                 # 训练脚本
```

## 🚀 Usage | 使用方法

```bash
# 训练 (无头模式, 4096个环境)
python train_bridge.py --task=sirius --headless

# 可视化 (100个环境)
python train_bridge.py --task=sirius

# 运行训练好的策略
python legged_gym/scripts/play.py --task=sirius
```

## 💡 Key Insights | 关键见解

1. **Curriculum Learning | 课程学习**: Difficulty adapts per-environment based on travel distance | 难度根据行进距离按环境自适应
2. **Visibility-based Observation | 基于可见性的观测**: Robot only sees pillars ahead (more realistic) | 机器人仅看到前方柱子（更真实）
3. **Static Geometry | 静态几何体**: Triangle mesh avoids actor overhead | 三角网格避免actor开销
4. **Batch Processing | 批处理**: GPU-optimized observation calculation | GPU优化的观测计算
5. **Height-based Task | 基于高度的任务**: Elevated bridge (1.445m target) vs. ground locomotion (0.445m) | 高架桥梁（1.445m目标）vs. 地面运动（0.445m）

## 📊 Differences from Original Fork | 与原版本的差异

| Aspect 方面 | Original 原版 | Modified 修改后 |
|--------|----------|----------|
| Observation dim 观测维度 | 45 | 61 (+16 pillar corners 柱子角点) |
| Terrain 地形 | Flat/heightfield 平地/高度场 | Bridge with pillars 带柱子的桥梁 |
| Movement 运动 | Omnidirectional 全向 | Forward-only 仅前进 |
| Base height target 基础高度目标 | 0.445m | 1.445m |
| Curriculum 课程 | Terrain-based 基于地形 | Gap-based per-env 基于间隙的每环境 |
| Termination 终止 | Contact-based 基于接触 | Contact + height < 1m 接触+高度<1m |
| Rewards 奖励 | Standard locomotion 标准运动 | + lateral_deviation, forward_progress 横向偏差,前进进度 |

## 📝 Notes | 注意事项

- Pillar layouts regenerate on environment reset (curriculum-based)
- 柱子布局在环境重置时重新生成（基于课程）
- Visualization shows green (1st pillar) and blue (2nd pillar) markers
- 可视化显示绿色（第1个柱子）和蓝色（第2个柱子）标记
- Each environment maintains independent difficulty level
- 每个环境维护独立的难度等级
- Triangle mesh is static (cannot be modified after creation)
- 三角网格是静态的（创建后无法修改）
