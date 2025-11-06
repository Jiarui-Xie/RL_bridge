# Bridge Crossing Task - Modifications Summary

This document describes the modifications made to the original Sirius RL Gym codebase to implement a bridge crossing task with curriculum learning.

## Overview

The robot is trained to cross a bridge made of pillars with variable gaps. The task uses curriculum learning to progressively increase difficulty based on performance.

## Key Changes

### 1. Environment Configuration (`sirius_flat_config.py`)

#### Observation Space
- **Increased from 45 to 61 dimensions**
- Added 16 values representing pillar corner coordinates (8 corners × 2 coordinates: x, z)
- Structure: `[ang_vel(3), gravity(3), commands(3), joint_pos(12), joint_vel(12), actions(12), pillar_corners(16)]`

#### Command Configuration
- **Forward-only movement**: `lin_vel_x=[0.5, 0.5]`, `lin_vel_y=[0.0, 0.0]`, `ang_vel_yaw=[0.0, 0.0]`
- Robot only moves forward without lateral or rotational movement

#### Terrain Configuration
```python
mesh_type = 'bridge'
env_spacing = 7.0  # meters between environments
pillar_gap_range = [0.05, 0.15]  # 5-15cm gap range
curriculum = True
```

#### Reward Scales
- `base_height_target`: 0.445m → **1.445m** (elevated bridge height)
- `tracking_lin_vel`: 1.0 → **0.3** (reduced for stability)
- `lateral_deviation`: **-10.0** (new, penalize deviation from centerline)
- `forward_progress`: **+2.0** (new, reward forward movement)

#### Termination
- Added height check: terminate if `robot_height < 1.0m`

### 2. Environment Implementation (`sirius_joystick.py`)

#### Bridge Terrain Creation
```python
def _create_bridge_pillars(self):
    """Creates bridge using triangle mesh (static geometry)"""
```
- Uses `gym.add_triangle_mesh()` to create static pillar geometry
- Each pillar: 8 vertices + 12 triangles
- Pillars are NOT actors (won't affect `root_states`)
- All pillars merged into single mesh for performance

#### Pillar Layout Generation
```python
def _generate_pillar_layout(self, difficulty=0.0):
    """Generate pillar positions based on difficulty (0.0-1.0)"""
```
- **Start pillar**: 1.0m × 1.0m × 1.0m
- **Middle pillars** (×10): 0.25m × 1.0m × 1.0m
- **End pillar**: 1.0m × 1.0m × 1.0m
- Gap size interpolated: `min_gap + difficulty × (max_gap - min_gap)`

#### Curriculum Learning
```python
def _update_terrain_curriculum(self, env_ids):
    """Update pillar difficulty based on performance"""
```
- **Increase difficulty** (+0.1): if robot travels > 2m
- **Decrease difficulty** (-0.05): if robot fails quickly (< 1m)
- Difficulty clamped to [0.0, 1.0]
- Each environment has independent difficulty level

#### Pillar Visibility System
```python
def _get_front_pillar_corners(self):
    """Get top corners of 2 pillars ahead of robot"""
```
- Robot only observes pillars ahead of its position
- Visibility check: `robot_x < pillar_front_edge`
- Returns 16 values in robot's local frame (x, z coordinates)
- Batch processing for GPU efficiency

#### New Reward Functions
```python
def _reward_lateral_deviation(self):
    """Penalize deviation from centerline (y=0)"""
    return torch.square(self.root_states[:, 1])

def _reward_forward_progress(self):
    """Reward forward movement in x direction"""
    return self.base_lin_vel[:, 0]
```

### 3. Training Script (`train_bridge.py`)

```python
# Renamed from train_bridge_visual.py
# Automatically switches between visualization and headless mode
num_envs = 100 if not args.headless else 4096
```

### 4. Multi-Environment Setup

- **Visualization mode**: 100 environments
- **Headless training**: 4096 environments
- **Environment spacing**: 7m (prevents interference)
- **Grid layout**: `sqrt(num_envs)` rows/columns
- **Visualization limit**: First 20 environments only (performance optimization)

## Technical Details

### Observation Calculation Optimization
- Batch coordinate transformation (4 corners at once)
- Early exit when 2 pillars found
- GPU-accelerated quaternion rotation

### Pillar Geometry
- **Large pillars** (start/end): 1.0m × 1.0m × 1.0m
- **Small pillars** (middle): 0.25m × 1.0m × 1.0m
- **Height**: 1.0m above ground
- **Material**: friction=0.8, restitution=0.1

### Performance Metrics
- **Triangle mesh**: ~9,600 triangles for 100 envs (12 pillars × 12 triangles × 100 envs)
- **Memory efficient**: Single mesh vs. individual actors
- **Collision detection**: O(1) with spatial hashing

## File Structure

```
legged_gym/envs/sirius_diff_release/
├── sirius_flat_config.py      # Configuration (61-dim obs, rewards, terrain)
├── sirius_joystick.py          # Environment implementation
└── __init__.py

train_bridge.py                 # Training script
```

## Usage

```bash
# Training (headless, 4096 envs)
python train_bridge.py --task=sirius --headless

# Visualization (100 envs)
python train_bridge.py --task=sirius

# Play trained policy
python legged_gym/scripts/play.py --task=sirius
```

## Key Insights

1. **Curriculum Learning**: Difficulty adapts per-environment based on travel distance
2. **Visibility-based Observation**: Robot only sees pillars ahead (more realistic)
3. **Static Geometry**: Triangle mesh avoids actor overhead
4. **Batch Processing**: GPU-optimized observation calculation
5. **Height-based Task**: Elevated bridge (1.445m target) vs. ground locomotion (0.445m)

## Differences from Original Fork

| Aspect | Original | Modified |
|--------|----------|----------|
| Observation dim | 45 | 61 (+16 pillar corners) |
| Terrain | Flat/heightfield | Bridge with pillars |
| Movement | Omnidirectional | Forward-only |
| Base height target | 0.445m | 1.445m |
| Curriculum | Terrain-based | Gap-based per-env |
| Termination | Contact-based | Contact + height < 1m |
| Rewards | Standard locomotion | + lateral_deviation, forward_progress |

## Notes

- Pillar layouts regenerate on environment reset (curriculum-based)
- Visualization shows green (1st pillar) and blue (2nd pillar) markers
- Each environment maintains independent difficulty level
- Triangle mesh is static (cannot be modified after creation)
