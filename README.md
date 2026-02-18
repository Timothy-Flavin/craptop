# Multi-Agent Coverage Environment

A high-performance batched multi-agent environment built with C++ (pybind11) and OpenMP for fast parallel simulation of agents exploring a grid world.

## Demo

![Multi-Agent Coverage Demo](demo.gif)

## Features

- **High-Performance**: ~11.5k FPS for single environment, ~134k FPS for 16 parallel environments
- **Batched Simulation**: Run multiple independent environments efficiently in parallel
- **Zero-Copy Memory**: Direct memory sharing between C++ backend and PyTorch tensors
- **Gymnasium Compatible**: Standard gym.vector.VectorEnv interface
- **Gravity-Based Attractions**: Query attraction vectors towards map features for each agent
- **PyGame Visualization**: Real-time rendering of environment state

## Installation

### From Source

```bash
# Clone repository
git clone <repository>
cd craptop

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .
```

### Requirements

- Python 3.10+
- pybind11
- OpenGL-compatible system (for rendering)
- GCC/Clang with OpenMP support

## API Reference

### `BatchedGridEnv`

High-level gymnasium-compatible wrapper around the C++ environment.

#### Constructor

```python
from env_wrapper import BatchedGridEnv, FeatureType

env = BatchedGridEnv(
    num_envs=16,              # Number of parallel environments
    n_agents=4,               # Agents per environment
    map_size=32,              # Grid size (32x32)
    device='cpu',             # PyTorch device
    render_mode=None          # 'human' for rendering, None for headless
)
```

#### Methods

##### `reset(seed=None, options=None)`
Reset all environments and return observations.

```python
obs, info = env.reset()
# obs shape: (num_envs, stride) where stride ≈ 15400 floats
```

##### `step(actions)`
Execute actions and return observations, rewards, and terminal flags.

```python
actions = np.random.uniform(-1, 1, (num_envs, n_agents, 2))
obs, rewards, terminated, truncated, info = env.step(actions)

# obs shape: (num_envs, stride)
# rewards shape: (num_envs, n_agents)
# terminated, truncated: (num_envs,) bool arrays
```

##### `get_gravity_attractions(feature_type, agent_mask=None, pow=2)`
Compute gravity attraction vectors towards a map feature.

```python
from env_wrapper import FeatureType

# Get attractions towards discovered areas
gravity = env.get_gravity_attractions(
    feature_type=FeatureType.GLOBAL_DISCOVERED,
    agent_mask=None,  # None = all agents
    pow=2             # Power parameter
)
# Returns torch.Tensor of shape (num_envs, n_agents, 2) with (gx, gy)
```

**Feature Types:**
- `FeatureType.EXPECTED_DANGER` - Expected danger map (global, all agents see same)
- `FeatureType.ACTUAL_DANGER` - True danger map (global)
- `FeatureType.OBSERVED_DANGER` - Per-agent observed danger map
- `FeatureType.OBS` - Per-agent observation mask (whether cell was seen)
- `FeatureType.EXPECTED_OBS` - Per-agent expected observation map
- `FeatureType.GLOBAL_DISCOVERED` - Global discovery map (shared across agents)

**Agent Mask:**
```python
# Only get attractions for first 2 agents
mask = np.array([True, True, False, False])
gravity = env.get_gravity_attractions(FeatureType.GLOBAL_DISCOVERED, agent_mask=mask)
# Masked agents have zero gravity vectors
```

##### `render()`
Render current state to screen (only if `render_mode='human'`).

```python
env = BatchedGridEnv(num_envs=4, render_mode='human')
obs, _ = env.reset()

while True:
    actions = np.random.uniform(-1, 1, (4, 4, 2))
    obs, r, term, trunc, info = env.step(actions)
    env.render()  # Called automatically in step() if render_mode='human'
```

##### `close()`
Clean up resources.

```python
env.close()
```

## Usage Examples

### Basic Loop

```python
import numpy as np
from env_wrapper import BatchedGridEnv

env = BatchedGridEnv(num_envs=8, n_agents=4)
obs, _ = env.reset()

for step in range(1000):
    # Random policy
    actions = np.random.uniform(-1, 1, (8, 4, 2))
    
    obs, rewards, terminated, truncated, info = env.step(actions)
    
    # Access observations or rewards
    print(f"Step {step}, Rewards: {rewards}")

env.close()
```

### With Rendering

```python
env = BatchedGridEnv(num_envs=1, render_mode='human')
obs, _ = env.reset()

try:
    for _ in range(500):
        actions = np.random.uniform(-1, 1, (1, 4, 2))
        env.step(actions)
        # Render is called automatically
except KeyboardInterrupt:
    env.close()
```

### Gravity-Based Navigation

```python
from env_wrapper import BatchedGridEnv, FeatureType

env = BatchedGridEnv(num_envs=4, n_agents=4)
obs, _ = env.reset()

for step in range(100):
    # Get gravity towards global discovered areas
    gravity = env.get_gravity_attractions(FeatureType.GLOBAL_DISCOVERED, pow=2)
    
    # Move toward high gravity (explored areas)
    actions = gravity.numpy() * 0.5
    actions = np.clip(actions, -1, 1)
    
    obs, rewards, _, _, _ = env.step(actions)
    print(f"Step {step}, Discovery rewards: {rewards[0]}")

env.close()
```

### Observation Space Layout

The observation is a flattened array with the following structure:

```
Index Range          | Content                 | Shape
0 - 1024             | Expected Danger         | (32, 32)
1024 - 2048          | Actual Danger           | (32, 32)
2048 - 6144          | Observed Danger         | (4, 32, 32)
6144 - 10240         | Observation Mask        | (4, 32, 32)
10240 - 10248        | Agent Locations         | (4, 2) - [y, x] per agent
10248 - 14344        | Expected Obs            | (4, 32, 32)
14344 - 14376        | Last Agent Locations    | (4, 2, 4) - history
14376 - 15400        | Global Discovered       | (32, 32)
```

Access slices:
```python
import numpy as np

obs = obs[0].numpy()  # Get first env's observation
fms = 32 * 32  # FLAT_MAP_SIZE

expected_danger = obs[0:fms].reshape(32, 32)
actual_danger = obs[fms:2*fms].reshape(32, 32)
agent_locs_offset = (2 + 2*4) * fms
agent_locations = obs[agent_locs_offset:agent_locs_offset+8].reshape(4, 2)
discovered = obs[-fms:].reshape(32, 32)
```

## Recording Demonstrations

Generate an animated GIF of the environment:

```bash
python gif.py
```

This creates `demo.gif` showing agents exploring a 32x32 grid. The GIF shows:
- Green cells: safe, explored areas
- Red cells: danger, explored areas  
- Dark gray: unexplored cells
- Light blue dots: agents

## Performance

Benchmark results (on typical Linux machine with OpenMP):

| Config | FPS |
|--------|-----|
| 1 env, 10k frames | ~11,500 |
| 16 envs, 10k frames | ~134,000 (scaled) |

## Environment Details

### State
- **Map Size**: 32×32 fixed grid
- **Agents per Env**: 4 fixed
- **Agent Speed**: 0.5 cells/step (reduced in danger zones)
- **View Range**: 3 cells (7×7 view window)

### Rewards
Agents receive +1.0 reward (divided equally) for discovering new cells. Total discoverable: 1024 cells.

### Dynamics
- Agent movement is normalized and clamped to map bounds
- Movement speed is reduced by `1 - 0.8 * danger_level` in each cell
- Agents cannot see beyond their 7×7view window

## Building from Source

The extension requires a C++ compiler with OpenMP:

```bash
# Install build dependencies
pip install pybind11 setuptools build

# Build in-place for testing
python setup.py build_ext --inplace

# Or use modern build system
python -m build
```

## Publishing to PyPI

```bash
# Local build and publish
export PYPI_API_TOKEN="your-token-here"
./build_and_publish.sh

# Or via GitHub Actions (requires PYPI_API_TOKEN secret):
git tag v0.1.0
git push origin v0.1.0
```

## License

MIT
