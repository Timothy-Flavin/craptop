# Multi-Agent Coverage Environment

A high-performance batched multi-agent environment built with C++ (pybind11) and OpenMP for fast parallel simulation of agents exploring a 32×32 grid world with configurable danger maps.

## Demo

![Multi-Agent Coverage Demo](demo.gif)

## Features

- **High-Performance**: ~11.5k FPS for single environment, ~134k FPS for 16 parallel environments
- **Batched Simulation**: Run multiple independent environments efficiently in parallel
- **Zero-Copy Memory**: Direct memory sharing between C++ backend and PyTorch tensors
- **Gymnasium Compatible**: Standard `gym.vector.VectorEnv` interface
- **Custom Maps**: Load PNG/JPG/BMP or raw binary danger maps; auto-conversion built in
- **Gravity-Based Attractions**: Query attraction vectors towards map features for each agent
- **PyGame Visualization**: Real-time rendering of environment state with fog-of-war

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
- Pillow (for PNG map conversion)
- OpenGL-compatible system (for rendering)
- GCC/Clang with OpenMP support

## Maps

The environment supports two map inputs per environment:

| Argument | Purpose |
|---|---|
| `maps` | Ground-truth danger map — what the environment actually uses for movement penalties and rewards |
| `expected_maps` | Prior belief map (e.g. satellite imagery) — used as the agents' initial expected danger before any exploration |

### Map Format

Maps are stored as **raw binary float32 files** (`.bin`) containing 1024 values (32×32 grid, row-major) in the range **`[-1.0, 1.0]`**:

- `-1.0` → completely safe
- ` 0.0` → neutral
- `+1.0` → maximum danger

### PNG / Image Maps

Any PNG, JPG, or BMP image can be passed directly — the wrapper auto-converts it:

1. Converts to grayscale
2. Resizes to 32×32 with Lanczos resampling
3. Normalizes pixel values from `[0, 255]` → `[-1.0, 1.0]`
4. Saves a `.bin` sidecar file next to the image

Light pixels (`255`) map to `+1.0` (danger); dark pixels (`0`) map to `-1.0` (safe).

### Provided Example Maps

- `map0.png` — ground-truth danger map used when no `maps` argument is provided in examples
- `expected_map0.png` — prior belief map used as `expected_maps` in examples

### Converting Maps Manually

```python
from env_wrapper import convert_map

# Convert a PNG to a .bin file (saved alongside the image)
bin_path = convert_map("my_map.png")            # -> "my_map.bin"
bin_path = convert_map("my_map.png", "out.bin") # explicit output path
```

Or use the standalone script:

```bash
python map_converter.py
# Enter path to input PNG: map0.png
# Enter path for output .bin: map0.bin
```

### Creating Maps Programmatically

```python
import numpy as np

# 32x32 map: danger concentrated in a circle in the center
y, x = np.mgrid[0:32, 0:32]
dist = np.sqrt((y - 16)**2 + (x - 16)**2)
danger = np.clip(1.0 - dist / 16.0, -1.0, 1.0).astype(np.float32)
danger.flatten().tofile("circle_danger.bin")
```

## API Reference

### `BatchedGridEnv`

High-level gymnasium-compatible wrapper around the C++ environment.

#### Constructor

```python
from env_wrapper import BatchedGridEnv, FeatureType

env = BatchedGridEnv(
    num_envs=16,              # Number of parallel environments
    n_agents=4,               # Agents per environment (fixed at 4 in C++ backend)
    map_size=32,              # Grid size (fixed at 32x32)
    device='cpu',             # PyTorch device ('cpu' or 'cuda')
    render_mode=None,         # 'human' for pygame window, None for headless
    seed=42,                  # Random seed for procedural map generation
    communication_prob=-1.0,  # Probability [0,1] of radio updates; -1 disables
    maps=None,                # str path or list of str paths to ground-truth maps
    expected_maps=None,       # str path or list of str paths to prior belief maps
)
```

**Map arguments** accept:
- `None` — procedural sine/cosine map is generated per environment
- `"map0.png"` — same image used for all `num_envs` environments (auto-converted)
- `"map0.bin"` — same binary file used for all environments
- `["map0.bin", "map1.bin", ...]` — one file per environment (list length must equal `num_envs`)

#### Methods

##### `reset(seed=None, options=None)`
Reset all environments and return observations.

```python
obs, info = env.reset()
# obs: torch.Tensor of shape (num_envs, stride) where stride = 15400
```

##### `step(actions)`
Execute actions and return observations, rewards, and terminal flags.

```python
actions = np.random.uniform(-1, 1, (num_envs, n_agents, 2))  # or torch.Tensor
obs, rewards, terminated, truncated, info = env.step(actions)

# obs:        torch.Tensor (num_envs, stride)
# rewards:    torch.Tensor (num_envs, n_agents)
# terminated: torch.Tensor (num_envs,) bool — True when all cells discovered
# truncated:  torch.Tensor (num_envs,) bool — always False (no time limit)
```

Environments that terminate are **automatically reset** at the start of their next step.

##### `get_gravity_attractions(feature_type, agent_mask=None, pow=2, normalize=False)`
Compute gravity attraction vectors for each agent towards cells of a given feature map.

The gravity force from each cell is: $\vec{F} = \text{mass} \cdot \hat{r} / r^{pow}$, summed over all cells.

```python
from env_wrapper import FeatureType

gravity = env.get_gravity_attractions(
    feature_type=FeatureType.GLOBAL_UNDISCOVERED,
    agent_mask=None,   # None = all agents; or np.array([True, True, False, False])
    pow=2,             # Distance power exponent (1 = linear falloff, 2 = quadratic)
    normalize=False,   # If True, scale output so max vector norm = 1.0
)
# Returns torch.Tensor of shape (num_envs, n_agents, 2) with (dy, dx) per agent
```

**Feature Types:**

| Feature Type | Description |
|---|---|
| `FeatureType.EXPECTED_DANGER` | Prior belief danger map (global, same for all agents) |
| `FeatureType.ACTUAL_DANGER` | True ground-truth danger map (global) |
| `FeatureType.OBSERVED_DANGER` | Per-agent observed danger (updated as cells are visited) |
| `FeatureType.OBS` | Per-agent binary observation mask (1 = cell has been seen) |
| `FeatureType.EXPECTED_OBS` | Per-agent expected observation map |
| `FeatureType.GLOBAL_DISCOVERED` | Global binary discovery map (union of all agents' obs) |
| `FeatureType.GLOBAL_UNDISCOVERED` | Inverse of global discovery (attracts toward unseen cells) |
| `FeatureType.OBS_UNDISCOVERED` | Per-agent undiscovered cells |
| `FeatureType.EXPECTED_OBS_UNDISCOVERED` | Per-agent expected undiscovered cells |
| `FeatureType.OTHER_AGENTS` | Gravity from current positions of other agents |
| `FeatureType.OTHER_AGENTS_LAST_KNOWN` | Gravity from last known positions of other agents |

**Agent Mask:**
```python
# Compute gravity only for the first two agents; others get zero vectors
mask = np.array([True, True, False, False])
gravity = env.get_gravity_attractions(FeatureType.GLOBAL_UNDISCOVERED, agent_mask=mask)
```

##### `render()`
Render the first environment to a pygame window. Called automatically each step when `render_mode='human'`.

The window shows:
- **Black cells**: undiscovered (fog of war)
- **Green cells**: discovered, safe (`danger ≈ -1.0`)
- **Yellow cells**: discovered, neutral danger (`danger ≈ 0.0`)
- **Red cells**: discovered, high danger (`danger ≈ +1.0`)
- **Blue circles**: agent positions with translucent 7×7 view-range boxes

```python
env = BatchedGridEnv(num_envs=4, render_mode='human')
obs, _ = env.reset()

try:
    while True:
        actions = np.random.uniform(-1, 1, (4, 4, 2))
        env.step(actions)  # render() called automatically
except KeyboardInterrupt:
    env.close()
```

##### `close()`
Close the pygame window and release resources.

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
    actions = np.random.uniform(-1, 1, (8, 4, 2))
    obs, rewards, terminated, truncated, info = env.step(actions)
    print(f"Step {step}, Rewards: {rewards}")

env.close()
```

### With Custom Maps

```python
from env_wrapper import BatchedGridEnv

# Same map for all envs (PNG auto-converted to .bin on first run)
env = BatchedGridEnv(
    num_envs=8,
    maps="map0.png",
    expected_maps="expected_map0.png",
)

# Different maps per env
env = BatchedGridEnv(
    num_envs=2,
    maps=["map0.bin", "map1.bin"],
    expected_maps=["expected_map0.bin", "expected_map1.bin"],
)
```

### Gravity-Based Navigation

```python
from env_wrapper import BatchedGridEnv, FeatureType
import numpy as np

env = BatchedGridEnv(num_envs=16, maps="map0.png", expected_maps="expected_map0.png")
obs, _ = env.reset()

for step in range(1000):
    # Pull toward undiscovered areas, away from danger and other agents
    toward_unknown = env.get_gravity_attractions(FeatureType.GLOBAL_UNDISCOVERED, normalize=True, pow=1)
    avoid_danger   = env.get_gravity_attractions(FeatureType.OBSERVED_DANGER,     normalize=True, pow=2)
    spread_out     = env.get_gravity_attractions(FeatureType.OTHER_AGENTS,         normalize=True, pow=1)

    actions = toward_unknown - avoid_danger - spread_out
    obs, rewards, terminated, truncated, info = env.step(actions)

env.close()
```

### Observation Space Layout

The observation is a flattened float32 tensor with the following structure (15400 values total):

```
Offset       | Size  | Content                  | Shape      | Range
-------------|-------|--------------------------|------------|----------
0            | 1024  | Expected Danger          | (32, 32)   | [-1, 1]
1024         | 1024  | Actual Danger            | (32, 32)   | [-1, 1]
2048         | 4096  | Observed Danger          | (4, 32, 32)| [-1, 1]
6144         | 4096  | Observation Mask         | (4, 32, 32)| {0, 1}
10240        | 8     | Agent Locations          | (4, 2)     | [0, 31] [y, x]
10248        | 4096  | Expected Obs             | (4, 32, 32)| [-1, 1]
14344        | 32    | Last Agent Locations     | (4, 2, 4)  | [0, 31]
14376        | 1024  | Global Discovered        | (32, 32)   | {0, 1}
```

Access slices:
```python
obs_np = obs[0].numpy()   # First environment
fms = 32 * 32             # FLAT_MAP_SIZE = 1024
n_agents = 4

expected_danger = obs_np[0:fms].reshape(32, 32)
actual_danger   = obs_np[fms:2*fms].reshape(32, 32)
obs_mask        = obs_np[2*fms:2*fms + n_agents*fms].reshape(n_agents, 32, 32)

agent_locs_offset = (2 + 2*n_agents) * fms   # = 10240
agent_locations   = obs_np[agent_locs_offset:agent_locs_offset + n_agents*2].reshape(n_agents, 2)

discovered = obs_np[-fms:].reshape(32, 32)   # Global discovered map
```

## Recording Demonstrations

Generate an animated GIF matching the pygame renderer:

```bash
python gif.py
```

The output `demo.gif` shows:
- **Black cells**: undiscovered (fog of war)
- **Green → Yellow → Red**: discovered cells from safe to dangerous
- **Translucent boxes**: each agent's 7×7 view range
- **Blue circles with white border**: agents

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
- **Danger Scale**: `[-1.0, 1.0]` — negative is safe, positive is dangerous

### Rewards
Agents receive `+1.0` reward (split equally) for each newly discovered cell. The episode terminates when all 1024 cells are discovered.

### Dynamics
- Action vectors are L2-normalized before being applied
- Effective speed per step: `SPEED × (1 - 0.8 × danger)` at the agent's current cell
- Agent positions are clamped to `[0, 31.99]` on both axes
- Terminated environments auto-reset at the start of their next `step()` call

## Building from Source

The extension requires a C++ compiler with OpenMP:

```bash
# Install build dependencies
pip install pybind11 setuptools build pillow

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
