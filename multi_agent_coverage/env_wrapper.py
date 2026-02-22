import gymnasium as gym
import numpy as np
import torch
from . import _core as _core_partial
from . import _core_global as _core_global_mod
from gymnasium import spaces
import pygame
import ctypes
import os

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
AGENT_COLORS = [
    (0, 100, 255),
    (255, 140, 0),
    (0, 210, 100),
    (220, 50, 220),
]
# Re-export FeatureType enum for convenience (same in both modules)
FeatureType = _core_partial.FeatureType


def convert_map(image_path, output_path=None):
    """
    Converts an image map (PNG/JPG) to the raw binary float32 format expected by the C++ backend.

    Args:
        image_path: Path to source image.
        output_path: Path to save binary file. If None, defaults to image_path with extension replaced by .bin

    Returns:
        The path to the binary file.
    """
    if not PIL_AVAILABLE:
        raise ImportError(
            "Pillow is required for map conversion. Install with `pip install Pillow`."
        )

    if output_path is None:
        base, _ = os.path.splitext(image_path)
        output_path = base + ".bin"

    # 1. Load image and convert to Grayscale ('L')
    img = Image.open(image_path).convert("L")

    # 2. Resize to 32x32 using Lanczos for high-quality downsampling
    img = img.resize((32, 32), Image.Resampling.LANCZOS)

    # 3. Convert to numpy array and normalize to [-1.0, 1.0]
    # PNG pixels are 0-255; dividing by 255.0 then scaling to [-1, 1]
    grid = np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0

    # 4. Flatten to a 1D array
    grid_1d = grid.flatten()

    # 5. Save as raw binary file
    grid_1d.tofile(output_path)
    print(f"converted map {image_path} to {output_path}")
    return output_path


def _process_map_list(maps_arg, num_envs, arg_name="maps"):
    """
    Helper to process a map argument (string, list, or None) into a list of binary file paths.
    Handles PNG->BIN conversion if needed.
    """
    if maps_arg is None:
        return []

    processed_maps = []
    if isinstance(maps_arg, str):
        map_list = [maps_arg] * num_envs
    elif isinstance(maps_arg, list):
        if len(maps_arg) != num_envs:
            raise ValueError(
                f"Length of {arg_name} list ({len(maps_arg)}) must match num_envs ({num_envs})"
            )
        map_list = maps_arg
    else:
        raise ValueError(f"{arg_name} must be a string path or a list of string paths")

    for m_path in map_list:
        if not os.path.exists(m_path):
            raise FileNotFoundError(f"Map file not found: {m_path}")

        _, ext = os.path.splitext(m_path)
        if ext.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
            # Auto-convert image to bin
            if not PIL_AVAILABLE:
                raise ImportError(
                    "Pillow is required for map conversion. Install with `pip install Pillow`."
                )
            bin_path = convert_map(m_path)
            processed_maps.append(bin_path)
        else:
            processed_maps.append(m_path)

    return processed_maps


class BatchedGridEnv(gym.vector.VectorEnv):
    def __init__(
        self,
        num_envs,
        n_agents=4,
        map_size=32,
        device="cpu",
        render_mode=None,
        seed=42,
        communication_prob=-1.0,
        maps=None,
        expected_maps=None,
        global_comms=False,
        reset_automatically=True,
        death_penalty=-20.0,
        share_danger=False,
    ):
        """
        maps: Path or list of paths to ground truth danger maps.
        expected_maps: Path or list of paths to prior belief maps (e.g. satellite data).
        global_comms: If True, use the global-communication backend (shared obs,
                      no expected_obs/last_agent_locations, ~2.4x smaller state).
        reset_automatically: If True (default), terminated environments are automatically
            reset at the start of their next step(). If False, terminated environments
            hold their last state and continue returning terminated=True and zero rewards
            until reset_env(i) or reset() is called explicitly.
        death_penalty: Reward applied to an agent the step it dies (default -20.0).
            Set to 0.0 to disable the penalty while keeping the death mechanic.
        share_danger: If True, when a radio communication fires between agents i and j,
            agent i also receives agent j's currently-observed danger map for j's view
            window (7×7 tiles around j). Has no effect in global_comms mode or when
            communication_prob <= 0.
        """
        self.num_envs = num_envs
        self.communication_prob = communication_prob
        self.global_comms = global_comms
        self.reset_automatically = reset_automatically
        self.death_penalty = death_penalty
        self.share_danger = share_danger
        self.n_agents = n_agents
        self.map_size = map_size
        self.device = device
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.cell_size = 20  # Pixels per grid cell

        # 1. Process Map Arguments
        processed_maps = _process_map_list(maps, num_envs, "maps")
        processed_expected_maps = _process_map_list(
            expected_maps, num_envs, "expected_maps"
        )

        # 2. Initialize C++ Backend (pick module based on mode)
        backend = _core_global_mod if global_comms else _core_partial
        self.env = backend.BatchedEnvironment(
            num_envs,
            seed,
            processed_maps,
            processed_expected_maps,
            reset_automatically,
            death_penalty,
        )

        # 3. Define Spaces
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_agents, 2), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_envs, n_agents, 2), dtype=np.float32
        )

        stride = self.env.get_stride()
        self.single_observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(stride,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_envs, stride), dtype=np.float32
        )

        super().__init__()

        # 4. Zero-Copy Memory Mapping
        ptr, size_bytes = self.env.get_memory_view()
        float_count = size_bytes // 4
        # Create a ctypes array view of the C++ vector memory
        ctypes_array = (ctypes.c_float * float_count).from_address(ptr)
        # Convert to numpy array (shares memory with ctypes wrapper)
        np_array = np.ctypeslib.as_array(ctypes_array)
        # Create torch tensor from numpy view
        self._raw_tensor = torch.from_numpy(np_array)
        self.state_tensor = self._raw_tensor.view(num_envs, stride)

        # Pre-calculate slice indices for faster rendering
        fms = self.env.get_flat_map_size()

        if global_comms:
            # Global layout: expected_danger | actual_danger | observed_danger | obs | agent_locs | recency[N] | agents_alive
            self.sl_actual_danger = slice(fms, 2 * fms)
            agent_locs_offset = 4 * fms
            self.sl_agent_locs = slice(
                agent_locs_offset, agent_locs_offset + n_agents * 2
            )
            # obs (= global_discovered) is at offset 3*fms
            self.sl_global_disc = slice(3 * fms, 4 * fms)
            # agents_alive: after recency
            recency_offset = agent_locs_offset + n_agents * 2
            agents_alive_offset = recency_offset + n_agents * fms
            self.sl_agents_alive = slice(
                agents_alive_offset, agents_alive_offset + n_agents
            )
        else:
            # Partial layout: expected | actual | observed_danger[N] | obs[N] | agent_locs | expected_obs[N] | last_locs | global_disc | recency[N] | agents_alive | agents_last_alive[N*N]
            self.sl_actual_danger = slice(fms, 2 * fms)
            agent_locs_offset = (2 + 2 * n_agents) * fms
            self.sl_agent_locs = slice(
                agent_locs_offset, agent_locs_offset + n_agents * 2
            )
            # global_discovered starts after last_agent_locations
            gd_offset = (
                agent_locs_offset
                + n_agents * 2
                + n_agents * fms
                + n_agents * 2 * n_agents
            )
            self.sl_global_disc = slice(gd_offset, gd_offset + fms)
            # agents_alive + agents_last_alive: after recency
            recency_offset = gd_offset + fms
            agents_alive_offset = recency_offset + n_agents * fms
            self.sl_agents_alive = slice(
                agents_alive_offset, agents_alive_offset + n_agents
            )
            agents_last_alive_offset = agents_alive_offset + n_agents
            self.sl_agents_last_alive = slice(
                agents_last_alive_offset, agents_last_alive_offset + n_agents * n_agents
            )

    def reset(self, seed=None, options=None):
        self.env.reset()
        obs = (
            self.state_tensor
            if self.device == "cpu"
            else self.state_tensor.to(self.device)
        )
        return obs, {}

    def reset_env(self, env_idx: int):
        """Reset a single environment by index without affecting others.

        Args:
            env_idx: Index of the environment to reset (0 <= env_idx < num_envs).

        Returns:
            obs: The state tensor row for that environment (shape [stride,]),
                 as a view into the shared state buffer.
        """
        self.env.reset_single(env_idx)
        obs = self.state_tensor[env_idx]
        return obs if self.device == "cpu" else obs.to(self.device)

    def sync_termination(self):
        """Clear env_terminated flags and recompute undiscovered_remaining
        from the current obs array.  Use after injecting belief state into
        a global-comms imaginary env so that step() is no longer a no-op,
        without destroying the injected recency / obs / danger / locations.

        Only available on global_comms backends.
        """
        self.env.sync_termination()

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        flat_actions = actions.reshape(self.num_envs, -1).astype(np.float32)

        if self.global_comms:
            rewards, terminated_np = self.env.step(flat_actions)
        else:
            rewards, terminated_np = self.env.step(
                flat_actions, self.communication_prob, self.share_danger
            )

        obs = self.state_tensor
        rewards = torch.from_numpy(rewards)
        terminated = torch.from_numpy(terminated_np)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)

        if self.render_mode == "human":
            self.render()

        return obs, rewards, terminated, truncated, {}

    def get_gravity_attractions(
        self, feature_type, agent_mask=None, pow=2, normalize=False
    ):
        """
        Compute gravity attraction vectors for agents towards a feature.

        Args:
            feature_type: FeatureType enum value (EXPECTED_DANGER, OBSERVED_DANGER, etc.)
            agent_mask: Optional boolean array of shape (n_agents,) or None for all agents
            pow: Power parameter for gravity calculation (default 2)
            normalize: If True, normalize gravity vectors to max norm 1.0 (default False)

        Returns:
            torch.Tensor of shape (num_envs, n_agents, 2) with (gx, gy) for each agent
        """
        if agent_mask is not None:
            agent_mask = np.asarray(agent_mask, dtype=bool)

        gravity_array = self.env.get_gravity_attractions(
            agent_mask, feature_type, pow, normalize
        )
        return torch.from_numpy(gravity_array)

    def render(self):
        if self.screen is None:
            pygame.init()
            width = self.map_size * self.cell_size
            height = self.map_size * self.cell_size
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Batched Env (Idx 0)")
            self.clock = pygame.time.Clock()

        # Fetch Data for Env 0
        state = self.state_tensor[0].numpy()

        # Extract Layers
        danger_map = state[self.sl_actual_danger].reshape(self.map_size, self.map_size)
        discovered_map = state[self.sl_global_disc].reshape(
            self.map_size, self.map_size
        )
        agents = state[self.sl_agent_locs].reshape(self.n_agents, 2)  # [y, x]

        self.screen.fill((0, 0, 0))

        # 1. Draw Grid (Danger Map)
        for y in range(self.map_size):
            for x in range(self.map_size):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                # Check if discovered
                if discovered_map[y, x] > 0.5:
                    # Visible: Color based on danger (-1.0 to 1.0)
                    # -1.0 (safe) -> Green, 0.0 (neutral) -> Yellow, 1.0 (danger) -> Red
                    d = danger_map[y, x]
                    # Map [-1, 1] to [0, 1] for visualization
                    d_normalized = (d + 1.0) / 2.0
                    color = (
                        int(255 * d_normalized),  # R: 0 at -1, 255 at +1
                        int(255 * (1 - abs(d))),  # G: 255 at 0, 0 at extremes
                        int(255 * (1 - d_normalized)),  # B: 255 at -1, 0 at +1
                    )
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)  # Border
                else:
                    # Undiscovered: Black (Fog of War)
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)
                    # Optional: faint grid line to show it exists
                    pygame.draw.rect(self.screen, (20, 20, 20), rect, 1)

        # 2. Draw Agents & View Range
        # Create a transparent surface for view range
        view_surface = pygame.Surface(
            (self.map_size * self.cell_size, self.map_size * self.cell_size),
            pygame.SRCALPHA,
        )

        for i in range(self.n_agents):
            ay, ax = agents[i]

            # Convert grid coords to pixel coords
            # Add 0.5 to center in the tile
            px = (ax + 0.5) * self.cell_size
            py = (ay + 0.5) * self.cell_size

            # Draw View Range (Translucent White Box)
            # Range is 3, so box width is (3+1+3) = 7 tiles?
            # Logic in C++: [y-3, y+3], so 7x7 area
            vr = 3
            rect_size = (vr * 2 + 1) * self.cell_size
            view_rect = pygame.Rect(0, 0, rect_size, rect_size)
            view_rect.center = (px, py)

            # Draw translucent white (255, 255, 255, 30)
            pygame.draw.rect(view_surface, (255, 255, 255, 30), view_rect)

            # Draw Agent (Colored Circle)
            pygame.draw.circle(
                self.screen,
                AGENT_COLORS[i % len(AGENT_COLORS)],
                (int(px), int(py)),
                int(self.cell_size * 0.4),
            )
            # Agent Border (White)
            pygame.draw.circle(
                self.screen,
                (255, 255, 255),
                (int(px), int(py)),
                int(self.cell_size * 0.4),
                2,
            )

        # Blit the transparent surface onto main screen
        self.screen.blit(view_surface, (0, 0))

        pygame.display.flip()

        # Cap framerate for visualization
        self.clock.tick(30)

        # Handle Window Close Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit("Pygame window closed by user")

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# --- Quick Test ---
if __name__ == "__main__":
    import time

    # FPS test: 1 env, 10k frames
    print("=== FPS Test: 1 env, 10k frames ===")
    env1 = BatchedGridEnv(num_envs=1, render_mode=None)
    obs, _ = env1.reset()
    start = time.time()
    for _ in range(10000):
        actions = np.random.uniform(-1, 1, (1, 4, 2))
        env1.step(actions)
    elapsed1 = time.time() - start
    fps1 = 10000 / elapsed1
    print(f"Time: {elapsed1:.2f}s, FPS: {fps1:.1f}")
    env1.close()

    # FPS test: 16 envs, 10k frames
    print("\n=== FPS Test: 16 envs, 10k frames ===")
    env16 = BatchedGridEnv(num_envs=16, render_mode=None)
    obs, _ = env16.reset()
    start = time.time()
    for _ in range(10000):
        actions = np.random.uniform(-1, 1, (16, 4, 2))
        env16.step(actions)
    elapsed16 = time.time() - start
    fps16 = 10000 / elapsed16 * 16
    print(f"Time: {elapsed16:.2f}s, FPS: {fps16:.1f}")

    # Gravity attractions test
    print("\n=== Gravity Attractions Test ===")
    gravity_observed = env16.get_gravity_attractions(FeatureType.OBSERVED_DANGER, pow=2)
    print(f"Gravity shape (should be [16, 4, 2]): {gravity_observed.shape}")
    print(f"Sample gravity for env 0, agent 0: {gravity_observed[0, 0].numpy()}")

    # Test with agent mask (only first 2 agents)
    mask = np.array([True, True, False, False])
    gravity_masked = env16.get_gravity_attractions(
        FeatureType.GLOBAL_DISCOVERED, agent_mask=mask
    )
    print(f"Masked gravity shape: {gravity_masked.shape}")
    print(
        f"Masked gravity for disabled agents (should be ~0): {gravity_masked[0, 2:4].numpy()}"
    )

    env16.close()

    # Render test
    print("\n=== Render Test: 16 envs ===")
    env = BatchedGridEnv(
        num_envs=16,
        render_mode="human",
        maps="map0.png",
        expected_maps="expected_map0.png",
    )
    obs, _ = env.reset()

    # Simple random walk
    try:
        while True:
            obs_masked = env.get_gravity_attractions(
                FeatureType.GLOBAL_UNDISCOVERED, agent_mask=None, normalize=True, pow=1
            )
            danger_masked = env.get_gravity_attractions(
                FeatureType.OBSERVED_DANGER, agent_mask=None, normalize=True, pow=2
            )
            other_masked = env.get_gravity_attractions(
                FeatureType.OTHER_AGENTS, agent_mask=None, normalize=True, pow=1
            )

            print(
                f"Obs gravity sample: {obs_masked[0, 0].numpy()}, Danger gravity sample: {danger_masked[0, 0].numpy()}, Other agents gravity sample: {other_masked[0, 0].numpy()}"
            )
            actions = np.random.uniform(-1, 1, (16, 4, 2))
            obs, r, term, trunc, info = env.step(
                -danger_masked + obs_masked - other_masked + actions
            )
            if sum(r[0]) > 0:
                print(f"Rewards: {r[0]}")
    except KeyboardInterrupt:
        env.close()
