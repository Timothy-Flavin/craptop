import gymnasium as gym
import numpy as np
import torch
import multi_agent_coverage
from gymnasium import spaces
import pygame
import ctypes

# Re-export FeatureType enum for convenience
FeatureType = multi_agent_coverage.FeatureType

class BatchedGridEnv(gym.vector.VectorEnv):
    def __init__(self, num_envs, n_agents=4, map_size=32, device='cpu', render_mode=None):
        self.num_envs = num_envs
        self.n_agents = n_agents
        self.map_size = map_size
        self.device = device
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.cell_size = 20  # Pixels per grid cell
        
        # 1. Initialize C++ Backend
        self.env = multi_agent_coverage.BatchedEnvironment(num_envs)
        
        # 2. Define Spaces
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

        # 3. Zero-Copy Memory Mapping
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
        self.sl_actual_danger = slice(fms, 2*fms)
        # agent_locations offset: (2 + 2*N_AGENTS)*FLAT_MAP_SIZE
        agent_locs_offset = (2 + 2*n_agents) * fms
        self.sl_agent_locs = slice(agent_locs_offset, agent_locs_offset + n_agents*2)
        self.sl_global_disc = slice(stride - fms, stride) # Last section

    def reset(self, seed=None, options=None):
        self.env.reset()
        obs = self.state_tensor if self.device == 'cpu' else self.state_tensor.to(self.device)
        return obs, {}

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        
        flat_actions = actions.reshape(self.num_envs, -1).astype(np.float32)
        rewards = self.env.step(flat_actions)
        
        obs = self.state_tensor
        rewards = torch.from_numpy(rewards)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        
        if self.render_mode == "human":
            self.render()

        return obs, rewards, terminated, truncated, {}

    def get_gravity_attractions(self, feature_type, agent_mask=None, pow=2, normalize=False):
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
        
        gravity_array = self.env.get_gravity_attractions(agent_mask, feature_type, pow, normalize)
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
        discovered_map = state[self.sl_global_disc].reshape(self.map_size, self.map_size)
        agents = state[self.sl_agent_locs].reshape(self.n_agents, 2) # [y, x]

        self.screen.fill((0, 0, 0))

        # 1. Draw Grid (Danger Map)
        for y in range(self.map_size):
            for x in range(self.map_size):
                rect = pygame.Rect(
                    x * self.cell_size, 
                    y * self.cell_size, 
                    self.cell_size, 
                    self.cell_size
                )
                
                # Check if discovered
                if discovered_map[y, x] > 0.5:
                    # Visible: Color based on danger (-1.0 to 1.0)
                    # -1.0 (safe) -> Green, 0.0 (neutral) -> Yellow, 1.0 (danger) -> Red
                    d = danger_map[y, x]
                    # Map [-1, 1] to [0, 1] for visualization
                    d_normalized = (d + 1.0) / 2.0
                    color = (
                        int(255 * d_normalized),           # R: 0 at -1, 255 at +1
                        int(255 * (1 - abs(d))),          # G: 255 at 0, 0 at extremes
                        int(255 * (1 - d_normalized))     # B: 255 at -1, 0 at +1
                    )
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, (50, 50, 50), rect, 1) # Border
                else:
                    # Undiscovered: Black (Fog of War)
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)
                    # Optional: faint grid line to show it exists
                    pygame.draw.rect(self.screen, (20, 20, 20), rect, 1)

        # 2. Draw Agents & View Range
        # Create a transparent surface for view range
        view_surface = pygame.Surface((self.map_size*self.cell_size, self.map_size*self.cell_size), pygame.SRCALPHA)
        
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

            # Draw Agent (Blue Circle)
            pygame.draw.circle(self.screen, (0, 100, 255), (int(px), int(py)), int(self.cell_size * 0.4))
            # Agent Border (White)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(px), int(py)), int(self.cell_size * 0.4), 2)

        # Blit the transparent surface onto main screen
        self.screen.blit(view_surface, (0,0))

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
    gravity_masked = env16.get_gravity_attractions(FeatureType.GLOBAL_DISCOVERED, agent_mask=mask)
    print(f"Masked gravity shape: {gravity_masked.shape}")
    print(f"Masked gravity for disabled agents (should be ~0): {gravity_masked[0, 2:4].numpy()}")
    
    env16.close()
    
    # Render test
    print("\n=== Render Test: 16 envs ===")
    env = BatchedGridEnv(num_envs=16, render_mode="human")
    obs, _ = env.reset()
    
    # Simple random walk
    try:
        while True:
            obs_masked = env.get_gravity_attractions(FeatureType.GLOBAL_UNDISCOVERED, agent_mask=None, normalize=True, pow=1)
            danger_masked = env.get_gravity_attractions(FeatureType.OBSERVED_DANGER, agent_mask=None, normalize=True, pow=2)
            other_masked = env.get_gravity_attractions(FeatureType.OTHER_AGENTS, agent_mask=None, normalize=True, pow=1)
            
            print(f"Obs gravity sample: {obs_masked[0, 0].numpy()}, Danger gravity sample: {danger_masked[0, 0].numpy()}, Other agents gravity sample: {other_masked[0, 0].numpy()}")
            actions = np.random.uniform(-1, 1, (16, 4, 2))
            obs, r, term, trunc, info = env.step(-danger_masked + obs_masked-other_masked+actions)#obs_masked.numpy() + actions - danger_masked.numpy() - other_masked.numpy())  # Add gravity to random actions
            if sum(r[0])>0:
                print(f"Rewards: {r[0]}")
    except KeyboardInterrupt:
        env.close()