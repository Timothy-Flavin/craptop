"""
Record a demonstration gif of the multi-agent coverage environment.
"""
import numpy as np
import pygame
from PIL import Image
import io
from env_wrapper import BatchedGridEnv

def record_gif(num_envs=4, n_agents=4, num_frames=300, output_path="demo.gif"):
    """
    Record an animated GIF of the environment running.
    
    Args:
        num_envs: Number of environments to run (only first is rendered)
        n_agents: Number of agents per environment
        num_frames: Number of frames to record
        output_path: Path to save the GIF
    """
    print(f"Recording {num_frames} frames to {output_path}...")
    
    # Create environment without rendering (we'll capture manually)
    env = BatchedGridEnv(num_envs=num_envs, n_agents=n_agents, render_mode=None)
    obs, _ = env.reset()
    
    frames = []
    
    for frame_idx in range(num_frames):
        # Take a random action
        actions = np.random.uniform(-1, 1, (num_envs, n_agents, 2))
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Capture frame data from environment state
        state = obs[0].numpy()
        
        # Extract visualization layers
        fms = env.env.get_flat_map_size()
        danger_map = state[fms:2*fms].reshape(32, 32)
        agent_locs_offset = (2 + 2*n_agents) * fms
        agent_locs = state[agent_locs_offset:agent_locs_offset + n_agents*2].reshape(n_agents, 2)
        discovered_map = state[-fms:].reshape(32, 32)
        
        # Render to PIL Image
        cell_size = 10  # Smaller size for gif
        img_size = 32 * cell_size
        img = Image.new('RGB', (img_size, img_size), color=(0, 0, 0))
        pixels = img.load()
        
        # Draw grid based on danger and discovery
        for y in range(32):
            for x in range(32):
                if discovered_map[y, x] > 0.5:
                    # Visible: Green (safe) to Red (danger)
                    d = danger_map[y, x]
                    r = int(255 * d)
                    g = int(255 * (1 - d))
                    b = 0
                    color = (r, g, b)
                else:
                    # Undiscovered: Dark gray
                    color = (15, 15, 15)
                
                # Fill cell
                for dy in range(cell_size):
                    for dx in range(cell_size):
                        pixels[x * cell_size + dx, y * cell_size + dy] = color
        
        # Draw agents as white circles
        for i in range(n_agents):
            ay, ax = agent_locs[i]
            px = int((ax + 0.5) * cell_size)
            py = int((ay + 0.5) * cell_size)
            radius = max(1, cell_size // 3)
            
            # Simple circle using bresenham-like approach
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx*dx + dy*dy <= radius*radius:
                        nx, ny = px + dx, py + dy
                        if 0 <= nx < img_size and 0 <= ny < img_size:
                            pixels[nx, ny] = (100, 150, 255)
        
        frames.append(img)
        
        if (frame_idx + 1) % 50 == 0:
            print(f"  Recorded {frame_idx + 1}/{num_frames} frames")
    
    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # milliseconds per frame
        loop=0  # Loop forever
    )
    
    print(f"✓ Saved GIF to {output_path}")
    env.close()

if __name__ == "__main__":
    record_gif(num_envs=4, n_agents=4, num_frames=300, output_path="demo.gif")
