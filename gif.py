"""
Record a demonstration gif of the multi-agent coverage environment.
Visual style matches the pygame renderer in env_wrapper.py.
"""
import numpy as np
from PIL import Image, ImageDraw
from env_wrapper import BatchedGridEnv, FeatureType


MAP_SIZE = 32
CELL_SIZE = 12        # Pixels per grid cell
VIEW_RANGE = 3        # Must match C++ VIEW_RANGE
VIEW_ALPHA = 40       # 0-255 opacity of agent view-range box
AGENT_COLORS = [
    (0, 100, 255),
    (255, 140, 0),
    (0, 210, 100),
    (220, 50, 220),
]
AGENT_BORDER = (255, 255, 255)


def _danger_color(d: float) -> tuple[int, int, int]:
    """Map danger value in [-1, 1] to RGB, matching the pygame renderer."""
    d_norm = (d + 1.0) / 2.0          # [-1,1] -> [0,1]
    r = int(255 * d_norm)
    g = int(255 * (1.0 - abs(d)))
    b = int(255 * (1.0 - d_norm))
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def _render_frame(state: np.ndarray, n_agents: int) -> Image.Image:
    """Render a single environment state to a PIL Image."""
    img_size = MAP_SIZE * CELL_SIZE

    # --- Extract layers ---
    fms = MAP_SIZE * MAP_SIZE
    danger_map    = state[fms : 2*fms].reshape(MAP_SIZE, MAP_SIZE)
    agent_locs    = state[(2 + 2*n_agents)*fms : (2 + 2*n_agents)*fms + n_agents*2].reshape(n_agents, 2)
    discovered    = state[-fms:].reshape(MAP_SIZE, MAP_SIZE)

    # --- Base layer: grid (RGB) ---
    base = Image.new('RGB', (img_size, img_size), (0, 0, 0))
    draw_base = ImageDraw.Draw(base)

    for y in range(MAP_SIZE):
        for x in range(MAP_SIZE):
            x0, y0 = x * CELL_SIZE, y * CELL_SIZE
            x1, y1 = x0 + CELL_SIZE - 1, y0 + CELL_SIZE - 1
            if discovered[y, x] > 0.5:
                color = _danger_color(danger_map[y, x])
                draw_base.rectangle([x0, y0, x1, y1], fill=color)
                draw_base.rectangle([x0, y0, x1, y1], outline=(50, 50, 50))
            else:
                # Fog of war: black with faint grid
                draw_base.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
                draw_base.rectangle([x0, y0, x1, y1], outline=(20, 20, 20))

    # --- View range overlay (RGBA, translucent) ---
    overlay = Image.new('RGBA', (img_size, img_size), (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)

    for i in range(n_agents):
        ay, ax = agent_locs[i]
        px = (ax + 0.5) * CELL_SIZE
        py = (ay + 0.5) * CELL_SIZE
        box_half = (VIEW_RANGE + 0.5) * CELL_SIZE
        draw_ov.rectangle(
            [px - box_half, py - box_half, px + box_half, py + box_half],
            fill=(255, 255, 255, VIEW_ALPHA),
        )

    # Composite overlay onto base
    base = base.convert('RGBA')
    base = Image.alpha_composite(base, overlay)
    base = base.convert('RGB')

    # --- Agents: filled circle + white border ---
    draw_agents = ImageDraw.Draw(base)
    radius = max(1, int(CELL_SIZE * 0.4))

    for i in range(n_agents):
        ay, ax = agent_locs[i]
        px = int((ax + 0.5) * CELL_SIZE)
        py = int((ay + 0.5) * CELL_SIZE)
        color = AGENT_COLORS[i % len(AGENT_COLORS)]

        # Filled body
        draw_agents.ellipse(
            [px - radius, py - radius, px + radius, py + radius],
            fill=color,
        )
        # White border
        draw_agents.ellipse(
            [px - radius, py - radius, px + radius, py + radius],
            outline=AGENT_BORDER,
            width=max(1, radius // 3),
        )

    return base


def record_gif(
    num_envs: int = 4,
    n_agents: int = 4,
    num_frames: int = 300,
    output_path: str = "demo.gif",
    maps=None,
    expected_maps=None,
):
    """
    Record an animated GIF of the environment.

    Args:
        num_envs:      Number of parallel environments (only first is rendered)
        n_agents:      Agents per environment
        num_frames:    Frames to capture
        output_path:   Output .gif path
        maps:          Ground-truth map path(s) passed to BatchedGridEnv
        expected_maps: Prior belief map path(s) passed to BatchedGridEnv
    """
    print(f"Recording {num_frames} frames to {output_path}...")

    env = BatchedGridEnv(
        num_envs=num_envs,
        n_agents=n_agents,
        render_mode=None,
        maps=maps,
        expected_maps=expected_maps,
    )
    obs, _ = env.reset()

    frames = []

    for frame_idx in range(num_frames):
        actions = np.random.uniform(-1, 1, (num_envs, n_agents, 2))
        obs, rewards, terminated, truncated, info = env.step(actions)

        frame = _render_frame(obs[0].numpy(), n_agents)
        frames.append(frame)

        if (frame_idx + 1) % 50 == 0:
            print(f"  Recorded {frame_idx + 1}/{num_frames} frames")

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=80,   # ms per frame (~12 fps)
        loop=0,
    )

    print(f"Saved {output_path}")
    env.close()


if __name__ == "__main__":
    record_gif(
        num_envs=4,
        n_agents=4,
        num_frames=300,
        output_path="demo.gif",
    )
