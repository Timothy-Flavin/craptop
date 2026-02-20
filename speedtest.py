import time, numpy as np
from multi_agent_coverage.env_wrapper import BatchedGridEnv, FeatureType as FT

NUM_ENVS = 16
NUM_FRAMES = 10_000
GRAVITY_ITERS = 1000

features = [
    FT.EXPECTED_DANGER,
    FT.ACTUAL_DANGER,
    FT.OBSERVED_DANGER,
    FT.OBS,
    FT.EXPECTED_OBS,
    FT.GLOBAL_DISCOVERED,
    FT.OTHER_AGENTS,
    FT.OTHER_AGENTS_LAST_KNOWN,
    FT.GLOBAL_UNDISCOVERED,
    FT.OBS_UNDISCOVERED,
    FT.EXPECTED_OBS_UNDISCOVERED,
    FT.RECENCY,
    FT.RECENCY_STALE,
    FT.WALL_REPEL,
    FT.WALL_ATTRACT,
    FT.GLOBAL_VORONOI_UNDISCOVERED,
    FT.EXPECTED_VORONOI_UNDISCOVERED,
]

def bench(label, global_comms):
    actions = np.random.uniform(-1, 1, (NUM_ENVS, 4, 2)).astype(np.float32)
    for run in range(5):
        env = BatchedGridEnv(num_envs=NUM_ENVS, render_mode=None, global_comms=global_comms)
        env.reset()

        start = time.time()
        for _ in range(NUM_FRAMES):
            env.step(actions)
        step_elapsed = time.time() - start

        gstart = time.time()
        for _ in range(GRAVITY_ITERS):
            for f in features:
                env.get_gravity_attractions(f, pow=2)
        gravity_elapsed = time.time() - gstart
        n_calls = GRAVITY_ITERS * len(features)

        stride = env.env.get_stride()
        print(f"[{label} run {run+1}] stride={stride}  "
              f"step FPS={NUM_FRAMES*NUM_ENVS/step_elapsed:.0f}  "
              f"gravity={n_calls/gravity_elapsed:.0f} calls/s")
        env.close()

print("=== Partial-obs mode ===")
bench("partial", global_comms=False)
print()
print("=== Global-comms mode ===")
bench("global", global_comms=True)
