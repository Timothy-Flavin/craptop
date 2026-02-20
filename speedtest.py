import time, numpy as np
from env_wrapper import BatchedGridEnv, FeatureType as FT

for test in range(5):
    env16 = BatchedGridEnv(num_envs=16, render_mode=None)
    obs, _ = env16.reset()

    start = time.time()
    for _ in range(10000):
        actions = np.random.uniform(-1, 1, (16, 4, 2))
        env16.step(actions)
    elapsed = time.time() - start
    print(f"Step FPS (16 envs): {10000*16/elapsed:.0f}")

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
    ]
    gstart = time.time()
    for _ in range(1000):
        for f in features:
            env16.get_gravity_attractions(f, pow=2)
    gelapsed = time.time() - gstart
    n_calls = 1000 * len(features)
    print(
        f"Gravity: {n_calls} calls in {gelapsed:.3f}s ({n_calls/gelapsed:.0f} calls/s)"
    )
    env16.close()
