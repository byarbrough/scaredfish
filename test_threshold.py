"""
Quick test of threshold experiment (reduced parameters for testing)
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool
from typing import Dict, Tuple
import time
from fish_school_simulator import run_simulation

# Tank dimensions
TANK_DIMENSIONS = (100, 200, 100)

# Test parameters (reduced for quick testing)
DELTA_VALUES = [1, 10, 20]  # Just 3 values for testing
N_ITERATIONS = 2  # Just 2 trials per delta
N_FISH = 40
GAMMA = 10
PREDATOR_SPAWN_FRAME = 400
N_STEPS = 800
BETA = 0.6


def run_single_trial(args: Tuple[int, int, int]) -> Dict:
    """Run a single simulation trial"""
    delta, trial_num, seed = args
    np.random.seed(seed)

    school = run_simulation(
        n_fish=N_FISH,
        n_steps=N_STEPS,
        predator_spawn_frame=PREDATOR_SPAWN_FRAME,
        predator_remove_frame=None,
        gamma=GAMMA,
        delta=delta,
        space_size=TANK_DIMENSIONS,
        beta=BETA,
        verbose=False,
    )

    # Get final state at frame 800
    final_s_proportion = school.history["S"][-1]
    final_i_proportion = school.history["I"][-1]
    final_r_proportion = school.history["R"][-1]

    final_s_count = int(round(final_s_proportion * N_FISH))
    final_i_count = int(round(final_i_proportion * N_FISH))
    final_r_count = int(round(final_r_proportion * N_FISH))

    print(
        f"Completed: delta={delta}, trial={trial_num}, S={final_s_count}, I={final_i_count}, R={final_r_count}"
    )

    return {
        "delta": delta,
        "trial": trial_num,
        "S_count": final_s_count,
        "I_count": final_i_count,
        "R_count": final_r_count,
        "S_proportion": final_s_proportion,
        "I_proportion": final_i_proportion,
        "R_proportion": final_r_proportion,
    }


def main():
    print("Testing threshold experiment with reduced parameters...")
    print(f"Delta values: {DELTA_VALUES}")
    print(f"Iterations per delta: {N_ITERATIONS}")
    print(f"Total simulations: {len(DELTA_VALUES) * N_ITERATIONS}")
    print()

    # Prepare tasks
    tasks = []
    for delta in DELTA_VALUES:
        for trial_num in range(N_ITERATIONS):
            seed = delta * 1000 + trial_num
            tasks.append((delta, trial_num, seed))

    print("Starting simulations...")
    start_time = time.time()

    # Run in parallel
    with Pool() as pool:
        results = pool.map(run_single_trial, tasks)

    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds")

    # Show results
    df = pd.DataFrame(results)
    print("\nResults:")
    print(df)

    summary = (
        df.groupby("delta")
        .agg(
            {
                "S_count": "mean",
                "I_count": "mean",
                "R_count": "mean",
            }
        )
        .round(1)
    )
    print("\nSummary (mean counts):")
    print(summary)


if __name__ == "__main__":
    main()
