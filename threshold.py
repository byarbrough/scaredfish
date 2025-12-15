"""
Threshold Experiment for Delta Parameter

This script runs multiple simulations to find the threshold value for delta
where endemic equilibrium occurs. It iterates through different delta values,
running multiple trials in parallel for each value.
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple
import time
import sys
import io
import argparse
from fish_school_simulator import FishSchool

# Tank dimensions (in cm): 1m x 2m x 1m = 100cm x 200cm x 100cm
TANK_DIMENSIONS: Tuple[int, int, int] = (100, 200, 100)

# Experiment parameters
DELTA_VALUES: list[float] = [float(x) for x in np.arange(0.0, 10.5, 0.5)]
N_ITERATIONS: int = 250  # Number of trials per delta value
N_FISH: int = 40
GAMMA: float = 10  # Infected duration (frames)
PREDATOR_SPAWN_FRAME: int = 400
N_STEPS: int = 800  # Run until frame 800
BETA: float = (
    0  # Not used - transmission probability calculated via empirical equation P = 1/(1 + e^(-β₁ - β₂·LMD - β₃·RAA))
)

# Couzin zone parameter presets
ZONE_PARAMS = {
    "relaxed": {
        "zone_repulsion": 5.0,
        "zone_orientation": 12.0,
        "zone_attraction": 36.0,
    },
    "alarmed": {
        "zone_repulsion": 2.75,
        "zone_orientation": 12.0,
        "zone_attraction": 48.0,
    },
}


def run_single_trial(args: Tuple[float, int, int, float, float, float]) -> Dict:
    """
    Run a single simulation trial

    Args:
        args: Tuple of (delta, trial_number, seed, zone_repulsion, zone_orientation, zone_attraction)

    Returns:
        Dictionary with delta, trial number, final S, I, R counts, and pre-startle NND
    """
    delta, trial_num, seed, zone_repulsion, zone_orientation, zone_attraction = args

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Suppress print statements
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        # Create school
        school = FishSchool(
            n_fish=N_FISH,
            space_size=TANK_DIMENSIONS,
            beta=BETA,
            gamma=GAMMA,
            delta=delta,
            zone_repulsion=zone_repulsion,
            zone_orientation=zone_orientation,
            zone_attraction=zone_attraction,
        )

        # Variable to store NND before predator spawn
        nnd_before_predator = None

        # Run simulation manually to capture NND at predator spawn frame
        for frame in range(N_STEPS):
            # Capture NND right before predator spawns
            if frame == PREDATOR_SPAWN_FRAME:
                nnd_before_predator = school.get_average_nearest_neighbor_distance()
                # Now spawn predator
                school.spawn_predator()

            # Update simulation
            school.update()

        # Get final state at frame 800 (last frame)
        final_s_proportion = school.history["S"][-1]
        final_i_proportion = school.history["I"][-1]
        final_r_proportion = school.history["R"][-1]

        # Convert proportions to counts
        final_s_count = int(round(final_s_proportion * N_FISH))
        final_i_count = int(round(final_i_proportion * N_FISH))
        final_r_count = int(round(final_r_proportion * N_FISH))

        return {
            "delta": delta,
            "trial": trial_num,
            "S_count": final_s_count,
            "I_count": final_i_count,
            "R_count": final_r_count,
            "S_proportion": final_s_proportion,
            "I_proportion": final_i_proportion,
            "R_proportion": final_r_proportion,
            "nnd_before_predator": nnd_before_predator,
        }
    finally:
        # Restore stdout
        sys.stdout = old_stdout


def main():
    """Run threshold experiment"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run threshold experiment for delta parameter"
    )
    parser.add_argument(
        "--state",
        type=str,
        choices=["relaxed", "alarmed"],
        default="relaxed",
        help="Fish school state: relaxed or alarmed (default: relaxed)",
    )
    args = parser.parse_args()

    state_name = args.state
    zone_params = ZONE_PARAMS[state_name]

    print("=" * 80)
    print(f"THRESHOLD EXPERIMENT FOR DELTA PARAMETER - {state_name.upper()} STATE")
    print("=" * 80)
    print(
        f"Tank dimensions: {TANK_DIMENSIONS[0]}cm x {TANK_DIMENSIONS[1]}cm x {TANK_DIMENSIONS[2]}cm"
    )
    print(f"Number of fish: {N_FISH}")
    print(f"Gamma (infected duration): {GAMMA} frames")
    print(
        f"Delta values: {DELTA_VALUES[0]} to {DELTA_VALUES[-1]}, step size {DELTA_VALUES[1] - DELTA_VALUES[0]}"
    )
    print(f"Iterations per delta: {N_ITERATIONS}")
    print(f"Predator spawn frame: {PREDATOR_SPAWN_FRAME}")
    print(f"Simulation end frame: {N_STEPS}")
    print(f"Total simulations: {len(DELTA_VALUES) * N_ITERATIONS}")
    print(f"\nCouzin Parameters ({state_name} state):")
    print(f"  Zone repulsion:   {zone_params['zone_repulsion']} cm")
    print(f"  Zone orientation: {zone_params['zone_orientation']} cm")
    print(f"  Zone attraction:  {zone_params['zone_attraction']} cm")
    print(f"\nCPU cores available: {cpu_count()}")
    print("=" * 80)
    print()

    # Prepare all simulation tasks
    tasks = []
    for delta in DELTA_VALUES:
        for trial_num in range(N_ITERATIONS):
            # Generate a unique seed for each trial (convert to int)
            seed = int(delta * 1000) + trial_num
            tasks.append(
                (
                    delta,
                    trial_num,
                    seed,
                    zone_params["zone_repulsion"],
                    zone_params["zone_orientation"],
                    zone_params["zone_attraction"],
                )
            )

    print(f"Starting {len(tasks)} simulations in parallel...")
    start_time = time.time()

    # Run simulations in parallel with progress tracking
    results = []
    with Pool() as pool:
        for i, result in enumerate(pool.imap_unordered(run_single_trial, tasks), 1):
            results.append(result)
            # Print progress every 50 simulations or at completion
            if i % 50 == 0 or i == len(tasks):
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (len(tasks) - i) / rate if rate > 0 else 0
                print(
                    f"Progress: {i}/{len(tasks)} ({i/len(tasks)*100:.1f}%) | "
                    f"Rate: {rate:.1f} sims/sec | "
                    f"ETA: {remaining/60:.1f} min"
                )

    elapsed_time = time.time() - start_time
    print(f"Completed all simulations in {elapsed_time:.2f} seconds")
    print(f"Average time per simulation: {elapsed_time / len(tasks):.2f} seconds")
    print()

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Calculate summary statistics for each delta
    summary = (
        df.groupby("delta")
        .agg(
            {
                "S_count": ["mean", "std"],
                "I_count": ["mean", "std"],
                "R_count": ["mean", "std"],
                "S_proportion": ["mean", "std"],
                "I_proportion": ["mean", "std"],
                "R_proportion": ["mean", "std"],
                "nnd_before_predator": ["mean", "std"],
            }
        )
        .round(3)
    )

    # Save detailed results
    detailed_filename = f"threshold_{state_name}_detailed_results.csv"
    df.to_csv(detailed_filename, index=False)
    print(f"Detailed results saved to: {detailed_filename}")

    # Save summary statistics
    summary_filename = f"threshold_{state_name}_summary.csv"
    summary.to_csv(summary_filename)
    print(f"Summary statistics saved to: {summary_filename}")

    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(summary)
    print()

    # Identify potential endemic equilibrium (I_count > 0 on average)
    i_proportion_mean = summary[("I_proportion", "mean")]
    endemic_deltas = summary[i_proportion_mean > 0.01]
    if len(endemic_deltas) > 0:
        print("Delta values with potential endemic equilibrium (mean I > 1%):")
        # Show I_proportion columns
        i_prop_cols = endemic_deltas["I_proportion"]
        print(i_prop_cols)
    else:
        print("No delta values showed endemic equilibrium (all infections died out)")

    print()
    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
