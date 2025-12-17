"""
Extinction Experiment for Maximum I and Time to Extinction

This script runs multiple simulations with fixed delta=40 to measure:
1. Maximum I (infected) proportion reached during simulation
2. Number of frames from predator spawn to I=0 (extinction)

Runs 500 trials each for "relaxed" and "alarmed" behavioral states.
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple, Optional
import time
import sys
import io
import argparse
from fish_school_simulator import FishSchool

# Tank dimensions (in cm): 1m x 2m x 1m = 100cm x 200cm x 100cm
TANK_DIMENSIONS: Tuple[int, int, int] = (100, 200, 100)

# Experiment parameters
DELTA: float = 40.0  # Fixed delta value
N_ITERATIONS: int = 500  # Number of trials per state
N_FISH: int = 40
GAMMA: float = 10  # Infected duration (frames)
PREDATOR_SPAWN_FRAME: int = 400  # Predator appears at frame 400
MAX_FRAMES: int = 1400  # Maximum frames (1000 frames after predator spawn)
BETA: float = 0  # Not used - empirical transmission model

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


def run_single_trial(args: Tuple[int, int, float, float, float]) -> Dict:
    """
    Run a single simulation trial tracking max I and time to extinction

    Args:
        args: Tuple of (trial_number, seed, zone_repulsion, zone_orientation, zone_attraction)

    Returns:
        Dictionary with trial number, max I, and frames to extinction
    """
    trial_num, seed, zone_repulsion, zone_orientation, zone_attraction = args

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
            delta=DELTA,
            zone_repulsion=zone_repulsion,
            zone_orientation=zone_orientation,
            zone_attraction=zone_attraction,
        )

        max_i_proportion = 0.0
        max_i_count = 0
        frames_to_extinction: Optional[int] = None
        predator_spawned = False

        # Run simulation
        for frame in range(MAX_FRAMES):
            # Spawn predator at frame 400
            if frame == PREDATOR_SPAWN_FRAME:
                school.spawn_predator()
                predator_spawned = True

            # Update simulation
            school.update()

            # Track max I only after predator spawns
            if predator_spawned:
                current_i_proportion = school.history["I"][-1]
                current_i_count = int(round(current_i_proportion * N_FISH))

                if current_i_proportion > max_i_proportion:
                    max_i_proportion = current_i_proportion
                    max_i_count = current_i_count

                # Check for extinction (I reaches 0)
                if current_i_count == 0 and frames_to_extinction is None:
                    frames_to_extinction = frame - PREDATOR_SPAWN_FRAME
                    # Exit early once extinction occurs
                    break

        # If simulation ended without extinction, record as None or max frames
        if frames_to_extinction is None:
            frames_to_extinction = MAX_FRAMES - PREDATOR_SPAWN_FRAME

        return {
            "trial": trial_num,
            "max_I_proportion": max_i_proportion,
            "max_I_count": max_i_count,
            "frames_to_extinction": frames_to_extinction,
        }
    finally:
        # Restore stdout
        sys.stdout = old_stdout


def run_state_experiment(state_name: str) -> pd.DataFrame:
    """Run experiment for a single state (relaxed or alarmed)"""
    zone_params = ZONE_PARAMS[state_name]

    print("=" * 80)
    print(f"EXTINCTION EXPERIMENT - {state_name.upper()} STATE")
    print("=" * 80)
    print(
        f"Tank dimensions: {TANK_DIMENSIONS[0]}cm x {TANK_DIMENSIONS[1]}cm x {TANK_DIMENSIONS[2]}cm"
    )
    print(f"Number of fish: {N_FISH}")
    print(f"Gamma (infected duration): {GAMMA} frames")
    print(f"Delta (recovered duration): {DELTA} frames")
    print(f"Iterations: {N_ITERATIONS}")
    print(f"Predator spawn frame: {PREDATOR_SPAWN_FRAME}")
    print(f"Max simulation frames: {MAX_FRAMES}")
    print(f"\nCouzin Parameters ({state_name} state):")
    print(f"  Zone repulsion:   {zone_params['zone_repulsion']} cm")
    print(f"  Zone orientation: {zone_params['zone_orientation']} cm")
    print(f"  Zone attraction:  {zone_params['zone_attraction']} cm")
    print(f"\nCPU cores available: {cpu_count()}")
    print("=" * 80)
    print()

    # Prepare all simulation tasks
    tasks = []
    state_offset = 0 if state_name == "relaxed" else 100000
    for trial_num in range(N_ITERATIONS):
        seed = (int(DELTA * 1000) + trial_num + state_offset) % (2**32)
        tasks.append(
            (
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
    df["state"] = state_name

    return df


def main():
    """Run extinction experiment for both states"""
    parser = argparse.ArgumentParser(
        description="Run extinction experiment for both relaxed and alarmed states"
    )
    parser.add_argument(
        "--state",
        type=str,
        choices=["relaxed", "alarmed", "both"],
        default="both",
        help="Which state(s) to run: relaxed, alarmed, or both (default: both)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("EXTINCTION EXPERIMENT: Maximum I and Time to Extinction")
    print("=" * 80)
    print(f"Fixed delta: {DELTA} frames")
    print(f"Tracking: (1) Max I proportion, (2) Frames to extinction")
    print(
        f"Exit condition: I=0 or {MAX_FRAMES - PREDATOR_SPAWN_FRAME} frames after predator spawn"
    )
    print("=" * 80 + "\n")

    all_results = []

    if args.state == "both":
        states_to_run = ["relaxed", "alarmed"]
    else:
        states_to_run = [args.state]

    for state in states_to_run:
        df = run_state_experiment(state)
        all_results.append(df)

    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Save detailed results
    detailed_filename = "extinction_detailed_results.csv"
    combined_df.to_csv(detailed_filename, index=False)
    print(f"\nDetailed results saved to: {detailed_filename}")

    # Calculate summary statistics for each state
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for state in states_to_run:
        state_df = combined_df[combined_df["state"] == state]

        print(f"\n{state.upper()} STATE:")
        print("-" * 40)
        print(
            f"Average max I proportion: {state_df['max_I_proportion'].mean():.4f} ± {state_df['max_I_proportion'].std():.4f}"
        )
        print(
            f"Average max I count: {state_df['max_I_count'].mean():.2f} ± {state_df['max_I_count'].std():.2f}"
        )
        print(
            f"Average frames to extinction: {state_df['frames_to_extinction'].mean():.2f} ± {state_df['frames_to_extinction'].std():.2f}"
        )
        print(
            f"Median frames to extinction: {state_df['frames_to_extinction'].median():.2f}"
        )
        print(f"Min frames to extinction: {state_df['frames_to_extinction'].min()}")
        print(f"Max frames to extinction: {state_df['frames_to_extinction'].max()}")

        # Count how many reached max frames (no extinction)
        no_extinction_count = (
            state_df["frames_to_extinction"] >= (MAX_FRAMES - PREDATOR_SPAWN_FRAME)
        ).sum()
        print(
            f"Trials without extinction: {no_extinction_count}/{len(state_df)} ({no_extinction_count/len(state_df)*100:.1f}%)"
        )

    # Save summary
    summary_data = []
    for state in states_to_run:
        state_df = combined_df[combined_df["state"] == state]
        summary_data.append(
            {
                "state": state,
                "avg_max_I_proportion": state_df["max_I_proportion"].mean(),
                "std_max_I_proportion": state_df["max_I_proportion"].std(),
                "avg_max_I_count": state_df["max_I_count"].mean(),
                "std_max_I_count": state_df["max_I_count"].std(),
                "avg_frames_to_extinction": state_df["frames_to_extinction"].mean(),
                "std_frames_to_extinction": state_df["frames_to_extinction"].std(),
                "median_frames_to_extinction": state_df[
                    "frames_to_extinction"
                ].median(),
                "trials_without_extinction": (
                    state_df["frames_to_extinction"]
                    >= (MAX_FRAMES - PREDATOR_SPAWN_FRAME)
                ).sum(),
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_filename = "extinction_summary.csv"
    summary_df.to_csv(summary_filename, index=False)
    print(f"\nSummary statistics saved to: {summary_filename}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
