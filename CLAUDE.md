# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based fish school simulator implementing the Couzin model for collective animal behavior combined with a SIRS epidemic model. The simulation models how fish schools respond to predators using epidemic dynamics: fish can be Susceptible (S), Infected (I, showing escape behavior), or Recovered (R, temporarily immune). Infection spreads through direct predator detection and social transmission using an empirically-determined probability model based on distance and visual prominence (ranked angular area).

## Development Environment

### Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# Install dependencies for simulation
pip install -r requirements.txt

# Additional dependencies for data analysis notebooks (optional)
pip install pandas scikit-learn statsmodels
```

### Running the Simulation
```bash
# Activate virtual environment first
source .venv/bin/activate

# Run the simulation
python fish_school_simulator.py
```

The simulation will open a 3D matplotlib visualization showing fish behavior over time. A predator spawns at t=200 frames (at the school's center) and is removed at t=400. After the simulation completes, a SIRS dynamics plot is displayed and data is exported to a CSV file named `{beta}_{gamma}_{delta}.csv`.

### Working with Empirical Data
The repository includes Jupyter notebooks for analyzing the empirical data that informed the model:

- **`regression.ipynb`**: Reproduces the logistic regression coefficients from research paper. Fits the startle transmission probability model P(s_i | s_j) using experimental data from `realdata/first_responders_srk*.csv`.
- **`datafit.ipynb`**: Explores different model formulations and feature selections for predicting startle responses. Tests various combinations of distance metrics, angular area, loom, etc.
- **`realdata/` folder**: Contains CSV and HDF5 files from actual golden shiner fish experiments, including cascade data and first responder analysis.

The current empirical coefficients (β₁=0.103641, β₂=-3.297823, β₃=-0.075034) were derived by fitting to "Schreck only" data (schreckstoff hormone condition), as shown in `regression.ipynb`.

## Architecture

### Core Components

**Fish Class** (`fish_school_simulator.py:18-60`)
- Individual fish agent with position, velocity, and SIRS state
- States: 'susceptible' (S), 'infected' (I), 'recovered' (R)
- Tracks state duration with timers (infected_duration, recovered_duration)
- Implements infection response: increases velocity to 1.75 cm/frame (35 cm/s at 20 fps) for escape behavior
  - Susceptible: 0.5 cm/frame (10 cm/s)
  - Infected: 1.75 cm/frame (35 cm/s)
  - Recovered: 0.25 cm/frame (5 cm/s)
- State transitions: S → I (via predator detection or social transmission) → R (after infected_duration) → S (after recovered_duration)

**FishSchool Class** (`fish_school_simulator.py:62-350`)
- Main simulation manager containing all fish agents
- Implements SIRS epidemic model combined with Couzin behavioral model:
  1. **Couzin Model** (`couzin_behavior` method): Three-zone interaction model for normal/recovered fish
     - Zone of repulsion (< 2.0 units): Avoid collisions
     - Zone of orientation (2.0-10.0 units): Align with neighbors
     - Zone of attraction (10.0-20.0 units): Stay with group
  2. **Predator Detection** (`check_predator_startle` method): Direct infection when susceptible fish detect predator within detection radius
  3. **Empirical Startle Transmission** (`check_startle_cascade` method): Social transmission using logistic regression model
     - Probability: P(s_i | s_j) = 1/(1 + exp(-β₁ - β₂·LMD - β₃·RAA))
     - LMD (Log Metric Distance): log₁₀ of Euclidean distance between fish
     - RAA (Ranked Angular Area): normalized rank of apparent size on observer's retina (accounts for field of view, occlusion, and visual prominence)
- Collects SIR dynamics data in `history` dict for analysis and export

### Key Parameters

**Couzin Model Parameters** (in `FishSchool.__init__`):
- `zone_repulsion`, `zone_orientation`, `zone_attraction`: Interaction zone distances (default: 4.5, 12.0, 48.0 cm)
  - Can be adjusted to model scared (tight) vs relaxed (loose) formations
- `max_speed`, `min_speed`: Movement constraints (1.0, 0.5 cm/frame at 20 fps)
- `max_turn_rate`: Turning angle limit per timestep (0.3 radians)

**SIRS Epidemic Parameters** (in `FishSchool.__init__`):
- `beta`: Kept for backward compatibility in plots/exports (default: 0) - **NOT used** for transmission probability
  - Transmission probability is calculated using the empirical equation: P = 1/(1 + e^(-β₁ - β₂·LMD - β₃·RAA))
- `gamma` (γ): Infected duration in frames before transition to recovered state (default: 10 frames = 0.5 seconds at 20 fps)
- `delta` (δ): Recovered duration in frames before returning to susceptible state (default: 60 frames = 3 seconds at 20 fps)
- `predator_detection_radius`: How far fish detect predators (50.0 cm)
- `visual_range`: Distance fish can see each other for social transmission (120.0 cm - extended to capture long-range cascades)
- `verbose`: Control debug output (default: False) - set to True to see detailed cascade information

**Empirical Startle Transmission Parameters** (in `FishSchool.__init__`):
- `beta_1`: Logistic regression intercept (0.103641)
- `beta_2`: Log₁₀ metric distance coefficient (-3.297823) - negative means closer fish have higher transmission probability
- `beta_3`: Ranked angular area coefficient (-0.075034) - negative means more visually prominent fish have higher transmission probability
- **Note**: LMD uses base-10 logarithm (`np.log10`), not natural log, for distance calculation

### Update Loop

The simulation follows this sequence each frame (`update` method):
1. Check for predator-induced infections (direct detection: S → I)
2. Check for social transmission (infected to susceptible neighbors: S → I)
3. For each fish:
   - Update SIRS state timer (handle I → R and R → S transitions)
   - Calculate desired velocity:
     - **Infected fish**: Escape away from predator at max speed
     - **Susceptible/Recovered fish**: Apply Couzin model for normal schooling
   - Apply smooth turning constraint (max_turn_rate)
   - Limit speed to valid range
   - Update position
   - Apply periodic boundary conditions
4. Record SIR proportions in history dict

### Visualization and Data Export

**3D Animation** (`visualize_simulation` function):
- Blue dots: Susceptible fish (S)
- Red dots: Infected fish (I)
- Green dots: Recovered fish (R)
- Black X: Predator position
- Red wireframe sphere: Predator detection radius
- Real-time state counts displayed on screen

**SIRS Dynamics Plot** (`plot_sir_dynamics` function):
- Line plot showing S, I, R proportions over time
- Vertical lines marking predator spawn/removal events
- Automatically displayed after simulation completes

**Data Export** (`export_sir_data` function):
- Exports time-series data to CSV file: `{beta}_{gamma}_{delta}.csv`
- Contains columns: time, S_proportion, I_proportion, R_proportion
- Enables quantitative analysis of epidemic dynamics

Console output shows real-time detection and cascade events.

## Code Patterns

### Type Annotations
The codebase uses Python type hints with numpy typing:
- `npt.NDArray[np.float64]` for numpy arrays
- `npt.ArrayLike` for inputs that accept array-like objects
- Return types specified for all methods

### Vector Operations
All spatial calculations use numpy array operations:
- Positions and velocities are 3D numpy arrays
- Helper methods `_normalize`, `_limit_speed`, `_smooth_turn` ensure realistic movement
- Periodic boundaries via modulo operator

### Neighbor Queries
The `get_neighbors` method returns a list of (fish, distance) tuples within a given radius. This is used for both Couzin behavioral interactions and epidemic transmission.

### Ranked Angular Area (RAA) Calculation
The `_calculate_raa` method computes how visually prominent a startled fish appears to an observer, using simplified heuristics:

**Field of View (FOV):**
- Fish have 333° field of view (25° blind spot directly behind)
- Calculate angle between observer's heading (velocity direction) and direction to target
- Target visible if angle < 166.5° from forward direction
- Uses dot product: `angle = arccos(observer_heading · direction_to_target)`

**Occlusion Detection (Simplified):**
- Fish B occludes fish C if B is significantly closer (20%+ threshold) AND in nearly same direction (< 5° angular difference)
- Skips complex ray-tracing; captures most obvious occlusions
- Compares all pairs of neighbors to determine visibility

**Angular Size Ranking:**
- Angular size on retina ∝ 1/distance² (assumes uniform fish size)
- Among visible neighbors, compute angular size for each
- Sort by angular size (largest/closest first)
- Assign ranks: 1 (most prominent), 2, 3, ...
- Normalize to [0, 1]: RAA = 1 - (rank-1)/(n_visible-1)
- Result: RAA=1.0 for closest visible fish, RAA=0.0 for farthest, RAA=0.0 if not visible

### SIRS State Management
- Fish states are tracked via string literals: 'susceptible', 'infected', 'recovered'
- State transitions managed by `update_state()` method using timers
- Only susceptible fish can be infected (via `infect()` method)
- Infected fish automatically escape (velocity manipulation in movement logic)
- State history tracked in `FishSchool.history` dict with normalized proportions

## Performance Optimizations

The simulation implements several performance optimizations to handle real-time visualization and large-scale experiments:

**Spatial Indexing with KDTree** (`get_neighbors_fast` method):
- Uses `scipy.spatial.KDTree` for efficient neighbor queries
- O(log n) lookup time instead of O(n²) brute force
- Critical for cascade checks and RAA calculations with many fish
- Dual implementation: `get_neighbors()` for Fish objects (backward compatibility), `get_neighbors_fast()` for index-based operations

**Numba JIT Compilation**:
- `@jit(nopython=True)` decorator on performance-critical functions
- `compute_startle_probabilities()`: Vectorized probability calculations
- Compiles to machine code for near-C performance
- Significant speedup for probability computation over many fish pairs

**Vectorized RAA Calculations** (`_calculate_raa_batch` method):
- Batch processing of RAA for all susceptible neighbors simultaneously
- NumPy array operations instead of Python loops
- Reduces overhead when checking cascades from multiple infected fish
- Dual implementation: `_calculate_raa()` for single calculations, `_calculate_raa_batch()` for batch operations

**Performance Impact**:
- Enables real-time 3D visualization with 40+ fish at 20 fps
- Threshold experiments can run 250 trials × 21 parameter values in reasonable time
- Essential for exploring parameter space and running statistical analyses

## Threshold Experiments

The `threshold.py` script runs systematic experiments to find endemic equilibrium points for the delta parameter.

**Purpose**: Identify threshold values where infection persists in the population (endemic) vs dies out (epidemic extinction).

**Experimental Design**:
- Tests 21 delta values from 0 to 10 frames (0 to 0.5 seconds at 20 fps)
- 250 independent trials per delta value for statistical power
- Parallel execution using multiprocessing (utilizes all CPU cores)
- Two behavioral state presets:

**Behavioral States**:
1. **Relaxed** (loose schooling):
   - zone_repulsion: 5.0 cm
   - zone_orientation: 12.0 cm
   - zone_attraction: 36.0 cm
   - Models unstressed fish with normal nearest-neighbor distances

2. **Alarmed** (tight schooling):
   - zone_repulsion: 2.75 cm
   - zone_orientation: 12.0 cm
   - zone_attraction: 48.0 cm
   - Models stressed fish with closer grouping behavior

**Output Files**:
- `threshold_detailed_results.csv`: All 5,250 trial results (250 trials × 21 deltas)
  - Columns: delta, trial, S_count, I_count, R_count, S_prop, I_prop, R_prop, mean_NND
- `threshold_summary.csv`: Aggregated statistics per delta value
  - Columns: delta, mean_S, std_S, mean_I, std_I, mean_R, std_R, mean_NND, std_NND

**Analysis Workflow**:
1. Run experiments: `python threshold.py --state relaxed` or `python threshold.py --state alarmed`
2. Analyze results: Open `threshold_analysis.ipynb` to visualize endemic equilibrium thresholds
3. Compare NND (nearest-neighbor distance) predictions against empirical data from `realdata/`

**Key Findings**:
- Lower delta (shorter recovered duration) → higher endemic prevalence
- Behavioral state affects transmission: alarmed fish (closer spacing) show different epidemic dynamics
- Endemic equilibrium occurs when delta < ~5-7 frames depending on behavioral state
