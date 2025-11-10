# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based fish school simulator implementing the Couzin model for collective animal behavior combined with a SIRS epidemic model. The simulation models how fish schools respond to predators using epidemic dynamics: fish can be Susceptible (S), Infected (I, showing escape behavior), or Recovered (R, temporarily immune). Infection spreads through direct predator detection and social transmission between visible neighbors.

## Development Environment

### Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Running the Simulation
```bash
# Activate virtual environment first
source .venv/bin/activate

# Run the simulation
python fish_school_simulator.py
```

The simulation will open a 3D matplotlib visualization showing fish behavior over time. A predator spawns at t=100 frames (at the school's center) and is removed at t=200. After the simulation completes, a SIRS dynamics plot is displayed and data is exported to a CSV file named `{beta}_{gamma}_{delta}.csv`.

## Architecture

### Core Components

**Fish Class** (`fish_school_simulator.py:18-60`)
- Individual fish agent with position, velocity, and SIRS state
- States: 'susceptible' (S), 'infected' (I), 'recovered' (R)
- Tracks state duration with timers (infected_duration, recovered_duration)
- Implements infection response: doubles velocity for escape behavior
- State transitions: S → I (via predator detection or social transmission) → R (after infected_duration) → S (after recovered_duration)

**FishSchool Class** (`fish_school_simulator.py:62-350`)
- Main simulation manager containing all fish agents
- Implements SIRS epidemic model combined with Couzin behavioral model:
  1. **Couzin Model** (`couzin_behavior` method): Three-zone interaction model for normal/recovered fish
     - Zone of repulsion (< 2.0 units): Avoid collisions
     - Zone of orientation (2.0-10.0 units): Align with neighbors
     - Zone of attraction (10.0-20.0 units): Stay with group
  2. **Predator Detection** (`check_predator_startle` method): Direct infection when susceptible fish detect predator within detection radius
  3. **Epidemic Transmission** (`check_startle_cascade` method): Social transmission of infection from infected to susceptible neighbors within visual range (probability β)
- Collects SIR dynamics data in `history` dict for analysis and export

### Key Parameters

**Couzin Model Parameters** (in `FishSchool.__init__`):
- `zone_repulsion`, `zone_orientation`, `zone_attraction`: Interaction zone distances (2.0, 10.0, 20.0 units)
- `max_speed`, `min_speed`: Movement constraints (2.0, 0.5 units/frame)
- `max_turn_rate`: Turning angle limit per timestep (0.3 radians)

**SIRS Epidemic Parameters** (in `FishSchool.__init__`):
- `beta` (β): Transmission probability when infected fish is within visual range of susceptible fish (default: 0.6)
- `gamma` (γ): Infected duration in frames before transition to recovered state (default: 10)
- `delta` (δ): Recovered duration in frames before returning to susceptible state (default: 20)
- `predator_detection_radius`: How far fish detect predators (25.0 units)
- `visual_range`: Distance fish can see each other for social transmission (15.0 units)

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

### SIRS State Management
- Fish states are tracked via string literals: 'susceptible', 'infected', 'recovered'
- State transitions managed by `update_state()` method using timers
- Only susceptible fish can be infected (via `infect()` method)
- Infected fish automatically escape (velocity manipulation in movement logic)
- State history tracked in `FishSchool.history` dict with normalized proportions
