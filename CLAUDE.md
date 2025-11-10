# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based fish school simulator implementing the Couzin model for collective animal behavior with startle cascade mechanics. The simulation models how fish schools respond to predators through direct detection and social transmission of startle responses.

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

The simulation will open a 3D matplotlib visualization showing fish behavior over time. A predator spawns at t=100 frames (at the school's center) and is removed at t=200.

## Architecture

### Core Components

**Fish Class** (`fish_school_simulator.py:17-50`)
- Individual fish agent with position, velocity, and startle state
- Tracks startle duration and timing
- Implements startle response (doubling velocity for escape)

**FishSchool Class** (`fish_school_simulator.py:52-314`)
- Main simulation manager containing all fish agents
- Implements three key behavioral systems:
  1. **Couzin Model** (`couzin_behavior` method): Three-zone interaction model
     - Zone of repulsion (< 2.0 units): Avoid collisions
     - Zone of orientation (2.0-10.0 units): Align with neighbors
     - Zone of attraction (10.0-20.0 units): Stay with group
  2. **Predator Detection** (`check_predator_startle` method): Direct startle response when fish detect predator within detection radius
  3. **Startle Cascade** (`check_startle_cascade` method): Social transmission of startle between visible neighbors (60% probability)

### Key Parameters

Located in `FishSchool.__init__`:
- `zone_repulsion`, `zone_orientation`, `zone_attraction`: Couzin model distances
- `max_speed`, `min_speed`: Movement constraints
- `max_turn_rate`: Turning angle limit per timestep (0.3 radians)
- `predator_detection_radius`: How far fish detect predators (25.0 units)
- `startle_transmission_prob`: Probability of startle cascade (0.6)
- `visual_range`: Distance fish can see each other (15.0 units)

### Update Loop

The simulation follows this sequence each frame (`update` method):
1. Check for predator-induced startles (direct detection)
2. Check for startle cascades (social transmission)
3. For each fish:
   - Update startle timer
   - Calculate desired velocity (escape if startled, Couzin model if normal)
   - Apply smooth turning constraint
   - Limit speed to valid range
   - Update position
   - Apply periodic boundary conditions

### Visualization

The `visualize_simulation` function creates a 3D animated plot with:
- Blue dots: Normal fish
- Red dots: Startled fish
- Black X: Predator position
- Red wireframe sphere: Predator detection radius

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
The `get_neighbors` method returns a list of (fish, distance) tuples within a given radius. This is used for both behavioral interactions and startle transmission.
