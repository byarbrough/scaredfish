# Scared Fish Simulation

Final project for Purdue University ECE 60283, Fall 2025

## Overview

A Python-based fish school simulator that combines the **Couzin model** for collective animal behavior with a **SIRS epidemic model** to study how predator-induced startle responses spread through fish schools. The simulation uses empirically-validated transmission probabilities based on distance and visual prominence, derived from real golden shiner fish experiments (Sosna et al. 2019).

**Key Features**:
- 3D real-time visualization of fish schooling behavior
- SIRS epidemic dynamics (Susceptible → Infected → Recovered → Susceptible)
- Empirical startle transmission: P(startle) = 1/(1 + e^(-β₁ - β₂·LMD - β₃·RAA))
- Performance-optimized with Numba JIT compilation and KDTree spatial indexing
- Threshold experiments to identify endemic equilibrium points
- Data export for quantitative analysis

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd scaredfish

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or: .venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install additional packages for data analysis
pip install pandas scikit-learn statsmodels
```

### Running the Simulation

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the main simulation (3D visualization)
python fish_school_simulator.py

# Run threshold experiments
python threshold.py --state relaxed
python threshold.py --state alarmed
```

The simulation opens a 3D matplotlib visualization showing:
- **Blue dots**: Susceptible fish (S)
- **Red dots**: Infected/startled fish (I)
- **Green dots**: Recovered fish (R)
- **Black X**: Predator position
- **Red sphere**: Predator detection radius

## Project Structure

```
scaredfish/
├── fish_school_simulator.py   # Main simulator with Couzin + SIRS models
├── threshold.py                # Threshold experiments for endemic equilibrium
├── test_empirical_model.py     # Unit tests for empirical transmission model
├── test_threshold.py           # Integration tests for threshold experiments
├── regression.ipynb            # Reproduces empirical model coefficients
├── threshold_analysis.ipynb    # Analyzes threshold experiment results
├── datafit.ipynb              # Explores model formulations
├── realdata/                   # Empirical data from Sosna et al. 2019
│   ├── README.md              # Data documentation
│   ├── cascade_sizes.h5       # Cascade size matrices
│   ├── first_responders_*.csv # First responder analysis data
│   └── NND_data.h5            # Nearest-neighbor distance arrays
├── references/                 # Research papers
└── CLAUDE.md                   # Detailed developer documentation
```

## Scientific Background

This simulation models **collective anti-predator behavior** in fish schools using:

1. **Couzin Model**: Three-zone interaction rules (repulsion, orientation, attraction) for realistic schooling
2. **SIRS Epidemic Model**: Startle responses spread like an infection with recovery and re-susceptibility
3. **Empirical Transmission**: Probability based on:
   - **LMD (Log Metric Distance)**: log₁₀ of Euclidean distance between fish
   - **RAA (Ranked Angular Area)**: Visual prominence accounting for field of view and occlusion

**Key Parameters**:
- γ (gamma): Infected duration (~0.5 seconds)
- δ (delta): Recovered duration (variable in threshold experiments)
- β₁, β₂, β₃: Empirical regression coefficients from Sosna et al. 2019

## Results

The simulation successfully:
- ✅ Reproduces empirical startle transmission probabilities
- ✅ Models realistic fish schooling with Couzin zones
- ✅ Identifies endemic equilibrium thresholds for delta parameter
- ✅ Matches nearest-neighbor distance distributions from empirical data
- ✅ Demonstrates cascade propagation dynamics

## References

Sosna, M. M. G., et al. (2019). Individual and collective encoding of risk in animal groups. *Proceedings of the National Academy of Sciences*, 116(50), 20556-20562.

## Documentation

- **[CLAUDE.md](CLAUDE.md)**: Comprehensive developer guide with architecture details
- **[EMPIRICAL_MODEL_SUMMARY.md](EMPIRICAL_MODEL_SUMMARY.md)**: Technical summary of the empirical model
- **[realdata/README.md](realdata/README.md)**: Documentation of empirical datasets

## Testing

```bash
# Run empirical model tests
python test_empirical_model.py

# Run threshold experiment tests
python test_threshold.py

# Type checking with pyright
pyright fish_school_simulator.py
```

## License

Educational project for Purdue University ECE 60283, Fall 2025.
