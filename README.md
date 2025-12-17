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

## Development History

The following LaTeX block documents all conversation prompts that guided the development of this project:

```latex
\begin{itemize}
\item \textbf{[2025-12-01 21:44]} In regression.ipynb there is the following block: \texttt{\# TODO: fit the logistic regression model, to verify author's findings}. It is refering to the work presented by the two PDFs in \texttt{references/}. Please review the sources and then implement the TODO block, so that I can check the coeffeciencts. I only care about the first exposure to Schreckstoff, not the other trials.

\item \textbf{[2025-12-01 22:30]} We are going to update \texttt{fish\_school\_simulator.py} to use emperically determined parameters. Spefcifically, the probability that an individual $i$ startels given that individual $j$ has startled as $P(s_i | s_j) = \frac{1}{1+e^{-\beta_1-\beta_2 LMD - \beta_3 RAA}}$. LMD is the log of the metric distance between the two fish. RAA is the ranked angular area of fish $j$ (compared to other fish) on the retina of fish $i$. For now, use $\beta_1 = 0.103641$, $\beta_2 = -3.297823$, $\beta_3 = -0.075034$. Strictly speaking, RAA requires identifying a) the fishes field fo view, which is 333 degrees from the heading - there is a 25 degree blnid spot behind the fish b) which other fish that fish can see because they are in the FOV and not obstructed by other fish and c) ranked order for which of those other fish takes up the most visual space. The origianl research used advanced ray tracing. We aren't doing that here! Recommend some simple heuristics that get us a pretty-good rank order.

\item \textbf{[2025-12-01 22:49]} \texttt{fish\_school\_simulator.py} runs smoothly until a fish startles; then it is very slow. Add in the ability to do things in parallel or use the GPU on my apple silicon macbook to speed things up.

\item \textbf{[2025-12-01 23:47]} Give yourself the skill to run \texttt{.venv/bin/black} after you make code changes.

\item \textbf{[2025-12-02 09:53]} We are continuing to make this model more realistic. 1) Make the tank 1m by 2m, 2) The unstartled fish should swim at 10 cm/s, 3) A startled fish should swim at twice that speed (20 cm/s), 4) A startle should last for 1 second before going into the recovered state, so change gamma to match this.

\item \textbf{[2025-12-02 10:17]} Time to change where the predator spawns and how startles initiate. I want it such that: 1) A fish is randomly selected, 2) The predator spawns near that fish, 3) The selected fish startles, 4) The startle of the other fish is entirely dependent on the cascade, no longer on seeing the predator, 5) Keep a count of the fish who are within range of the predator that do not startle and keep a count of the fish who are not within range of the predator but startle anyways; as well as the opposites, 6) Put the startle results into a confusion matrix.

\item \textbf{[2025-12-02 12:18]} \texttt{/init}

\item \textbf{[2025-12-02 14:41]} In \texttt{regression.ipynb} there is a TODO for plotting the data. I would like two simple plots. The first plot should show a subset of the data (keeping in mind that \texttt{Response=1} is rare compared to \texttt{=0}) such that Rank\_area is on the y-axis, Log\_dist\_metric is on the x-axis, the color is red if \texttt{Response=1} else blue, and the shape is circle if \texttt{When=Before} and triangle if \texttt{=After}. The second plot should help visualize the selected regression model; I will let you decide on the specifics.

\item \textbf{[2025-12-02 21:34]} Summarize the SIRS dynamics in a latex algorithmic block. Simple pseudocode.

\item \textbf{[2025-12-02 22:58]} I want to adjust the velocity of a fish upon a state change: S $\rightarrow$ I velocity should be 35 cm/s, I $\rightarrow$ R velocity should be 5 cm/s. In state S, average velocity should be 10 cm/s... perhaps this means that it should be set to 10cm/s upon R $\rightarrow$ S.

\item \textbf{[2025-12-07 14:27]} The simulation is working well. It is now time to actually start doing experiments! Specifically, I want to find the threshold value for delta such that endemic equilibrium occurs a certain percentage of the time. Here's what you need to do: 1) Without breaking the existing \texttt{python fish\_school\_simulator.py} have a way to tweak paramaters \texttt{gamma}, \texttt{delta}, \texttt{n\_fish} currently at the top of \_\_main\_\_, 1a) allow specification of which frame the initial predator appears on, 2) Have a way to suspend animations and run the simulation quickly, 3) Create a new file named \texttt{threshold.py} and in this file have a for loop that simulates the fish school, 4) \texttt{threshold.py} should iterate through values of \texttt{delta} with values $[1,100]$ and a step size of $5$, 5) The initial preadtor should appear at from 400, then dissapear. There is never another predator, 6) At frame 800, count how many fish are in each state, S, I, R and record this in an array alongside the \texttt{delta} value, 7) Run 25 iterations at each delta value, 8) These simulation should be able to occur in parallel.

\item \textbf{[2025-12-07 19:26]} \texttt{threshold.py} outputs two CSVs. I want you to create a jupyter notebook that reads the summary and plots delta on the x axis and average S, I, R on the y axis.

\item \textbf{[2025-12-10 16:36]} Change \texttt{fish\_school\_simulator.py} to accept arguments for the couzing zone\_repulsion, zone\_orientation, and zone\_attraction. Then, add two cells the top of \texttt{threshold\_analysis.ipynb}; we want to model the dynamics with the fish in a scared state, swimming closely together vs. fish in a relaxed state swimming further apart. Review literature to find appropriate zone values; keep in mind that we are respecting the approximated size of the fish and their average speed.

\item \textbf{[2025-12-10 22:05]} I want to use the following epericals: Median Schreckstoff nearest neighbor distance: 3.54 cm, Median water nearest neighbor distance: 6.01 cm. How do I translate those two distances into parameters for the couzin model. Does the size of the fish have to do with it. Explain your reasoning.

\item \textbf{[2025-12-14 09:03]} Remind me how to use \texttt{threshold.py} if I want to test delta from 0.5 to 10, incrementing by .05, with 1000 iterations at each step.

\item \textbf{[2025-12-14 15:31]} Currently \texttt{fish\_school\_simulator} has default zone params defined in at least three places. That's confusing because you don't know which one is master. It should only be defined in a sinlge place, in such a way that it can be (but doesn't have to be) overriden when \texttt{threshold.py} calls it.

\item \textbf{[2025-12-14 22:22]} I am trying to better fit \texttt{DEFAULT\_ZONE\_ATTRACTION} for the relaxed state and the alarmed state. We know that the average nearest negihbor distance in relaxed should be 6cm and the alarmed averaged nearest neighbor should be 3.5 cm. Please add a cell block to \texttt{regression.ipynb} that runs 10 simulations of \texttt{fish\_school\_simulator.py} setting \texttt{DEFAULT\_ZONE\_ATTRACTION=24} and \texttt{=36}. Get the average nearest neighbor distance at time step 400 and then stop. No predator should appear. Furthermore, do get those numbers is \texttt{DEFAULT\_ZONE\_REPULSION=6} and \texttt{=3.5} respectively the right way, or do they need to be lower? You may need to add a function to clculate average nearest neighbor distance.

\item \textbf{[2025-12-15 12:53]} Simulation complete! Time to update \texttt{threshold\_analysis.py}. 1) Remove all of the stuff that saves images to file, 2) See some of the TODOs... add in a comparative analysis between relaxed and alarmed: Add a line to the ``Endemic Equilibrium: Infected Proportion vs Delta'' chart for alarmed, Duplicate the ``SIRS Equilibrium Proportion vs Delta Parameter'' chart for the other state, Duplicate ``Bifurcation Analysis'' chart and put side-by-side.

\item \textbf{[2025-12-15 13:44]} Two taks for you. 1) General refactor and code cleanup. Removing unused functions, clarifying comments, fixing descrepencies..., 2) Update \texttt{CLAUDE.md} and other machine-maintained documentation.

\end{itemize}
```

## License

Educational project for Purdue University ECE 60283, Fall 2025.
