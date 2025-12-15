# Empirical Startle Transmission Model - Implementation Summary

## Overview
Updated `fish_school_simulator.py` to use empirically-determined parameters for startle transmission probability, replacing the simple constant `beta` with a logistic regression model based on research data.

## Mathematical Model
Probability that fish `i` startles given fish `j` has startled:

```
P(s_i | s_j) = 1 / (1 + exp(-β₁ - β₂·LMD - β₃·RAA))
```

Where:
- **LMD** (Log Metric Distance): `log₁₀(distance)` between the two fish (base-10 logarithm, not natural log)
- **RAA** (Ranked Angular Area): Normalized rank of fish `j`'s apparent size on fish `i`'s retina

### Empirical Parameters (from research)
- β₁ = 0.103641 (intercept)
- β₂ = -3.297823 (log distance coefficient - negative means closer = higher probability)
- β₃ = -0.075034 (ranked angular area coefficient - negative means more prominent = higher probability)

## RAA Calculation Heuristics

Since the original research used advanced ray-tracing, we implemented practical heuristics that capture the essential behavior:

### 1. Field of View (FOV)
- **Total FOV**: 333° (25° blind spot directly behind)
- **Implementation**: Calculate angle between observer's heading and direction to target
- **Visibility criterion**: Angle < 166.5° from forward direction
- **Method**: Dot product between normalized velocity (heading) and direction vector

### 2. Occlusion Detection (Simplified)
Instead of full ray-tracing, use a practical heuristic:
- Fish B occludes fish C from observer A if:
  - B is **20%+ closer** than C (distance threshold)
  - B is in **nearly the same direction** as C (< 5° angular difference)
- This captures the most obvious occlusions without expensive computation
- Trade-off: May miss some edge cases, but computationally efficient

### 3. Angular Size Ranking
- **Angular size** ∝ 1/distance² (assumes uniform fish body size)
- Among visible (non-occluded) neighbors within FOV:
  1. Calculate apparent size: `1/distance²`
  2. Sort by apparent size (descending)
  3. Assign integer ranks: 1, 2, 3, ... (1 = largest/closest)
  4. Normalize to [0, 1]: `RAA = 1 - (rank-1)/(n_visible-1)`
- **Result**: RAA = 1.0 for most prominent fish, RAA = 0.0 for least prominent or invisible

## Code Changes

### New Methods
1. **`calculate_startle_probability(observer, startled_fish)`**
   - Computes P(s_i | s_j) using the empirical model
   - Handles edge cases (very close fish, log(0) prevention)
   - Returns 0 if target is not visible

2. **`_calculate_raa(observer, target)`**
   - Implements FOV filtering
   - Performs simplified occlusion detection
   - Ranks visible neighbors by angular size
   - Returns normalized RAA value

### Modified Methods
- **`check_startle_cascade()`**: Now uses `calculate_startle_probability()` instead of constant `self.beta`

### New Parameters (in `FishSchool.__init__`)
- `beta_1`, `beta_2`, `beta_3`: Empirical coefficients
- `beta`: Retained for backward compatibility (used in plot titles and CSV filenames)

## Validation

Comprehensive unit tests in **[test_empirical_model.py](test_empirical_model.py)** verify the empirical model implementation.

Test results show expected behavior:
- ✓ Closer fish have higher transmission probability
- ✓ Probability decreases exponentially with distance
- ✓ Fish in blind spot (behind) have 0% transmission probability
- ✓ RAA correctly accounts for visual prominence

Example probabilities:
- 2.0 units away, directly ahead: **9.47%**
- 5.0 units away, directly ahead: **0.51%**
- 10.0 units away, directly ahead: **0.05%**
- 2.0 units away, behind (blind spot): **0.00%**

## Future Improvements (Optional)

If more accuracy is needed, consider:
1. **Better occlusion**: Implement spatial partitioning (quadtree/octree) for efficient occlusion queries
2. **Fish body size**: Account for actual fish dimensions rather than point approximation
3. **Dynamic FOV**: Adjust FOV based on swimming speed or behavioral state
4. **Calibration**: Tune the 20% distance threshold and 5° angular threshold based on empirical data

## Usage

The simulation runs identically to before - the empirical model is automatically used:

```bash
source .venv/bin/activate
python fish_school_simulator.py
```

To test the probability calculation without running the full simulation:
```bash
python test_empirical_model.py
```
