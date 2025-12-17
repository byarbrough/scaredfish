"""
Fish School Simulator using Couzin Model with Empirical Startle Transmission

Based on the Couzin model for collective animal behavior with added
startle response mechanics using empirical parameters from research.

Startle transmission probability: P(s_i | s_j) = 1/(1 + exp(-β1 - β2*LMD - β3*RAA))
where LMD = log(metric distance) and RAA = ranked angular area on retina
"""

from typing import Any, List, Tuple, Optional
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
from numba import jit
from scipy.spatial import KDTree

# Default Couzin model zone parameters (cm)
# Fish body size: 5.5 cm long ≈ 3 cm diameter sphere
# Default zones maintain proper schooling behavior (4x and 8x ratios)
# Can be adjusted to model scared (tight) vs relaxed (loose) formations
DEFAULT_ZONE_REPULSION = 4.5  # Distance for repulsion - prevents overlap
DEFAULT_ZONE_ORIENTATION = 12.0  # Distance for alignment (4× repulsion)
DEFAULT_ZONE_ATTRACTION = 24.0  # Distance for attraction (8× repulsion)


# JIT-compiled helper functions for performance
@jit(nopython=True, cache=True)
def compute_distances_vectorized(
    positions: npt.NDArray[np.float64], target_position: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Vectorized distance computation for all fish to a target position"""
    diff = positions - target_position
    return np.sqrt(np.sum(diff * diff, axis=1))


@jit(nopython=True, cache=True)
def compute_pairwise_distances(
    positions: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute all pairwise distances between positions"""
    n = positions.shape[0]
    distances = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = positions[i] - positions[j]
            dist = np.sqrt(np.sum(diff * diff))
            distances[i, j] = dist
            distances[j, i] = dist
    return distances


@jit(nopython=True, cache=True)
def check_field_of_view(
    observer_velocity: npt.NDArray[np.float64],
    directions_to_targets: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Vectorized FOV check for multiple targets

    Returns boolean array indicating which targets are in FOV
    """
    observer_heading = observer_velocity / np.linalg.norm(observer_velocity)
    max_fov_angle = np.radians(166.5)

    n_targets = directions_to_targets.shape[0]
    in_fov = np.zeros(n_targets, dtype=np.bool_)

    for i in range(n_targets):
        direction_normalized = directions_to_targets[i] / np.linalg.norm(
            directions_to_targets[i]
        )
        dot_product = np.dot(observer_heading, direction_normalized)
        # Manual clip instead of np.clip for numba compatibility
        dot_product = max(-1.0, min(1.0, dot_product))
        angle = np.arccos(dot_product)
        in_fov[i] = angle <= max_fov_angle

    return in_fov


@jit(nopython=True, cache=True)
def compute_occlusion_mask(
    positions: npt.NDArray[np.float64],
    distances: npt.NDArray[np.float64],
    observer_idx: int,
) -> npt.NDArray[np.bool_]:
    """Compute which targets are occluded by closer fish

    Returns boolean array where True means NOT occluded
    """
    n = positions.shape[0]
    not_occluded = np.ones(n, dtype=np.bool_)
    not_occluded[observer_idx] = False  # Can't see self

    observer_pos = positions[observer_idx]

    for i in range(n):
        if i == observer_idx or not_occluded[i] == False:
            continue

        direction_i = positions[i] - observer_pos
        direction_i_norm = direction_i / np.linalg.norm(direction_i)
        dist_i = distances[observer_idx, i]

        # Check if any other fish occludes this one
        for j in range(n):
            if j == observer_idx or j == i:
                continue

            dist_j = distances[observer_idx, j]

            # Is j significantly closer? (20% threshold)
            if dist_j < dist_i * 0.8:
                direction_j = positions[j] - observer_pos
                direction_j_norm = direction_j / np.linalg.norm(direction_j)

                # Are they in nearly the same direction? (< 10 degrees)
                # Updated from 5° to 10° to account for 3 cm fish body size
                dot_prod = np.dot(direction_i_norm, direction_j_norm)
                # Manual clip for numba compatibility
                dot_prod = max(-1.0, min(1.0, dot_prod))
                angular_diff = np.arccos(dot_prod)

                if angular_diff < np.radians(10):
                    not_occluded[i] = False
                    break

    return not_occluded


@jit(nopython=True, cache=True)
def compute_startle_probabilities(
    distances: npt.NDArray[np.float64],
    raas: npt.NDArray[np.float64],
    beta_1: float,
    beta_2: float,
    beta_3: float,
) -> npt.NDArray[np.float64]:
    """Vectorized startle probability computation"""
    n = len(distances)
    probs = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if raas[i] == 0.0:
            probs[i] = 0.0
        else:
            # Avoid log(0)
            dist = max(distances[i], 0.01)
            lmd = np.log10(dist)
            logit = beta_1 + beta_2 * lmd + beta_3 * raas[i]
            probs[i] = 1.0 / (1.0 + np.exp(-logit))

    return probs


class Fish:
    """Individual fish agent with Couzin model behavior and SIRS startle response"""

    def __init__(
        self,
        position: npt.ArrayLike,
        velocity: npt.ArrayLike,
        fish_id: int,
        infected_duration: float = 10,
        recovered_duration: float = 20,
    ) -> None:
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.id = fish_id
        self.state = "susceptible"
        self.state_timer = 0
        self.infected_duration = infected_duration
        self.recovered_duration = recovered_duration

    def update_state(self) -> None:
        """Update SIRS state transitions"""
        self.state_timer += 1

        if self.state == "infected":
            if self.state_timer >= self.infected_duration:
                self.state = "recovered"
                # Slow down to 5 cm/s (0.25 cm/frame at 20 fps) after startle response
                velocity_direction = self.velocity / np.linalg.norm(self.velocity)
                self.velocity = velocity_direction * 0.25
                self.state_timer = 0

        elif self.state == "recovered":
            if self.state_timer >= self.recovered_duration:
                self.state = "susceptible"
                # Return to average susceptible speed: 10 cm/s (0.5 cm/frame at 20 fps)
                velocity_direction = self.velocity / np.linalg.norm(self.velocity)
                self.velocity = velocity_direction * 0.5
                self.state_timer = 0

    def infect(self) -> None:
        """Trigger infection (startle response) - only works if susceptible"""
        if self.state == "susceptible":
            self.state = "infected"
            self.state_timer = 0
            # Startle causes rapid acceleration to 35 cm/s (1.75 cm/frame at 20 fps)
            velocity_direction = self.velocity / np.linalg.norm(self.velocity)
            self.velocity = velocity_direction * 1.75


class FishSchool:
    """Simulation of fish school using Couzin model with empirical startle transmission

    Startle transmission uses logistic regression model:
    P(s_i | s_j) = 1 / (1 + exp(-β1 - β2*LMD - β3*RAA))
    where LMD = log(metric distance) and RAA = ranked angular area
    """

    def __init__(
        self,
        n_fish: int = 50,
        space_size: float = 100,
        beta: float = 0.6,
        gamma: float = 20,
        delta: float = 20,
        zone_repulsion: float = DEFAULT_ZONE_REPULSION,
        zone_orientation: float = DEFAULT_ZONE_ORIENTATION,
        zone_attraction: float = DEFAULT_ZONE_ATTRACTION,
        verbose: bool = False,
    ) -> None:
        self.n_fish = n_fish
        # Support both single space_size (for backward compatibility) and tuple
        if isinstance(space_size, (tuple, list, np.ndarray)):
            self.space_dimensions = np.array(space_size, dtype=float)
        else:
            self.space_dimensions = np.array(
                [space_size, space_size, space_size], dtype=float
            )
        self.space_size = space_size  # Keep for backward compatibility

        # Couzin model parameters
        self.zone_repulsion = zone_repulsion
        self.zone_orientation = zone_orientation
        self.zone_attraction = zone_attraction

        # Speed parameters (in cm/frame at 20 fps)
        # min_speed: 0.5 cm/frame = 10 cm/s at 20 fps
        # max_speed: 1.0 cm/frame = 20 cm/s at 20 fps
        self.max_speed = 1.0
        self.min_speed = 0.5
        self.max_turn_rate = 0.3  # Maximum turning angle per step

        # SIRS epidemic model parameters
        self.beta = beta  # Kept for backward compatibility (used in plots/export)
        self.gamma = gamma  # Infected duration (frames before I -> R)
        self.delta = delta  # Recovered duration (frames before R -> S)

        # Empirical startle transmission parameters (logistic regression model)
        self.beta_1 = 0.103641  # Intercept
        self.beta_2 = -3.297823  # Log metric distance coefficient
        self.beta_3 = -0.075034  # Ranked angular area coefficient

        # Startle cascade parameters
        self.predator_position = None
        self.predator_detection_radius = 50.0  # Increased for better detection
        self.visual_range = (
            120.0  # How far fish can see each other for social transmission
        )

        # Debug tracking
        self.total_startles = 0
        self.cascade_startles = 0
        self.verbose = verbose  # Control debug output

        # Confusion matrix tracking for cascade effectiveness
        self.selected_fish_id: Optional[int] = None  # ID of fish that spawns predator
        self.fish_within_range_at_spawn: set[int] = (
            set()
        )  # IDs of fish within range when predator spawned
        self.fish_that_startled: set[int] = (
            set()
        )  # IDs of fish that startled during predator presence
        self.confusion_matrix = {
            "true_positive": 0,  # Within range, startled via cascade
            "false_positive": 0,  # NOT within range, startled via cascade
            "true_negative": 0,  # NOT within range, did NOT startle
            "false_negative": 0,  # Within range, did NOT startle
        }

        # Data collection for SIR dynamics
        self.history = {"time": [], "S": [], "I": [], "R": []}

        # Initialize fish
        self.fish = []
        for i in range(n_fish):
            # Random positions within the space dimensions
            pos = np.array(
                [
                    np.random.uniform(0, self.space_dimensions[0]),
                    np.random.uniform(0, self.space_dimensions[1]),
                    np.random.uniform(0, self.space_dimensions[2]),
                ]
            )
            # Random initial velocities
            vel = np.random.uniform(-1, 1, 3)
            vel = self._normalize(vel) * self.min_speed
            self.fish.append(
                Fish(pos, vel, i, infected_duration=gamma, recovered_duration=delta)
            )

        self.time_step = 0

        # Spatial indexing for fast neighbor queries
        self.spatial_tree: Optional[KDTree] = None
        self._cached_positions: Optional[npt.NDArray[np.float64]] = None
        self._cached_velocities: Optional[npt.NDArray[np.float64]] = None

    def _normalize(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Normalize a vector"""
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            return np.array([1.0, 0.0, 0.0])
        return vector / norm

    def _limit_speed(
        self, velocity: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Limit velocity to min/max speed"""
        speed = np.linalg.norm(velocity)
        if speed > self.max_speed:
            return velocity * (self.max_speed / speed)
        elif speed < self.min_speed and speed > 1e-10:
            return velocity * (self.min_speed / speed)
        return velocity

    def _smooth_turn(
        self, current_vel: npt.NDArray[np.float64], desired_vel: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Limit turning rate for realistic movement"""
        current_dir = self._normalize(current_vel)
        desired_dir = self._normalize(desired_vel)

        # Calculate angle between directions
        dot_product = np.clip(np.dot(current_dir, desired_dir), -1.0, 1.0)
        angle = np.arccos(dot_product)

        # If turn is too sharp, interpolate
        if angle > self.max_turn_rate:
            # Spherical linear interpolation
            t = self.max_turn_rate / angle
            new_dir = current_dir + t * (desired_dir - current_dir)
            new_dir = self._normalize(new_dir)
        else:
            new_dir = desired_dir

        # Maintain speed
        speed = np.linalg.norm(current_vel)
        return new_dir * speed

    def _rebuild_spatial_index(self) -> None:
        """Rebuild spatial index and cache position/velocity arrays"""
        self._cached_positions = np.array(
            [f.position for f in self.fish], dtype=np.float64
        )
        self._cached_velocities = np.array(
            [f.velocity for f in self.fish], dtype=np.float64
        )
        self.spatial_tree = KDTree(self._cached_positions)

    def get_neighbors(self, fish: Fish, radius: float) -> List[Tuple[Fish, float]]:
        """Get all neighbors within radius using spatial indexing"""
        if self.spatial_tree is None:
            self._rebuild_spatial_index()

        assert self.spatial_tree is not None
        # Query spatial tree for neighbors
        indices = self.spatial_tree.query_ball_point(fish.position, radius)

        neighbors = []
        for idx in indices:
            if self.fish[idx].id != fish.id:
                dist = np.linalg.norm(fish.position - self.fish[idx].position)
                neighbors.append((self.fish[idx], dist))
        return neighbors

    def get_neighbors_fast(
        self, fish_idx: int, radius: float
    ) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Fast neighbor query returning indices and distances

        Returns:
            Tuple of (neighbor_indices, distances)
        """
        if self.spatial_tree is None or self._cached_positions is None:
            self._rebuild_spatial_index()

        assert self._cached_positions is not None
        assert self.spatial_tree is not None
        indices = self.spatial_tree.query_ball_point(
            self._cached_positions[fish_idx], radius
        )

        # Filter out self
        indices = [i for i in indices if i != fish_idx]

        if not indices:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        indices = np.array(indices, dtype=np.int64)
        # Compute distances vectorized
        distances = np.linalg.norm(
            self._cached_positions[indices] - self._cached_positions[fish_idx], axis=1
        )

        return indices, distances

    def _calculate_raa_batch(
        self, observer_idx: int, target_indices: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """Calculate RAA for multiple targets at once (vectorized)

        Args:
            observer_idx: Index of observing fish
            target_indices: Array of target fish indices

        Returns:
            Array of RAA values for each target
        """
        if len(target_indices) == 0:
            return np.array([], dtype=np.float64)

        if self._cached_positions is None or self._cached_velocities is None:
            self._rebuild_spatial_index()

        assert self._cached_positions is not None
        assert self._cached_velocities is not None

        # Get all neighbors within visual range
        neighbor_indices, neighbor_distances = self.get_neighbors_fast(
            observer_idx, self.visual_range
        )

        if len(neighbor_indices) == 0:
            return np.zeros(len(target_indices), dtype=np.float64)

        # Calculate directions from observer to all neighbors
        observer_pos = self._cached_positions[observer_idx]
        observer_vel = self._cached_velocities[observer_idx]
        neighbor_positions = self._cached_positions[neighbor_indices]
        directions = neighbor_positions - observer_pos

        # Check field of view for all neighbors
        in_fov = check_field_of_view(observer_vel, directions)

        # Compute occlusion mask
        all_distances = np.linalg.norm(self._cached_positions - observer_pos, axis=1)
        not_occluded = compute_occlusion_mask(
            self._cached_positions,
            all_distances.reshape(1, -1).repeat(len(self._cached_positions), axis=0),
            observer_idx,
        )

        # Combine FOV and occlusion to get visible fish
        visible_mask = np.zeros(len(self._cached_positions), dtype=bool)
        visible_mask[neighbor_indices] = in_fov & not_occluded[neighbor_indices]

        if not np.any(visible_mask):
            return np.zeros(len(target_indices), dtype=np.float64)

        # Calculate angular sizes for visible fish
        visible_indices = np.where(visible_mask)[0]
        visible_distances = all_distances[visible_indices]
        angular_sizes = 1.0 / (visible_distances * visible_distances)

        # Sort by angular size (largest first) and get ranks
        sorted_order = np.argsort(-angular_sizes)  # Negative for descending
        ranks = np.empty(len(sorted_order), dtype=np.int64)
        ranks[sorted_order] = np.arange(1, len(sorted_order) + 1)

        # Create mapping from fish index to RAA
        raa_map = np.zeros(len(self._cached_positions), dtype=np.float64)
        n_visible = len(visible_indices)

        if n_visible == 1:
            raa_map[visible_indices] = 1.0
        else:
            # Normalize ranks to [0, 1]
            raa_values = 1.0 - (ranks - 1) / (n_visible - 1)
            raa_map[visible_indices] = raa_values

        # Return RAA values for requested targets
        return raa_map[target_indices]

    def _calculate_raa(self, observer: Fish, target: Fish) -> float:
        """Calculate Ranked Angular Area of target fish on observer's retina

        Uses simplified heuristics:
        1. Field of view: 333 degrees (25-degree blind spot behind)
        2. Occlusion: Simplified (fish occluded if another is 20%+ closer in same direction)
        3. Angular size: Proportional to 1/distance^2
        4. Ranking: Normalize ranks to [0, 1] where 1 = largest/closest visible fish

        Args:
            observer: The fish doing the observing
            target: The startled fish being observed

        Returns:
            RAA value between 0 and 1 (1 = most prominent, 0 = least prominent or not visible)
        """
        # Get all potential neighbors within visual range
        neighbors = self.get_neighbors(observer, self.visual_range)

        if not neighbors:
            return 0.0

        # Calculate observer's heading direction
        observer_heading = self._normalize(observer.velocity)

        # Filter to fish in field of view and compute their angular properties
        visible_fish = []

        for neighbor, dist in neighbors:
            # Direction from observer to neighbor
            direction_to_neighbor = self._normalize(
                neighbor.position - observer.position
            )

            # Check field of view (333 degrees = can see 166.5° to each side)
            dot_product = np.clip(
                np.dot(observer_heading, direction_to_neighbor), -1.0, 1.0
            )
            angle_to_neighbor = np.arccos(dot_product)

            # Can see if angle < 166.5 degrees (333/2 converted to radians)
            max_fov_angle = np.radians(166.5)
            if angle_to_neighbor > max_fov_angle:
                continue  # In blind spot

            # Simple occlusion check: is there a much closer fish in nearly the same direction?
            occluded = False
            for other_neighbor, other_dist in neighbors:
                if other_neighbor.id == neighbor.id:
                    continue

                # Check if other fish is significantly closer (20% threshold)
                if other_dist < dist * 0.8:
                    direction_to_other = self._normalize(
                        other_neighbor.position - observer.position
                    )
                    angular_diff = np.arccos(
                        np.clip(
                            np.dot(direction_to_neighbor, direction_to_other), -1.0, 1.0
                        )
                    )
                    # If angular difference < 10 degrees, consider occluded
                    # Updated from 5° to 10° to account for 3 cm fish body size
                    if angular_diff < np.radians(10):
                        occluded = True
                        break

            if not occluded:
                # Calculate apparent angular size (proportional to 1/distance^2)
                angular_size = 1.0 / (dist * dist)
                visible_fish.append((neighbor, dist, angular_size))

        if not visible_fish:
            return 0.0

        # Find the target fish in visible list
        target_visible = False
        for fish, dist, ang_size in visible_fish:
            if fish.id == target.id:
                target_visible = True
                break

        if not target_visible:
            # Target is not visible (occluded or out of FOV)
            return 0.0

        # Sort by angular size (largest first)
        visible_fish.sort(key=lambda x: x[2], reverse=True)

        # Find rank of target fish (1-indexed)
        rank = None
        for r, (fish, dist, ang_size) in enumerate(visible_fish, start=1):
            if fish.id == target.id:
                rank = r
                break

        if rank is None:
            return 0.0

        # Normalize rank to [0, 1] where 1 = largest (rank 1), 0 = smallest
        n_visible = len(visible_fish)
        if n_visible == 1:
            raa = 1.0
        else:
            raa = 1.0 - (rank - 1) / (n_visible - 1)

        return raa

    def calculate_startle_probability(
        self, observer: Fish, startled_fish: Fish
    ) -> float:
        """Calculate probability that observer startles given startled_fish has startled

        Uses empirical model from research:
        P(s_i | s_j) = 1 / (1 + exp(-β1 - β2*LMD - β3*RAA))

        where:
        - LMD = log(metric distance between fish)
        - RAA = ranked angular area (normalized rank by apparent size on retina)

        Args:
            observer: Susceptible fish that might startle
            startled_fish: Infected fish that has already startled

        Returns:
            Probability between 0 and 1
        """
        # Calculate distance
        distance = np.linalg.norm(observer.position - startled_fish.position)

        # Avoid log(0) - if fish are extremely close, use small positive value
        if distance < 0.01:
            distance = 0.01

        # LMD: log10 of metric distance
        lmd = np.log10(distance)

        # RAA: Calculate ranked angular area
        raa = self._calculate_raa(observer, startled_fish)

        # If target is not visible, probability is 0
        if raa == 0.0:
            return 0.0

        # Logistic regression model
        logit = self.beta_1 + self.beta_2 * lmd + self.beta_3 * raa
        probability = 1.0 / (1.0 + np.exp(-logit))
        # NOTE: This function is NOT used during cascade checks!
        # Cascades use compute_startle_probabilities() (JIT-compiled) which has its own RAA==0 check
        return probability

    def couzin_behavior(self, fish: Fish) -> npt.NDArray[np.float64]:
        """Apply Couzin model rules to determine desired velocity"""

        # Get neighbors in different zones
        all_neighbors = self.get_neighbors(fish, self.zone_attraction)

        repulsion_force = np.zeros(3)
        orientation_force = np.zeros(3)
        attraction_force = np.zeros(3)

        n_repulsion = 0
        n_orientation = 0
        n_attraction = 0

        for neighbor, dist in all_neighbors:
            direction = fish.position - neighbor.position

            # Zone of repulsion
            if dist < self.zone_repulsion:
                repulsion_force += self._normalize(direction)
                n_repulsion += 1

            # Zone of orientation
            elif dist < self.zone_orientation:
                orientation_force += neighbor.velocity
                n_orientation += 1

            # Zone of attraction
            elif dist < self.zone_attraction:
                attraction_force += neighbor.position - fish.position
                n_attraction += 1

        # Calculate desired velocity based on Couzin rules
        desired_velocity = np.zeros(3)

        # Priority 1: Repulsion (highest priority)
        if n_repulsion > 0:
            desired_velocity = repulsion_force / n_repulsion

        # Priority 2: Orientation + Attraction
        else:
            if n_orientation > 0:
                desired_velocity += orientation_force / n_orientation

            if n_attraction > 0:
                desired_velocity += attraction_force / n_attraction

        # If no neighbors, maintain current direction
        if np.linalg.norm(desired_velocity) < 1e-10:
            desired_velocity = fish.velocity

        return desired_velocity

    def check_predator_startle(self) -> None:
        """DISABLED: Direct predator detection is now disabled.

        The selected fish is infected directly in spawn_predator().
        All other startles must occur via cascade (social transmission).
        This allows testing cascade effectiveness with a confusion matrix.
        """
        # Do nothing - cascade-only transmission
        return

    def check_startle_cascade(self) -> None:
        """Check if infected fish trigger cascade in susceptible neighbors using empirical model (optimized)"""
        # Rebuild spatial index with current positions
        self._rebuild_spatial_index()

        assert self._cached_positions is not None

        # Get indices of infected and susceptible fish
        infected_indices = [i for i, f in enumerate(self.fish) if f.state == "infected"]
        susceptible_indices = np.array(
            [i for i, f in enumerate(self.fish) if f.state == "susceptible"],
            dtype=np.int64,
        )

        if len(infected_indices) == 0 or len(susceptible_indices) == 0:
            return

        newly_infected_count = 0

        # Debug tracking
        total_neighbors_in_range = 0
        total_with_nonzero_raa = 0
        total_susceptible_checked = 0

        # Process each infected fish
        for infected_idx in infected_indices:
            # Get susceptible neighbors within visual range
            neighbor_indices, neighbor_distances = self.get_neighbors_fast(
                infected_idx, self.visual_range
            )

            if len(neighbor_indices) == 0:
                continue

            # Filter to only susceptible fish
            susceptible_mask = np.isin(neighbor_indices, susceptible_indices)
            susceptible_neighbors = neighbor_indices[susceptible_mask]
            susceptible_distances = neighbor_distances[susceptible_mask]

            if len(susceptible_neighbors) == 0:
                continue

            total_neighbors_in_range += len(susceptible_neighbors)
            total_susceptible_checked += 1

            # Batch calculate RAA values (observer = susceptible, target = infected)
            # We need RAA from each susceptible fish's perspective looking at the infected fish
            raas = np.zeros(len(susceptible_neighbors), dtype=np.float64)
            for i, susc_idx in enumerate(susceptible_neighbors):
                # Calculate RAA for this susceptible fish observing the infected fish
                raa_values = self._calculate_raa_batch(
                    int(susc_idx), np.array([infected_idx], dtype=np.int64)
                )
                raas[i] = raa_values[0] if len(raa_values) > 0 else 0.0

            # Track RAA visibility
            nonzero_raas = np.sum(raas > 0.0)
            total_with_nonzero_raa += nonzero_raas

            # Calculate probabilities using vectorized function
            probs = compute_startle_probabilities(
                susceptible_distances, raas, self.beta_1, self.beta_2, self.beta_3
            )

            # Debug: print detailed info for visible fish
            if self.verbose and nonzero_raas > 0:
                for i in range(len(raas)):
                    if raas[i] > 0:
                        print(
                            f"  Susceptible fish {susceptible_neighbors[i]}: "
                            f"dist={susceptible_distances[i]:.2f}cm, "
                            f"RAA={raas[i]:.3f}, "
                            f"prob={probs[i]:.3f}"
                        )

            # Randomly determine infections
            random_values = np.random.random(len(probs))
            infected_mask = random_values < probs

            # Infect susceptible fish
            for susc_idx in susceptible_neighbors[infected_mask]:
                self.fish[susc_idx].infect()
                newly_infected_count += 1
                self.cascade_startles += 1
                # Track that this fish startled (for confusion matrix)
                self.fish_that_startled.add(self.fish[susc_idx].id)

        # Debug output every frame with infected fish
        if self.verbose and total_susceptible_checked > 0:
            print(
                f"[Frame {self.time_step}] CASCADE CHECK: {len(infected_indices)} infected, "
                f"{total_neighbors_in_range} susceptible in range, "
                f"{total_with_nonzero_raa} visible (RAA>0), "
                f"{newly_infected_count} newly infected"
            )

        if self.verbose and newly_infected_count > 0:
            print(
                f"[Frame {self.time_step}] CASCADE SUCCESS: {newly_infected_count} fish infected by seeing others!"
            )

    def apply_boundaries(self, fish: Fish) -> None:
        """Apply reflective boundary conditions (bounce off walls like a tank)"""
        for i in range(3):
            # Check lower boundary
            if fish.position[i] < 0:
                fish.position[i] = -fish.position[i]  # Reflect position
                fish.velocity[i] = -fish.velocity[i]  # Reverse velocity

            # Check upper boundary
            elif fish.position[i] > self.space_dimensions[i]:
                fish.position[i] = (
                    2 * self.space_dimensions[i] - fish.position[i]
                )  # Reflect position
                fish.velocity[i] = -fish.velocity[i]  # Reverse velocity

    def update(self) -> None:
        """Update simulation one time step"""
        self.time_step += 1

        # Rebuild spatial index with current positions for fast neighbor queries
        self._rebuild_spatial_index()

        # Check for predator-induced infections
        self.check_predator_startle()

        # Check for infection cascades
        self.check_startle_cascade()

        # Update each fish
        new_velocities = []
        for fish in self.fish:
            # Update SIRS state
            fish.update_state()

            if fish.state == "infected":
                # Infected fish: escape behavior (move away from predator)
                if self.predator_position is not None:
                    escape_direction = fish.position - self.predator_position
                    desired_velocity = (
                        self._normalize(escape_direction) * self.max_speed
                    )
                else:
                    desired_velocity = fish.velocity * 1.5
            else:
                # Normal behavior (susceptible or recovered): apply Couzin model
                desired_velocity = self.couzin_behavior(fish)

            # Smooth turning
            desired_velocity = self._normalize(desired_velocity) * np.linalg.norm(
                fish.velocity
            )
            new_velocity = self._smooth_turn(fish.velocity, desired_velocity)

            # Limit speed
            new_velocity = self._limit_speed(new_velocity)
            new_velocities.append(new_velocity)

        # Update velocities and positions
        for fish, new_vel in zip(self.fish, new_velocities):
            fish.velocity = new_vel
            fish.position += fish.velocity
            self.apply_boundaries(fish)

        # Collect SIR data
        s_count = sum(1 for f in self.fish if f.state == "susceptible")
        i_count = sum(1 for f in self.fish if f.state == "infected")
        r_count = sum(1 for f in self.fish if f.state == "recovered")

        self.history["time"].append(self.time_step)
        self.history["S"].append(s_count / self.n_fish)
        self.history["I"].append(i_count / self.n_fish)
        self.history["R"].append(r_count / self.n_fish)

    def get_school_center(self) -> npt.NDArray[np.float64]:
        """Calculate center of mass of the fish school"""
        positions = np.array([fish.position for fish in self.fish])
        return np.mean(positions, axis=0)

    def get_average_nearest_neighbor_distance(self) -> float:
        """Calculate average nearest neighbor distance across all fish

        Returns:
            Average of the nearest neighbor distance for each fish (in cm)
        """
        if self._cached_positions is None:
            self._rebuild_spatial_index()

        assert self._cached_positions is not None

        # Compute pairwise distances using vectorized operation
        pairwise_distances = compute_pairwise_distances(self._cached_positions)

        # For each fish, find its nearest neighbor (excluding itself)
        nearest_neighbor_distances = []
        for i in range(len(self.fish)):
            # Set self-distance to infinity to exclude it
            distances_from_i = pairwise_distances[i].copy()
            distances_from_i[i] = np.inf

            # Find minimum distance
            min_dist = np.min(distances_from_i)
            nearest_neighbor_distances.append(min_dist)

        return float(np.mean(nearest_neighbor_distances))

    def spawn_predator(
        self, selected_fish: Optional[Fish] = None, offset_distance: float = 5.0
    ) -> None:
        """Spawn predator near a randomly selected fish

        Args:
            selected_fish: Fish to spawn near (if None, randomly selects one)
            offset_distance: Distance offset from selected fish (in cm)
        """
        # Randomly select a fish if not provided
        if selected_fish is None:
            selected_fish = np.random.choice(self.fish)

        # Type narrowing: ensure selected_fish is not None
        assert selected_fish is not None

        # Store the selected fish ID for confusion matrix tracking
        self.selected_fish_id = selected_fish.id

        # Spawn predator near the selected fish with a small random offset
        random_direction = np.random.randn(3)
        random_direction = self._normalize(random_direction)
        self.predator_position = (
            selected_fish.position + random_direction * offset_distance
        )

        # Immediately infect the selected fish (initial startle)
        selected_fish.infect()
        self.total_startles = 1  # Count the initial selected fish
        self.cascade_startles = 0  # Reset cascade counter

        # Reset tracking for this predator event
        self.fish_within_range_at_spawn = set()
        self.fish_that_startled = set()

        # Store which fish are within detection range at spawn time (for confusion matrix)
        for fish in self.fish:
            distance = np.linalg.norm(fish.position - self.predator_position)
            if distance < self.predator_detection_radius:
                self.fish_within_range_at_spawn.add(fish.id)

        print(
            f"\n[Frame {self.time_step}] PREDATOR SPAWNED near fish {selected_fish.id}"
        )
        print(f"  Predator position: {self.predator_position}")
        print(f"  Selected fish position: {selected_fish.position}")
        print(f"  Distance to selected fish: {offset_distance:.2f} cm")
        print(f"  Detection radius: {self.predator_detection_radius} cm")
        print(
            f"  Fish within detection range: {len(self.fish_within_range_at_spawn)}/{len(self.fish)}"
        )
        print(f"  Fish IDs within range: {sorted(self.fish_within_range_at_spawn)}")

    def calculate_confusion_matrix(self) -> None:
        """Calculate confusion matrix comparing predator range vs cascade effectiveness

        Ground truth: Whether fish was within predator detection radius AT SPAWN TIME
        Prediction: Whether fish startled (infected or recovered state) by removal time

        Excludes the initially selected fish from the matrix.
        """
        if self.selected_fish_id is None:
            print("Warning: Cannot calculate confusion matrix - selected fish not set")
            return

        # Reset confusion matrix
        self.confusion_matrix = {
            "true_positive": 0,
            "false_positive": 0,
            "true_negative": 0,
            "false_negative": 0,
        }

        for fish in self.fish:
            # Skip the initially selected fish
            if fish.id == self.selected_fish_id:
                continue

            # Use stored information about whether fish was within range at spawn
            within_range = fish.id in self.fish_within_range_at_spawn

            # Check if fish startled at any point during predator presence (via cascade)
            startled = fish.id in self.fish_that_startled

            # Populate confusion matrix
            if within_range and startled:
                self.confusion_matrix["true_positive"] += 1
            elif not within_range and startled:
                self.confusion_matrix["false_positive"] += 1
            elif not within_range and not startled:
                self.confusion_matrix["true_negative"] += 1
            elif within_range and not startled:
                self.confusion_matrix["false_negative"] += 1

        # Print results
        print("\n" + "=" * 60)
        print("CONFUSION MATRIX - Cascade Effectiveness")
        print("=" * 60)
        print("Ground Truth: Within predator detection range AT SPAWN TIME")
        print("Prediction: Fish startled via cascade DURING PREDATOR PRESENCE")
        print(f"(Excluding initially selected fish #{self.selected_fish_id})")
        print(
            f"({len(self.fish_within_range_at_spawn) - 1} fish were within range at spawn, excluding selected)"
        )
        print(f"({len(self.fish_that_startled)} fish startled via cascade)")
        print()
        print(f"                    Startled    Not Startled")
        print(
            f"Within Range        {self.confusion_matrix['true_positive']:4d} (TP)    {self.confusion_matrix['false_negative']:4d} (FN)"
        )
        print(
            f"NOT Within Range    {self.confusion_matrix['false_positive']:4d} (FP)    {self.confusion_matrix['true_negative']:4d} (TN)"
        )
        print()

        # Calculate metrics
        tp = self.confusion_matrix["true_positive"]
        fp = self.confusion_matrix["false_positive"]
        tn = self.confusion_matrix["true_negative"]
        fn = self.confusion_matrix["false_negative"]

        total = tp + fp + tn + fn
        if total > 0:
            accuracy = (tp + tn) / total
            print(f"Accuracy: {accuracy:.2%} ({tp + tn}/{total})")

        if tp + fn > 0:
            sensitivity = tp / (tp + fn)  # True Positive Rate
            print(
                f"Sensitivity (TPR): {sensitivity:.2%} - Fish in danger that startled"
            )

        if tn + fp > 0:
            specificity = tn / (tn + fp)  # True Negative Rate
            print(
                f"Specificity (TNR): {specificity:.2%} - Fish safe that didn't startle"
            )

        if tp + fp > 0:
            precision = tp / (tp + fp)
            print(
                f"Precision: {precision:.2%} - Startled fish that were actually in danger"
            )

        print("=" * 60 + "\n")

    def remove_predator(self) -> None:
        """Remove predator from simulation and calculate confusion matrix"""
        if self.predator_position is not None:
            # Calculate confusion matrix before removing predator
            self.calculate_confusion_matrix()

        self.predator_position = None


def export_sir_data(school: FishSchool, beta: float, gamma: float, delta: float) -> str:
    """Export SIR dynamics data to CSV file

    Returns:
        str: Path to the created CSV file
    """
    filename = f"{beta}_{gamma}_{delta}.csv"

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["time", "S_proportion", "I_proportion", "R_proportion"])

        for i in range(len(school.history["time"])):
            writer.writerow(
                [
                    school.history["time"][i],
                    school.history["S"][i],
                    school.history["I"][i],
                    school.history["R"][i],
                ]
            )

    print(f"SIR data exported to {filename}")
    return filename


def plot_sir_dynamics(
    school: FishSchool, predator_spawn: int = 100, predator_remove: int = 200
) -> None:
    """Plot SIRS dynamics over time showing proportions of S, I, R states"""
    fig, ax = plt.subplots(figsize=(12, 6))

    times = school.history["time"]
    s_prop = school.history["S"]
    i_prop = school.history["I"]
    r_prop = school.history["R"]

    # Plot S, I, R proportions
    ax.plot(times, s_prop, label="Susceptible (S)", color="blue", linewidth=2)
    ax.plot(times, i_prop, label="Infected (I)", color="red", linewidth=2)
    ax.plot(times, r_prop, label="Recovered (R)", color="green", linewidth=2)

    # Add vertical lines for predator events
    ax.axvline(
        x=predator_spawn,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Predator Spawn",
    )
    ax.axvline(
        x=predator_remove,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Predator Removed",
    )

    ax.set_xlabel("Time (frames)", fontsize=12)
    ax.set_ylabel("Proportion of Fish", fontsize=12)
    ax.set_title(
        f"SIRS Dynamics (γ={school.gamma}, δ={school.delta})",
        fontsize=14,
    )
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(school: FishSchool) -> None:
    """Plot confusion matrix as a heatmap"""
    # Extract values
    tp = school.confusion_matrix["true_positive"]
    fp = school.confusion_matrix["false_positive"]
    tn = school.confusion_matrix["true_negative"]
    fn = school.confusion_matrix["false_negative"]

    # Create matrix
    confusion_data = np.array([[tp, fn], [fp, tn]])

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    im = ax.imshow(confusion_data, cmap="Blues", aspect="auto")

    # Set ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Startled", "Not Startled"])
    ax.set_yticklabels(["Within Range", "NOT Within Range"])

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(
                j,
                i,
                f"{confusion_data[i, j]}\n({['TP', 'FN', 'FP', 'TN'][i * 2 + j]})",
                ha="center",
                va="center",
                color=(
                    "white"
                    if confusion_data[i, j] > confusion_data.max() / 2
                    else "black"
                ),
                fontsize=14,
                weight="bold",
            )

    # Labels and title
    ax.set_xlabel("Prediction (Cascade Result)", fontsize=12)
    ax.set_ylabel("Ground Truth (Predator Range)", fontsize=12)
    ax.set_title(
        f"Cascade Effectiveness - Confusion Matrix\n(Excluding initially selected fish #{school.selected_fish_id})",
        fontsize=14,
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Number of Fish", rotation=270, labelpad=20)

    # Calculate and display metrics
    total = tp + fp + tn + fn
    metrics_text = ""
    if total > 0:
        accuracy = (tp + tn) / total
        metrics_text += f"Accuracy: {accuracy:.1%}\n"
    if tp + fn > 0:
        sensitivity = tp / (tp + fn)
        metrics_text += f"Sensitivity: {sensitivity:.1%}\n"
    if tn + fp > 0:
        specificity = tn / (tn + fp)
        metrics_text += f"Specificity: {specificity:.1%}\n"
    if tp + fp > 0:
        precision = tp / (tp + fp)
        metrics_text += f"Precision: {precision:.1%}"

    # Add metrics text box
    if metrics_text:
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(
            1.35,
            0.5,
            metrics_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="center",
            bbox=props,
        )

    plt.tight_layout()
    plt.show()


def run_simulation(
    n_fish: int = 50,
    n_steps: int = 1000,
    predator_spawn_frame: int = 100,
    predator_remove_frame: Optional[int] = None,
    gamma: float = 20,
    delta: float = 20,
    space_size: Any = 100,
    beta: float = 0.6,
    verbose: bool = False,
    zone_repulsion: float = DEFAULT_ZONE_REPULSION,
    zone_orientation: float = DEFAULT_ZONE_ORIENTATION,
    zone_attraction: float = DEFAULT_ZONE_ATTRACTION,
) -> FishSchool:
    """Run simulation in headless mode (no animation) for programmatic access

    Args:
        n_fish: Number of fish in the school
        n_steps: Number of simulation steps
        predator_spawn_frame: Frame when predator spawns
        predator_remove_frame: Frame when predator is removed (optional, never removed if None)
        gamma: Infected duration in frames (I->R)
        delta: Recovered duration in frames (R->S)
        space_size: Size of simulation space (single value or tuple)
        beta: Kept for backward compatibility (not used in transmission)
        verbose: Whether to print debug output
        zone_repulsion: Couzin model repulsion zone radius (cm)
        zone_orientation: Couzin model orientation zone radius (cm)
        zone_attraction: Couzin model attraction zone radius (cm)

    Returns:
        FishSchool object with history data
    """
    # Suppress print statements if not verbose
    import sys
    import io

    old_stdout = sys.stdout
    if not verbose:
        sys.stdout = io.StringIO()

    try:
        school = FishSchool(
            n_fish=n_fish,
            space_size=space_size,
            beta=beta,
            gamma=gamma,
            delta=delta,
            zone_repulsion=zone_repulsion,
            zone_orientation=zone_orientation,
            zone_attraction=zone_attraction,
        )

        for frame in range(n_steps):
            # Spawn predator at specific time
            if frame == predator_spawn_frame:
                school.spawn_predator()

            # Remove predator at specified time (if specified)
            if predator_remove_frame is not None and frame == predator_remove_frame:
                school.remove_predator()

            # Update simulation
            school.update()

        return school
    finally:
        # Restore stdout
        sys.stdout = old_stdout


def visualize_simulation(
    n_fish: int = 50,
    n_steps: int = 1000,
    predator_time: int = 100,
    predator_remove_time: Optional[int] = None,
    beta: float = 0.6,
    gamma: float = 20,
    delta: float = 20,
    space_size: Any = 100,
    show_plot: bool = True,
    show_animation: bool = True,
    zone_repulsion: float = DEFAULT_ZONE_REPULSION,
    zone_orientation: float = DEFAULT_ZONE_ORIENTATION,
    zone_attraction: float = DEFAULT_ZONE_ATTRACTION,
) -> Tuple[Optional[FuncAnimation], FishSchool]:
    """Run and visualize the simulation with SIRS dynamics

    Args:
        n_fish: Number of fish in the school
        n_steps: Number of simulation steps
        predator_time: Frame when predator spawns
        predator_remove_time: Frame when predator is removed (default: predator_time + 100)
        beta: Transmission probability (S->I)
        gamma: Infected duration in frames (I->R)
        delta: Recovered duration in frames (R->S)
        space_size: Size of simulation space (single value or tuple)
        show_plot: Whether to show SIR dynamics plot after simulation
        show_animation: Whether to show 3D animation (False for headless mode)
        zone_repulsion: Couzin model repulsion zone radius (cm)
        zone_orientation: Couzin model orientation zone radius (cm)
        zone_attraction: Couzin model attraction zone radius (cm)

    Returns:
        Tuple of (animation, school) for further analysis
    """

    # Set default predator removal time
    if predator_remove_time is None:
        predator_remove_time = predator_time + 100

    # Use space_size as-is (can be single value or tuple)
    school = FishSchool(
        n_fish=n_fish,
        space_size=space_size,
        beta=beta,
        gamma=gamma,
        delta=delta,
        zone_repulsion=zone_repulsion,
        zone_orientation=zone_orientation,
        zone_attraction=zone_attraction,
    )

    # If headless mode, run simulation without animation
    if not show_animation:
        for frame in range(n_steps):
            # Spawn predator at specific time
            if frame == predator_time:
                school.spawn_predator()

            # Remove predator at specified time
            if frame == predator_remove_time:
                school.remove_predator()

            # Update simulation
            school.update()

        # Show plots if requested
        if show_plot:
            plot_sir_dynamics(
                school,
                predator_spawn=predator_time,
                predator_remove=predator_remove_time,
            )
            plot_confusion_matrix(school)

        export_sir_data(school, beta, gamma, delta)
        return None, school

    # Setup plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Initialize scatter plots for three states
    susceptible_fish = ax.scatter([], [], [], c="blue", marker="o", s=30, alpha=0.6, label="Susceptible")  # type: ignore
    infected_fish = ax.scatter([], [], [], c="red", marker="o", s=50, alpha=0.8, label="Infected")  # type: ignore
    recovered_fish = ax.scatter([], [], [], c="green", marker="o", s=40, alpha=0.7, label="Recovered")  # type: ignore
    predator_plot = ax.scatter([], [], [], c="black", marker="x", s=200, label="Predator")  # type: ignore

    # Detection radius sphere (will be shown when predator is active)
    detection_sphere = None

    ax.set_xlim(0, school.space_dimensions[0])
    ax.set_ylim(0, school.space_dimensions[1])
    ax.set_zlim(0, school.space_dimensions[2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
    state_text = ax.text2D(0.05, 0.90, "", transform=ax.transAxes)

    def init() -> Tuple:
        return (
            susceptible_fish,
            infected_fish,
            recovered_fish,
            predator_plot,
            time_text,
            state_text,
        )

    def update(frame: int) -> Tuple:
        # Spawn predator at specific time
        if frame == predator_time:
            # Spawn at school center (no position argument)
            school.spawn_predator()

        # Remove predator after some time
        if frame == predator_remove_time:
            school.remove_predator()
            print(f"\n[Frame {frame}] PREDATOR REMOVED")
            print(f"  Total direct infections: {school.total_startles}")
            print(f"  Total cascade infections: {school.cascade_startles}")
            print(
                f"  Total infections: {school.total_startles + school.cascade_startles}\n"
            )

        # Update simulation
        school.update()

        # Separate fish by SIRS state
        susceptible_positions = []
        infected_positions = []
        recovered_positions = []

        for fish in school.fish:
            if fish.state == "susceptible":
                susceptible_positions.append(fish.position)
            elif fish.state == "infected":
                infected_positions.append(fish.position)
            elif fish.state == "recovered":
                recovered_positions.append(fish.position)

        # Update plots for each state
        if susceptible_positions:
            susceptible_positions = np.array(susceptible_positions)
            susceptible_fish._offsets3d = (  # type: ignore
                susceptible_positions[:, 0],
                susceptible_positions[:, 1],
                susceptible_positions[:, 2],
            )
        else:
            susceptible_fish._offsets3d = ([], [], [])  # type: ignore

        if infected_positions:
            infected_positions = np.array(infected_positions)
            infected_fish._offsets3d = (  # type: ignore
                infected_positions[:, 0],
                infected_positions[:, 1],
                infected_positions[:, 2],
            )
        else:
            infected_fish._offsets3d = ([], [], [])  # type: ignore

        if recovered_positions:
            recovered_positions = np.array(recovered_positions)
            recovered_fish._offsets3d = (  # type: ignore
                recovered_positions[:, 0],
                recovered_positions[:, 1],
                recovered_positions[:, 2],
            )
        else:
            recovered_fish._offsets3d = ([], [], [])  # type: ignore

        # Update predator and detection sphere
        nonlocal detection_sphere
        if school.predator_position is not None:
            predator_plot._offsets3d = (  # type: ignore
                [school.predator_position[0]],
                [school.predator_position[1]],
                [school.predator_position[2]],
            )

            # Draw detection radius sphere
            if detection_sphere is None:
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x = (
                    school.predator_detection_radius * np.outer(np.cos(u), np.sin(v))
                    + school.predator_position[0]
                )
                y = (
                    school.predator_detection_radius * np.outer(np.sin(u), np.sin(v))
                    + school.predator_position[1]
                )
                z = (
                    school.predator_detection_radius
                    * np.outer(np.ones(np.size(u)), np.cos(v))
                    + school.predator_position[2]
                )
                detection_sphere = ax.plot_wireframe(
                    x, y, z, color="red", alpha=0.1, linewidth=0.5
                )
        else:
            predator_plot._offsets3d = ([], [], [])  # type: ignore
            # Remove detection sphere
            if detection_sphere is not None:
                detection_sphere.remove()
                detection_sphere = None

        # Update text
        time_text.set_text(f"Time: {frame}")
        n_s = len(susceptible_positions)
        n_i = len(infected_positions)
        n_r = len(recovered_positions)
        state_text.set_text(f"S:{n_s} I:{n_i} R:{n_r}")

        return (
            susceptible_fish,
            infected_fish,
            recovered_fish,
            predator_plot,
            time_text,
            state_text,
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=n_steps,
        init_func=init,
        blit=False,
        interval=50,
        repeat=True,
    )

    plt.show()

    # After simulation completes, show SIR dynamics plot and export data
    if show_plot:
        plot_sir_dynamics(
            school, predator_spawn=predator_time, predator_remove=predator_remove_time
        )
        # Show confusion matrix plot
        plot_confusion_matrix(school)

    export_sir_data(school, beta, gamma, delta)

    return anim, school


if __name__ == "__main__":
    # Tank dimensions (in cm): 1m x 2m x 1m = 100cm x 200cm x 100cm
    tank_dimensions = (100, 200, 100)

    # SIRS model parameters at 20 fps
    beta = 0  # Not used
    gamma = 10  # Infected duration (frames)
    delta = 40  # Recovered duration (frames)
    n_fish = 40

    print("Starting Fish School Simulator with SIRS Epidemic Model")
    print("=" * 60)
    print("Simulation parameters:")
    print(f"  - Number of fish: {n_fish}")
    print(
        f"  - Tank dimensions: {tank_dimensions[0]}cm x {tank_dimensions[1]}cm x {tank_dimensions[2]}cm (1m x 2m x 1m)"
    )
    print("  - Susceptible fish speed: 10 cm/s (0.5 cm/frame at 20 fps)")
    print("  - Infected fish speed: 35 cm/s (1.75 cm/frame at 20 fps)")
    print("  - Recovered fish speed: 5 cm/s (0.25 cm/frame at 20 fps)")
    print(f"  - Predator appears at t={200} (spawns at school center)")
    print(f"  - Predator removed at t={400}")
    print("  - Predator detection radius: 50 cm")
    print("\nSIRS Model Parameters:")
    print(
        f"  - β (transmission probability): {beta} (not used - empirical model instead)"
    )
    print(
        f"  - γ (infected duration): {gamma} frames ({gamma/20:.2f} seconds at 20 fps)"
    )
    print(
        f"  - δ (recovered duration): {delta} frames ({delta/20:.2f} seconds at 20 fps)"
    )
    print("  - Visual range (fish-to-fish): 120 cm")
    print("=" * 60)
    print("\nLegend:")
    print("  Blue dots = Susceptible (S)")
    print("  Red dots = Infected (I)")
    print("  Green dots = Recovered (R)")
    print("  Black X = Predator")
    print("  Red wireframe sphere = Detection radius")
    print("=" * 60)
    print("\nWatch for infection cascades when predator appears!")
    print("Console will show detection and cascade events...\n")

    visualize_simulation(
        n_fish=n_fish,
        n_steps=600,
        predator_time=400,
        beta=beta,
        gamma=gamma,
        delta=delta,
        space_size=tank_dimensions,
        show_plot=True,
    )
