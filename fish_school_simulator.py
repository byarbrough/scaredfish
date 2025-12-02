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


class Fish:
    """Individual fish agent with Couzin model behavior and SIRS startle response"""

    def __init__(
        self,
        position: npt.ArrayLike,
        velocity: npt.ArrayLike,
        fish_id: int,
        infected_duration: int = 10,
        recovered_duration: int = 20,
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
                self.velocity = self.velocity / 2.0  # Slow down after startle response
                self.state_timer = 0

        elif self.state == "recovered":
            if self.state_timer >= self.recovered_duration:
                self.state = "susceptible"
                self.state_timer = 0

    def infect(self) -> None:
        """Trigger infection (startle response) - only works if susceptible"""
        if self.state == "susceptible":
            self.state = "infected"
            self.state_timer = 0
            # Startle causes rapid acceleration away
            self.velocity *= 2.0


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
        gamma: int = 10,
        delta: int = 20,
    ) -> None:
        self.n_fish = n_fish
        self.space_size = space_size

        # Couzin model parameters
        self.zone_repulsion = 2.0  # Distance for repulsion
        self.zone_orientation = 10.0  # Distance for alignment
        self.zone_attraction = 20.0  # Distance for attraction

        self.max_speed = 2.0
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
        self.predator_detection_radius = 25.0  # Increased for better detection
        self.visual_range = 15.0  # How far fish can see each other

        # Debug tracking
        self.total_startles = 0
        self.cascade_startles = 0

        # Data collection for SIR dynamics
        self.history = {"time": [], "S": [], "I": [], "R": []}

        # Initialize fish
        self.fish = []
        for i in range(n_fish):
            # Random positions
            pos = np.random.uniform(0, space_size, 3)
            # Random initial velocities
            vel = np.random.uniform(-1, 1, 3)
            vel = self._normalize(vel) * self.min_speed
            self.fish.append(
                Fish(pos, vel, i, infected_duration=gamma, recovered_duration=delta)
            )

        self.time_step = 0

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

    def get_neighbors(self, fish: Fish, radius: float) -> List[Tuple[Fish, float]]:
        """Get all neighbors within radius"""
        neighbors = []
        for other in self.fish:
            if other.id != fish.id:
                dist = np.linalg.norm(fish.position - other.position)
                if dist < radius:
                    neighbors.append((other, dist))
        return neighbors

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
                    # If angular difference < 5 degrees, consider occluded
                    if angular_diff < np.radians(5):
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

        # LMD: log of metric distance
        lmd = np.log(distance)

        # RAA: Calculate ranked angular area
        raa = self._calculate_raa(observer, startled_fish)

        # If target is not visible, probability is 0
        if raa == 0.0:
            return 0.0

        # Logistic regression model
        logit = self.beta_1 + self.beta_2 * lmd + self.beta_3 * raa
        probability = 1.0 / (1.0 + np.exp(-logit))

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
        """Check if any fish detect predator and become infected"""
        if self.predator_position is None:
            return

        startled_this_frame = 0
        for fish in self.fish:
            if fish.state == "susceptible":
                dist = np.linalg.norm(fish.position - self.predator_position)
                if dist < self.predator_detection_radius:
                    # Fish detects predator and becomes infected
                    fish.infect()
                    startled_this_frame += 1
                    self.total_startles += 1

        if startled_this_frame > 0:
            print(
                f"[Frame {self.time_step}] {startled_this_frame} fish detected predator!"
            )

    def check_startle_cascade(self) -> None:
        """Check if infected fish trigger cascade in susceptible neighbors using empirical model"""
        newly_infected = []

        for fish in self.fish:
            # Only infected fish can transmit
            if fish.state == "infected" and fish not in newly_infected:
                # Get visible neighbors
                neighbors = self.get_neighbors(fish, self.visual_range)

                for neighbor, dist in neighbors:
                    # Only susceptible fish can be infected
                    if neighbor.state == "susceptible":
                        # Calculate probability using empirical model (LMD + RAA)
                        prob = self.calculate_startle_probability(neighbor, fish)

                        if np.random.random() < prob:
                            neighbor.infect()
                            newly_infected.append(neighbor)
                            self.cascade_startles += 1

        if len(newly_infected) > 0:
            print(
                f"[Frame {self.time_step}] CASCADE: {len(newly_infected)} fish infected by seeing others!"
            )

    def apply_boundaries(self, fish: Fish) -> None:
        """Apply periodic boundary conditions"""
        fish.position = fish.position % self.space_size

    def update(self) -> None:
        """Update simulation one time step"""
        self.time_step += 1

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

    def spawn_predator(self, position: Optional[npt.ArrayLike] = None) -> None:
        """Spawn predator at position (defaults to school center)"""
        if position is None:
            # Spawn at school center for guaranteed interaction
            position = self.get_school_center()
        self.predator_position = np.array(position)
        print(
            f"\n[Frame {self.time_step}] PREDATOR SPAWNED at {self.predator_position}"
        )
        print(f"  Detection radius: {self.predator_detection_radius}")

        # Print distances to nearest fish
        distances = [
            np.linalg.norm(fish.position - self.predator_position) for fish in self.fish
        ]
        distances.sort()
        print(f"  Nearest 5 fish: {distances[:5]}")

    def remove_predator(self) -> None:
        """Remove predator from simulation"""
        self.predator_position = None


def export_sir_data(school: FishSchool, beta: float, gamma: int, delta: int) -> str:
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
        f"SIRS Dynamics (β={school.beta}, γ={school.gamma}, δ={school.delta})",
        fontsize=14,
    )
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    plt.show()


def visualize_simulation(
    n_fish: int = 50,
    n_steps: int = 1000,
    predator_time: int = 100,
    beta: float = 0.6,
    gamma: int = 10,
    delta: int = 20,
    show_plot: bool = True,
) -> Tuple[FuncAnimation, FishSchool]:
    """Run and visualize the simulation with SIRS dynamics

    Args:
        n_fish: Number of fish in the school
        n_steps: Number of simulation steps
        predator_time: Frame when predator spawns
        beta: Transmission probability (S->I)
        gamma: Infected duration in frames (I->R)
        delta: Recovered duration in frames (R->S)
        show_plot: Whether to show SIR dynamics plot after simulation

    Returns:
        Tuple of (animation, school) for further analysis
    """

    school = FishSchool(
        n_fish=n_fish, space_size=100, beta=beta, gamma=gamma, delta=delta
    )

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

    ax.set_xlim(0, school.space_size)
    ax.set_ylim(0, school.space_size)
    ax.set_zlim(0, school.space_size)
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
        if frame == predator_time + 100:
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
            susceptible_fish._offsets3d = (
                susceptible_positions[:, 0],  # type: ignore
                susceptible_positions[:, 1],
                susceptible_positions[:, 2],
            )
        else:
            susceptible_fish._offsets3d = ([], [], [])  # type: ignore

        if infected_positions:
            infected_positions = np.array(infected_positions)
            infected_fish._offsets3d = (
                infected_positions[:, 0],  # type: ignore
                infected_positions[:, 1],
                infected_positions[:, 2],
            )
        else:
            infected_fish._offsets3d = ([], [], [])  # type: ignore

        if recovered_positions:
            recovered_positions = np.array(recovered_positions)
            recovered_fish._offsets3d = (
                recovered_positions[:, 0],  # type: ignore
                recovered_positions[:, 1],
                recovered_positions[:, 2],
            )
        else:
            recovered_fish._offsets3d = ([], [], [])  # type: ignore

        # Update predator and detection sphere
        nonlocal detection_sphere
        if school.predator_position is not None:
            predator_plot._offsets3d = (
                [school.predator_position[0]],  # type: ignore
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
        n_s = (
            len(susceptible_positions)
            if isinstance(susceptible_positions, list)
            else len(susceptible_positions)
        )
        n_i = (
            len(infected_positions)
            if isinstance(infected_positions, list)
            else len(infected_positions)
        )
        n_r = (
            len(recovered_positions)
            if isinstance(recovered_positions, list)
            else len(recovered_positions)
        )
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
            school, predator_spawn=predator_time, predator_remove=predator_time + 100
        )

    export_sir_data(school, beta, gamma, delta)

    return anim, school


if __name__ == "__main__":
    # SIRS model parameters
    beta = 0.6  # Transmission probability (S -> I)
    gamma = 10  # Infected duration (frames)
    delta = 20  # Recovered duration (frames)
    n_fish = 100

    print("Starting Fish School Simulator with SIRS Epidemic Model")
    print("=" * 60)
    print("Simulation parameters:")
    print(f"  - Number of fish: {n_fish}")
    print("  - Predator appears at t=100 (spawns at school center)")
    print("  - Predator removed at t=200")
    print("  - Predator detection radius: 25 units")
    print("\nSIRS Model Parameters:")
    print(f"  - β (transmission probability): {beta}")
    print(f"  - γ (infected duration): {gamma} frames")
    print(f"  - δ (recovered duration): {delta} frames")
    print("  - Visual range (fish-to-fish): 15 units")
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
        n_steps=500,
        predator_time=100,
        beta=beta,
        gamma=gamma,
        delta=delta,
        show_plot=True,
    )
