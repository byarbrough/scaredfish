"""
Fish School Simulator using Couzin Model with Startle Cascades

Based on the Couzin model for collective animal behavior with added
startle response mechanics inspired by "Individual and collective
encoding of risk in animal groups"
"""

from typing import List, Tuple, Optional
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


class Fish:
    """Individual fish agent with Couzin model behavior and startle response"""

    position: npt.NDArray[np.float64]
    velocity: npt.NDArray[np.float64]
    id: int
    is_startled: bool
    startle_timer: int
    startle_duration: int

    def __init__(self, position: npt.ArrayLike, velocity: npt.ArrayLike, fish_id: int) -> None:
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.id = fish_id
        self.is_startled = False
        self.startle_timer = 0
        self.startle_duration = 30  # frames the startle lasts (increased for visibility)

    def update_startle(self) -> None:
        """Update startle state"""
        if self.is_startled:
            self.startle_timer += 1
            if self.startle_timer >= self.startle_duration:
                self.is_startled = False
                self.startle_timer = 0

    def startle(self) -> None:
        """Trigger startle response"""
        if not self.is_startled:
            self.is_startled = True
            self.startle_timer = 0
            # Startle causes rapid acceleration away
            self.velocity *= 2.0


class FishSchool:
    """Simulation of fish school using Couzin model"""

    n_fish: int
    space_size: float
    zone_repulsion: float
    zone_orientation: float
    zone_attraction: float
    max_speed: float
    min_speed: float
    max_turn_rate: float
    predator_position: Optional[npt.NDArray[np.float64]]
    predator_detection_radius: float
    startle_transmission_prob: float
    visual_range: float
    total_startles: int
    cascade_startles: int
    fish: List[Fish]
    time_step: int

    def __init__(self, n_fish: int = 50, space_size: float = 100) -> None:
        self.n_fish = n_fish
        self.space_size = space_size

        # Couzin model parameters
        self.zone_repulsion = 2.0      # Distance for repulsion
        self.zone_orientation = 10.0    # Distance for alignment
        self.zone_attraction = 20.0     # Distance for attraction

        self.max_speed = 2.0
        self.min_speed = 0.5
        self.max_turn_rate = 0.3  # Maximum turning angle per step

        # Startle cascade parameters
        self.predator_position = None
        self.predator_detection_radius = 25.0  # Increased for better detection
        self.startle_transmission_prob = 0.6  # Probability of transmitting startle
        self.visual_range = 15.0  # How far fish can see each other

        # Debug tracking
        self.total_startles = 0
        self.cascade_startles = 0

        # Initialize fish
        self.fish = []
        for i in range(n_fish):
            # Random positions
            pos = np.random.uniform(0, space_size, 3)
            # Random initial velocities
            vel = np.random.uniform(-1, 1, 3)
            vel = self._normalize(vel) * self.min_speed
            self.fish.append(Fish(pos, vel, i))

        self.time_step = 0

    def _normalize(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Normalize a vector"""
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            return np.array([1.0, 0.0, 0.0])
        return vector / norm

    def _limit_speed(self, velocity: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Limit velocity to min/max speed"""
        speed = np.linalg.norm(velocity)
        if speed > self.max_speed:
            return velocity * (self.max_speed / speed)
        elif speed < self.min_speed and speed > 1e-10:
            return velocity * (self.min_speed / speed)
        return velocity

    def _smooth_turn(self, current_vel: npt.NDArray[np.float64],
                     desired_vel: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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
        neighbors: List[Tuple[Fish, float]] = []
        for other in self.fish:
            if other.id != fish.id:
                dist = np.linalg.norm(fish.position - other.position)
                if dist < radius:
                    neighbors.append((other, dist))
        return neighbors

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
        """Check if any fish detect predator and startle"""
        if self.predator_position is None:
            return

        startled_this_frame = 0
        for fish in self.fish:
            if not fish.is_startled:
                dist = np.linalg.norm(fish.position - self.predator_position)
                if dist < self.predator_detection_radius:
                    # Fish detects predator and startles
                    fish.startle()
                    startled_this_frame += 1
                    self.total_startles += 1

        if startled_this_frame > 0:
            print(f"[Frame {self.time_step}] {startled_this_frame} fish detected predator!")

    def check_startle_cascade(self) -> None:
        """Check if startled fish trigger cascade in neighbors"""
        newly_startled = []

        for fish in self.fish:
            if fish.is_startled and not fish in newly_startled:
                # Get visible neighbors
                neighbors = self.get_neighbors(fish, self.visual_range)

                for neighbor, dist in neighbors:
                    if not neighbor.is_startled:
                        # Probability of transmission
                        if np.random.random() < self.startle_transmission_prob:
                            neighbor.startle()
                            newly_startled.append(neighbor)
                            self.cascade_startles += 1

        if len(newly_startled) > 0:
            print(f"[Frame {self.time_step}] CASCADE: {len(newly_startled)} fish startled by seeing others!")

    def apply_boundaries(self, fish: Fish) -> None:
        """Apply periodic boundary conditions"""
        fish.position = fish.position % self.space_size

    def update(self) -> None:
        """Update simulation one time step"""
        self.time_step += 1

        # Check for predator-induced startles
        self.check_predator_startle()

        # Check for startle cascades
        self.check_startle_cascade()

        # Update each fish
        new_velocities = []
        for fish in self.fish:
            # Update startle state
            fish.update_startle()

            if fish.is_startled:
                # Startled fish: escape behavior (move away from predator)
                if self.predator_position is not None:
                    escape_direction = fish.position - self.predator_position
                    desired_velocity = self._normalize(escape_direction) * self.max_speed
                else:
                    desired_velocity = fish.velocity * 1.5
            else:
                # Normal behavior: apply Couzin model
                desired_velocity = self.couzin_behavior(fish)

            # Smooth turning
            desired_velocity = self._normalize(desired_velocity) * np.linalg.norm(fish.velocity)
            new_velocity = self._smooth_turn(fish.velocity, desired_velocity)

            # Limit speed
            new_velocity = self._limit_speed(new_velocity)
            new_velocities.append(new_velocity)

        # Update velocities and positions
        for fish, new_vel in zip(self.fish, new_velocities):
            fish.velocity = new_vel
            fish.position += fish.velocity
            self.apply_boundaries(fish)

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
        print(f"\n[Frame {self.time_step}] PREDATOR SPAWNED at {self.predator_position}")
        print(f"  Detection radius: {self.predator_detection_radius}")

        # Print distances to nearest fish
        distances = [np.linalg.norm(fish.position - self.predator_position) for fish in self.fish]
        distances.sort()
        print(f"  Nearest 5 fish: {distances[:5]}")

    def remove_predator(self) -> None:
        """Remove predator from simulation"""
        self.predator_position = None


def visualize_simulation(n_fish: int = 50, n_steps: int = 1000, predator_time: int = 100) -> FuncAnimation:
    """Run and visualize the simulation"""

    school = FishSchool(n_fish=n_fish, space_size=100)

    # Setup plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Initialize scatter plots
    positions = np.array([fish.position for fish in school.fish])
    normal_fish = ax.scatter([], [], [], c='blue', marker='o', s=30, alpha=0.6, label='Normal')
    startled_fish = ax.scatter([], [], [], c='red', marker='o', s=50, alpha=0.8, label='Startled')
    predator_plot = ax.scatter([], [], [], c='black', marker='x', s=200, label='Predator')

    # Detection radius sphere (will be shown when predator is active)
    detection_sphere = None

    ax.set_xlim(0, school.space_size)
    ax.set_ylim(0, school.space_size)
    ax.set_zlim(0, school.space_size)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
    startle_text = ax.text2D(0.05, 0.90, '', transform=ax.transAxes)

    def init() -> Tuple:
        return normal_fish, startled_fish, predator_plot, time_text, startle_text

    def update(frame: int) -> Tuple:
        # Spawn predator at specific time
        if frame == predator_time:
            # Spawn at school center (no position argument)
            school.spawn_predator()

        # Remove predator after some time
        if frame == predator_time + 100:
            school.remove_predator()
            print(f"\n[Frame {frame}] PREDATOR REMOVED")
            print(f"  Total direct startles: {school.total_startles}")
            print(f"  Total cascade startles: {school.cascade_startles}")
            print(f"  Total startles: {school.total_startles + school.cascade_startles}\n")

        # Update simulation
        school.update()

        # Separate normal and startled fish
        normal_positions = []
        startled_positions = []

        for fish in school.fish:
            if fish.is_startled:
                startled_positions.append(fish.position)
            else:
                normal_positions.append(fish.position)

        # Update plots
        if normal_positions:
            normal_positions = np.array(normal_positions)
            normal_fish._offsets3d = (normal_positions[:, 0],
                                      normal_positions[:, 1],
                                      normal_positions[:, 2])
        else:
            normal_fish._offsets3d = ([], [], [])

        if startled_positions:
            startled_positions = np.array(startled_positions)
            startled_fish._offsets3d = (startled_positions[:, 0],
                                        startled_positions[:, 1],
                                        startled_positions[:, 2])
        else:
            startled_fish._offsets3d = ([], [], [])

        # Update predator and detection sphere
        nonlocal detection_sphere
        if school.predator_position is not None:
            predator_plot._offsets3d = ([school.predator_position[0]],
                                       [school.predator_position[1]],
                                       [school.predator_position[2]])

            # Draw detection radius sphere
            if detection_sphere is None:
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x = school.predator_detection_radius * np.outer(np.cos(u), np.sin(v)) + school.predator_position[0]
                y = school.predator_detection_radius * np.outer(np.sin(u), np.sin(v)) + school.predator_position[1]
                z = school.predator_detection_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + school.predator_position[2]
                detection_sphere = ax.plot_wireframe(x, y, z, color='red', alpha=0.1, linewidth=0.5)
        else:
            predator_plot._offsets3d = ([], [], [])
            # Remove detection sphere
            if detection_sphere is not None:
                detection_sphere.remove()
                detection_sphere = None

        # Update text
        time_text.set_text(f'Time: {frame}')
        n_startled = len(startled_positions)
        startle_text.set_text(f'Startled: {n_startled}/{n_fish}')

        return normal_fish, startled_fish, predator_plot, time_text, startle_text

    anim = FuncAnimation(fig, update, frames=n_steps, init_func=init,
                        blit=False, interval=50, repeat=True)

    plt.show()

    return anim


if __name__ == "__main__":
    print("Starting Fish School Simulator with Startle Cascades")
    print("=" * 60)
    print("Simulation parameters:")
    print("  - Number of fish: 50")
    print("  - Predator appears at t=100 (spawns at school center)")
    print("  - Predator detection radius: 25 units")
    print("  - Startle transmission probability: 60%")
    print("  - Visual range (fish-to-fish): 15 units")
    print("=" * 60)
    print("\nLegend:")
    print("  Blue dots = Normal fish")
    print("  Red dots = Startled fish")
    print("  Black X = Predator")
    print("  Red wireframe sphere = Detection radius")
    print("=" * 60)
    print("\nWatch for startle cascades when predator appears!")
    print("Console will show detection and cascade events...\n")

    visualize_simulation(n_fish=50, n_steps=500, predator_time=100)
