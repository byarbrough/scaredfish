"""
Quick test of empirical startle transmission model
"""
import numpy as np
from fish_school_simulator import Fish, FishSchool


def test_startle_probability():
    """Test the empirical probability calculation"""
    print("Testing Empirical Startle Transmission Model")
    print("=" * 60)

    # Create a small school
    school = FishSchool(n_fish=10, space_size=100)

    # Create two test fish at known positions
    # Fish A at origin, heading in +x direction
    fish_observer = Fish(
        position=[50, 50, 50],
        velocity=[1, 0, 0],  # heading in +x direction
        fish_id=100
    )

    # Test different scenarios
    scenarios = [
        ("Very close, directly ahead", [52, 50, 50], [0, 0, 0]),
        ("Medium distance, ahead", [55, 50, 50], [0, 0, 0]),
        ("Far distance, ahead", [60, 50, 50], [0, 0, 0]),
        ("Close, to the side", [52, 52, 50], [0, 0, 0]),
        ("Close, behind (blind spot)", [48, 50, 50], [0, 0, 0]),
    ]

    print("\nScenario Tests:")
    print("-" * 60)

    for scenario_name, pos, vel in scenarios:
        fish_startled = Fish(
            position=pos,
            velocity=vel,
            fish_id=101
        )

        # Add both fish to school temporarily
        school.fish = [fish_observer, fish_startled]

        # Calculate probability
        prob = school.calculate_startle_probability(fish_observer, fish_startled)

        # Calculate distance and RAA for display
        distance = np.linalg.norm(
            np.array(fish_observer.position) - np.array(fish_startled.position)
        )
        raa = school._calculate_raa(fish_observer, fish_startled)
        lmd = np.log(distance) if distance > 0 else 0

        print(f"\n{scenario_name}:")
        print(f"  Position: {pos}")
        print(f"  Distance: {distance:.2f}")
        print(f"  LMD (log distance): {lmd:.3f}")
        print(f"  RAA (ranked angular area): {raa:.3f}")
        print(f"  → Startle Probability: {prob:.4f} ({prob*100:.2f}%)")

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("\nExpected behavior:")
    print("  - Closer fish should have HIGHER probability")
    print("  - Fish in blind spot should have probability = 0")
    print("  - More visually prominent fish (higher RAA) should have higher probability")


if __name__ == "__main__":
    test_startle_probability()
