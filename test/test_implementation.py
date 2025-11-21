"""
Quick test script to verify all implemented components work correctly.
"""

from environment.traffic_env import TrafficIntersectionEnv
from environment.multi_intersection_env import MultiIntersectionEnv


def test_single_intersection():
    """Test single intersection environment."""
    print("=" * 60)
    print("Testing Single Intersection Environment")
    print("=" * 60)
    
    env = TrafficIntersectionEnv(min_green_time=5, yellow_time=2, switching_penalty=10.0)
    obs, info = env.reset()
    
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print(f"Action space: {env.action_space}")
    
    # Run a few steps
    for step in range(20):
        # Alternate between keeping and switching
        action = 1 if step > 10 else 0
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 5 == 0:
            print(f"\nStep {step}:")
            print(f"  Observation: {obs}")
            print(f"  Reward: {reward:.2f}")
            print(f"  Throughput: {info['throughput']}")
            print(f"  Total Wait: {info['total_wait']:.1f}")
            print(f"  Yellow Timer: {info['yellow_timer']}")
    
    print("\n✅ Single Intersection test passed!\n")


def test_multi_intersection():
    """Test multi-intersection environment."""
    print("=" * 60)
    print("Testing Multi-Intersection Environment")
    print("=" * 60)
    
    env = MultiIntersectionEnv(min_green_time=5, yellow_time=2, switching_penalty=10.0)
    obs, info = env.reset()
    
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print(f"Action space: {env.action_space}")
    
    # Run a few steps
    for step in range(20):
        # Different actions for each intersection
        action = [1 if step > 10 else 0, 0]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 5 == 0:
            print(f"\nStep {step}:")
            print(f"  Observation: {obs}")
            print(f"  Reward: {reward:.2f}")
            print(f"  Throughput: {info['throughput']}")
            print(f"  Total Wait: {info['total_wait']:.1f}")
            print(f"  Int1 Phase: {info['intersection_1_phase']}, Int2 Phase: {info['intersection_2_phase']}")
    
    print("\n✅ Multi-Intersection test passed!\n")


if __name__ == "__main__":
    try:
        test_single_intersection()
        test_multi_intersection()
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
