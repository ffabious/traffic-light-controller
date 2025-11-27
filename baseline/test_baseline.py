import sys
import os

# Add parent directory to path to import environment
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environment.multi_env import MultiIntersectionEnv
from baseline.regular_controller import FixedTimeController, RegularTrafficLightController
import numpy as np


def test_fixed_time_controller():
    """Test the FixedTimeController (faster, stateless version)."""
    print("=" * 60)
    print("Testing FixedTimeController")
    print("=" * 60)
    
    # Initialize environment
    env = MultiIntersectionEnv(
        min_green_time=5,
        yellow_time=2,
        switching_penalty=10.0
    )
    
    # Initialize controller with realistic parameters
    # Cycle time = 12 (NS) + 2 (yellow) + 12 (EW) + 2 (yellow) = 28
    controller = FixedTimeController(
        cycle_time=28,
        green_time_ns=12,
        green_time_ew=12,
        yellow_time=2,
        offset=None,  # Auto-calculate offset
        travel_time=8
    )
    
    print(f"Controller Configuration:")
    print(f"  Cycle Time: {controller.cycle_time} steps")
    print(f"  NS Green Time: {controller.green_time_ns} steps")
    print(f"  EW Green Time: {controller.green_time_ew} steps")
    print(f"  Yellow Time: {controller.yellow_time} steps")
    print(f"  Offset: {controller.offset} steps")
    print()
    
    # Run simulation
    obs, info = env.reset()
    controller.reset()
    
    total_reward = 0
    total_throughput = 0
    total_wait = 0
    
    num_steps = 200
    
    for step in range(num_steps):
        # Get action from controller
        action = controller.get_action(env.current_time)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        total_throughput += info['throughput']
        total_wait += info['total_wait']
        
        # Print status every 20 steps
        if step % 20 == 0:
            phases = controller.get_current_phases(env.current_time)
            print(f"Step {step:3d} | "
                  f"Phase 1: {'NS' if phases[0]==0 else 'EW'} | "
                  f"Phase 2: {'NS' if phases[1]==0 else 'EW'} | "
                  f"Queues: {obs[:4].astype(int)} | {obs[4:8].astype(int)} | "
                  f"Throughput: {info['throughput']}")
    
    print()
    print("Simulation Results:")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Total Throughput: {total_throughput} vehicles")
    print(f"  Average Wait Time: {total_wait / num_steps:.2f} vehicle-steps")
    print(f"  Average Reward per Step: {total_reward / num_steps:.2f}")
    print()


def test_regular_controller():
    """Test the RegularTrafficLightController (stateful version)."""
    print("=" * 60)
    print("Testing RegularTrafficLightController")
    print("=" * 60)
    
    # Initialize environment
    env = MultiIntersectionEnv(
        min_green_time=5,
        yellow_time=2,
        switching_penalty=10.0
    )
    
    # Initialize controller
    # Cycle time = 12 (NS) + 2 (yellow) + 12 (EW) + 2 (yellow) = 28
    controller = RegularTrafficLightController(
        cycle_time=28,
        green_time_ns=12,
        green_time_ew=12,
        yellow_time=2,
        offset=None,
        travel_time=8
    )
    
    print(f"Controller Configuration:")
    print(f"  Cycle Time: {controller.cycle_time} steps")
    print(f"  Offset: {controller.offset} steps")
    print()
    
    # Run simulation
    obs, info = env.reset()
    controller.reset()
    
    total_reward = 0
    total_throughput = 0
    
    num_steps = 200
    
    for step in range(num_steps):
        # Get action from controller
        action = controller.get_action(env.current_time)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        total_throughput += info['throughput']
        
        # Print status every 20 steps
        if step % 20 == 0:
            phases = controller.get_current_phases()
            ctrl_info = controller.get_info()
            print(f"Step {step:3d} | "
                  f"Phase 1: {'NS' if phases[0]==0 else 'EW'} | "
                  f"Phase 2: {'NS' if phases[1]==0 else 'EW'} | "
                  f"State 1: {ctrl_info['phase_state_1']} | "
                  f"State 2: {ctrl_info['phase_state_2']}")
    
    print()
    print("Simulation Results:")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Total Throughput: {total_throughput} vehicles")
    print(f"  Average Reward per Step: {total_reward / num_steps:.2f}")
    print()


def compare_controllers():
    """Compare both controllers to ensure they produce similar results."""
    print("=" * 60)
    print("Comparing Controllers")
    print("=" * 60)
    
    # Test parameters
    num_runs = 5
    num_steps = 200
    
    results_fixed = []
    results_regular = []
    
    for run in range(num_runs):
        # Test FixedTimeController
        env1 = MultiIntersectionEnv(min_green_time=5, yellow_time=2, switching_penalty=10.0)
        controller1 = FixedTimeController(cycle_time=28, green_time_ns=12, green_time_ew=12, 
                                          yellow_time=2, offset=None, travel_time=8)
        obs1, _ = env1.reset()
        controller1.reset()
        
        reward1 = 0
        for _ in range(num_steps):
            action = controller1.get_action(env1.current_time)
            obs1, r, term, trunc, info1 = env1.step(action)
            reward1 += r
        
        results_fixed.append(reward1)
        
        # Test RegularTrafficLightController
        env2 = MultiIntersectionEnv(min_green_time=5, yellow_time=2, switching_penalty=10.0)
        controller2 = RegularTrafficLightController(cycle_time=28, green_time_ns=12, green_time_ew=12,
                                                   yellow_time=2, offset=None, travel_time=8)
        obs2, _ = env2.reset()
        controller2.reset()
        
        reward2 = 0
        for _ in range(num_steps):
            action = controller2.get_action(env2.current_time)
            obs2, r, term, trunc, info2 = env2.step(action)
            reward2 += r
        
        results_regular.append(reward2)
    
    print(f"FixedTimeController - Mean Reward: {np.mean(results_fixed):.2f} "
          f"(std: {np.std(results_fixed):.2f})")
    print(f"RegularTrafficLightController - Mean Reward: {np.mean(results_regular):.2f} "
          f"(std: {np.std(results_regular):.2f})")
    print()


if __name__ == "__main__":
    # Run tests
    test_fixed_time_controller()
    print()
    test_regular_controller()
    print()
    compare_controllers()

