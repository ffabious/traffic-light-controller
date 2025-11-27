"""
Evaluation script for trained DQN agent.

Usage:
    python rl/evaluate_dqn.py --model_path models/dqn_model.pth --episodes 100
"""

import argparse
import os
import sys
import numpy as np
from collections import defaultdict

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.single_env import SingleIntersectionEnv
from rl.dqn_agent import DQNAgent


def evaluate_agent(
    model_path,
    num_episodes=100,
    max_steps=1000,
    min_green_time=5,
    yellow_time=2,
    switching_penalty=10.0,
    render=False,
    verbose=True
):
    """
    Evaluate trained DQN agent.
    
    Args:
        model_path: Path to saved model
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        min_green_time: Minimum green time for environment
        yellow_time: Yellow light duration
        switching_penalty: Switching penalty
        render: Whether to render episodes
        verbose: Whether to print detailed information
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Create environment
    env = SingleIntersectionEnv(
        min_green_time=min_green_time,
        yellow_time=yellow_time,
        switching_penalty=switching_penalty
    )
    
    # Create agent and load model
    agent = DQNAgent(state_dim=5, action_dim=2)
    agent.load(model_path)
    agent.epsilon = 0.0  # No exploration during evaluation
    
    # Metrics
    episode_rewards = []
    episode_lengths = []
    episode_throughputs = []
    episode_wait_times = []
    episode_switches = []
    
    # Per-episode detailed metrics
    lane_queue_stats = defaultdict(list)
    phase_distribution = defaultdict(int)
    
    print("=" * 60)
    print("Evaluating DQN Agent")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_switches_count = 0
        previous_phase = obs[4]
        
        # Track cumulative metrics
        cumulative_throughput = 0
        cumulative_wait_time = 0
        
        for step in range(max_steps):
            # Select action (no exploration)
            action = agent.select_action(obs, training=False)
            
            # Track phase switches
            if action == 1:  # Switch action
                episode_switches_count += 1
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            
            # Accumulate cumulative metrics (info contains per-step values)
            cumulative_throughput += info.get('throughput', 0)
            cumulative_wait_time += info.get('total_wait', 0)
            
            # Track metrics
            for i in range(4):
                lane_queue_stats[i].append(obs[i])
            
            phase_distribution[int(obs[4])] += 1
            
            if render:
                env.render()
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        episode_throughputs.append(cumulative_throughput)
        episode_wait_times.append(cumulative_wait_time)
        episode_switches.append(episode_switches_count)
        
        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Length: {step + 1} | "
                  f"Throughput: {cumulative_throughput} | "
                  f"Wait: {cumulative_wait_time:.2f}")
    
    # Compute statistics
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_throughputs': episode_throughputs,
        'episode_wait_times': episode_wait_times,
        'episode_switches': episode_switches,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'mean_throughput': np.mean(episode_throughputs),
        'std_throughput': np.std(episode_throughputs),
        'mean_wait_time': np.mean(episode_wait_times),
        'std_wait_time': np.std(episode_wait_times),
        'mean_switches': np.mean(episode_switches),
        'std_switches': np.std(episode_switches),
        'lane_queue_stats': {k: {
            'mean': np.mean(v),
            'std': np.std(v),
            'max': np.max(v)
        } for k, v in lane_queue_stats.items()},
        'phase_distribution': dict(phase_distribution)
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Mean Episode Length: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
    print(f"Mean Throughput (cumulative): {metrics['mean_throughput']:.2f} ± {metrics['std_throughput']:.2f}")
    print(f"Mean Wait Time (cumulative): {metrics['mean_wait_time']:.2f} ± {metrics['std_wait_time']:.2f}")
    print(f"Mean Switches per Episode: {metrics['mean_switches']:.2f} ± {metrics['std_switches']:.2f}")
    print("\nLane Queue Statistics:")
    lane_names = ['North', 'South', 'East', 'West']
    for i, name in enumerate(lane_names):
        stats = metrics['lane_queue_stats'][i]
        print(f"  {name}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, Max={stats['max']:.1f}")
    print("\nPhase Distribution:")
    print(f"  Phase 0 (NS Green): {metrics['phase_distribution'].get(0, 0)} steps")
    print(f"  Phase 1 (EW Green): {metrics['phase_distribution'].get(1, 0)} steps")
    print("=" * 60)
    
    return metrics


def compare_with_baseline(
    model_path,
    num_episodes=100,
    max_steps=1000,
    min_green_time=5,
    yellow_time=2,
    switching_penalty=10.0
):
    """
    Compare DQN agent with a simple baseline (fixed-time controller).
    
    Args:
        model_path: Path to saved DQN model
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        min_green_time: Minimum green time
        yellow_time: Yellow light duration
        switching_penalty: Switching penalty
    
    Returns:
        Comparison metrics
    """
    # Evaluate DQN
    print("Evaluating DQN Agent...")
    dqn_metrics = evaluate_agent(
        model_path, num_episodes, max_steps,
        min_green_time, yellow_time, switching_penalty,
        render=False, verbose=False
    )
    
    # Evaluate baseline (fixed-time: switch every min_green_time steps)
    print("\nEvaluating Baseline (Fixed-Time Controller)...")
    env = SingleIntersectionEnv(
        min_green_time=min_green_time,
        yellow_time=yellow_time,
        switching_penalty=switching_penalty
    )
    
    baseline_rewards = []
    baseline_throughputs = []
    baseline_wait_times = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        # Track cumulative metrics
        cumulative_throughput = 0
        cumulative_wait_time = 0
        
        for step in range(max_steps):
            # Fixed-time: switch every min_green_time steps
            action = 1 if (step_count % (min_green_time + yellow_time) == 0 and step_count > 0) else 0
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step_count += 1
            
            # Accumulate cumulative metrics
            cumulative_throughput += info.get('throughput', 0)
            cumulative_wait_time += info.get('total_wait', 0)
            
            if done:
                break
        
        baseline_rewards.append(episode_reward)
        baseline_throughputs.append(cumulative_throughput)
        baseline_wait_times.append(cumulative_wait_time)
    
    baseline_metrics = {
        'mean_reward': np.mean(baseline_rewards),
        'mean_throughput': np.mean(baseline_throughputs),
        'mean_wait_time': np.mean(baseline_wait_times)
    }
    
    # Print comparison
    print("\n" + "=" * 60)
    print("Comparison: DQN vs Baseline")
    print("=" * 60)
    print(f"{'Metric':<20} {'DQN':<15} {'Baseline':<15} {'Improvement':<15}")
    print("-" * 60)
    
    reward_improvement = ((dqn_metrics['mean_reward'] - baseline_metrics['mean_reward']) / 
                         abs(baseline_metrics['mean_reward']) * 100)
    print(f"{'Mean Reward':<20} {dqn_metrics['mean_reward']:<15.2f} "
          f"{baseline_metrics['mean_reward']:<15.2f} {reward_improvement:>14.1f}%")
    
    throughput_improvement = ((dqn_metrics['mean_throughput'] - baseline_metrics['mean_throughput']) / 
                             baseline_metrics['mean_throughput'] * 100)
    print(f"{'Mean Throughput (cum)':<20} {dqn_metrics['mean_throughput']:<15.2f} "
          f"{baseline_metrics['mean_throughput']:<15.2f} {throughput_improvement:>14.1f}%")
    
    wait_improvement = ((baseline_metrics['mean_wait_time'] - dqn_metrics['mean_wait_time']) / 
                       baseline_metrics['mean_wait_time'] * 100)
    print(f"{'Mean Wait Time (cum)':<20} {dqn_metrics['mean_wait_time']:<15.2f} "
          f"{baseline_metrics['mean_wait_time']:<15.2f} {wait_improvement:>14.1f}%")
    print("=" * 60)
    
    return {
        'dqn': dqn_metrics,
        'baseline': baseline_metrics
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained DQN agent')
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--min_green_time', type=int, default=5, help='Minimum green time')
    parser.add_argument('--yellow_time', type=int, default=2, help='Yellow light duration')
    parser.add_argument('--switching_penalty', type=float, default=10.0, help='Switching penalty')
    parser.add_argument('--render', action='store_true', help='Render episodes')
    parser.add_argument('--compare_baseline', action='store_true', help='Compare with baseline')
    
    args = parser.parse_args()
    
    if args.compare_baseline:
        compare_with_baseline(
            args.model_path,
            args.episodes,
            args.max_steps,
            args.min_green_time,
            args.yellow_time,
            args.switching_penalty
        )
    else:
        evaluate_agent(
            args.model_path,
            args.episodes,
            args.max_steps,
            args.min_green_time,
            args.yellow_time,
            args.switching_penalty,
            args.render
        )


if __name__ == "__main__":
    main()

