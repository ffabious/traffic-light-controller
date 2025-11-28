"""
Evaluation script for trained PPO agent on Multi-Intersection Environment.
Compares performance against a Fixed-Time Controller baseline.

Usage:
    python rl/evaluate_ppo.py --model_path models/ppo_model.pth --episodes 100 --compare_baseline
"""

import argparse
import os
import sys
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.multi_env import MultiIntersectionEnv
from rl.ppo_agent import PPOAgent
from baseline.regular_controller import FixedTimeController


def evaluate_agent(
    model_path,
    num_episodes=100,
    min_green_time=5,
    yellow_time=2,
    switching_penalty=10.0,
    verbose=True
):
    """Evaluate trained PPO agent."""
    env = MultiIntersectionEnv(
        min_green_time=min_green_time,
        yellow_time=yellow_time,
        switching_penalty=switching_penalty
    )
    
    # Load PPO Agent
    agent = PPOAgent(state_dim=12, hidden_dim=256)
    agent.load(model_path)
    
    episode_rewards = []
    episode_waits = []
    episode_throughputs = []
    
    if verbose:
        print("=" * 60)
        print("Evaluating PPO Agent")
        print("=" * 60)
    
    for i in range(num_episodes):
        obs, _ = env.reset(seed=42 + i)
        done = False
        total_reward = 0
        total_wait = 0
        total_throughput = 0
        
        while not done:
            # PPO action (deterministic)
            action, _, _ = agent.get_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            total_wait += info["total_wait"]
            total_throughput += info["throughput"]
            
        episode_rewards.append(total_reward)
        episode_waits.append(total_wait)
        episode_throughputs.append(total_throughput)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"Episode {i + 1}/{num_episodes} | Reward: {total_reward:.2f}")

    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "mean_wait": np.mean(episode_waits),
        "mean_throughput": np.mean(episode_throughputs)
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("PPO Evaluation Summary")
        print("=" * 60)
        print(f"Mean Reward:     {metrics['mean_reward']:.2f}")
        print(f"Mean Wait Time:  {metrics['mean_wait']:.2f}")
        print(f"Mean Throughput: {metrics['mean_throughput']:.2f}")
        print("=" * 60)
        
    return metrics


def evaluate_baseline_agent(
    num_episodes=100,
    min_green_time=5,
    yellow_time=2,
    switching_penalty=10.0,
    green_ns=None,
    green_ew=None,
    verbose=True
):
    """Evaluate FixedTimeController baseline."""
    env = MultiIntersectionEnv(
        min_green_time=min_green_time,
        yellow_time=yellow_time,
        switching_penalty=switching_penalty
    )
    
    # Defaults to min_green_time if not provided
    g_ns = green_ns if green_ns else min_green_time
    g_ew = green_ew if green_ew else min_green_time
    cycle_time = g_ns + yellow_time + g_ew + yellow_time
    
    # Baseline controller
    controller = FixedTimeController(
        cycle_time=cycle_time,
        green_time_ns=g_ns,
        green_time_ew=g_ew,
        yellow_time=yellow_time,
        offset=None
    )
    
    episode_rewards = []
    episode_waits = []
    episode_throughputs = []
    
    if verbose:
        print("=" * 60)
        print("Evaluating Baseline (FixedTimeController)")
        print("=" * 60)
        print(f"Cycle Time: {cycle_time} | NS: {g_ns} | EW: {g_ew}")
    
    for i in range(num_episodes):
        obs, _ = env.reset(seed=42 + i)
        controller.reset()
        done = False
        total_reward = 0
        total_wait = 0
        total_throughput = 0
        
        while not done:
            action = controller.get_action(env.current_time)
            obs, reward, terminated, truncated, info = env.step(np.array(action))
            done = terminated or truncated
            
            total_reward += reward
            total_wait += info["total_wait"]
            total_throughput += info["throughput"]
            
        episode_rewards.append(total_reward)
        episode_waits.append(total_wait)
        episode_throughputs.append(total_throughput)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"Episode {i + 1}/{num_episodes} | Reward: {total_reward:.2f}")

    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "mean_wait": np.mean(episode_waits),
        "mean_throughput": np.mean(episode_throughputs)
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("Baseline Evaluation Summary")
        print("=" * 60)
        print(f"Mean Reward:     {metrics['mean_reward']:.2f}")
        print(f"Mean Wait Time:  {metrics['mean_wait']:.2f}")
        print(f"Mean Throughput: {metrics['mean_throughput']:.2f}")
        print("=" * 60)
        
    return metrics


def compare_with_baseline(args):
    """Compare PPO agent with Baseline controller."""
    # Run PPO Evaluation
    ppo_metrics = evaluate_agent(
        args.model_path, args.episodes,
        args.min_green_time, args.yellow_time, args.switching_penalty,
        verbose=False
    )
    
    # Run Baseline Evaluation
    baseline_metrics = evaluate_baseline_agent(
        args.episodes,
        args.min_green_time, args.yellow_time, args.switching_penalty,
        args.baseline_green_ns, args.baseline_green_ew,
        verbose=False
    )
    
    # Display Results
    print(f"{"Metric":<25} {"PPO":<20} {"Baseline":<20} {"Improvement":<15}")
    print("-" * 80)
    
    # Reward Improvement (Higher is better)
    r_imp = ((ppo_metrics["mean_reward"] - baseline_metrics["mean_reward"]) / 
             abs(baseline_metrics["mean_reward"]) * 100)
    print(f"{"Mean Reward":<25} {ppo_metrics["mean_reward"]:<20.0f} "
          f"{baseline_metrics["mean_reward"]:<20.0f} {r_imp:>14.2f}%")
    
    # Wait Time Improvement (Lower is better, so inverted)
    w_imp = ((baseline_metrics["mean_wait"] - ppo_metrics["mean_wait"]) / 
             baseline_metrics["mean_wait"] * 100)
    print(f"{"Mean Wait Time":<25} {ppo_metrics["mean_wait"]:<20.0f} "
          f"{baseline_metrics["mean_wait"]:<20.0f} {w_imp:>14.2f}%")

    # Throughput Improvement (Higher is better)
    t_imp = ((ppo_metrics["mean_throughput"] - baseline_metrics["mean_throughput"]) / 
             baseline_metrics["mean_throughput"] * 100)
    print(f"{"Mean Throughput":<25} {ppo_metrics["mean_throughput"]:<20.0f} "
          f"{baseline_metrics["mean_throughput"]:<20.0f} {t_imp:>14.2f}%")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--compare_baseline", action="store_true", help="Compare with baseline")
    
    # Environment parameters
    parser.add_argument("--min_green_time", type=int, default=5, help="Minimum green time")
    parser.add_argument("--yellow_time", type=int, default=2, help="Yellow light duration")
    parser.add_argument("--switching_penalty", type=float, default=10.0, help="Switching penalty")
    
    # Baseline parameters
    parser.add_argument("--baseline_green_ns", type=int, default=None, help="Baseline NS green time")
    parser.add_argument("--baseline_green_ew", type=int, default=None, help="Baseline EW green time")
    
    args = parser.parse_args()
    
    if args.compare_baseline:
        compare_with_baseline(args)
    else:
        evaluate_agent(
            args.model_path, args.episodes,
            args.min_green_time, args.yellow_time, args.switching_penalty
        )


if __name__ == "__main__":
    main()
