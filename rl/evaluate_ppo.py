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
from metrics.traffic_metrics import create_metrics_tracker


def evaluate_agent(
    model_path,
    num_episodes=100,
    min_green_time=5,
    yellow_time=2,
    switching_penalty=10.0,
    verbose=True,
    save_dir="metrics/output/ppo"
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
    
    metrics_tracker = create_metrics_tracker(
        env_type="multi",
        num_intersections=2,
        lanes_per_intersection=4
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if verbose:
        print("=" * 80)
        print("Evaluating PPO Agent with Comprehensive Metrics")
        print("=" * 80)
        print(f"Model: {model_path}")
        print(f"Episodes: {num_episodes}")
        print(f"Device: {agent.device}")
        print("=" * 80)


    for episode in range(num_episodes):
        obs, _ = env.reset(seed=42 + episode)
        done = False

        while not done:
            # PPO action (deterministic)
            action, _, _ = agent.get_action(obs, training=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            metrics_tracker.update_multi_intersection_step(obs, action, reward, info)

            obs = next_obs

        episode_metrics = metrics_tracker.finalize_episode()

        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {episode_metrics['total_reward']:>8.2f} | "
                  f"Throughput: {episode_metrics['total_throughput']:>6.0f} | "
                  f"Avg Queue: {episode_metrics['mean_queue_length']:>6.2f} | "
                  f"Service Level: {episode_metrics['service_level']:>5.1%} | "
                  f"Coordination: {episode_metrics.get('phase_sync_rate', 0):>5.1%}")

    if verbose:
        metrics_tracker.print_summary(detailed=True)

    if save_dir:
        metrics_file = os.path.join(save_dir, "ppo_metrics.json")
        metrics_tracker.save_to_file(metrics_file)

    return metrics_tracker


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

    metrics_tracker = create_metrics_tracker(
        env_type="multi",
        num_intersections=2,
        lanes_per_intersection=4
    )

    if verbose:
        print("=" * 80)
        print("Evaluating Baseline (FixedTimeController) with Comprehensive Metrics")
        print("=" * 80)
        print(f"Cycle Time: {cycle_time} | NS: {g_ns} | EW: {g_ew}")
        print(f"Episodes: {num_episodes}")
        print("=" * 80)

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=42 + episode)
        controller.reset()
        done = False

        while not done:
            action = controller.get_action(env.current_time)
            next_obs, reward, terminated, truncated, info = env.step(np.array(action))
            done = terminated or truncated

            metrics_tracker.update_multi_intersection_step(obs, action, reward, info)

            obs = next_obs

        episode_metrics = metrics_tracker.finalize_episode()

        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {episode_metrics['total_reward']:>8.2f} | "
                  f"Throughput: {episode_metrics['total_throughput']:>6.0f} | "
                  f"Avg Queue: {episode_metrics['mean_queue_length']:>6.2f} | "
                  f"Service Level: {episode_metrics['service_level']:>5.1%} | "
                  f"Coordination: {episode_metrics.get('phase_sync_rate', 0):>5.1%}")
    if verbose:
        metrics_tracker.print_summary(detailed=True)

    return metrics_tracker


def compare_with_baseline(args):
    """Compare PPO agent with Baseline controller."""
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    print("Evaluating PPO Agent...")
    ppo_metrics = evaluate_agent(
        args.model_path, args.episodes,
        args.min_green_time, args.yellow_time, args.switching_penalty,
        verbose=False
    )

    print("Evaluating Baseline Controller...")
    baseline_metrics = evaluate_baseline_agent(
        args.episodes,
        args.min_green_time, args.yellow_time, args.switching_penalty,
        args.baseline_green_ns, args.baseline_green_ew,
        verbose=False
    )

    ppo_summary = ppo_metrics.get_summary_statistics()
    baseline_summary = baseline_metrics.get_summary_statistics()

    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    comparison_metrics = [
        ('total_reward', 'Total Reward', 'higher', True),
        ('mean_queue_length', 'Mean Queue Length', 'lower', False),
        ('max_queue_length', 'Max Queue Length', 'lower', False),
        ('total_throughput', 'Total Throughput', 'higher', True),
        ('mean_wait_time', 'Mean Wait Time', 'lower', False),
        ('throughput_rate', 'Throughput Rate', 'higher', True),
        ('system_efficiency', 'System Efficiency', 'higher', True),
        ('fairness_index', 'Fairness Index', 'higher', True),
        ('service_level', 'Service Level', 'higher', True),
        ('congestion_level', 'Congestion Level', 'lower', False),
        ('queue_imbalance', 'Queue Imbalance', 'lower', False),
        ('queue_oscillation', 'Queue Oscillation', 'lower', False),
        ('phase_switches', 'Phase Switches', 'lower', False),
        ('avg_delay_per_vehicle', 'Avg Delay/Vehicle', 'lower', False),
    ]

    print(f"\n{'Metric':<30} {'PPO':<20} {'Baseline':<20} {'Improvement':<15} {'Winner':<10}")
    print("-" * 95)

    comparison_results = {}

    for metric_key, metric_name, direction, higher_better in comparison_metrics:
        if metric_key in ppo_summary and metric_key in baseline_summary:
            ppo_val = ppo_summary[metric_key]['mean']
            baseline_val = baseline_summary[metric_key]['mean']

            # Calculate improvement
            if baseline_val != 0:
                if higher_better:
                    improvement = ((ppo_val - baseline_val) / abs(baseline_val)) * 100
                else:
                    improvement = ((baseline_val - ppo_val) / abs(baseline_val)) * 100
            else:
                improvement = 0.0

            # Determine winner
            if higher_better:
                winner = "PPO" if ppo_val > baseline_val else "Baseline"
            else:
                winner = "PPO" if ppo_val < baseline_val else "Baseline"

            # Format improvement
            if improvement > 0:
                imp_str = f"+{improvement:>6.2f}%"
            else:
                imp_str = f"{improvement:>7.2f}%"

            print(f"{metric_name:<30} {ppo_val:<20.4f} {baseline_val:<20.4f} "
                  f"{imp_str:<15} {winner:<10}")

            comparison_results[metric_key] = {
                'ppo': ppo_val,
                'baseline': baseline_val,
                'improvement': improvement,
                'winner': winner
            }

    print("=" * 95)

    # Multi-intersection specific metrics
    print(f"\n{'─'*95}")
    print("Multi-Intersection Coordination Metrics:")
    print(f"{'─'*95}")

    # Phase synchronization
    if 'phase_sync_rate' in ppo_metrics.episode_data and 'phase_sync_rate' in baseline_metrics.episode_data:
        ppo_sync = np.mean(ppo_metrics.episode_data['phase_sync_rate'])
        baseline_sync = np.mean(baseline_metrics.episode_data['phase_sync_rate'])
        sync_improvement = ((ppo_sync - baseline_sync) / baseline_sync * 100) if baseline_sync != 0 else 0

        print(f"  Phase Sync Rate:            PPO={ppo_sync:>6.2%}, "
              f"Baseline={baseline_sync:>6.2%}, Δ={sync_improvement:>+7.2f}%")

    if 'mean_phase_offset' in ppo_metrics.episode_data and 'mean_phase_offset' in baseline_metrics.episode_data:
        ppo_offset = np.mean(ppo_metrics.episode_data['mean_phase_offset'])
        baseline_offset = np.mean(baseline_metrics.episode_data['mean_phase_offset'])

        print(f"  Mean Phase Offset:          PPO={ppo_offset:>6.2f}, "
              f"Baseline={baseline_offset:>6.2f}")

    # Calculate overall score
    ppo_wins = sum(1 for v in comparison_results.values() if v['winner'] == 'PPO')
    baseline_wins = sum(1 for v in comparison_results.values() if v['winner'] == 'Baseline')

    print(f"\n{'─'*95}")
    print(f"Overall Performance:")
    print(f"  PPO Wins: {ppo_wins}/{len(comparison_results)}")
    print(f"  Baseline Wins: {baseline_wins}/{len(comparison_results)}")

    avg_improvement = np.mean([v['improvement'] for v in comparison_results.values()])
    print(f"  Average Improvement: {avg_improvement:+.2f}%")

    # Key Performance Indicators
    print(f"\n{'─'*95}")
    print("Key performance indicators:")
    print(f"{'─'*95}")

    kpis = [
        ('total_reward', 'Cumulative Reward'),
        ('throughput_rate', 'Vehicles/Step'),
        ('avg_delay_per_vehicle', 'Delay/Vehicle'),
        ('service_level', 'Service Level'),
        ('fairness_index', 'Fairness Index'),
        ('system_efficiency', 'System Efficiency')
    ]

    for metric_key, metric_name in kpis:
        if metric_key in comparison_results:
            result = comparison_results[metric_key]
            print(f"  {metric_name:<25}: PPO={result['ppo']:>10.4f}, "
                  f"Baseline={result['baseline']:>10.4f}, "
                  f"Δ={result['improvement']:>+7.2f}%")

    print("=" * 95 + "\n")

    comparison_file = os.path.join(args.save_dir, "comparison_results.txt")

    with open(comparison_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPARISON RESULTS (Multi-Intersection)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"PPO Model: {args.model_path}\n")
        f.write(f"Episodes Evaluated: {args.episodes}\n\n")

        f.write(f"{'Metric':<30} {'PPO':<20} {'Baseline':<20} {'Improvement':<15}\n")
        f.write("-" * 85 + "\n")

        for metric_key, metric_name, _, _ in comparison_metrics:
            if metric_key in comparison_results:
                result = comparison_results[metric_key]
                f.write(f"{metric_name:<30} {result['ppo']:<20.4f} "
                        f"{result['baseline']:<20.4f} {result['improvement']:>+7.2f}%\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"PPO Wins: {ppo_wins}/{len(comparison_results)}\n")
        f.write(f"Baseline Wins: {baseline_wins}/{len(comparison_results)}\n")
        f.write(f"Average Improvement: {avg_improvement:+.2f}%\n")

    print(f"Comparison results saved to {comparison_file}")

    ppo_metrics.save_to_file(os.path.join(args.save_dir, "ppo_metrics.json"))
    baseline_metrics.save_to_file(os.path.join(args.save_dir, "baseline_metrics.json"))

    return {
        'ppo': ppo_metrics,
        'baseline': baseline_metrics,
        'comparison': comparison_results
    }

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

    parser.add_argument("--save_dir", type=str, default="metrics/output/ppo")
    
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
