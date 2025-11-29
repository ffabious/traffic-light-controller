"""
Evaluation script for trained DQN agent.

Usage:
    python rl/evaluate_dqn.py --model_path models/dqn_model.pth --episodes 100
"""

import argparse
import os
import sys
import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.single_env import SingleIntersectionEnv
from rl.dqn_agent import DQNAgent
from baseline import FixedTimeController
from metrics.traffic_metrics import create_metrics_tracker


def evaluate_agent(
    model_path,
    num_episodes=100,
    max_steps=1000,
    min_green_time=5,
    yellow_time=2,
    switching_penalty=10.0,
    render=False,
    verbose=True,
    save_dir="metrics/output/dqn"
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

    metrics_tracker = create_metrics_tracker(env_type="single", num_lanes=4)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("Evaluating DQN Agent")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)
    
    for episode in range(num_episodes):
        obs, info = env.reset()

        for step in range(max_steps):
            # Select action (no exploration)
            action = agent.select_action(obs, training=False)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            yellow_active = info.get('yellow_timer', 0) > 0

            metrics_tracker.update_step(
                queues=obs[:4].tolist(),
                throughput=info.get('throughput', 0),
                wait_time=info.get('total_wait', 0),
                phase=int(obs[4]),
                reward=reward,
                action=action,
                yellow_active=yellow_active,
                info=info
            )

            obs = next_obs

            if render:
                env.render()
            
            if done:
                break

        episode_metrics = metrics_tracker.finalize_episode()

        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {episode_metrics['total_reward']:>8.2f} | "
                  f"Throughput: {episode_metrics['total_throughput']:>6.0f} | "
                  f"Avg Queue: {episode_metrics['mean_queue_length']:>6.2f} | "
                  f"Service Level: {episode_metrics['service_level']:>5.1%} | "
                  f"Fairness: {episode_metrics['fairness_index']:>5.3f}"
            )


    print()
    metrics_tracker.print_summary(detailed=True)

    if save_dir:
        metrics_file = os.path.join(save_dir, "dqn_metrics.json")
        metrics_tracker.save_to_file(metrics_file)

    return metrics_tracker


def evaluate_baseline_agent(
    num_episodes=100,
    max_steps=1000,
    min_green_time=5,
    yellow_time=2,
    switching_penalty=10.0,
    green_time_ns=None,
    green_time_ew=None,
    verbose=True
):
    """
    Evaluate baseline FixedTimeController agent.
    
    Args:
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        min_green_time: Minimum green time for environment
        yellow_time: Yellow light duration
        switching_penalty: Switching penalty
        green_time_ns: NS green time for baseline controller (defaults to min_green_time)
        green_time_ew: EW green time for baseline controller (defaults to min_green_time)
        verbose: Whether to print detailed information
    
    Returns:
        Dictionary with evaluation metrics (same format as evaluate_agent)
    """
    # Set default green times
    if green_time_ns is None:
        green_time_ns = min_green_time
    if green_time_ew is None:
        green_time_ew = min_green_time
    
    # Calculate cycle time
    cycle_time = green_time_ns + yellow_time + green_time_ew + yellow_time
    
    # Create environment
    env = SingleIntersectionEnv(
        min_green_time=min_green_time,
        yellow_time=yellow_time,
        switching_penalty=switching_penalty
    )
    
    # Create baseline controller
    # Note: Controller returns (action1, action2) for two intersections,
    # but we only use action1 for single intersection
    controller = FixedTimeController(
        cycle_time=cycle_time,
        green_time_ns=green_time_ns,
        green_time_ew=green_time_ew,
        yellow_time=yellow_time,
        offset=None,
        travel_time=8
    )
    
    metrics_tracker = create_metrics_tracker(env_type="single", num_lanes=4)

    if verbose:
        print("=" * 60)
        print("Evaluating Baseline (FixedTimeController)")
        print("=" * 60)
        print(f"Cycle Time: {cycle_time} steps")
        print(f"NS Green Time: {green_time_ns} steps")
        print(f"EW Green Time: {green_time_ew} steps")
        print(f"Yellow Time: {yellow_time} steps")
        print(f"Episodes: {num_episodes}")
        print("=" * 60)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        controller.reset()

        for step in range(max_steps):
            # Get action from baseline controller
            # Controller returns tuple (action1, action2), use action1 for single intersection
            # Use step count as time (env.current_time increments inside step(), so we use step)
            actions = controller.get_action(step)
            action = actions[0]  # Use first intersection's action
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            yellow_active = info.get('yellow_timer', 0) > 0

            metrics_tracker.update_step(
                queues=obs[:4].tolist(),
                throughput=info.get('throughput', 0),
                wait_time=info.get('total_wait', 0),
                phase=int(obs[4]),
                reward=reward,
                action=action,
                yellow_active=yellow_active,
                info=info
            )

            obs = next_obs

            if done:
                break

        episode_metrics = metrics_tracker.finalize_episode()

        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {episode_metrics['total_reward']:>8.2f} | "
                  f"Throughput: {episode_metrics['total_throughput']:>6.0f} | "
                  f"Avg Queue: {episode_metrics['mean_queue_length']:>6.2f} | "
                  f"Service Level: {episode_metrics['service_level']:>5.1%} | "
                  f"Fairness: {episode_metrics['fairness_index']:>5.3f}")
    
    if verbose:
        metrics_tracker.print_summary(detailed=True)

    return metrics_tracker


def compare_with_baseline(
    model_path,
    num_episodes=100,
    max_steps=1000,
    min_green_time=5,
    yellow_time=2,
    switching_penalty=10.0,
    green_time_ns=None,
    green_time_ew=None,
    save_dir="metrics/output/dqn"

):
    """
    Compare DQN agent with baseline FixedTimeController.
    
    Args:
        model_path: Path to saved DQN model
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        min_green_time: Minimum green time
        yellow_time: Yellow light duration
        switching_penalty: Switching penalty
        green_time_ns: NS green time for baseline controller (defaults to min_green_time)
        green_time_ew: EW green time for baseline controller (defaults to min_green_time)
    
    Returns:
        Comparison metrics
    """

    os.makedirs(save_dir, exist_ok=True)

    # Evaluate DQN
    print("Evaluating DQN Agent...")
    dqn_metrics = evaluate_agent(
        model_path, num_episodes, max_steps,
        min_green_time, yellow_time, switching_penalty,
        render=False, verbose=False
    )

    # Evaluate baseline
    print("\nEvaluating Baseline (FixedTimeController)...")
    baseline_metrics = evaluate_baseline_agent(
        num_episodes, max_steps,
        min_green_time, yellow_time, switching_penalty,
        green_time_ns, green_time_ew,
        verbose=False
    )

    # Print detailed comparison
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    dqn_summary = dqn_metrics.get_summary_statistics()
    baseline_summary = baseline_metrics.get_summary_statistics()

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

    print(f"\n{'Metric':<30} {'DQN':<20} {'Baseline':<20} {'Improvement':<15} {'Winner':<10}")
    print("-" * 95)

    comparison_results = {}

    for metric_key, metric_name, direction, higher_better in comparison_metrics:
        if metric_key in dqn_summary and metric_key in baseline_summary:
            dqn_val = dqn_summary[metric_key]['mean']
            baseline_val = baseline_summary[metric_key]['mean']

            # Calculate improvement
            if baseline_val != 0:
                if higher_better:
                    improvement = ((dqn_val - baseline_val) / abs(baseline_val)) * 100
                else:
                    improvement = ((baseline_val - dqn_val) / abs(baseline_val)) * 100
            else:
                improvement = 0.0

            # Determine winner
            if higher_better:
                winner = "DQN" if dqn_val > baseline_val else "Baseline"
            else:
                winner = "DQN" if dqn_val < baseline_val else "Baseline"

            # Color code improvement
            if improvement > 0:
                imp_str = f"+{improvement:>6.2f}%"
            else:
                imp_str = f"{improvement:>7.2f}%"

            print(f"{metric_name:<30} {dqn_val:<20.4f} {baseline_val:<20.4f} "
                  f"{imp_str:<15} {winner:<10}")

            comparison_results[metric_key] = {
                'dqn': dqn_val,
                'baseline': baseline_val,
                'improvement': improvement,
                'winner': winner
            }

    print("=" * 95)

    # Calculate overall score
    dqn_wins = sum(1 for v in comparison_results.values() if v['winner'] == 'DQN')
    baseline_wins = sum(1 for v in comparison_results.values() if v['winner'] == 'Baseline')

    print(f"\nOverall performance:")
    print(f"  DQN wins: {dqn_wins}/{len(comparison_results)}")
    print(f"  Baseline wins: {baseline_wins}/{len(comparison_results)}")

    avg_improvement = np.mean([v['improvement'] for v in comparison_results.values()])
    print(f"  Average Improvement: {avg_improvement:+.2f}%")

    # Statistical summary
    print(f"\n{'─'*95}")
    print("Key performance indicators:")
    print(f"{'─'*95}")

    kpis = [
        ('total_reward', 'Cumulative Reward'),
        ('throughput_rate', 'Vehicles/Step'),
        ('avg_delay_per_vehicle', 'Delay/Vehicle'),
        ('service_level', 'Service Level'),
        ('fairness_index', 'Fairness Index')
    ]

    for metric_key, metric_name in kpis:
        if metric_key in comparison_results:
            result = comparison_results[metric_key]
            print(f"  {metric_name:<25}: DQN={result['dqn']:>10.4f}, "
                  f"Baseline={result['baseline']:>10.4f}, "
                  f"Δ={result['improvement']:>+7.2f}%")

    print("=" * 95 + "\n")

    # Save comparison results
    comparison_file = os.path.join(save_dir, "comparison_results.txt")
    with open(comparison_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPARISON RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"DQN Model: {model_path}\n")
        f.write(f"Episodes Evaluated: {num_episodes}\n")
        f.write(f"Max Steps per Episode: {max_steps}\n\n")

        f.write(f"{'Metric':<30} {'DQN':<20} {'Baseline':<20} {'Improvement':<15}\n")
        f.write("-" * 85 + "\n")

        for metric_key, metric_name, _, _ in comparison_metrics:
            if metric_key in comparison_results:
                result = comparison_results[metric_key]
                f.write(f"{metric_name:<30} {result['dqn']:<20.4f} "
                        f"{result['baseline']:<20.4f} {result['improvement']:>+7.2f}%\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"DQN Wins: {dqn_wins}/{len(comparison_results)}\n")
        f.write(f"Baseline Wins: {baseline_wins}/{len(comparison_results)}\n")
        f.write(f"Average Improvement: {avg_improvement:+.2f}%\n")

    print(f"Comparison results saved to {comparison_file}")

    # Save individual metrics
    dqn_metrics.save_to_file(os.path.join(save_dir, "dqn_metrics.json"))
    baseline_metrics.save_to_file(os.path.join(save_dir, "baseline_metrics.json"))

    return {
        'dqn': dqn_metrics,
        'baseline': baseline_metrics,
        'comparison': comparison_results
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
    parser.add_argument('--baseline_green_ns', type=int, default=None, 
                       help='Baseline NS green time (defaults to min_green_time)')
    parser.add_argument('--baseline_green_ew', type=int, default=None,
                       help='Baseline EW green time (defaults to min_green_time)')
    
    args = parser.parse_args()
    
    if args.compare_baseline:
        compare_with_baseline(
            args.model_path,
            args.episodes,
            args.max_steps,
            args.min_green_time,
            args.yellow_time,
            args.switching_penalty,
            args.baseline_green_ns,
            args.baseline_green_ew
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

