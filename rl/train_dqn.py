"""
Training script for DQN agent on single intersection traffic light control.

Usage:
    python rl/train_dqn.py --episodes 1000 --save_path models/dqn_model.pth
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.single_env import SingleIntersectionEnv
from rl.dqn_agent import DQNAgent


def train_dqn(
    episodes=1000,
    max_steps=1000,
    save_path="models/dqn_model.pth",
    save_freq=100,
    eval_freq=50,
    eval_episodes=10,
    min_green_time=5,
    yellow_time=2,
    switching_penalty=10.0,
    lr=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    batch_size=64,
    buffer_size=10000,
    target_update_freq=100,
    hidden_dims=[128, 128],
    warmup_steps=1000
):
    """
    Train DQN agent on single intersection environment.
    
    Args:
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        save_path: Path to save the trained model
        save_freq: Frequency of saving model (in episodes)
        eval_freq: Frequency of evaluation (in episodes)
        eval_episodes: Number of episodes for evaluation
        min_green_time: Minimum green time for environment
        yellow_time: Yellow light duration
        switching_penalty: Penalty for switching phases
        lr: Learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Epsilon decay rate
        batch_size: Batch size for training
        buffer_size: Size of replay buffer
        target_update_freq: Target network update frequency
        hidden_dims: Hidden layer dimensions
        warmup_steps: Steps before starting training
    """
    # Create environment
    env = SingleIntersectionEnv(
        min_green_time=min_green_time,
        yellow_time=yellow_time,
        switching_penalty=switching_penalty
    )
    
    # Create agent
    agent = DQNAgent(
        state_dim=5,
        action_dim=2,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        buffer_size=buffer_size,
        target_update_freq=target_update_freq,
        hidden_dims=hidden_dims
    )
    
    # Create save directory
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    eval_rewards = []
    eval_throughputs = []
    eval_wait_times = []
    
    # Rolling averages for tracking
    recent_rewards = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)
    
    print("=" * 60)
    print("Starting DQN Training")
    print("=" * 60)
    print(f"Environment: Single Intersection")
    print(f"Episodes: {episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Device: {agent.device}")
    print("=" * 60)
    
    total_steps = 0
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_loss = 0
        loss_count = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(obs, training=True)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(obs, action, reward, next_obs, done)
            
            # Train agent (after warmup)
            if total_steps >= warmup_steps:
                loss = agent.train()
                if loss is not None:
                    episode_loss += loss
                    loss_count += 1
            
            obs = next_obs
            episode_reward += reward
            total_steps += 1
            
            if done:
                break
        
        # Update epsilon
        agent.update_epsilon()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        recent_rewards.append(episode_reward)
        recent_lengths.append(step + 1)
        
        if loss_count > 0:
            episode_losses.append(episode_loss / loss_count)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(recent_lengths)
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Reward: {episode_reward:.2f} (avg: {avg_reward:.2f}) | "
                  f"Length: {step + 1} (avg: {avg_length:.1f}) | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Buffer: {len(agent.replay_buffer)}")
        
        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_reward, eval_throughput, eval_wait = evaluate_agent(
                env, agent, eval_episodes, max_steps
            )
            eval_rewards.append(eval_reward)
            eval_throughputs.append(eval_throughput)
            eval_wait_times.append(eval_wait)
            print(f"\n[Evaluation] Episode {episode + 1}")
            print(f"  Avg Reward: {eval_reward:.2f}")
            print(f"  Avg Throughput: {eval_throughput:.2f}")
            print(f"  Avg Wait Time: {eval_wait:.2f}\n")
        
        # Save model
        if (episode + 1) % save_freq == 0:
            agent.save(save_path)
            print(f"Model saved to {save_path}")
    
    # Final save
    agent.save(save_path)
    print(f"\nTraining completed! Final model saved to {save_path}")
    
    # Plot training curves
    plot_training_curves(
        episode_rewards,
        episode_lengths,
        episode_losses,
        eval_rewards,
        eval_throughputs,
        eval_wait_times,
        save_path.replace(".pth", "_training_curves.png")
    )
    
    return agent, {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_losses': episode_losses,
        'eval_rewards': eval_rewards,
        'eval_throughputs': eval_throughputs,
        'eval_wait_times': eval_wait_times
    }


def evaluate_agent(env, agent, num_episodes=10, max_steps=1000):
    """
    Evaluate agent performance.
    
    Args:
        env: Environment instance
        agent: Trained agent
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
    
    Returns:
        Average reward, throughput, and wait time
    """
    total_rewards = []
    total_throughputs = []
    total_wait_times = []
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        
        # Track cumulative metrics
        cumulative_throughput = 0
        cumulative_wait_time = 0
        
        for step in range(max_steps):
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            
            # Accumulate cumulative metrics
            cumulative_throughput += info.get('throughput', 0)
            cumulative_wait_time += info.get('total_wait', 0)
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        total_throughputs.append(cumulative_throughput)
        total_wait_times.append(cumulative_wait_time)
    
    return (
        np.mean(total_rewards),
        np.mean(total_throughputs),
        np.mean(total_wait_times)
    )


def plot_training_curves(
    episode_rewards,
    episode_lengths,
    episode_losses,
    eval_rewards,
    eval_throughputs,
    eval_wait_times,
    save_path
):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    if len(episode_rewards) >= 100:
        window = 100
        smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(episode_rewards)), smoothed, color='red', label='Smoothed (100)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(episode_lengths, alpha=0.3, color='green')
    if len(episode_lengths) >= 100:
        window = 100
        smoothed = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(episode_lengths)), smoothed, color='red')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].grid(True)
    
    # Loss
    if episode_losses:
        axes[0, 2].plot(episode_losses, alpha=0.3, color='orange')
        if len(episode_losses) >= 100:
            window = 100
            smoothed = np.convolve(episode_losses, np.ones(window)/window, mode='valid')
            axes[0, 2].plot(range(window-1, len(episode_losses)), smoothed, color='red')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('Training Loss')
        axes[0, 2].grid(True)
    
    # Evaluation rewards
    if eval_rewards:
        axes[1, 0].plot(eval_rewards, marker='o', color='purple')
        axes[1, 0].set_xlabel('Evaluation (every N episodes)')
        axes[1, 0].set_ylabel('Avg Reward')
        axes[1, 0].set_title('Evaluation Rewards')
        axes[1, 0].grid(True)
    
    # Evaluation throughput
    if eval_throughputs:
        axes[1, 1].plot(eval_throughputs, marker='o', color='cyan')
        axes[1, 1].set_xlabel('Evaluation (every N episodes)')
        axes[1, 1].set_ylabel('Avg Throughput')
        axes[1, 1].set_title('Evaluation Throughput')
        axes[1, 1].grid(True)
    
    # Evaluation wait times
    if eval_wait_times:
        axes[1, 2].plot(eval_wait_times, marker='o', color='red')
        axes[1, 2].set_xlabel('Evaluation (every N episodes)')
        axes[1, 2].set_ylabel('Avg Wait Time')
        axes[1, 2].set_title('Evaluation Wait Times')
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training curves saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train DQN agent for traffic light control')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--save_path', type=str, default='models/dqn_model.pth', help='Path to save model')
    parser.add_argument('--save_freq', type=int, default=100, help='Save frequency (episodes)')
    parser.add_argument('--eval_freq', type=int, default=50, help='Evaluation frequency (episodes)')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Episodes for evaluation')
    
    # Environment parameters
    parser.add_argument('--min_green_time', type=int, default=5, help='Minimum green time')
    parser.add_argument('--yellow_time', type=int, default=2, help='Yellow light duration')
    parser.add_argument('--switching_penalty', type=float, default=10.0, help='Switching penalty')
    
    # Agent parameters
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='Initial epsilon')
    parser.add_argument('--epsilon_end', type=float, default=0.01, help='Final epsilon')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--buffer_size', type=int, default=10000, help='Replay buffer size')
    parser.add_argument('--target_update_freq', type=int, default=100, help='Target network update frequency')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps before training')
    
    # Network architecture
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 128], help='Hidden layer dimensions')
    
    args = parser.parse_args()
    
    train_dqn(
        episodes=args.episodes,
        max_steps=args.max_steps,
        save_path=args.save_path,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        min_green_time=args.min_green_time,
        yellow_time=args.yellow_time,
        switching_penalty=args.switching_penalty,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        target_update_freq=args.target_update_freq,
        hidden_dims=args.hidden_dims,
        warmup_steps=args.warmup_steps
    )


if __name__ == "__main__":
    main()

