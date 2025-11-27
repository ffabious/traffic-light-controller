"""
Simple example demonstrating how to use the DQN agent.

This script shows a minimal training loop and evaluation.
"""

import os
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.single_env import SingleIntersectionEnv
from rl.dqn_agent import DQNAgent


def simple_training_example():
    """Simple training example."""
    print("=" * 60)
    print("Simple DQN Training Example")
    print("=" * 60)
    
    # Create environment
    env = SingleIntersectionEnv(
        min_green_time=5,
        yellow_time=2,
        switching_penalty=10.0
    )
    
    # Create agent
    agent = DQNAgent(
        state_dim=5,
        action_dim=2,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        buffer_size=10000,
        target_update_freq=100
    )
    
    print(f"Device: {agent.device}")
    print(f"Starting training...\n")
    
    # Training parameters
    num_episodes = 100
    max_steps = 1000
    warmup_steps = 1000
    
    total_steps = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        
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
            
            obs = next_obs
            episode_reward += reward
            total_steps += 1
            
            if done:
                break
        
        # Update epsilon
        agent.update_epsilon()
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Buffer: {len(agent.replay_buffer)}")
    
    # Save model
    save_path = "models/example_dqn.pth"
    import os
    os.makedirs("models", exist_ok=True)
    agent.save(save_path)
    print(f"\nModel saved to {save_path}")
    
    return agent


def evaluation_example(agent):
    """Simple evaluation example."""
    print("\n" + "=" * 60)
    print("Evaluation Example")
    print("=" * 60)
    
    # Create environment
    env = SingleIntersectionEnv(
        min_green_time=5,
        yellow_time=2,
        switching_penalty=10.0
    )
    
    # Set agent to evaluation mode (no exploration)
    agent.epsilon = 0.0
    
    num_episodes = 10
    max_steps = 1000
    
    total_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action (no exploration)
            action = agent.select_action(obs, training=False)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
              f"Throughput = {info.get('throughput', 0)}, "
              f"Wait Time = {info.get('total_wait', 0):.2f}")
    
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\nAverage Reward: {avg_reward:.2f}")


if __name__ == "__main__":
    # Train agent
    agent = simple_training_example()
    
    # Evaluate agent
    evaluation_example(agent)
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)

