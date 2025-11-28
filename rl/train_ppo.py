"""
Training script for PPO agent on multi-intersection environment.

Usage:
    python rl/train_ppo.py --timesteps 1000000 --save_path models/ppo_agent.pth
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.multi_env import MultiIntersectionEnv
from rl.ppo_agent import PPOAgent


def get_scheduler(total_timesteps, initial_val, final_val=0.0):
    """Linear decay scheduler."""
    def scheduler(step):
        frac = 1.0 - (step / total_timesteps)
        return final_val + (initial_val - final_val) * max(0.0, frac)
    return scheduler


def train_ppo(
    total_timesteps=1000000,
    steps_per_batch=2048,
    save_path="models/ppo_agent.pth",
    eval_freq=20000,
    min_green_time=5,
    yellow_time=2,
    switching_penalty=10.0
):
    """
    Train PPO agent.
    
    Args:
        total_timesteps: Total environment steps to train for
        steps_per_batch: Number of steps to collect before updating
        save_path: Path to save the model
        eval_freq: Frequency of evaluation (in timesteps)
        min_green_time: Minimum green time
        yellow_time: Yellow light duration
        switching_penalty: Penalty for switching phases
    """
    # Create environments (separate for training and eval)
    env = MultiIntersectionEnv(
        min_green_time=min_green_time,
        yellow_time=yellow_time,
        switching_penalty=switching_penalty
    )
    
    eval_env = MultiIntersectionEnv(
        min_green_time=min_green_time,
        yellow_time=yellow_time,
        switching_penalty=switching_penalty
    )
    
    # Hyperparameters
    initial_lr = 3e-4
    initial_ent = 0.05
    
    # Initialize Agent
    agent = PPOAgent(
        state_dim=12,
        lr=initial_lr,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coef=initial_ent,
        batch_size=64,
        hidden_dim=256,
        n_epochs=20
    )
    
    # Schedulers
    lr_scheduler = get_scheduler(total_timesteps, initial_lr, 0.0)
    ent_scheduler = get_scheduler(total_timesteps, initial_ent, 0.0)
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    
    # Metrics
    obs, _ = env.reset()
    current_ep_reward = 0
    
    ep_rewards = []
    avg_rewards = []
    eval_rewards = []
    
    print("=" * 60)
    print("Starting PPO Training")
    print("=" * 60)
    print(f"Total Timesteps: {total_timesteps}")
    print(f"Update Frequency: {steps_per_batch}")
    print(f"Device: {agent.device}")
    print("=" * 60)
    
    timestep = 0
    while timestep < total_timesteps:
        
        # Collection Phase
        for _ in range(steps_per_batch):
            action, log_prob, value = agent.get_action(obs)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(obs, action, log_prob, reward, done, value)
            
            obs = next_obs
            current_ep_reward += reward
            timestep += 1
            
            if done:
                ep_rewards.append(current_ep_reward)
                avg_rewards.append(np.mean(ep_rewards[-100:]))
                
                if len(ep_rewards) % 20 == 0:
                    current_lr = agent.optimizer.param_groups[0]["lr"]
                    print(f"Step {timestep} | Reward: {current_ep_reward:.0f} | LR: {current_lr:.6f}")
                
                obs, _ = env.reset()
                current_ep_reward = 0
                
                if timestep >= total_timesteps:
                    break
        
        # Update Phase
        loss = agent.train()
        
        # Scheduler Step
        new_lr = lr_scheduler(timestep)
        new_ent = ent_scheduler(timestep)
        
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = new_lr
        agent.entropy_coef = new_ent
        
        # Evaluation Phase
        if timestep % eval_freq < steps_per_batch:
            print(f"  >> Update | Loss: {loss:.4f} | LR: {new_lr:.6f} | Ent: {new_ent:.4f}")
            eval_reward = evaluate(eval_env, agent)
            eval_rewards.append(eval_reward)
            agent.save(save_path)
            print(f"  Eval Reward: {eval_reward:.2f} | Model Saved.\n")

    # Final Save
    agent.save(save_path)
    print("Training Complete.")
    
    plot_results(avg_rewards, eval_rewards, save_path.replace(".pth", "_curve.png"))


def evaluate(env, agent, episodes=5):
    """Evaluate agent performance."""
    total_reward = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _, _ = agent.get_action(obs, training=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
    return total_reward / episodes


def plot_results(train_rewards, eval_rewards, path):
    """Plot training curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_rewards, label="Training Avg Reward")
    eval_x = np.linspace(0, len(train_rewards), len(eval_rewards))
    plt.plot(eval_x, eval_rewards, "r-o", label="Evaluation")
    plt.title("PPO Training Curve")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    print(f"Training curves saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for traffic light control")
    
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total training timesteps")
    parser.add_argument("--save_path", type=str, default="models/ppo_agent.pth", help="Path to save model")
    parser.add_argument("--min_green_time", type=int, default=5, help="Minimum green time")
    parser.add_argument("--yellow_time", type=int, default=2, help="Yellow light duration")
    parser.add_argument("--switching_penalty", type=float, default=10.0, help="Switching penalty")
    
    args = parser.parse_args()
    
    train_ppo(
        total_timesteps=args.timesteps,
        save_path=args.save_path,
        min_green_time=args.min_green_time,
        yellow_time=args.yellow_time,
        switching_penalty=args.switching_penalty
    )


if __name__ == "__main__":
    main()
