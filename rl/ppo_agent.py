"""
Proximal Policy Optimization (PPO) Agent for Multi-Intersection Control.

This implementation includes:
- Actor-Critic architecture with shared feature extractor
- Generalized Advantage Estimation (GAE)
- Input Normalization (RunningMeanStd)
- Linear Learning Rate and Entropy Decay support
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class RunningMeanStd:
    """Dynamically normalizes inputs to mean 0 and std 1."""
    
    def __init__(self, shape=()):
        self.n = 0
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)

    def update(self, x):
        x = np.array(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if len(x.shape) > 0 else 1

        if self.n == 0:
            self.mean = batch_mean
            self.var = batch_var
            self.n = batch_count
        else:
            delta = batch_mean - self.mean
            total_count = self.n + batch_count
            
            new_mean = self.mean + delta * batch_count / total_count
            m_a = self.var * self.n
            m_b = batch_var * batch_count
            m_2 = m_a + m_b + delta**2 * self.n * batch_count / total_count
            
            self.var = m_2 / total_count
            self.mean = new_mean
            self.n = total_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class ActorCritic(nn.Module):
    """Combined Actor-Critic network."""
    
    def __init__(self, state_dim, action_dims=[2, 2], hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor heads (one per intersection)
        self.actor_head_1 = nn.Linear(hidden_dim, action_dims[0])
        self.actor_head_2 = nn.Linear(hidden_dim, action_dims[1])
        
        # Critic head
        self.critic_head = nn.Linear(hidden_dim, 1)
        
        self._init_weights()

    def _init_weights(self):
        for layer in [self.feature_extractor[0], self.feature_extractor[2], 
                      self.actor_head_1, self.actor_head_2, self.critic_head]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)

    def act(self, state):
        """Select action during rollout."""
        features = self.feature_extractor(state)
        
        logits_1 = self.actor_head_1(features)
        logits_2 = self.actor_head_2(features)
        
        dist_1 = Categorical(logits=logits_1)
        dist_2 = Categorical(logits=logits_2)
        
        action_1 = dist_1.sample()
        action_2 = dist_2.sample()
        
        log_prob = dist_1.log_prob(action_1) + dist_2.log_prob(action_2)
        value = self.critic_head(features)
        
        return np.array([action_1.item(), action_2.item()]), log_prob, value

    def evaluate(self, state, action):
        """Evaluate actions for training update."""
        features = self.feature_extractor(state)
        
        logits_1 = self.actor_head_1(features)
        logits_2 = self.actor_head_2(features)
        
        dist_1 = Categorical(logits=logits_1)
        dist_2 = Categorical(logits=logits_2)
        
        action_1 = action[:, 0]
        action_2 = action[:, 1]
        
        log_prob = dist_1.log_prob(action_1) + dist_2.log_prob(action_2)
        entropy = dist_1.entropy() + dist_2.entropy()
        value = self.critic_head(features)
        
        return log_prob, entropy, value


class PPOMemory:
    """Buffer to store trajectories."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
    def get(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            torch.tensor(self.log_probs),
            np.array(self.rewards),
            np.array(self.dones),
            torch.tensor(self.values)
        )


class PPOAgent:
    """PPO Agent for traffic light control."""
    
    def __init__(
        self,
        state_dim=12,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        n_epochs=20,
        batch_size=64,
        hidden_dim=128,
        device=None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Normalizers
        self.state_normalizer = RunningMeanStd(shape=(state_dim,))
        self.return_normalizer = RunningMeanStd(shape=())
        
        # Network and Optimizer
        self.policy = ActorCritic(state_dim, hidden_dim=hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = PPOMemory()
        
    def get_action(self, state, training=True):
        """Select action. Updates normalizer if training."""
        if training:
            self.state_normalizer.update(state)
            
        norm_state = self.state_normalizer.normalize(state)
        state_tensor = torch.FloatTensor(norm_state).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.act(state_tensor)
            
        return action, log_prob.item(), value.item()

    def store_transition(self, state, action, log_prob, reward, done, value):
        self.memory.store(state, action, log_prob, reward, done, value)

    def train(self):
        """Update policy using collected trajectories."""
        states, actions, old_log_probs, rewards, dones, values = self.memory.get()
        
        # Calculate true returns for reward normalization
        true_returns = np.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            true_returns[t] = running_return
        
        self.return_normalizer.update(true_returns)
        
        # Scale rewards
        scale_factor = np.sqrt(self.return_normalizer.var + 1e-8)
        norm_rewards = rewards / scale_factor

        # Calculate GAE
        advantages = np.zeros(len(norm_rewards), dtype=np.float32)
        last_gae_lam = 0
        
        for t in reversed(range(len(norm_rewards))):
            if t == len(norm_rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0 
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t+1]
                
            delta = norm_rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
            
        returns = advantages + values.numpy()
        
        # Prepare tensors
        norm_states = self.state_normalizer.normalize(states)
        states_t = torch.FloatTensor(norm_states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = old_log_probs.to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        total_loss = 0
        n_samples = len(states)
        indices = np.arange(n_samples)
        
        # PPO Epochs
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                # Get batch
                b_states = states_t[batch_idx]
                b_actions = actions_t[batch_idx]
                b_old_log_probs = old_log_probs_t[batch_idx]
                b_advantages = advantages_t[batch_idx]
                b_returns = returns_t[batch_idx]
                
                # Forward pass
                new_log_probs, entropy, new_values = self.policy.evaluate(b_states, b_actions)
                new_values = new_values.squeeze()
                
                # Ratio and Clipping
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                value_loss = 0.5 * ((new_values - b_returns) ** 2).mean()
                
                # Entropy Loss
                entropy_loss = -entropy.mean()
                
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
                
        self.memory.clear()
        return total_loss / (self.n_epochs * (n_samples // self.batch_size))

    def save(self, filepath):
        """Save the model and normalization stats."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state_mean": self.state_normalizer.mean,
            "state_var": self.state_normalizer.var,
            "return_var": self.return_normalizer.var
        }, filepath)

    def load(self, filepath):
        """Load the model and normalization stats."""
        # weights_only=False required for loading numpy scalars in normalization stats
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.state_normalizer.mean = checkpoint["state_mean"]
        self.state_normalizer.var = checkpoint["state_var"]
        self.state_normalizer.n = 100
        
        if "return_var" in checkpoint:
            self.return_normalizer.var = checkpoint["return_var"]
            self.return_normalizer.n = 100
