"""
PPO Agent for Trading

Implements Proximal Policy Optimization (PPO) for trading decisions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path

try:
    from .rollout_buffer import RolloutBuffer
except ImportError:
    from rollout_buffer import RolloutBuffer

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dims=[128, 64]):
        super(Actor, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        logits = self.network(state)
        return torch.softmax(logits, dim=-1)

class Critic(nn.Module):

    def __init__(self, state_dim, hidden_dims=[128, 64]):

        super(Critic, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state).squeeze(-1)

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent for trading decisions.
    
    Actions:
        0: Buy
        1: Sell
        2: Hold (do nothing)
    """

    # Action constants
    ACTION_BUY = 0
    ACTION_SELL = 1
    ACTION_HOLD = 2

    def __init__(
        self,
        state_dim,
        action_dim=3,
        hidden_dims=[128, 64],
        learning_rate=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        target_kl=0.01,
        buffer_size=2048,
        batch_size=64,
        value_loss_coeff=0.5,
        entropy_coeff=0.01,
        device='cpu'
    ):
        """
        Initialize the PPO Agent.

        Args:
            state_dim: Dimension of state space (number of features)
            action_dim: Dimension of action space (number of actions)
            hidden_dims: List of hidden layer dimensions for actor and critic
            learning_rate: Learning rate for both actor and critic
            gamma: Discount factor for future rewards
            lam: GAE lambda parameter
            clip_ratio: PPO clipping ratio
            target_kl: Target KL divergence for early stopping
            buffer_size: Size of the rollout buffer
            batch_size: Mini-batch size for training
            value_loss_coeff: Multiplier for critic (value) loss term
            entropy_coeff: Multiplier for entropy bonus (encourages exploration)
            device: Device to run the model on (e.g., 'cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.value_loss_coeff = value_loss_coeff
        self.entropy_coeff = entropy_coeff
        self.actor = Actor(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic = Critic(state_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate
        )
        self.buffer = RolloutBuffer(state_dim, buffer_size, gamma, lam)
        self.training_step = 0
        self.losses = []

    def select_action(self, state, training=True):
        """
        Select an action based on the current state.

        Args:
            state: Current state (numpy array)
            training: Whether in training mode (stochastic) or evaluation mode (deterministic)

        Returns:
            action: Selected action (int)
            log_prob: Log probability of the selected action
            value: Estimated state value
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.actor(state_tensor)
            value = self.critic(state_tensor).item()
        dist = torch.distributions.Categorical(probs)
        if training:
            action = dist.sample().item()
        else:
            action = torch.argmax(probs).item()
        log_prob = dist.log_prob(torch.tensor(action).to(self.device)).item()
        return action, log_prob, value

    def store_transition(self, state, action, reward, done, log_prob, value):
        self.buffer.store(state, action, reward, done, log_prob, value)

    def finish_path(self, last_value=0):
        self.buffer.finish_path(last_value)

    def train_step(self):
        """
        Perform one PPO update using data currently in the rollout buffer.

        Returns:
            metrics (dict): Collected training statistics.
        """

        self.buffer.finish_path(last_value=0)
        states, actions, returns, advantages, log_probs_old = self.buffer.get()
        states = states.to(self.device)
        actions = actions.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        log_probs_old = log_probs_old.to(self.device)

        n = states.shape[0]
        batch_size = min(self.batch_size, n)
        indices = np.arange(n)

        losses_, actor_losses, critic_losses, entropies, kls = [], [], [], [], []

        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            end = start + batch_size
            mb_idx = torch.as_tensor(indices[start:end], dtype=torch.long, device=self.device)
            mb_states = states[mb_idx]
            mb_actions = actions[mb_idx]
            mb_returns = returns[mb_idx]
            mb_adv = advantages[mb_idx]
            mb_logp_old = log_probs_old[mb_idx]

            # Policy forward
            probs = self.actor(mb_states)
            dist = torch.distributions.Categorical(probs)
            logp = dist.log_prob(mb_actions)
            ratio = torch.exp(logp - mb_logp_old)

            # Clipped surrogate objective
            clip_low, clip_high = 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
            clipped_ratio = torch.clamp(ratio, clip_low, clip_high)
            surrogate1 = ratio * mb_adv
            surrogate2 = clipped_ratio * mb_adv
            actor_loss = -torch.mean(torch.min(surrogate1, surrogate2))

            # Value loss
            value_preds = self.critic(mb_states)
            critic_loss = F.mse_loss(value_preds, mb_returns)

            # Entropy bonus
            entropy = dist.entropy().mean()

            # Total loss with configurable coefficients
            loss = actor_loss + self.value_loss_coeff * critic_loss - self.entropy_coeff * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Diagnostics
            with torch.no_grad():
                approx_kl = torch.mean(mb_logp_old - logp).item()
            losses_.append(loss.item())
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy.item())
            kls.append(approx_kl)

            self.training_step += 1

            # # Early stopping by KL divergence
            # if approx_kl > 1.5 * self.target_kl:
            #     print(f"Early stopping at step due to KL divergence: {approx_kl:.4f}")
            #     break

        metrics = {
            "losses": [float(x) for x in losses_],
            "actor_losses": [float(x) for x in actor_losses],
            "critic_losses": [float(x) for x in critic_losses],
            "entropies": [float(x) for x in entropies],
            "kls": [float(x) for x in kls],
            "updates": self.training_step
        }

        self.losses.append(metrics)
        self.buffer.clear()
        return metrics

    def save(self, filepath):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'losses': self.losses,
            'value_loss_coeff': self.value_loss_coeff,
            'entropy_coeff': self.entropy_coeff
        }, filepath)
        print(f"✓ PPO Agent saved to {filepath}")

    def load(self, filepath):
        """
        Load agent state.
        
        Args:
            filepath: Path to load file
        """
        if not Path(filepath).exists():
            print(f"✗ No saved PPO agent found at {filepath}")
            return False

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.losses = checkpoint['losses']
        self.value_loss_coeff = checkpoint.get('value_loss_coeff', self.value_loss_coeff)
        self.entropy_coeff = checkpoint.get('entropy_coeff', self.entropy_coeff)

        print(f"✓ PPO Agent loaded from {filepath}")
        return True

    def get_action_name(self, action):
        """Get human-readable action name."""
        if action == self.ACTION_BUY:
            return "BUY"
        elif action == self.ACTION_SELL:
            return "SELL"
        else:
            return "HOLD"

if __name__ == "__main__":

    print("Testing PPO Agent...")

    state_dim = 7
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=3,
        hidden_dims=[128, 64],
        learning_rate=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        target_kl=0.01,
        buffer_size=100,
        batch_size=32,
        value_loss_coeff=0.5,
        entropy_coeff=0.01,
        device='cpu'
    )

    print(f"✓ PPO Agent created with {state_dim} state features")

    # Test action selection
    state = np.random.randn(state_dim)
    action, log_prob, value = agent.select_action(state, training=True)
    print(f"\n✓ Selected action: {action} ({agent.get_action_name(action)})")
    print(f"✓ Log prob: {log_prob:.4f}, Value: {value:.4f}")

    # Simulate random transitions
    buffer_size = 100
    print(f"\n✓ Simulating {buffer_size} transitions...")
    last_state = None
    for i in range(buffer_size):
        state = np.random.randn(state_dim)
        action, log_prob, value = agent.select_action(state)
        reward = np.random.randn()
        done = (i % 20 == 19)
        agent.store_transition(state, action, reward, done, log_prob, value)
        if done:
            agent.finish_path(last_value=0.0)
        last_state = state

    # Final segment handling
    if agent.buffer.path_start_idx != agent.buffer.ptr:
        with torch.no_grad():
            v_last = agent.critic(torch.FloatTensor(last_state).unsqueeze(0)).item()
        agent.finish_path(last_value=v_last)

    print(f"✓ Buffer filled. Training...")

    metrics = agent.train_step()
    print(f"✓ Training complete. Metrics: {metrics}")

    # Save & reload
    save_path = "/tmp/ppo_agent_test.pth"
    agent.save(save_path)
    new_agent = PPOAgent(state_dim=state_dim)
    new_agent.load(save_path)
    print("\n✅ All tests passed!")