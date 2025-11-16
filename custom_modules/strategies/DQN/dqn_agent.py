"""
DQN Agent for Trading

Combines the DQN model, replay buffer, and training logic.
"""

from venv import logger
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from pathlib import Path

# Handle imports for both module and standalone execution
try:
    from .dqn_model import DQN, DuelingDQN
    from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
except ImportError:
    from dqn_model import DQN, DuelingDQN
    from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent:
    """
    Deep Q-Learning Agent for trading decisions.

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
        state_dim: int,
        action_dim: int = 3,
        hidden_dims: list = [128, 64],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        use_dueling: bool = False,
        use_prioritized_replay: bool = False,
        device: str = "cpu",
    ):
        """
        Initialize DQN agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions (default: 3 for Buy/Sell/Hold)
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
            buffer_capacity: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: Frequency of target network updates (episodes)
            use_dueling: Whether to use Dueling DQN architecture
            use_prioritized_replay: Whether to use prioritized experience replay
            device: Device to use ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # Create policy and target networks
        if use_dueling:
            self.policy_net = DuelingDQN(state_dim, action_dim, hidden_dims[0]).to(
                self.device
            )
            self.target_net = DuelingDQN(state_dim, action_dim, hidden_dims[0]).to(
                self.device
            )
        else:
            self.policy_net = DQN(state_dim, action_dim, hidden_dims).to(self.device)
            self.target_net = DQN(state_dim, action_dim, hidden_dims).to(self.device)

        # Copy policy net weights to target net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net is not trained directly

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.SmoothL1Loss(
            reduction="none"
        )  # 3/11/25 prevents the loss from exploding.

        # Replay buffer
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)
        else:
            self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self.use_prioritized_replay = use_prioritized_replay

        # Training statistics
        self.training_step = 0
        self.episode_count = 0
        self.losses = []

    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state (numpy array)
            training: Whether in training mode (uses epsilon) or eval mode (greedy)

        Returns:
            action: Selected action (0: Buy, 1: Sell, 2: Hold)
        """
        epsilon = self.epsilon if training else 0.0

        if np.random.random() < epsilon:
            # Explore: random action
            return np.random.randint(self.action_dim)
        else:
            # Exploit: choose best action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax(dim=1).item()
            return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        Perform one training step.

        Returns:
            loss: Training loss value (None if buffer not ready)
        """
        # Check if buffer has enough samples
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        if self.use_prioritized_replay:
            states, actions, rewards, next_states, dones, indices, weights = (
                self.replay_buffer.sample(self.batch_size, beta=0.4)
            )
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                self.batch_size
            )
            weights = torch.ones(self.batch_size).to(self.device)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute current Q values
        current_q_values = (
            self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        # Compute next Q values using target network (Double DQN)
        with torch.no_grad():
            # Select best actions using policy network
            next_actions = self.policy_net(next_states).argmax(dim=1)
            # Evaluate actions using target network
            next_q_values = (
                self.target_net(next_states)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze(1)
            )
            # Compute target Q values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        # loss = (weights * (current_q_values - target_q_values).pow(2)).mean()
        element_wise_loss = self.loss_fn(current_q_values, target_q_values)
        loss = (weights * element_wise_loss).mean()  # 3/11/25 use SmoothL1Loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), max_norm=0.5
        )  # (3/11/25 reduced from 1.0)
        self.optimizer.step()

        # Update priorities if using prioritized replay
        if self.use_prioritized_replay:
            td_errors = (
                (current_q_values - target_q_values).abs().detach().cpu().numpy()
            )
            self.replay_buffer.update_priorities(indices, td_errors + 1e-6)

        self.training_step += 1
        self.losses.append(loss.item())

        # Update target network every 'target_update_freq' steps
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
            logger.info(
                f"Target network updated at step {self.training_step}"
            )  # Optional: for logging

        return loss.item()

    def update_target_network(self):
        """Copy weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episode_count += 1

        # # Update target network periodically
        # if self.episode_count % self.target_update_freq == 0:
        #     self.update_target_network()

    def save(self, filepath):
        """
        Save agent state.

        Args:
            filepath: Path to save file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "episode_count": self.episode_count,
                "training_step": self.training_step,
                "losses": self.losses,
            },
            filepath,
        )

        print(f"✓ Agent saved to {filepath}")

    def load(self, filepath):
        """
        Load agent state.

        Args:
            filepath: Path to load file
        """
        if not os.path.exists(filepath):
            print(f"✗ No saved agent found at {filepath}")
            return False

        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.episode_count = checkpoint["episode_count"]
        self.training_step = checkpoint["training_step"]
        self.losses = checkpoint["losses"]

        print(f"✓ Agent loaded from {filepath}")
        print(f"  Episode: {self.episode_count}, Epsilon: {self.epsilon:.4f}")
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
    # Test DQN agent
    print("Testing DQN Agent...")

    # Create agent
    state_dim = 7  # price, volume, 3 indicators, wallet ratio
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=3,
        hidden_dims=[128, 64],
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=32,
    )

    print(f"✓ Agent created with {state_dim} state features")
    print(f"✓ Epsilon: {agent.epsilon}")

    # Test action selection
    state = np.random.randn(state_dim)
    action = agent.select_action(state, training=True)
    print(f"\n✓ Selected action: {action} ({agent.get_action_name(action)})")

    # Simulate some transitions
    print("\n✓ Simulating 100 transitions...")
    for i in range(100):
        state = np.random.randn(state_dim)
        action = agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = i % 20 == 19

        agent.store_transition(state, action, reward, next_state, done)

    print(f"✓ Replay buffer size: {len(agent.replay_buffer)}")

    # Train for a few steps
    print("\n✓ Training for 10 steps...")
    losses = []
    for i in range(10):
        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)

    print(f"✓ Average loss: {np.mean(losses):.6f}")

    # Test save/load
    save_path = "/tmp/dqn_agent_test.pth"
    agent.save(save_path)

    new_agent = DQNAgent(state_dim=state_dim)
    new_agent.load(save_path)

    print("\n✅ All tests passed!")
