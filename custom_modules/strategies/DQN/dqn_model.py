"""
Deep Q-Network (DQN) Model for Trading

This module contains the neural network architecture for the DQN agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQN(nn.Module):
    """
    Deep Q-Network for trading decisions.

    Architecture:
    - Input layer: state features
    - Hidden layers: 2 fully connected layers with ReLU activation
    - Output layer: Q-values for each action (Buy, Sell, Hold)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 64]):
        """
        Initialize the DQN network.

        Args:
            state_dim: Dimension of state space (number of features)
            action_dim: Dimension of action space (number of actions: 3)
            hidden_dims: List of hidden layer dimensions
        """
        super(DQN, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build network layers
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.2))  # Prevent overfitting # 3/11/25 Do not use dropout in dqn
            """
            Dropout is a technique for supervised learning where it randomly 
            "shuts off" 20% of the network to prevent memorization. In a DQN, 
            this is disastrous. It adds random noise to your agent's "brain" 
            while it's trying to make precise calculations. The agent needs a 
            stable, deterministic network to learn, and Dropout makes that 
            impossible.
            """
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state: Input state tensor [batch_size, state_dim] or [state_dim]

        Returns:
            Q-values for each action [batch_size, action_dim] or [action_dim]
        """
        return self.network(state)

    def get_action(self, state, epsilon=0.0):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state (numpy array)
            epsilon: Exploration rate (0.0 = greedy, 1.0 = random)

        Returns:
            action: Selected action index (0: Buy, 1: Sell, 2: Hold)
        """
        if np.random.random() < epsilon:
            # Explore: random action
            return np.random.randint(self.action_dim)
        else:
            # Exploit: choose best action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.forward(state_tensor)
                action = q_values.argmax(dim=1).item()
            return action


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture (optional advanced version).

    Separates state value and advantage functions:
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))

    This helps learning by separating state value from action advantage.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DuelingDQN, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state):
        """Forward pass through dueling architecture."""
        features = self.feature_layer(state)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def get_action(self, state, epsilon=0.0):
        """Select action using epsilon-greedy policy."""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.forward(state_tensor)
                action = q_values.argmax(dim=1).item()
            return action


if __name__ == "__main__":
    # Test the DQN model
    print("Testing DQN Model...")

    # Example: 7 state features (price, volume, 3 indicators, wallet ratio)
    state_dim = 7
    action_dim = 3  # Buy, Sell, Hold

    # Create model
    model = DQN(state_dim, action_dim)
    print(f"✓ DQN created with {state_dim} state features and {action_dim} actions")
    print(f"✓ Model architecture:\n{model}")

    # Test forward pass
    sample_state = torch.randn(1, state_dim)
    q_values = model(sample_state)
    print(f"\n✓ Forward pass test:")
    print(f"  Input shape: {sample_state.shape}")
    print(f"  Output shape: {q_values.shape}")
    print(f"  Q-values: {q_values}")

    # Test action selection
    state_np = np.random.randn(state_dim)
    action_greedy = model.get_action(state_np, epsilon=0.0)
    action_random = model.get_action(state_np, epsilon=1.0)
    print(f"\n✓ Action selection test:")
    print(f"  Greedy action: {action_greedy}")
    print(f"  Random action: {action_random}")

    # Test Dueling DQN
    print("\n\nTesting Dueling DQN Model...")
    dueling_model = DuelingDQN(state_dim, action_dim)
    print(f"✓ Dueling DQN created")
    print(f"✓ Model architecture:\n{dueling_model}")

    q_values_dueling = dueling_model(sample_state)
    print(f"\n✓ Forward pass test:")
    print(f"  Output shape: {q_values_dueling.shape}")
    print(f"  Q-values: {q_values_dueling}")

    print("\n✅ All tests passed!")
