"""
Experience Replay Buffer for DQN

Stores and samples transitions (state, action, reward, next_state, done)
for training the DQN agent.
"""

import numpy as np
import random
from collections import deque, namedtuple


# Transition tuple
Transition = namedtuple('Transition', 
                       ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN training.
    
    Stores transitions and provides random sampling for breaking correlation
    between consecutive samples.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state (numpy array)
            action: Action taken (int)
            reward: Reward received (float)
            next_state: Next state (numpy array)
            done: Whether episode ended (bool)
        """
        transition = Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size: int):
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        transitions = random.sample(self.buffer, batch_size)
        
        # Convert to numpy arrays
        states = np.array([t.state for t in transitions], dtype=np.float32)
        actions = np.array([t.action for t in transitions], dtype=np.int64)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        next_states = np.array([t.next_state for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.bool_)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
    
    def is_ready(self, min_size: int):
        """Check if buffer has enough samples."""
        return len(self.buffer) >= min_size


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer (Advanced).
    
    Samples transitions based on their TD error, giving priority to
    transitions that are more "surprising" (higher learning potential).
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        """Add transition with maximum priority."""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = Transition(state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """
        Sample batch with prioritization.
        
        Args:
            batch_size: Number of samples
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        if self.size < batch_size:
            batch_size = self.size
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get transitions
        transitions = [self.buffer[idx] for idx in indices]
        
        states = np.array([t.state for t in transitions], dtype=np.float32)
        actions = np.array([t.action for t in transitions], dtype=np.int64)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        next_states = np.array([t.next_state for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.bool_)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities based on TD errors."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return self.size


if __name__ == "__main__":
    # Test replay buffer
    print("Testing Replay Buffer...")
    
    buffer = ReplayBuffer(capacity=1000)
    
    # Add some transitions
    for i in range(100):
        state = np.random.randn(7)
        action = np.random.randint(3)
        reward = np.random.randn()
        next_state = np.random.randn(7)
        done = (i % 20 == 19)  # Episode ends every 20 steps
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"✓ Added 100 transitions to buffer")
    print(f"✓ Buffer size: {len(buffer)}")
    
    # Sample a batch
    batch_size = 32
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    
    print(f"\n✓ Sampled batch of {batch_size} transitions:")
    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Rewards shape: {rewards.shape}")
    print(f"  Next states shape: {next_states.shape}")
    print(f"  Dones shape: {dones.shape}")
    print(f"  Sample actions: {actions[:5]}")
    print(f"  Sample rewards: {rewards[:5]}")
    print(f"  Sample dones: {dones[:5]}")
    
    # Test prioritized buffer
    print("\n\nTesting Prioritized Replay Buffer...")
    
    pri_buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.6)
    
    for i in range(100):
        state = np.random.randn(7)
        action = np.random.randint(3)
        reward = np.random.randn()
        next_state = np.random.randn(7)
        done = (i % 20 == 19)
        
        pri_buffer.push(state, action, reward, next_state, done)
    
    print(f"✓ Added 100 transitions to prioritized buffer")
    print(f"✓ Buffer size: {len(pri_buffer)}")
    
    # Sample with priorities
    states, actions, rewards, next_states, dones, indices, weights = pri_buffer.sample(32, beta=0.4)
    
    print(f"\n✓ Sampled prioritized batch:")
    print(f"  Importance weights: {weights[:5]}")
    print(f"  Sampled indices: {indices[:5]}")
    
    # Update priorities
    new_priorities = np.random.rand(len(indices)) + 0.1
    pri_buffer.update_priorities(indices, new_priorities)
    print(f"✓ Updated priorities for sampled transitions")
    
    print("\n✅ All tests passed!")

