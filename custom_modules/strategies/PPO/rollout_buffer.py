"""
On-policy Rollout Buffer for PPO

Collects short-horizon trajectories, computes GAE and returns, and is cleared
after each policy update. This is not experience replay.
"""

import numpy as np
import torch

class RolloutBuffer:
    """
    On-policy rollout storage for PPO.

    Collects a short horizon of trajectories, computes GAE/returns, and is
    cleared after each policy update.
    """
    def __init__(self, state_dim, size, gamma=0.99, lam=0.95):
        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.gamma, self.lam = gamma, lam

    def store(self, state, action, reward, done, log_prob, value):
        assert self.ptr < self.max_size
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def finish_path(self, last_value=0.0):
        """
        Finish a trajectory segment and compute GAE advantages and returns.

        We use bootstrapped value for the state following the last one in the
        path. Rewards are NOT appended with last_value; only values are.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = self.rewards[path_slice]
        dones = self.dones[path_slice]
        values = np.append(self.values[path_slice], float(last_value))

        last_gae_lambda = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            self.advantages[self.path_start_idx + t] = last_gae_lambda = delta + \
                self.gamma * self.lam * next_non_terminal * last_gae_lambda

        self.returns[path_slice] = self.advantages[path_slice] + values[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        # Return only the filled portion, normalize advantages on that slice
        end = self.ptr
        adv_slice = self.advantages[:end]
        adv_mean = np.mean(adv_slice)
        adv_std = np.std(adv_slice) + 1e-8
        adv_norm = (adv_slice - adv_mean) / adv_std
        return [
            torch.FloatTensor(self.states[:end]),
            torch.LongTensor(self.actions[:end]),
            torch.FloatTensor(self.returns[:end]),
            torch.FloatTensor(adv_norm),
            torch.FloatTensor(self.log_probs[:end])
        ]

    def clear(self):
        """
        Clear the rollout buffer for the next on-policy collection.

        Resets pointers and zeroes internal storages to avoid leaking stale
        values. Pre-allocated arrays are reused to minimize allocations.
        """
        self.ptr = 0
        self.path_start_idx = 0
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.dones.fill(0)
        self.log_probs.fill(0)
        self.values.fill(0)
        self.returns.fill(0)
        self.advantages.fill(0)

if __name__ == "__main__":
    # Test rollout buffer
    print("Testing Rollout Buffer...")

    buffer = RolloutBuffer(state_dim=7, size=100, gamma=0.99, lam=0.95)

    # Add some transitions
    for t in range(100):
        buffer.store(
            state=np.random.rand(7), 
            action=np.random.randint(3), 
            reward=1.0, 
            done=0.0, 
            log_prob=-0.5, 
            value=0.5
        )
    buffer.finish_path(last_value=0.0)

    states, actions, returns, advantages, log_probs = buffer.get()
    print("States shape:", states.shape)
    print("Actions shape:", actions.shape)
    print("Returns shape:", returns.shape)
    print("Advantages shape:", advantages.shape)
    print("Log Probs shape:", log_probs.shape)

    # Check if all shapes are as expected
    assert states.shape == (100, 7), "Unexpected states shape"
    assert actions.shape == (100,), "Unexpected actions shape"
    assert returns.shape == (100,), "Unexpected returns shape"
    assert advantages.shape == (100,), "Unexpected advantages shape"
    assert log_probs.shape == (100,), "Unexpected log_probs shape"

    # Clear the buffer and check pointers
    buffer.clear()
    assert buffer.ptr == 0, "Buffer pointer not reset"
    assert buffer.path_start_idx == 0, "Path start index not reset"

    print("✓ RolloutBuffer test completed successfully.")
