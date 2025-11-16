#!/usr/bin/env python3
"""
DQN Training Script for Trading Strategy

This script trains a DQN agent by running multiple backtesting episodes
through the ATS API. Each episode is a complete backtest run, and the
agent learns from the collected experiences.

Usage:
    python train_dqn.py --episodes 100 --save_dir models/
"""

import sys
import os
import argparse
import json
import time
import requests
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.DQN.dqn_agent import DQNAgent


class DQNTrainer:
    """
    Trainer for DQN agent using ATS backtesting API.

    Training Loop:
    1. Create trading job via API
    2. Run episode (backtest)
    3. Collect rewards/states from strategy
    4. Train DQN agent
    5. Update epsilon
    6. Repeat
    """

    def __init__(
        self,
        api_base_url="http://localhost:5010",
        data_source="ats/data/BTC_USDT_short.csv",
        save_dir="models/dqn",
        config_template=None,
        monitor=None,
        learning_rate=0.0001,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_decay=0.995,
    ):
        """
        Initialize trainer.

        Args:
            api_base_url: Base URL for ATS API
            data_source: Path to CSV data file
            save_dir: Directory to save models
            config_template: Job configuration template (optional)
            monitor: TrainingMonitor instance for real-time dashboard (optional)
            learning_rate: DQN learning rate (default: 0.0001)
            batch_size: Training batch size (default: 64)
            epsilon_start: Initial epsilon for exploration (default: 1.0)
            epsilon_decay: Epsilon decay factor (default: 0.995)
        """
        self.api_base_url = api_base_url
        self.data_source = data_source
        self.save_dir = Path(save_dir)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor  # Store monitor reference

        # State configuration - UPDATED for Tier 1 features
        # State = [price, volume, MA, RSI, wallet_ratio, position_value, cash,
        #          volume_imbalance, return_1, return_5, return_20, buy_ratio_recent, unrealized_pnl]
        self.state_dim = 13  # Updated from 7 to 13 for Tier 1 features

        # Create DQN agent with provided hyperparameters
        self.agent = DQNAgent(
            state_dim=self.state_dim,
            action_dim=3,  # Buy, Sell, Hold
            hidden_dims=[128, 64],
            learning_rate=self.learning_rate,
            gamma=0.99,
            epsilon_start=self.epsilon_start,
            epsilon_end=0.05,
            epsilon_decay=self.epsilon_decay,
            buffer_capacity=100000,
            batch_size=self.batch_size,
            target_update_freq=100,  # (3/11/25 increase to 100 after target_update_freq to measure in steps)
            use_dueling=False,
            use_prioritized_replay=False,
        )

        # Job configuration template
        if config_template is None:
            self.config_template = self._get_default_config()
        else:
            self.config_template = config_template

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.recent_losses = deque(maxlen=100)  # For smoothed loss plot

        # Fix: Properly construct transitions directory path
        # Create a more reliable path for transitions
        try:
            # Try to get transitions directory relative to the current file
            current_file = Path(__file__).resolve()
            self.transitions_dir = current_file.parent / "transitions"

            # Ensure directory exists
            self.transitions_dir.mkdir(parents=True, exist_ok=True)

            # Debug: Print the path we're using
            print(f"Using transitions directory: {self.transitions_dir}")

        except Exception as e:
            # Fallback to a known-safe location
            import tempfile

            self.transitions_dir = Path(tempfile.gettempdir()) / "ats_transitions"
            self.transitions_dir.mkdir(exist_ok=True)
            print(f"Fallback transitions directory: {self.transitions_dir}")

        # Real-time monitoring
        self.loss_log_file = self.save_dir / "training_loss.csv"
        self.metrics_log_file = self.save_dir / "training_metrics.csv"

        # Initialize log files
        self._init_log_files()

    def _get_total_steps(self):
        """Get total number of steps from the dataset (without pandas)."""
        with open(self.data_source, "r") as f:
            # Subtract 1 for header row
            return sum(1 for line in f) - 1

    def _get_default_config(self):
        """Get default job configuration."""
        return {
            "exchange": {
                "namespace": "exchanges:back_trading",
                "config": {
                    "symbol": {"base": "BTC", "quote": "USDT"},
                    "plot_data_max_len": -1,
                    "extra": {
                        "data_source": self.data_source,
                        "wallet": {"assets": {"USDT": 10000, "BTC": 0}},
                        "min_trading_size": 0.0001,
                        "fees": {
                            "namespace": "fees:generic",
                            "config": {
                                "limit": {
                                    "buy": {"base": 0.001, "quote": 0},
                                    "sell": {"base": 0, "quote": 0.001},
                                },
                                "market": {
                                    "buy": {"base": 0.001, "quote": 0},
                                    "sell": {"base": 0, "quote": 0.001},
                                },
                            },
                        },
                    },
                },
            },
            "strategy": {
                "namespace": "strategies:DQN",
                "config": {
                    "base_symbol": "BTC",
                    "quote_symbol": "USDT",
                    "order_value_pct": 10,
                    "max_position_size": 0.8,
                    "log_trade_activity": False,  # Disable logging during training
                    "use_indicators": True,
                    "ma_window": 20,
                    "rsi_window": 14,
                    "training_mode": True,  # Enable training mode
                    "epsilon": 1.0,  # Will be updated each episode,
                    "total_steps": self._get_total_steps(),
                },
            },
        }

    def _init_log_files(self):
        """Initialize CSV log files for real-time monitoring."""
        # Loss log
        with open(self.loss_log_file, "w") as f:
            f.write("episode,step,loss,avg_loss_100\n")

        # Metrics log
        with open(self.metrics_log_file, "w") as f:
            f.write("episode,reward,return_pct,length,epsilon,buffer_size,num_trades\n")

    def create_job(self, epsilon, episode_num):
        """
        Create a new trading job via API.

        Args:
            epsilon: Current exploration rate
            episode_num: Episode number for tracking

        Returns:
            tuple: (job_id, episode_id)
        """
        # Generate unique episode ID
        episode_id = f"ep{episode_num:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Update epsilon and episode_id in config
        config = self.config_template.copy()
        config["strategy"]["config"]["epsilon"] = epsilon
        config["strategy"]["config"]["episode_id"] = episode_id

        config["strategy"]["config"]["transitions_dir"] = str(
            self.transitions_dir.absolute()
        )

        # Create job
        response = requests.post(
            f"{self.api_base_url}/trading_job/create",
            json=config,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            raise Exception(f"Failed to create job: {response.text}")

        job_data = response.json()
        return job_data["job_id"], episode_id

    def run_job(self, job_id):
        """
        Run trading job (episode).

        Args:
            job_id: Job ID to run
        """
        response = requests.get(f"{self.api_base_url}/trading_job/run/{job_id}")

        if response.status_code != 200:
            raise Exception(f"Failed to run job: {response.text}")

    def stop_job(self, job_id):
        """Stop a running trading job to ensure on_stop() is called."""
        try:
            response = requests.get(
                f"{self.api_base_url}/trading_job/stop/{job_id}", timeout=5
            )
            if response.status_code == 200:
                print(f"✓ Job {job_id} stopped")
            else:
                print(f"⚠️  Failed to stop job {job_id}")
        except Exception as e:
            print(f"⚠️  Error stopping job {job_id}: {e}")

    def wait_for_completion(self, job_id, timeout=300):
        """
        Wait for job to complete.

        Args:
            job_id: Job ID
            timeout: Maximum wait time in seconds

        Returns:
            bool: True if completed successfully
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            response = requests.get(f"{self.api_base_url}/trading_job/status/{job_id}")

            if response.status_code != 200:
                print(f"Warning: Failed to get status: {response.text}")
                time.sleep(1)
                continue

            status = response.json()

            if not status.get("is_running", True):
                return True

            time.sleep(1)

        print(f"Warning: Job {job_id} timed out after {timeout}s")
        return False

    def get_episode_data(self, episode_id):
        """
        Get episode data from transition file.

        Args:
            episode_id: Episode identifier

        Returns:
            dict: Episode data with transitions and metrics
        """
        transition_file = self.transitions_dir / f"episode_{episode_id}.json"

        # Wait a bit for file to be written
        max_wait = 30  # seconds (increased from 10 for robustness)
        wait_time = 0
        while not transition_file.exists() and wait_time < max_wait:
            time.sleep(0.5)
            wait_time += 0.5
            if wait_time % 5 == 0:  # Log every 5 seconds
                print(f"Waiting for transition file... ({wait_time}s elapsed)")

        if not transition_file.exists():
            print(f"✗ Transition file not found: {transition_file}")
            return None

        try:
            # Read transition file
            with open(transition_file, "r") as f:
                episode_data = json.load(f)

            # Convert transitions back to tuples with numpy arrays
            transitions = []
            for t in episode_data["transitions"]:
                transitions.append(
                    (
                        np.array(t["state"], dtype=np.float32),
                        t["action"],
                        t["reward"],
                        np.array(t["next_state"], dtype=np.float32),
                        t["done"],
                    )
                )

            result = {
                "total_reward": episode_data["metrics"]["total_reward"],
                "total_return": episode_data["metrics"]["total_return"],
                "episode_length": episode_data["metrics"]["num_steps"],
                "num_trades": episode_data["metrics"]["num_trades"],
                "transitions": transitions,
            }

            # Clean up transition file after reading
            try:
                transition_file.unlink()
            except:
                pass

            return result

        except Exception as e:
            print(f"✗ Error reading transition file: {e}")
            import traceback

            traceback.print_exc()
            return None

    def train_episode(self, episode_num, total_episodes):
        """
        Train for one episode.

        Args:
            episode_num: Current episode number
            total_episodes: Total number of episodes

        Returns:
            dict: Episode statistics
        """
        print(f"\n=== Episode {episode_num} (ε={self.agent.epsilon:.4f}) ===")

        # Notify monitor
        if self.monitor:
            self.monitor.episode_start(episode_num, total_episodes)
            self.monitor.log(
                f"Episode {episode_num} started (epsilon={self.agent.epsilon:.4f})"
            )

        # Create and run job
        try:
            job_id, episode_id = self.create_job(self.agent.epsilon, episode_num)
            print(f"✓ Created job: {job_id} (episode_id: {episode_id})")

            if self.monitor:
                self.monitor.log(f"Job created: {job_id}")

            self.run_job(job_id)
            print(f"✓ Running episode...")

            # Wait for completion
            if not self.wait_for_completion(job_id):
                print("✗ Episode failed or timed out")
                self.stop_job(job_id)  # Ensure cleanup even on failure
                return None

            print(f"✓ Episode completed")
            # Explicitly stop the job to trigger on_stop() → export transitions
            self.stop_job(job_id)

            # Get episode data from transition file
            episode_data = self.get_episode_data(episode_id)

            if episode_data is None:
                print("✗ Failed to load episode data")
                return None

            # Store transitions in replay buffer
            num_transitions = len(episode_data.get("transitions", []))
            for transition in episode_data.get("transitions", []):
                self.agent.store_transition(*transition)

            print(f"✓ Loaded {num_transitions} transitions")

            # Train agent (multiple gradient steps)
            if len(self.agent.replay_buffer) >= self.agent.batch_size:
                losses = []
                train_steps = max(episode_data["episode_length"] // 100, 10) #3/11/25 reduce the number of training steps from 10

                for step in range(train_steps):
                    loss = self.agent.train_step()
                    if loss is not None:
                        losses.append(loss)
                        self.recent_losses.append(loss)

                        # Log loss in real-time
                        avg_loss_100 = np.mean(self.recent_losses)
                        self._log_loss(episode_num, step, loss, avg_loss_100)

                        # Send to monitor (throttled: only every 10 steps)
                        if self.monitor and step % 10 == 0:
                            total_steps = (episode_num - 1) * train_steps + step
                            self.monitor.training_step(total_steps, train_steps, loss)

                avg_loss = np.mean(losses) if losses else 0.0
                print(f"✓ Training: {len(losses)} steps, avg loss = {avg_loss:.6f}")
            else:
                avg_loss = 0.0
                print(
                    f"✓ Buffer filling... ({len(self.agent.replay_buffer)}/{self.agent.batch_size})"
                )

            # Decay epsilon
            self.agent.decay_epsilon()

            # Notify monitor about epsilon update
            if self.monitor:
                self.monitor.epsilon_update(self.agent.epsilon)

            # Store statistics
            stats = {
                "episode": episode_num,
                "reward": episode_data["total_reward"],
                "return_pct": episode_data["total_return"],
                "length": episode_data["episode_length"],
                "num_trades": episode_data["num_trades"],
                "loss": avg_loss,
                "epsilon": self.agent.epsilon,
                "buffer_size": len(self.agent.replay_buffer),
            }

            self.episode_rewards.append(episode_data["total_reward"])
            self.episode_lengths.append(episode_data["episode_length"])
            if avg_loss > 0:
                self.losses.append(avg_loss)

            # Log metrics
            self._log_metrics(stats)

            print(
                f"✓ Reward: {episode_data['total_reward']:.2f}, Return: {episode_data['total_return']:.2f}%, Trades: {episode_data['num_trades']}"
            )

            # Notify monitor about episode completion
            if self.monitor:
                self.monitor.episode_complete(
                    episode_num,
                    total_episodes,
                    {
                        "total_reward": episode_data["total_reward"],
                        "return_pct": episode_data["total_return"],
                        "num_transitions": num_transitions,
                        "epsilon": self.agent.epsilon,
                    },
                )
                self.monitor.log(
                    f"Episode {episode_num} complete: Reward={episode_data['total_reward']:.2f}, Return={episode_data['total_return']:.2f}%",
                    "success",
                )

            return stats

        except Exception as e:
            print(f"✗ Error in episode {episode_num}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _log_loss(self, episode, step, loss, avg_loss_100):
        """Log training loss in real-time."""
        with open(self.loss_log_file, "a") as f:
            f.write(f"{episode},{step},{loss:.6f},{avg_loss_100:.6f}\n")

    def _log_metrics(self, stats):
        """Log episode metrics."""
        with open(self.metrics_log_file, "a") as f:
            f.write(
                f"{stats['episode']},{stats['reward']:.2f},{stats['return_pct']:.2f},"
                f"{stats['length']},{stats['epsilon']:.4f},{stats['buffer_size']},"
                f"{stats['num_trades']}\n"
            )

    def train(self, num_episodes=100, save_freq=10):
        """
        Main training loop.

        Args:
            num_episodes: Number of episodes to train
            save_freq: Frequency of model saving (episodes)
        """
        print(f"\n{'='*60}")
        print(f"DQN Training Started")
        print(f"{'='*60}")
        print(f"Episodes: {num_episodes}")
        print(f"State dim: {self.state_dim}")
        print(f"Action dim: {self.agent.action_dim}")
        print(f"Save dir: {self.save_dir}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for episode in range(1, num_episodes + 1):
            stats = self.train_episode(episode, num_episodes)

            # Save model periodically
            if episode % save_freq == 0:
                self.save_checkpoint(episode)

            # Print progress
            if episode % 10 == 0:
                self.print_progress(episode, start_time)

        # Final save
        self.save_checkpoint(num_episodes, final=True)

        print(f"\n{'='*60}")
        print(f"Training Completed!")
        print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")
        print(f"{'='*60}\n")

    def save_checkpoint(self, episode, final=False):
        """Save model checkpoint."""
        if final:
            filename = "dqn_final.pth"
        else:
            filename = f"dqn_episode_{episode}.pth"

        filepath = self.save_dir / filename
        self.agent.save(filepath)

        # Save training statistics
        stats_file = self.save_dir / f"training_stats_ep{episode}.json"
        with open(stats_file, "w") as f:
            json.dump(
                {
                    "episode": episode,
                    "episode_rewards": self.episode_rewards,
                    "episode_lengths": self.episode_lengths,
                    "losses": self.losses,
                    "epsilon": self.agent.epsilon,
                },
                f,
                indent=2,
            )

        print(f"✓ Checkpoint saved: {filepath}")

        # Notify monitor
        if self.monitor:
            self.monitor.checkpoint_saved(filename)

    def plot_training_progress(self, episode):
        """Generate and save training progress plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f"Training Progress - Episode {episode}", fontsize=16)

            # Plot 1: Episode Rewards
            if len(self.episode_rewards) > 0:
                axes[0, 0].plot(self.episode_rewards, alpha=0.6, label="Episode Reward")
                if len(self.episode_rewards) >= 10:
                    # Moving average
                    window = min(10, len(self.episode_rewards))
                    moving_avg = np.convolve(
                        self.episode_rewards, np.ones(window) / window, mode="valid"
                    )
                    axes[0, 0].plot(
                        range(window - 1, len(self.episode_rewards)),
                        moving_avg,
                        "r-",
                        linewidth=2,
                        label=f"MA({window})",
                    )
                axes[0, 0].set_xlabel("Episode")
                axes[0, 0].set_ylabel("Reward")
                axes[0, 0].set_title("Episode Rewards")
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Training Loss
            if len(self.losses) > 0:
                axes[0, 1].plot(self.losses, alpha=0.6, label="Loss")
                if len(self.losses) >= 10:
                    window = min(10, len(self.losses))
                    moving_avg = np.convolve(
                        self.losses, np.ones(window) / window, mode="valid"
                    )
                    axes[0, 1].plot(
                        range(window - 1, len(self.losses)),
                        moving_avg,
                        "r-",
                        linewidth=2,
                        label=f"MA({window})",
                    )
                axes[0, 1].set_xlabel("Training Step")
                axes[0, 1].set_ylabel("Loss")
                axes[0, 1].set_title("Training Loss")
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Episode Lengths
            if len(self.episode_lengths) > 0:
                axes[1, 0].bar(
                    range(len(self.episode_lengths)), self.episode_lengths, alpha=0.6
                )
                axes[1, 0].set_xlabel("Episode")
                axes[1, 0].set_ylabel("Steps")
                axes[1, 0].set_title("Episode Lengths")
                axes[1, 0].grid(True, alpha=0.3, axis="y")

            # Plot 4: Epsilon Decay
            axes[1, 1].plot(
                range(episode + 1),
                [
                    self.agent.epsilon_start * (self.agent.epsilon_decay**i)
                    for i in range(episode + 1)
                ],
                "g-",
                linewidth=2,
            )
            axes[1, 1].axhline(
                y=self.agent.epsilon,
                color="r",
                linestyle="--",
                label=f"Current ε={self.agent.epsilon:.4f}",
            )
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Epsilon")
            axes[1, 1].set_title("Exploration Rate (ε)")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = self.save_dir / f"training_progress_ep{episode}.png"
            plt.savefig(plot_file, dpi=100, bbox_inches="tight")
            plt.close()

            print(f"✓ Saved training plot: {plot_file}")

        except Exception as e:
            print(f"Warning: Could not generate plot: {e}")

    def print_progress(self, episode, start_time):
        """Print training progress."""
        elapsed = (time.time() - start_time) / 60

        if len(self.episode_rewards) >= 10:
            recent_rewards = self.episode_rewards[-10:]
            avg_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
            max_reward = max(recent_rewards)
            min_reward = min(recent_rewards)
        else:
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            std_reward = 0
            max_reward = max(self.episode_rewards) if self.episode_rewards else 0
            min_reward = min(self.episode_rewards) if self.episode_rewards else 0

        if len(self.losses) > 0:
            avg_loss = np.mean(self.losses[-100:])
            recent_loss = self.losses[-1] if self.losses else 0
        else:
            avg_loss = 0
            recent_loss = 0

        print(f"\n{'='*70}")
        print(f"📊 Progress Report - Episode {episode}")
        print(f"{'='*70}")
        print(f"⏱️  Time Elapsed: {elapsed:.2f} min ({elapsed/episode:.2f} min/episode)")
        print(f"🎯 Rewards (last 10 episodes):")
        print(f"   Average: {avg_reward:+.2f} ± {std_reward:.2f}")
        print(f"   Range: [{min_reward:+.2f}, {max_reward:+.2f}]")
        print(f"📉 Training Loss:")
        print(f"   Recent: {recent_loss:.6f}")
        print(f"   Average (last 100): {avg_loss:.6f}")
        print(f"🔍 Exploration:")
        print(
            f"   Epsilon: {self.agent.epsilon:.4f} ({self.agent.epsilon*100:.1f}% random)"
        )
        print(f"💾 Memory:")
        print(
            f"   Buffer Size: {len(self.agent.replay_buffer):,} / {self.agent.replay_buffer.capacity:,}"
        )
        print(
            f"   Buffer Fill: {len(self.agent.replay_buffer)/self.agent.replay_buffer.capacity*100:.1f}%"
        )
        print(f"{'='*70}\n")

        # Generate progress plot
        self.plot_training_progress(episode)


def main():
    parser = argparse.ArgumentParser(description="Train DQN agent for trading")
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of training episodes"
    )
    parser.add_argument(
        "--save_dir", type=str, default="models/dqn", help="Directory to save models"
    )
    parser.add_argument(
        "--save_freq", type=int, default=10, help="Save frequency (episodes)"
    )
    parser.add_argument(
        "--api_url", type=str, default="http://localhost:5010", help="API base URL"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="ats/data/BTC_USDT_short.csv",
        help="Path to data CSV file",
    )

    args = parser.parse_args()

    # Create trainer
    trainer = DQNTrainer(
        api_base_url=args.api_url, data_source=args.data, save_dir=args.save_dir
    )

    # Train
    trainer.train(num_episodes=args.episodes, save_freq=args.save_freq)


if __name__ == "__main__":
    main()
