#!/usr/bin/env python3
"""
PPO Training Script for Trading Strategy

This script trains a PPO agent by running multiple backtesting iterations
through the ATS API. Each iteration is a complete backtest run, and the
agent learns from the collected experiences.

Usage:
    python train_ppo.py --iterations 100 --save_dir models/
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
import torch
import matplotlib
# Force a headless backend to avoid Qt plugin errors on Linux/CI or headless servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from custom_modules.strategies.PPO.ppo_agent import PPOAgent


class PPOTrainer:
    """
    Trainer for PPO agent using ATS backtesting API.
    
    Training Loop:
    1. Create trading job via API
    2. Run iteration (backtest)
    3. Collect rewards/states from strategy
    4. Train PPO agent
    6. Repeat
    """
    
    def __init__(
        self,
        api_base_url="http://localhost:5010",
        data_source="ats/data/BTC_USDT_short.csv",  # Relative to ATS root
        save_dir="models/ppo",
        config_template=None,
        monitor=None,
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
        Initialize trainer.
        
        Args:
            api_base_url: Base URL for ATS API
            data_source: Path to CSV data file
            save_dir: Directory to save models
            config_template: Job configuration template (optional)
            monitor: TrainingMonitor instance for real-time dashboard (optional)
            learning_rate: PPO learning rate (default: 0.0001)
            batch_size: Training batch size (default: 64)
        """
        self.api_base_url = api_base_url
        self.data_source = data_source
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor  # Store monitor reference
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.value_loss_coeff = value_loss_coeff
        self.entropy_coeff = entropy_coeff
        self.device = device
        
        # State configuration
        # State = [price, volume, MA, RSI, wallet_ratio, position_value, cash]
        self.state_dim = 7
        # Create PPO agent with provided hyperparameters
        self.agent = PPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            lam=self.lam,
            clip_ratio=self.clip_ratio,
            target_kl=self.target_kl,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            value_loss_coeff=self.value_loss_coeff,
            entropy_coeff=self.entropy_coeff,
            device=self.device
        )
        
        # Job configuration template
        if config_template is None:
            self.config_template = self._get_default_config()
        else:
            self.config_template = config_template
        
        # Training statistics
        self.iteration_rewards = []
        self.iteration_lengths = []
        self.losses = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_values = []
        self.kl_values = []
        self.recent_actor_losses = deque(maxlen=100)  # Smoothed actor loss
        
        # Paths for transition files
        self.transitions_dir = Path(__file__).parent / "transitions"
        self.transitions_dir.mkdir(exist_ok=True)
        
        # Real-time monitoring
        self.loss_log_file = self.save_dir / "training_loss.csv"
        self.metrics_log_file = self.save_dir / "training_metrics.csv"
        
        # Initialize log files
        self._init_log_files()
    
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
                        "wallet": {
                            "assets": {
                                "USDT": 10000,
                                "BTC": 0
                            }
                        },
                        "min_trading_size": 0.0001,
                        "fees": {
                            "namespace": "fees:generic",
                            "config": {
                                "limit": {
                                    "buy": {"base": 0.001, "quote": 0},
                                    "sell": {"base": 0, "quote": 0.001}
                                },
                                "market": {
                                    "buy": {"base": 0.001, "quote": 0},
                                    "sell": {"base": 0, "quote": 0.001}
                                }
                            }
                        }
                    }
                }
            },
            "strategy": {
                "namespace": "strategies:PPO",
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
                }
            }
        }
    
    def _init_log_files(self):
        """Initialize CSV log files for real-time monitoring."""
        # Loss log
        with open(self.loss_log_file, 'w') as f:
            f.write("iteration,loss,actor_loss,critic_loss,entropy,kl,avg_actor_loss_100\n")
        
        # Metrics log
        with open(self.metrics_log_file, 'w') as f:
            f.write("iteration,reward,return_pct,length,buffer_fill_pct,num_trades\n")
    
    def create_job(self, iteration_num):
        """
        Create a new trading job via API.
        
        Args:
            iteration_num: Iteration number for tracking
        
        Returns:
            tuple: (job_id, iteration_id)
        """
        # Generate unique iteration ID
        iteration_id = f"itr{iteration_num:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Update iteration_id in config
        config = self.config_template.copy()
        config["strategy"]["config"]["iteration_id"] = iteration_id
        
        # Create job
        response = requests.post(
            f"{self.api_base_url}/trading_job/create",
            json=config,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to create job: {response.text}")
        
        job_data = response.json()
        return job_data["job_id"], iteration_id
    
    def run_job(self, job_id):
        """
        Run trading job (iteration).
        
        Args:
            job_id: Job ID to run
        """
        response = requests.get(f"{self.api_base_url}/trading_job/run/{job_id}")
        
        if response.status_code != 200:
            raise Exception(f"Failed to run job: {response.text}")
    
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
    
    def get_iteration_data(self, iteration_id):
        """
        Get iteration data from transition file.
        
        Args:
            iteration_id: Iteration identifier
        
        Returns:
            dict: Iteration data with transitions and metrics
        """
        transition_file = self.transitions_dir / f"iteration_{iteration_id}.json"
        
        # Wait a bit for file to be written
        max_wait = 10  # seconds
        wait_time = 0
        while not transition_file.exists() and wait_time < max_wait:
            time.sleep(0.5)
            wait_time += 0.5
        
        if not transition_file.exists():
            print(f"✗ Transition file not found: {transition_file}")
            return None
        
        try:
            # Read transition file
            with open(transition_file, 'r') as f:
                iteration_data = json.load(f)
            
            # Convert transitions back to tuples with numpy arrays
            transitions = []
            for t in iteration_data['transitions']:
                transitions.append((
                    np.array(t['state'], dtype=np.float32),
                    t['action'],
                    t['reward'],
                    np.array(t['next_state'], dtype=np.float32),
                    t['done']
                ))
            
            result = {
                'total_reward': iteration_data['metrics']['total_reward'],
                'total_return': iteration_data['metrics']['total_return'],
                'iteration_length': iteration_data['metrics']['num_steps'],
                'num_trades': iteration_data['metrics']['num_trades'],
                'transitions': transitions
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
    
    def train_iteration(self, iteration_num, total_iterations):
        """
        Train for one iteration.
        
        Args:
            iteration_num: Current iteration number
            total_iterations: Total number of iterations
        
        Returns:
            dict: Iteration statistics
        """
        print(f"\n=== Iteration {iteration_num} ===")
        
        # Notify monitor
        if self.monitor:
            self.monitor.iteration_start(iteration_num, total_iterations)
            self.monitor.log(f"Iteration {iteration_num} started")
        
        # Create and run job
        try:
            job_id, iteration_id = self.create_job(iteration_num)
            print(f"✓ Created job: {job_id} (iteration_id: {iteration_id})")
            
            if self.monitor:
                self.monitor.log(f"Job created: {job_id}")
            
            self.run_job(job_id)
            print(f"✓ Running iteration...")
            
            # Wait for completion
            if not self.wait_for_completion(job_id):
                print("✗ Iteration failed or timed out")
                return None
            
            print(f"✓ Iteration completed")
            
            # Get iteration data from transition file
            iteration_data = self.get_iteration_data(iteration_id)
            
            if iteration_data is None:
                print("✗ Failed to load iteration data")
                return None
            
            # Clear buffer before storing new iteration's transitions (on-policy learning)
            self.agent.buffer.clear()
            
            # Store transitions in rollout buffer (convert format)
            num_transitions = len(iteration_data.get("transitions", []))
            
            # Safety check: ensure buffer can hold all transitions
            if num_transitions > self.agent.buffer.max_size:
                raise ValueError(
                    f"❌ Buffer overflow! Iteration collected {num_transitions} transitions "
                    f"but buffer size is only {self.agent.buffer.max_size}.\n"
                    f"Solution: Increase buffer size to at least {num_transitions}\n"
                    f"  --buffer_size {int(num_transitions * 1.2)}"
                )
            
            for (state, action, reward, _next_state, done) in iteration_data.get("transitions", []):
                # Derive log_prob & value for given state/action under current policy
                state_tensor = np.array(state, dtype=np.float32)
                probs = self.agent.actor(torch.FloatTensor(state_tensor).unsqueeze(0))
                dist = torch.distributions.Categorical(probs)
                log_prob = dist.log_prob(torch.tensor(action)).item()
                with torch.no_grad():
                    value = self.agent.critic(torch.FloatTensor(state_tensor).unsqueeze(0)).item()
                self.agent.store_transition(state_tensor, action, reward, done, log_prob, value)
                if done:
                    # Bootstrap with 0 for simplicity (could use critic for last state)
                    self.agent.finish_path(last_value=0.0)

            # If last path not finished, bootstrap value
            if self.agent.buffer.path_start_idx != self.agent.buffer.ptr:
                # Use critic estimate for last state
                last_state = iteration_data['transitions'][-1][0]
                with torch.no_grad():
                    last_value = self.agent.critic(torch.FloatTensor(last_state).unsqueeze(0)).item()
                self.agent.finish_path(last_value=last_value)

            print(f"✓ Loaded {num_transitions} transitions into PPO buffer")

            # Train agent once per collected iteration (on-policy)
            metrics = self.agent.train_step() if self.agent.buffer.ptr > 0 else None
            if metrics:
                loss = metrics.get('losses', 0.0)
                actor_loss = metrics.get('actor_losses', 0.0)
                critic_loss = metrics.get('critic_losses', 0.0)
                entropy = metrics.get('entropies', 0.0)
                kl = metrics.get('kls', 0.0)
                # Compute averaged metrics
                loss = np.array(loss).mean()
                actor_loss = np.array(actor_loss).mean()
                critic_loss = np.array(critic_loss).mean()
                entropy = np.array(entropy).mean()
                kl = np.array(kl).mean()
                
                # Send ONE aggregated update to monitor (not per mini-batch)
                if self.monitor:
                    total_steps = metrics.get('updates')
                    self.monitor.training_step(
                        iteration_num, 
                        total_steps, 
                        loss=loss,
                        actor_loss=actor_loss,
                        critic_loss=critic_loss,
                        entropy=entropy,
                        kl=kl
                    )
                
                self.losses.append(loss)
                self.actor_losses.append(actor_loss)
                self.critic_losses.append(critic_loss)
                self.entropy_values.append(entropy)
                self.kl_values.append(kl)
                self.recent_actor_losses.append(actor_loss)
                avg_actor_loss_100 = np.mean(self.recent_actor_losses)
                self._log_loss(iteration_num, loss, actor_loss, critic_loss, entropy, kl, avg_actor_loss_100)
                print(f"✓ PPO Update: actor_loss={actor_loss:.6f} critic_loss={critic_loss:.6f} entropy={entropy:.4f} kl={kl:.4f}")
            else:
                print("✓ No data collected for PPO update this iteration")
            
            # Store statistics
            buffer_fill_pct = (self.agent.buffer.ptr / self.agent.buffer.max_size) * 100.0
            stats = {
                "iteration": iteration_num,
                "reward": iteration_data["total_reward"],
                "return_pct": iteration_data["total_return"],
                "length": iteration_data["iteration_length"],
                "num_trades": iteration_data["num_trades"],
                "buffer_fill_pct": buffer_fill_pct
            }
            
            self.iteration_rewards.append(iteration_data["total_reward"])
            self.iteration_lengths.append(iteration_data["iteration_length"])
            # Log metrics
            self._log_metrics(stats)
            
            print(f"✓ Reward: {iteration_data['total_reward']:.2f}, Return: {iteration_data['total_return']:.2f}%, Trades: {iteration_data['num_trades']}")
            
            # Notify monitor about iteration completion
            if self.monitor:
                self.monitor.iteration_complete(iteration_num, total_iterations, {
                    'total_reward': iteration_data["total_reward"],
                    'return_pct': iteration_data["total_return"],
                    'num_transitions': num_transitions,
                    'buffer_fill_pct': buffer_fill_pct
                })
                self.monitor.log(f"Iteration {iteration_num} complete: Reward={iteration_data['total_reward']:.2f}, Return={iteration_data['total_return']:.2f}%", 'success')
            
            return stats
            
        except Exception as e:
            print(f"✗ Error in iteration {iteration_num}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _log_loss(self, iteration, loss, actor_loss, critic_loss, entropy, kl, avg_actor_loss_100):
        """Log PPO update metrics."""
        with open(self.loss_log_file, 'a') as f:
            f.write(f"{iteration},{loss:.6f},{actor_loss:.6f},{critic_loss:.6f},{entropy:.6f},{kl:.6f},{avg_actor_loss_100:.6f}\n")
    
    def _log_metrics(self, stats):
        """Log iteration-level (environment) metrics."""
        with open(self.metrics_log_file, 'a') as f:
            f.write(f"{stats['iteration']},{stats['reward']:.2f},{stats['return_pct']:.2f},"
                    f"{stats['length']},{stats['buffer_fill_pct']:.2f},{stats['num_trades']}\n")
    
    def train(self, num_iterations=100, save_freq=10):
        """
        Iteration-based training loop (standard PPO).
        Each iteration collects a rollout and trains.
        
        Args:
            num_iterations: Number of iterations to train
            save_freq: Frequency of model saving (iterations)
        """
        print(f"\n{'='*60}")
        print(f"PPO Iteration-Based Training Started")
        print(f"{'='*60}")
        print(f"Iterations: {num_iterations}")
        print(f"State dim: {self.state_dim}")
        print(f"Action dim: {self.agent.action_dim}")
        print(f"Save dir: {self.save_dir}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Notify monitor about training start (iteration mode)
        if self.monitor:
            self.monitor.training_start(num_iterations=num_iterations)
            self.monitor.log(f"Starting iteration-based training: {num_iterations} iterations", 'info')
        
        for iteration in range(1, num_iterations + 1):
            stats = self.train_iteration(iteration, num_iterations)
            
            # Save model periodically
            if iteration % save_freq == 0:
                self.save_checkpoint(iteration)
            
            # Print progress
            if iteration % 10 == 0:
                self.print_progress(iteration, start_time)
        
        # Final save
        self.save_checkpoint(num_iterations, final=True)
        
        # Notify monitor about training completion
        if self.monitor:
            self.monitor.training_complete({
                'total_iterations': num_iterations,
                'total_time': time.time() - start_time
            })
        
        print(f"\n{'='*60}")
        print(f"Training Completed!")
        print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, iteration, final=False):
        """Save model checkpoint."""
        if final:
            filename = "ppo_final.pth"
        else:
            filename = f"ppo_iteration_{iteration}.pth"
        
        filepath = self.save_dir / filename
        self.agent.save(filepath)
        
        # Save training statistics
        stats_file = self.save_dir / f"training_stats_itr{iteration}.json"
        with open(stats_file, 'w') as f:
            json.dump({
                "iteration": iteration,
                "iteration_rewards": self.iteration_rewards,
                "iteration_lengths": self.iteration_lengths,
                "losses": self.losses,
                "actor_losses": self.actor_losses,
                "critic_losses": self.critic_losses,
                "entropy_values": self.entropy_values,
                "kl_values": self.kl_values
            }, f, indent=2)
        
        print(f"✓ Checkpoint saved: {filepath}")
        
        # Notify monitor
        if self.monitor:
            self.monitor.checkpoint_saved(filename)
    
    def plot_training_progress(self, iteration):
        """Generate and save training progress plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training Progress - Iteration {iteration}', fontsize=16)
            
            # Plot 1: Iteration Rewards
            if len(self.iteration_rewards) > 0:
                axes[0, 0].plot(self.iteration_rewards, alpha=0.6, label='Iteration Reward')
                if len(self.iteration_rewards) >= 10:
                    # Moving average
                    window = min(10, len(self.iteration_rewards))
                    moving_avg = np.convolve(self.iteration_rewards, np.ones(window)/window, mode='valid')
                    axes[0, 0].plot(range(window-1, len(self.iteration_rewards)), moving_avg, 
                                   'r-', linewidth=2, label=f'MA({window})')
                axes[0, 0].set_xlabel('Iteration')
                axes[0, 0].set_ylabel('Reward')
                axes[0, 0].set_title('Iteration Rewards')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Training Loss
            if len(self.losses) > 0:
                axes[0, 1].plot(self.losses, alpha=0.6, label='Loss')
                if len(self.losses) >= 10:
                    window = min(10, len(self.losses))
                    moving_avg = np.convolve(self.losses, np.ones(window)/window, mode='valid')
                    axes[0, 1].plot(range(window-1, len(self.losses)), moving_avg,
                                   'r-', linewidth=2, label=f'MA({window})')
                axes[0, 1].set_xlabel('Training Step')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].set_title('Training Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Actor Loss
            if len(self.actor_losses) > 0:
                axes[1, 0].plot(self.actor_losses, 'b-', linewidth=2, label='Actor Loss')
                axes[1, 0].set_xlabel('Training Step')
                axes[1, 0].set_ylabel('Actor Loss')
                axes[1, 0].set_title('Actor Loss Trend')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Critic Loss trend
            if len(self.critic_losses) > 0:
                axes[1, 1].plot(self.critic_losses, 'g-', linewidth=2, label='Critic Loss')
                axes[1, 1].set_xlabel('Training Step')
                axes[1, 1].set_ylabel('Critic Loss')
                axes[1, 1].set_title('Critic Loss Trend')
                axes[1, 1].legend()

            # Plot 5: Entropy trend
            if len(self.entropy_values) > 0:
                axes[1, 1].plot(self.entropy_values, 'c-', linewidth=2, label='Entropy')
                axes[1, 1].set_xlabel('Update')
                axes[1, 1].set_ylabel('Entropy')
                axes[1, 1].set_title('Policy Entropy')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

            # Plot 6: KL Divergence trend
            if len(self.kl_values) > 0:
                axes[1, 1].plot(self.kl_values, 'm-', linewidth=2, label='KL Divergence')
                axes[1, 1].set_xlabel('Update')
                axes[1, 1].set_ylabel('KL')
                axes[1, 1].set_title('Policy KL Divergence')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = self.save_dir / f'training_progress_ep{iteration}.png'
            plt.savefig(plot_file, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved training plot: {plot_file}")
            
        except Exception as e:
            print(f"Warning: Could not generate plot: {e}")
    
    def print_progress(self, iteration, start_time):
        """Print training progress."""
        elapsed = (time.time() - start_time) / 60
        
        if len(self.iteration_rewards) >= 10:
            recent_rewards = self.iteration_rewards[-10:]
            avg_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
            max_reward = max(recent_rewards)
            min_reward = min(recent_rewards)
        else:
            avg_reward = np.mean(self.iteration_rewards) if self.iteration_rewards else 0
            std_reward = 0
            max_reward = max(self.iteration_rewards) if self.iteration_rewards else 0
            min_reward = min(self.iteration_rewards) if self.iteration_rewards else 0
        
        if len(self.losses) > 0:
            avg_loss = np.mean(self.losses[-100:])
            recent_loss = self.losses[-1] if self.losses else 0
        else:
            avg_loss = 0
            recent_loss = 0
        
        print(f"\n{'='*70}")
        print(f"📊 Progress Report - Iteration {iteration}")
        print(f"{'='*70}")
        print(f"⏱️  Time Elapsed: {elapsed:.2f} min ({elapsed/iteration:.2f} min/iteration)")
        print(f"🎯 Rewards (last 10 iterations):")
        print(f"   Average: {avg_reward:+.2f} ± {std_reward:.2f}")
        print(f"   Range: [{min_reward:+.2f}, {max_reward:+.2f}]")
        print(f"📉 Training Loss:")
        print(f"   Recent: {recent_loss:.6f}")
        print(f"   Average (last 100): {avg_loss:.6f}")
        if self.losses:
            print(f"📐 Actor Loss (recent): {self.losses[-1]:.6f}")
        if self.critic_losses:
            print(f"🧮 Critic Loss (recent): {self.critic_losses[-1]:.6f}")
        if self.entropy_values:
            print(f"🌀 Entropy (recent): {self.entropy_values[-1]:.4f}")
        if self.kl_values:
            print(f"🔁 KL (recent): {self.kl_values[-1]:.4f}")
        print(f"{'='*70}\n")
        
        # Generate progress plot
        self.plot_training_progress(iteration)


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for trading")
    parser.add_argument("--iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--save_dir", type=str, default="models/ppo", help="Directory to save models")
    parser.add_argument("--save_freq", type=int, default=10, help="Save frequency (iterations)")
    parser.add_argument("--api_url", type=str, default="http://localhost:5010", help="API base URL")
    parser.add_argument("--data", type=str,
                        default="ats/data/BTC_USDT_short.csv",
                        help="Path to data CSV file (relative to ATS root)")
    # Optional PPO hyperparameters overrides
    parser.add_argument("--state_dim", type=int, default=7, help="State feature dimension")
    parser.add_argument("--action_dim", type=int, default=3, help="Number of discrete actions")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="PPO learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--target_kl", type=float, default=0.01, help="Target KL for early stopping")
    parser.add_argument("--buffer_size", type=int, default=204800, help="Rollout buffer size")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size for PPO updates")
    parser.add_argument("--value_loss_coeff", type=float, default=0.5, help="Critic loss coefficient")
    parser.add_argument("--entropy_coeff", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--device", type=str, default='cpu', help="Compute device: cpu or cuda")

    args = parser.parse_args()

    # Create trainer
    trainer = PPOTrainer(
        api_base_url=args.api_url,
        data_source=args.data,
        save_dir=args.save_dir,
        action_dim=args.action_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        lam=args.lam,
        clip_ratio=args.clip_ratio,
        target_kl=args.target_kl,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        value_loss_coeff=args.value_loss_coeff,
        entropy_coeff=args.entropy_coeff,
        device=args.device
    )

    # Train
    trainer.train(
        num_iterations=args.iterations,
        save_freq=args.save_freq
    )


if __name__ == "__main__":
    main()

