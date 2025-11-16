#!/usr/bin/env python3
"""
Training Monitor Client - Connects to monitor server and sends events
This is what the training script should use!
"""

import socketio
import time

class TrainingMonitorClient:
    """
    Client that connects to the monitor server and sends training events.
    This is the CORRECT way to send events from the training script.
    """
    def __init__(self, server_url='http://localhost:5050'):
        self.server_url = server_url
        self.client = socketio.Client()
        self.connected = False
        
    def connect(self):
        """Connect to the monitor server"""
        try:
            self.client.connect(self.server_url)
            self.connected = True
            print(f"✓ Connected to monitor at {self.server_url}")
            time.sleep(0.5)  # Give it a moment to stabilize
        except Exception as e:
            print(f"⚠️  Could not connect to monitor: {e}")
            self.connected = False
    
    def disconnect(self):
        """Disconnect from monitor"""
        if self.connected:
            self.client.disconnect()
            self.connected = False
    
    def log(self, message, level='info'):
        """Send a log message"""
        if self.connected:
            self.client.emit('log', {'message': message, 'level': level})
    
    def training_start(self, num_iterations):
        """Signal training start"""
        if self.connected:
            self.client.emit('training_start', {'num_iterations': num_iterations})
    
    def iteration_start(self, iteration, total_iterations):
        """Signal iteration start"""
        if self.connected:
            self.client.emit('iteration_start', {
                'iteration': iteration,
                'total_iterations': total_iterations
            })
    
    def training_step(self, step, total_steps, loss=None, actor_loss=None, critic_loss=None, entropy=None, kl=None):
        """Update training step progress, supports PPO metrics"""
        if self.connected:
            payload = {
                'step': step,
                'total_steps': total_steps,
            }
            if loss is not None:
                payload['loss'] = loss
            if actor_loss is not None:
                payload['actor_loss'] = actor_loss
            if critic_loss is not None:
                payload['critic_loss'] = critic_loss
            if entropy is not None:
                payload['entropy'] = entropy
            if kl is not None:
                payload['kl'] = kl
            self.client.emit('training_step', payload)

    def iteration_complete(self, iteration, total_iterations, metrics):
        """Signal iteration completion with metrics"""
        if self.connected:
            self.client.emit('iteration_complete', {
                'iteration': iteration,
                'total_iterations': total_iterations,
                **metrics
            })
    
    def checkpoint_saved(self, filename):
        """Signal checkpoint saved"""
        if self.connected:
            self.client.emit('checkpoint_saved', {'filename': filename})
    
    def training_complete(self, metrics):
        """Signal training completion"""
        if self.connected:
            self.client.emit('training_complete', metrics)

if __name__ == '__main__':
    # Test the client
    print("Testing PPO monitor client...")
    
    monitor = TrainingMonitorClient()
    monitor.connect()
    
    if monitor.connected:
        print("✓ Testing events (iterations)...")
        
        monitor.training_start(3)
        time.sleep(0.5)
        
        for it in range(1, 4):
            monitor.iteration_start(it, 3)
            monitor.log(f'Iteration {it} started', 'info')
            time.sleep(1)
            
            for step in range(1, 6):
                # Simulate PPO metrics (actor/critic/entropy/kl)
                monitor.training_step(
                    step,
                    5,
                    loss=100.0/step,
                    actor_loss=0.5/step,
                    critic_loss=0.8/step,
                    entropy=1.0 - 0.05*step,
                    kl=0.01*step
                )
                time.sleep(0.3)
            
            monitor.iteration_complete(it, 3, {
                'total_reward': 150.0,
                'return_pct': 1.5,
                'num_transitions': 1000
            })
            monitor.log(f'Iteration {it} complete', 'success')
            time.sleep(1)
        
        monitor.training_complete(10.5)
        print("\n✓ Test complete! Check the dashboard.")
        
        time.sleep(2)
        monitor.disconnect()
    else:
        print("❌ Could not connect to monitor server")
        print("Make sure the monitor server is running:")
        print("  python custom_modules/strategies/PPO/monitor_server.py")

