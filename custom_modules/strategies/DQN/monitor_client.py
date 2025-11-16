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
    
    def training_start(self, num_episodes):
        """Signal training start"""
        if self.connected:
            self.client.emit('training_start', {'num_episodes': num_episodes})
    
    def episode_start(self, episode, total_episodes):
        """Signal episode start"""
        if self.connected:
            self.client.emit('episode_start', {
                'episode': episode,
                'total_episodes': total_episodes
            })
    
    def training_step(self, step, total_steps, loss):
        """Update training step progress"""
        if self.connected:
            self.client.emit('training_step', {
                'step': step,
                'total_steps': total_steps,
                'loss': loss
            })
    
    def episode_complete(self, episode, total_episodes, metrics):
        """Signal episode completion with metrics"""
        if self.connected:
            self.client.emit('episode_complete', {
                'episode': episode,
                'total_episodes': total_episodes,
                **metrics
            })
    
    def epsilon_update(self, epsilon):
        """Update epsilon value"""
        if self.connected:
            self.client.emit('epsilon_update', {'epsilon': epsilon})
    
    def checkpoint_saved(self, filename):
        """Signal checkpoint saved"""
        if self.connected:
            self.client.emit('checkpoint_saved', {'filename': filename})
    
    def training_complete(self, total_time):
        """Signal training completion"""
        if self.connected:
            self.client.emit('training_complete', {'total_time': total_time})

if __name__ == '__main__':
    # Test the client
    print("Testing monitor client...")
    
    monitor = TrainingMonitorClient()
    monitor.connect()
    
    if monitor.connected:
        print("✓ Testing events...")
        
        monitor.training_start(3)
        time.sleep(0.5)
        
        for ep in range(1, 4):
            monitor.episode_start(ep, 3)
            monitor.log(f'Episode {ep} started', 'info')
            time.sleep(1)
            
            for step in range(1, 6):
                monitor.training_step(step, 5, 100.0 / step)
                time.sleep(0.3)
            
            monitor.episode_complete(ep, 3, {
                'total_reward': 150.0,
                'return_pct': 1.5,
                'num_transitions': 1000,
                'epsilon': 1.0 - ep * 0.1
            })
            monitor.log(f'Episode {ep} complete', 'success')
            time.sleep(1)
        
        monitor.training_complete(10.5)
        print("\n✓ Test complete! Check the dashboard.")
        
        time.sleep(2)
        monitor.disconnect()
    else:
        print("❌ Could not connect to monitor server")
        print("Make sure the monitor server is running:")
        print("  python custom_modules/strategies/DQN/monitor_server.py")

