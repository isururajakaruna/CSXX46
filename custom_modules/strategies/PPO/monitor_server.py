#!/usr/bin/env python3
"""
Real-time Training Monitor Server
Streams training metrics and logs to a web dashboard
"""

from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
import webbrowser
import time
import os
from pathlib import Path

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ppo-training-monitor'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# HTML Template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>PPO Training Monitor</title>
    <meta charset="utf-8">
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            padding: 20px 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .header h1 {
            color: #667eea;
            font-size: 32px;
            margin-bottom: 10px;
        }
        
        .status {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        
        .status-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 14px;
        }
        
        .status-badge.running {
            background: #10b981;
            color: white;
        }
        
        .status-badge.waiting {
            background: #f59e0b;
            color: white;
        }
        
        .status-badge.completed {
            background: #6366f1;
            color: white;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .metric-card h3 {
            font-size: 14px;
            color: #6b7280;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metric-card .value {
            font-size: 32px;
            font-weight: 700;
            color: #667eea;
        }
        
        .metric-card .subvalue {
            font-size: 14px;
            color: #9ca3af;
            margin-top: 4px;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .chart-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .chart-card h2 {
            font-size: 18px;
            color: #374151;
            margin-bottom: 15px;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
        }
        
        .logs-container {
            background: #1f2937;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            max-height: 400px;
            overflow-y: auto;
        }
        
        .logs-container h2 {
            color: #f3f4f6;
            font-size: 18px;
            margin-bottom: 15px;
        }
        
        .log-entry {
            font-family: 'Courier New', monospace;
            font-size: 13px;
            color: #d1d5db;
            padding: 4px 0;
            border-bottom: 1px solid #374151;
        }
        
        .log-entry.info {
            color: #60a5fa;
        }
        
        .log-entry.success {
            color: #34d399;
        }
        
        .log-entry.warning {
            color: #fbbf24;
        }
        
        .log-entry.error {
            color: #f87171;
        }
        
        .log-entry .timestamp {
            color: #9ca3af;
            margin-right: 10px;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 PPO Training Monitor</h1>
            <div class="status">
                <span id="status-badge" class="status-badge waiting">⏳ Waiting</span>
                <span id="current-iteration">Iteration: 0/0</span>
            </div>
            <div class="progress-bar">
                <div id="progress-fill" class="progress-fill" style="width: 0%"></div>
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Current Loss</h3>
                <div class="value" id="current-loss">--</div>
                <div class="subvalue">Average over last batch</div>
            </div>
            <div class="metric-card">
                <h3>Iteration Reward</h3>
                <div class="value" id="iteration-reward">--</div>
                <div class="subvalue">Total iteration return</div>
            </div>
            <div class="metric-card">
                <h3>Iteration Return</h3>
                <div class="value" id="iteration-return">--</div>
                <div class="subvalue">Percentage gain/loss</div>
            </div>
            <div class="metric-card">
                <h3>Transitions</h3>
                <div class="value" id="transitions">--</div>
                <div class="subvalue">Collected this iteration</div>
            </div>
            <div class="metric-card">
                <h3>Training Steps</h3>
                <div class="value" id="training-steps">--</div>
                <div class="subvalue">Total gradient updates</div>
            </div>
            <div class="metric-card">
                <h3>Actor Loss</h3>
                <div class="value" id="actor-loss">--</div>
                <div class="subvalue">Policy loss</div>
            </div>
            <div class="metric-card">
                <h3>Critic Loss</h3>
                <div class="value" id="critic-loss">--</div>
                <div class="subvalue">Value function loss</div>
            </div>
            <div class="metric-card">
                <h3>Entropy</h3>
                <div class="value" id="entropy">--</div>
                <div class="subvalue">Policy entropy</div>
            </div>
            <div class="metric-card">
                <h3>KL Divergence</h3>
                <div class="value" id="kl">--</div>
                <div class="subvalue">Approx KL</div>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-card">
                <h2>📉 Training Loss</h2>
                <div class="chart-container">
                    <canvas id="lossChart"></canvas>
                </div>
            </div>
            <div class="chart-card">
                <h2>💰 Iteration Rewards</h2>
                <div class="chart-container">
                    <canvas id="rewardChart"></canvas>
                </div>
            </div>
            <div class="chart-card">
                <h2>📊 Iteration Returns (%)</h2>
                <div class="chart-container">
                    <canvas id="returnChart"></canvas>
                </div>
            </div>
            <div class="chart-card">
                <h2>🧠 Actor Loss</h2>
                <div class="chart-container">
                    <canvas id="actorLossChart"></canvas>
                </div>
            </div>
            <div class="chart-card">
                <h2>🏛️ Critic Loss</h2>
                <div class="chart-container">
                    <canvas id="criticLossChart"></canvas>
                </div>
            </div>
            <div class="chart-card">
                <h2>🎲 Entropy</h2>
                <div class="chart-container">
                    <canvas id="entropyChart"></canvas>
                </div>
            </div>
            <div class="chart-card">
                <h2>📐 KL Divergence</h2>
                <div class="chart-container">
                    <canvas id="klChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="logs-container">
            <h2>📋 Training Logs</h2>
            <div id="logs"></div>
        </div>
    </div>
    
    <script>
        // Connect to Socket.IO
        const socket = io();
        
        // Chart configurations
        const chartConfig = {
            type: 'line',
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 300
                },
                scales: {
                    y: {
                        beginAtZero: false
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        };
        
        // Initialize charts
        const lossChart = new Chart(
            document.getElementById('lossChart'),
            {
                ...chartConfig,
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Loss',
                        data: [],
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        tension: 0.4
                    }]
                }
            }
        );
        
        const rewardChart = new Chart(
            document.getElementById('rewardChart'),
            {
                ...chartConfig,
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Reward',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4
                    }]
                }
            }
        );
        
        const returnChart = new Chart(
            document.getElementById('returnChart'),
            {
                ...chartConfig,
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Return %',
                        data: [],
                        borderColor: '#6366f1',
                        backgroundColor: 'rgba(99, 102, 241, 0.1)',
                        tension: 0.4
                    }]
                }
            }
        );

        const actorLossChart = new Chart(
            document.getElementById('actorLossChart'),
            {
                ...chartConfig,
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Actor Loss',
                        data: [],
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        tension: 0.4
                    }]
                }
            }
        );

        const criticLossChart = new Chart(
            document.getElementById('criticLossChart'),
            {
                ...chartConfig,
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Critic Loss',
                        data: [],
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.1)',
                        tension: 0.4
                    }]
                }
            }
        );

        const entropyChart = new Chart(
            document.getElementById('entropyChart'),
            {
                ...chartConfig,
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Entropy',
                        data: [],
                        borderColor: '#22c55e',
                        backgroundColor: 'rgba(34, 197, 94, 0.1)',
                        tension: 0.4
                    }]
                }
            }
        );

        const klChart = new Chart(
            document.getElementById('klChart'),
            {
                ...chartConfig,
                data: {
                    labels: [],
                    datasets: [{
                        label: 'KL',
                        data: [],
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        tension: 0.4
                    }]
                }
            }
        );

        // Update chart helper
        function updateChart(chart, label, value, maxPoints = 50) {
            chart.data.labels.push(label);
            chart.data.datasets[0].data.push(value);
            
            // Keep only last maxPoints
            if (chart.data.labels.length > maxPoints) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
            }
            
            chart.update('none');
        }
        
        // Add log entry
        function addLog(message, level = 'info') {
            const logsDiv = document.getElementById('logs');
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry ${level}`;
            logEntry.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${message}`;
            logsDiv.appendChild(logEntry);
            logsDiv.scrollTop = logsDiv.scrollHeight;
            
            // Keep only last 100 logs
            while (logsDiv.children.length > 100) {
                logsDiv.removeChild(logsDiv.firstChild);
            }
        }
        
        // Socket.IO event handlers
        socket.on('connect', () => {
            addLog('✓ Connected to training monitor', 'success');
        });
        
        // Track training mode (episode vs iteration)
        let trainingMode = 'iteration';  // default
        
        socket.on('training_start', (data) => {
            document.getElementById('status-badge').className = 'status-badge running';
            document.getElementById('status-badge').textContent = '🚀 Training';
            
            // Detect mode from training_start event
            if (data.num_episodes !== undefined) {
                trainingMode = 'episode';
                addLog(`Training started: ${data.num_episodes} episodes`, 'success');
            } else {
                trainingMode = 'iteration';
                const count = data.num_iterations || 0;
                addLog(`Training started: ${count} iterations`, 'success');
            }
        });
        
        function onEpisodeStart(data) {
            trainingMode = 'episode';
            const current = (data.episode !== undefined) ? data.episode : 0;
            const total = (data.total_episodes !== undefined) ? data.total_episodes : 0;
            document.getElementById('current-iteration').textContent = 
                `Episode: ${current}/${total}`;
            addLog(`Episode ${current} started`, 'info');
        }
        socket.on('episode_start', onEpisodeStart);
        
        function onIterationStart(data) {
            trainingMode = 'iteration';
            const current = (data.iteration !== undefined) ? data.iteration : 0;
            const total = (data.total_iterations !== undefined) ? data.total_iterations : 0;
            document.getElementById('current-iteration').textContent = 
                `Iteration: ${current}/${total}`;
            addLog(`Iteration ${current} started`, 'info');
        }
        socket.on('iteration_start', onIterationStart);
        
        function onEpisodeComplete(data) {
            trainingMode = 'episode';
            const episode = (data.episode !== undefined) ? data.episode : 0;
            const total = (data.total_episodes !== undefined) ? data.total_episodes : 0;
            updateChart(rewardChart, `Ep ${episode}`, data.total_reward);
            updateChart(returnChart, `Ep ${episode}`, data.return_pct);
            
            if (data.total_reward !== undefined) {
                document.getElementById('iteration-reward').textContent = Number(data.total_reward).toFixed(2);
            }
            if (data.return_pct !== undefined) {
                document.getElementById('iteration-return').textContent = `${Number(data.return_pct) >= 0 ? '+' : ''}${Number(data.return_pct).toFixed(2)}%`;
            }
            if (data.num_transitions !== undefined) {
                document.getElementById('transitions').textContent = Number(data.num_transitions).toLocaleString();
            }
            
            if (total) {
                const progress = (episode / total) * 100;
                document.getElementById('progress-fill').style.width = progress + '%';
            }
            
            addLog(`Episode ${episode} complete: Reward=${Number(data.total_reward || 0).toFixed(2)}, Return=${Number(data.return_pct || 0).toFixed(2)}%`, 'success');
        }
        socket.on('episode_complete', onEpisodeComplete);
        
        function onIterationComplete(data) {
            trainingMode = 'iteration';
            const iteration = (data.iteration !== undefined) ? data.iteration : 0;
            const total = (data.total_iterations !== undefined) ? data.total_iterations : 0;
            updateChart(rewardChart, `It ${iteration}`, data.total_reward);
            updateChart(returnChart, `It ${iteration}`, data.return_pct);
            
            if (data.total_reward !== undefined) {
                document.getElementById('iteration-reward').textContent = Number(data.total_reward).toFixed(2);
            }
            if (data.return_pct !== undefined) {
                document.getElementById('iteration-return').textContent = `${Number(data.return_pct) >= 0 ? '+' : ''}${Number(data.return_pct).toFixed(2)}%`;
            }
            if (data.num_transitions !== undefined) {
                document.getElementById('transitions').textContent = Number(data.num_transitions).toLocaleString();
            }
            
            if (total) {
                const progress = (iteration / total) * 100;
                document.getElementById('progress-fill').style.width = `${progress}%`;
            }
            
            addLog(`Iteration ${iteration} complete: Reward=${Number(data.total_reward).toFixed(2)}, Return=${Number(data.return_pct).toFixed(2)}%`, 'success');
        }
        socket.on('iteration_complete', onIterationComplete);
        
        socket.on('training_step', (data) => {
            if (data.loss !== undefined) {
                updateChart(lossChart, data.step, data.loss);
                document.getElementById('current-loss').textContent = Number(data.loss).toFixed(4);
            }
            if (data.total_steps !== undefined) {
                document.getElementById('training-steps').textContent = Number(data.total_steps).toLocaleString();
            }
            
            if (data.loss !== undefined && data.step % 100 === 0) {
                addLog(`Training step ${data.step}: loss=${Number(data.loss).toFixed(4)}`, 'info');
            }

            // PPO-specific metrics (optional)
            if (data.actor_loss !== undefined) {
                updateChart(actorLossChart, data.step, data.actor_loss);
                document.getElementById('actor-loss').textContent = Number(data.actor_loss).toFixed(4);
            }
            if (data.critic_loss !== undefined) {
                updateChart(criticLossChart, data.step, data.critic_loss);
                document.getElementById('critic-loss').textContent = Number(data.critic_loss).toFixed(4);
            }
            if (data.entropy !== undefined) {
                updateChart(entropyChart, data.step, data.entropy);
                document.getElementById('entropy').textContent = Number(data.entropy).toFixed(4);
            }
            if (data.kl !== undefined) {
                updateChart(klChart, data.step, data.kl);
                document.getElementById('kl').textContent = Number(data.kl).toFixed(4);
            }
        });
  
        
        socket.on('checkpoint_saved', (data) => {
            addLog(`✓ Checkpoint saved: ${data.filename}`, 'success');
        });
        
        socket.on('training_complete', (data) => {
            document.getElementById('status-badge').className = 'status-badge completed';
            document.getElementById('status-badge').textContent = '✓ Complete';
            addLog(`Training complete! Total time: ${data.total_time}`, 'success');
        });
        
        socket.on('log', (data) => {
            addLog(data.message, data.level || 'info');
        });
        
        socket.on('error', (data) => {
            addLog(`✗ Error: ${data.message}`, 'error');
        });
        
        // Initial log
        addLog('Training monitor initialized', 'info');
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the monitoring dashboard"""
    return render_template_string(DASHBOARD_HTML)

# Socket.IO Event Handlers - Relay events from external clients to all dashboards
@socketio.on('training_start')
def handle_training_start(data):
    """Relay training_start event"""
    emit('training_start', data, broadcast=True)

@socketio.on('episode_start')
def handle_episode_start(data):
    """Relay episode_start event"""
    emit('episode_start', data, broadcast=True)

@socketio.on('iteration_start')
def handle_iteration_start(data):
    """Relay iteration_start event"""
    emit('iteration_start', data, broadcast=True)

@socketio.on('training_step')
def handle_training_step(data):
    """Relay training_step event"""
    emit('training_step', data, broadcast=True)

@socketio.on('episode_complete')
def handle_episode_complete(data):
    """Relay episode_complete event"""
    emit('episode_complete', data, broadcast=True)

@socketio.on('iteration_complete')
def handle_iteration_complete(data):
    """Relay iteration_complete event"""
    emit('iteration_complete', data, broadcast=True)

@socketio.on('log')
def handle_log(data):
    """Relay log event"""
    emit('log', data, broadcast=True)

@socketio.on('checkpoint_saved')
def handle_checkpoint_saved(data):
    """Relay checkpoint_saved event"""
    emit('checkpoint_saved', data, broadcast=True)

@socketio.on('training_complete')
def handle_training_complete(data):
    """Relay training_complete event"""
    emit('training_complete', data, broadcast=True)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"✓ Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"✗ Client disconnected")

class TrainingMonitor:
    """
    Training monitor that can be used by the training script
    """
    def __init__(self, port=5050, auto_open=True):
        self.port = port
        self.auto_open = auto_open
        self.server_thread = None
        self.running = False
        
    def start(self):
        """Start the monitoring server in a background thread"""
        if self.running:
            print("Monitor already running")
            return
            
        self.running = True
        
        # Start server in background thread
        self.server_thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        self.server_thread.start()
        
        # Wait a bit for server to start
        time.sleep(2)
        
        # Auto-open browser
        if self.auto_open:
            url = f"http://localhost:{self.port}"
            print(f"\n{'='*60}")
            print(f"🌐 Training Monitor: {url}")
            print(f"{'='*60}\n")
            webbrowser.open(url)
    
    def _run_server(self):
        """Run the Flask-SocketIO server"""
        socketio.run(
            app,
            host='0.0.0.0',
            port=self.port,
            debug=False,
            use_reloader=False,
            log_output=False,
            allow_unsafe_werkzeug=True
        )
    
    def emit(self, event, data):
        """Emit an event to all connected clients"""
        if self.running:
            socketio.emit(event, data, namespace='/')
    
    def log(self, message, level='info'):
        """Send a log message to the dashboard"""
        self.emit('log', {'message': message, 'level': level})
    
    def training_start(self, num_iterations):
        """Signal training start"""
        self.emit('training_start', {'num_iterations': num_iterations})

    def iteration_start(self, iteration, total_iterations):
        """Signal iteration start"""
        self.emit('iteration_start', {
            'iteration': iteration,
            'total_iterations': total_iterations
        })

    def iteration_complete(self, iteration, total_iterations, metrics):
        """Signal iteration completion with metrics"""
        self.emit('iteration_complete', {
            'iteration': iteration,
            'total_iterations': total_iterations,
            **metrics
        })
    
    def training_step(self, step, total_steps, loss):
        """Update training step progress"""
        self.emit('training_step', {
            'step': step,
            'total_steps': total_steps,
            'loss': loss
        })
    
    def checkpoint_saved(self, filename):
        """Signal checkpoint saved"""
        self.emit('checkpoint_saved', {'filename': filename})
    
    def training_complete(self, total_time):
        """Signal training completion"""
        self.emit('training_complete', {'total_time': total_time})
    
    def error(self, message):
        """Send error message"""
        self.emit('error', {'message': message})

if __name__ == '__main__':
    # Test the monitor
    monitor = TrainingMonitor(port=5050, auto_open=True)
    monitor.start()
    
    print("Monitor server running. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping monitor...")

