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
import argparse

app = Flask(__name__)
app.config["SECRET_KEY"] = "dqn-training-monitor"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# HTML Template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>DQN Training Monitor</title>
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
            <h1>🤖 DQN Training Monitor</h1>
            <div class="status">
                <span id="status-badge" class="status-badge waiting">⏳ Waiting</span>
                <span id="current-episode">Episode: 0/0</span>
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
                <h3>Episode Reward</h3>
                <div class="value" id="episode-reward">--</div>
                <div class="subvalue">Total episode return</div>
            </div>
            <div class="metric-card">
                <h3>Episode Return</h3>
                <div class="value" id="episode-return">--</div>
                <div class="subvalue">Percentage gain/loss</div>
            </div>
            <div class="metric-card">
                <h3>Epsilon</h3>
                <div class="value" id="epsilon">--</div>
                <div class="subvalue">Exploration rate</div>
            </div>
            <div class="metric-card">
                <h3>Transitions</h3>
                <div class="value" id="transitions">--</div>
                <div class="subvalue">Collected this episode</div>
            </div>
            <div class="metric-card">
                <h3>Training Steps</h3>
                <div class="value" id="training-steps">--</div>
                <div class="subvalue">Total gradient updates</div>
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
                <h2>💰 Episode Rewards</h2>
                <div class="chart-container">
                    <canvas id="rewardChart"></canvas>
                </div>
            </div>
            <div class="chart-card">
                <h2>📊 Episode Returns (%)</h2>
                <div class="chart-container">
                    <canvas id="returnChart"></canvas>
                </div>
            </div>
            <div class="chart-card">
                <h2>🎲 Epsilon Decay</h2>
                <div class="chart-container">
                    <canvas id="epsilonChart"></canvas>
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
                        tension: 0.4,
                        pointRadius: 0,           // Hide points by default
                        pointHoverRadius: 4,      // Show on hover
                        borderWidth: 2            // Slightly thicker line
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
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        borderWidth: 2
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
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        borderWidth: 2
                    }]
                }
            }
        );
        
        const epsilonChart = new Chart(
            document.getElementById('epsilonChart'),
            {
                ...chartConfig,
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Epsilon',
                        data: [],
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        borderWidth: 2
                    }]
                }
            }
        );
        
        // Update chart helper - shows ALL data (no sliding window)
        function updateChart(chart, label, value) {
            chart.data.labels.push(label);
            chart.data.datasets[0].data.push(value);
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
        
        socket.on('training_start', (data) => {
            document.getElementById('status-badge').className = 'status-badge running';
            document.getElementById('status-badge').textContent = '🚀 Training';
            addLog(`Training started: ${data.num_episodes} episodes`, 'success');
        });
        
        socket.on('episode_start', (data) => {
            document.getElementById('current-episode').textContent = 
                `Episode: ${data.episode}/${data.total_episodes}`;
            addLog(`Episode ${data.episode} started`, 'info');
        });
        
        socket.on('episode_complete', (data) => {
            const episode = data.episode;
            updateChart(rewardChart, `Ep ${episode}`, data.total_reward);
            updateChart(returnChart, `Ep ${episode}`, data.return_pct);
            updateChart(epsilonChart, `Ep ${episode}`, data.epsilon);
            
            document.getElementById('episode-reward').textContent = data.total_reward.toFixed(2);
            document.getElementById('episode-return').textContent = `${data.return_pct >= 0 ? '+' : ''}${data.return_pct.toFixed(2)}%`;
            document.getElementById('transitions').textContent = data.num_transitions.toLocaleString();
            
            const progress = (data.episode / data.total_episodes) * 100;
            document.getElementById('progress-fill').style.width = `${progress}%`;
            
            addLog(`Episode ${episode} complete: Reward=${data.total_reward.toFixed(2)}, Return=${data.return_pct.toFixed(2)}%`, 'success');
        });
        
        socket.on('training_step', (data) => {
            updateChart(lossChart, data.step, data.loss);
            document.getElementById('current-loss').textContent = data.loss.toFixed(4);
            document.getElementById('training-steps').textContent = data.total_steps.toLocaleString();
            
            if (data.step % 100 === 0) {
                addLog(`Training step ${data.step}: loss=${data.loss.toFixed(4)}`, 'info');
            }
        });
        
        socket.on('epsilon_update', (data) => {
            document.getElementById('epsilon').textContent = data.epsilon.toFixed(4);
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


@app.route("/")
def index():
    """Serve the monitoring dashboard"""
    return render_template_string(DASHBOARD_HTML)


# Socket.IO Event Handlers - Relay events from external clients to all dashboards
@socketio.on("training_start")
def handle_training_start(data):
    """Relay training_start event"""
    emit("training_start", data, broadcast=True)


@socketio.on("episode_start")
def handle_episode_start(data):
    """Relay episode_start event"""
    emit("episode_start", data, broadcast=True)


@socketio.on("training_step")
def handle_training_step(data):
    """Relay training_step event"""
    emit("training_step", data, broadcast=True)


@socketio.on("episode_complete")
def handle_episode_complete(data):
    """Relay episode_complete event"""
    emit("episode_complete", data, broadcast=True)


@socketio.on("epsilon_update")
def handle_epsilon_update(data):
    """Relay epsilon_update event"""
    emit("epsilon_update", data, broadcast=True)


@socketio.on("log")
def handle_log(data):
    """Relay log event"""
    emit("log", data, broadcast=True)


@socketio.on("checkpoint_saved")
def handle_checkpoint_saved(data):
    """Relay checkpoint_saved event"""
    emit("checkpoint_saved", data, broadcast=True)


@socketio.on("training_complete")
def handle_training_complete(data):
    """Relay training_complete event"""
    emit("training_complete", data, broadcast=True)


@socketio.on("connect")
def handle_connect():
    """Handle client connection"""
    print(f"✓ Client connected")


@socketio.on("disconnect")
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
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
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
            host="0.0.0.0",
            port=self.port,
            debug=False,
            use_reloader=False,
            log_output=False,
            allow_unsafe_werkzeug=True,
        )

    def emit(self, event, data):
        """Emit an event to all connected clients"""
        if self.running:
            socketio.emit(event, data, namespace="/")

    def log(self, message, level="info"):
        """Send a log message to the dashboard"""
        self.emit("log", {"message": message, "level": level})

    def training_start(self, num_episodes):
        """Signal training start"""
        self.emit("training_start", {"num_episodes": num_episodes})

    def episode_start(self, episode, total_episodes):
        """Signal episode start"""
        self.emit(
            "episode_start", {"episode": episode, "total_episodes": total_episodes}
        )

    def episode_complete(self, episode, total_episodes, metrics):
        """Signal episode completion with metrics"""
        self.emit(
            "episode_complete",
            {"episode": episode, "total_episodes": total_episodes, **metrics},
        )

    def training_step(self, step, total_steps, loss):
        """Update training step progress"""
        self.emit(
            "training_step", {"step": step, "total_steps": total_steps, "loss": loss}
        )

    def epsilon_update(self, epsilon):
        """Update epsilon value"""
        self.emit("epsilon_update", {"epsilon": epsilon})

    def checkpoint_saved(self, filename):
        """Signal checkpoint saved"""
        self.emit("checkpoint_saved", {"filename": filename})

    def training_complete(self, total_time):
        """Signal training completion"""
        self.emit("training_complete", {"total_time": total_time})

    def error(self, message):
        """Send error message"""
        self.emit("error", {"message": message})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Training Monitor Server")
    parser.add_argument(
        "--port",
        type=int,
        default=5050,
        help="Port to run the monitor server on (default: 5050)",
    )
    parser.add_argument(
        "--no-open", action="store_true", help="Do not automatically open browser"
    )
    args = parser.parse_args()

    monitor = TrainingMonitor(port=args.port, auto_open=not args.no_open)
    monitor.start()

    print(f"Monitor server running on port {args.port}. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping monitor...")
