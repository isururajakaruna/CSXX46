#!/bin/bash
# Interactive DQN Training Launcher
# Asks for all parameters and starts training with live dashboard

clear
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║        🤖 Interactive DQN Training Launcher 🤖                 ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Navigate to the project root (go up 2 levels: custom_scripts -> custom_modules -> ats)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
echo "📁 Project root detected as: $PROJECT_ROOT"

# Change to project directory
cd "$PROJECT_ROOT"

# ============================================================
# 1. SELECT CONDA ENVIRONMENT
# ============================================================
echo "📦 Step 1/5: Select Conda Environment"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Available conda environments:"
conda env list | grep -v "^#" | nl
echo ""
read -p "Enter environment name (or press Enter for 'ats-nus-project'): " CONDA_ENV
CONDA_ENV=${CONDA_ENV:-ats-nus-project}
echo "✅ Selected: $CONDA_ENV"
echo ""

# Get conda base and Python path
CONDA_BASE=$(conda info --base)
PYTHON_BIN="$CONDA_BASE/envs/$CONDA_ENV/bin/python"

# Validate conda environment exists
if [ ! -f "$PYTHON_BIN" ]; then
    echo "❌ Error: Conda environment '$CONDA_ENV' not found!"
    echo "   Expected Python at: $PYTHON_BIN"
    exit 1
fi
echo "✅ Using Python: $PYTHON_BIN"

# ============================================================
# 2. SELECT DATASET (with auto-detection)
# ============================================================
echo ""
echo "📊 Step 2/5: Select Dataset"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Candidate data directories (in order of preference)
CANDIDATE_DIRS=(
    "$PROJECT_ROOT/ats/data"                 # Inside project ats/data
    "$PROJECT_ROOT/data"                     # Inside project root data
    "$PROJECT_ROOT/custom_modules/data"      # Inside custom_modules
    "/opt/ats/data"                          # System-wide (optional)
)

# Auto-find first valid data dir with required files
DATA_ROOT=""
for dir in "${CANDIDATE_DIRS[@]}"; do
    if [ -d "$dir" ] && [ -f "$dir/BTC_USDT_very_short.csv" ]; then
        DATA_ROOT="$dir"
        break
    fi
done

if [ -z "$DATA_ROOT" ]; then
    echo "⚠️  No default dataset directory found."
    echo "   Please place BTC_USDT_*.csv files in one of:"
    for d in "${CANDIDATE_DIRS[@]}"; do
        echo "     - $d"
    done
    echo ""
    DEFAULT_DATA_DIR=""
else
    echo "📁 Auto-detected data directory: $DATA_ROOT"
    DEFAULT_DATA_DIR="$DATA_ROOT"
fi

echo ""
echo "Available datasets:"
echo "  1) BTC_USDT_very_short.csv  (Fastest - ~10s/episode)"
echo "  2) BTC_USDT_short.csv       (Medium - ~30s/episode)"
echo "  3) Custom path"
echo ""

read -p "Select dataset [1-3] (default: 1): " DATASET_CHOICE
DATASET_CHOICE=${DATASET_CHOICE:-1}

case $DATASET_CHOICE in
    1)
        if [ -n "$DEFAULT_DATA_DIR" ]; then
            DATASET="$DEFAULT_DATA_DIR/BTC_USDT_very_short.csv"
        else
            echo "❌ No default data directory found. Please choose option 3."
            DATASET_CHOICE=3
        fi
        ;;
    2)
        if [ -n "$DEFAULT_DATA_DIR" ]; then
            DATASET="$DEFAULT_DATA_DIR/BTC_USDT_short.csv"
        else
            echo "❌ No default data directory found. Please choose option 3."
            DATASET_CHOICE=3
        fi
        ;;
    3)
        read -p "Enter dataset path: " DATASET
        if command -v realpath >/dev/null 2>&1; then
            DATASET="$(realpath "$DATASET")"
        fi
        ;;
    *)
        if [ -n "$DEFAULT_DATA_DIR" ]; then
            DATASET="$DEFAULT_DATA_DIR/BTC_USDT_very_short.csv"
        else
            echo "❌ Invalid choice and no default data found."
            exit 1
        fi
        ;;
esac

# Final validation
if [ ! -f "$DATASET" ]; then
    echo "⚠️  Dataset not found at: $DATASET"
    read -p "Enter correct dataset path: " DATASET
    if [ ! -f "$DATASET" ]; then
        echo "❌ Error: Dataset not found at: $DATASET"
        exit 1
    fi
fi

echo "✅ Selected: $DATASET"
echo ""

# ============================================================
# 3. TRAINING PARAMETERS
# ============================================================
echo ""
echo "⚙️  Step 3/5: Training Parameters"
echo "════════════════════════════════════════════════════════════════"
echo ""

read -p "Number of episodes (default: 5): " NUM_EPISODES
NUM_EPISODES=${NUM_EPISODES:-5}

read -p "Learning rate (default: 0.00001): " LEARNING_RATE
LEARNING_RATE=${LEARNING_RATE:-0.00001} #(3/11/25 reduced from 0.0001)

read -p "Batch size (default: 128): " BATCH_SIZE
BATCH_SIZE=${BATCH_SIZE:-128} #(3/11/25 increased from 64)

read -p "Epsilon start (default: 1.0): " EPSILON_START
EPSILON_START=${EPSILON_START:-1.0}

read -p "Epsilon decay (default: 0.995): " EPSILON_DECAY
EPSILON_DECAY=${EPSILON_DECAY:-0.995}

echo ""
echo "✅ Configuration:"
echo "   Episodes: $NUM_EPISODES"
echo "   Learning Rate: $LEARNING_RATE"
echo "   Batch Size: $BATCH_SIZE"
echo "   Epsilon: $EPSILON_START (decay: $EPSILON_DECAY)"
echo ""

# ============================================================
# 4. SAVE DIRECTORY
# ============================================================
echo ""
echo "💾 Step 4/5: Save Directory"
echo "════════════════════════════════════════════════════════════════"
echo ""

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DEFAULT_SAVE_DIR="$PROJECT_ROOT/strategies/DQN/saved_models/run_${TIMESTAMP}"

read -p "Save directory (default: $DEFAULT_SAVE_DIR): " SAVE_DIR
SAVE_DIR=${SAVE_DIR:-$DEFAULT_SAVE_DIR}

mkdir -p "$SAVE_DIR"
echo "✅ Models will be saved to: $SAVE_DIR"
echo ""

# ============================================================
# 5. MONITOR PORT
# ============================================================
echo ""
echo "🌐 Step 5/5: Dashboard Settings"
echo "════════════════════════════════════════════════════════════════"
echo ""

read -p "Monitor port (default: 5050): " MONITOR_PORT
MONITOR_PORT=${MONITOR_PORT:-5050}

echo "✅ Dashboard will run at: http://localhost:$MONITOR_PORT"

# Kill any existing process on this port
echo "🔍 Checking port $MONITOR_PORT..."
PID_ON_PORT=$(lsof -ti:$MONITOR_PORT 2>/dev/null)
if [ ! -z "$PID_ON_PORT" ]; then
    echo "⚠️  Port $MONITOR_PORT is in use (PID: $PID_ON_PORT)"
    echo "🔪 Killing existing process..."
    kill -9 $PID_ON_PORT 2>/dev/null
    sleep 1
    echo "✅ Port $MONITOR_PORT is now free"
else
    echo "✅ Port $MONITOR_PORT is available"
fi
echo ""

# ============================================================
# SUMMARY & CONFIRMATION
# ============================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                  📋 TRAINING SUMMARY                           ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║                                                                ║"
echo "║  Environment:    $CONDA_ENV"
echo "║  Dataset:        $DATASET"
echo "║  Episodes:       $NUM_EPISODES"
echo "║  Learning Rate:  $LEARNING_RATE"
echo "║  Batch Size:     $BATCH_SIZE"
echo "║  Epsilon:        $EPSILON_START → decay $EPSILON_DECAY"
echo "║  Save Dir:       $SAVE_DIR"
echo "║  Dashboard:      http://localhost:$MONITOR_PORT"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

read -p "Start training? [Y/n]: " CONFIRM
CONFIRM=${CONFIRM:-Y}

if [[ ! $CONFIRM =~ ^[Yy]$ ]]; then
    echo "❌ Training cancelled"
    exit 0
fi

# ============================================================
# CHECK ATS SERVER
# ============================================================
echo ""
echo "🔍 Checking ATS server..."
curl -s http://localhost:5010/trading_job/list > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ ATS server not running on port 5010!"
    echo ""
    echo "Please start the ATS server first:"
    echo "  conda activate $CONDA_ENV"
    echo "  python ats/main.py"
    echo ""
    exit 1
fi
echo "✅ ATS server is running"

# ============================================================
# START MONITOR SERVER
# ============================================================
echo ""
echo "🌐 Starting monitor server..."

# Kill any existing monitor
pkill -f monitor_server.py 2>/dev/null
sleep 1

# Start monitor in background
$PYTHON_BIN custom_modules/strategies/DQN/monitor_server.py --port "$MONITOR_PORT" > /tmp/monitor_$MONITOR_PORT.log 2>&1 &
MONITOR_PID=$!

# Wait for monitor to start
sleep 3

# Check if monitor is running
curl -s http://localhost:$MONITOR_PORT > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Failed to start monitor server"
    echo "Check logs: tail /tmp/monitor_$MONITOR_PORT.log"
    exit 1
fi

echo "✅ Monitor server started (PID: $MONITOR_PID)"

# Open browser
echo "🌐 Opening dashboard in browser..."
sleep 1
open "http://localhost:$MONITOR_PORT" 2>/dev/null || \
xdg-open "http://localhost:$MONITOR_PORT" 2>/dev/null || \
echo "   👉 Manually open: http://localhost:$MONITOR_PORT"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║              🚀 STARTING TRAINING! 🚀                          ║"
echo "║                                                                ║"
echo "║  Dashboard: http://localhost:$MONITOR_PORT                     ║"
echo "║  Press Ctrl+C to stop training                                 ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

sleep 3

# ============================================================
# RUN TRAINING
# ============================================================

# Create temporary Python training script with parameters
cat > /tmp/train_session.py << EOF
import sys
import os
sys.path.insert(0, '$PROJECT_ROOT')

from custom_modules.strategies.DQN.train_dqn import DQNTrainer
from custom_modules.strategies.DQN.monitor_client import TrainingMonitorClient

# Training configuration
api_base_url = "http://localhost:5010"
data_source = "$DATASET"
save_dir = "$SAVE_DIR"
num_episodes = $NUM_EPISODES
learning_rate = $LEARNING_RATE
batch_size = $BATCH_SIZE
epsilon_start = $EPSILON_START
epsilon_decay = $EPSILON_DECAY
monitor_port = $MONITOR_PORT

# Connect to monitor
print("🔗 Connecting to dashboard...")
monitor = TrainingMonitorClient(f"http://localhost:{monitor_port}")
monitor.connect()
print(f"🔗 Monitor connected: {monitor.connected}")

if not monitor.connected:
    print("⚠️  Dashboard not connected - training will continue without live updates")

# Initialize trainer
print("🤖 Initializing DQN trainer...")
trainer = DQNTrainer(
    api_base_url=api_base_url,
    data_source=data_source,
    save_dir=save_dir,
    monitor=monitor,
    learning_rate=learning_rate,
    batch_size=batch_size,
    epsilon_start=epsilon_start,
    epsilon_decay=epsilon_decay
)

# Start training
print(f"\\n🎯 Starting {num_episodes}-episode training...\\n")
print("="*70)

try:
    trainer.train(num_episodes=num_episodes)
    
    print("\\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"\\n📊 Dashboard: http://localhost:{monitor_port}")
    print(f"💾 Models saved in: {save_dir}/")
    print("\\n" + "="*70 + "\\n")
    
except KeyboardInterrupt:
    print("\\n\\n⏸️  Training interrupted by user")
except Exception as e:
    print(f"\\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    if monitor.connected:
        monitor.disconnect()

print("\\n💡 Monitor server still running at http://localhost:$MONITOR_PORT")
print("   Kill with: pkill -f monitor_server.py\\n")
EOF

# Run the training
$PYTHON_BIN /tmp/train_session.py

# ============================================================
# CLEANUP & FINAL MESSAGE
# ============================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║                  ✅ TRAINING SESSION COMPLETE                  ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Results saved to: $SAVE_DIR"
echo "📊 Dashboard still running at: http://localhost:$MONITOR_PORT"
echo ""
echo "To stop the monitor server:"
echo "  pkill -f monitor_server.py"
echo ""
echo "To view training logs:"
echo "  ls -lh $SAVE_DIR"
echo "  cat $SAVE_DIR/../training_logs/training_metrics.csv"
echo ""