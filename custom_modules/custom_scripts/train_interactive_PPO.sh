#!/bin/bash
# Interactive PPO Training Launcher
# Asks for all parameters and starts training with live dashboard

clear
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║        🤖 Interactive PPO Training Launcher 🤖                 ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Detect project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"
echo "📁 Project root: $PROJECT_ROOT"

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
PYTHON_PATH="$CONDA_BASE/envs/$CONDA_ENV/bin/python"

# Validate conda environment exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "❌ Error: Conda environment '$CONDA_ENV' not found!"
    echo "   Expected Python at: $PYTHON_PATH"
    exit 1
fi
echo "✅ Using Python: $PYTHON_PATH"

# ============================================================
# 2. SELECT DATASET
# ============================================================
echo ""
echo "📊 Step 2/5: Select Dataset"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Available datasets:"
echo "  1) BTC_USDT_very_short.csv  (Fastest - ~10 seconds per episode)"
echo "  2) BTC_USDT_short.csv       (Medium - ~30 seconds per episode)"
echo "  3) Custom path"
echo ""
read -p "Select dataset [1-3] (default: 1): " DATASET_CHOICE
DATASET_CHOICE=${DATASET_CHOICE:-1}

case $DATASET_CHOICE in
    1)
        DATASET="ats/data/BTC_USDT_very_short.csv"
        ;;
    2)
        DATASET="ats/data/BTC_USDT_short.csv"
        ;;
    3)
        read -p "Enter dataset path: " DATASET
        ;;
    *)
        DATASET="ats/data/BTC_USDT_very_short.csv"
        ;;
esac
echo "✅ Selected: $DATASET"
echo ""

# ============================================================
# 3. TRAINING PARAMETERS
# ============================================================
echo ""
echo "⚙️  Step 3/5: Training Parameters (PPO)"
echo "════════════════════════════════════════════════════════════════"
echo ""

read -p "Number of iterations (default: 10): " NUM_ITERATIONS
NUM_ITERATIONS=${NUM_ITERATIONS:-10}
echo "✅ Iterations: $NUM_ITERATIONS"

# Auto-calculate recommended buffer size from dataset
if [ -f "$DATASET" ]; then
    # Count lines in CSV (subtract 1 for header) and add 20% safety margin
    DATASET_ROWS=$(wc -l < "$DATASET")
    DATASET_ROWS=$((DATASET_ROWS - 1))
    RECOMMENDED_BUFFER=$((DATASET_ROWS + DATASET_ROWS / 5))
    echo ""
    echo "📊 Dataset analysis:"
    echo "   Rows in dataset: $DATASET_ROWS"
    echo "   Recommended buffer size: $RECOMMENDED_BUFFER (with 20% safety margin)"
    echo ""
    read -p "PPO buffer size (default: $RECOMMENDED_BUFFER): " BUFFER_SIZE
    BUFFER_SIZE=${BUFFER_SIZE:-$RECOMMENDED_BUFFER}
else
    echo "⚠️  Could not auto-detect dataset size"
    read -p "PPO buffer size (default: 10240): " BUFFER_SIZE
    BUFFER_SIZE=${BUFFER_SIZE:-10240}
fi

read -p "Batch size (default: 64): " BATCH_SIZE
BATCH_SIZE=${BATCH_SIZE:-64}

read -p "Learning rate (default: 0.0003): " LEARNING_RATE
LEARNING_RATE=${LEARNING_RATE:-0.0003}

read -p "Device [cpu/cuda] (default: cpu): " DEVICE
DEVICE=${DEVICE:-cpu}

echo ""
echo "✅ Configuration:"
echo "   Iterations:      $NUM_ITERATIONS"
echo "   Buffer size:     $BUFFER_SIZE"
echo "   Batch size:      $BATCH_SIZE"
echo "   Learning rate:   $LEARNING_RATE"
echo "   Device:          $DEVICE"
echo ""

# ============================================================
# 4. SAVE DIRECTORY
# ============================================================
echo ""
echo "💾 Step 4/5: Save Directory"
echo "════════════════════════════════════════════════════════════════"
echo ""

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DEFAULT_SAVE_DIR="custom_modules/strategies/PPO/saved_models/run_${TIMESTAMP}"

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
echo "║  Iterations:     $NUM_ITERATIONS"
echo "║  Buffer Size:    $BUFFER_SIZE"
echo "║  Batch Size:     $BATCH_SIZE"
echo "║  Learning Rate:  $LEARNING_RATE"
echo "║  Device:         $DEVICE"
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

# Start monitor in background (PPO monitor)
$PYTHON_PATH custom_modules/strategies/PPO/monitor_server.py > /tmp/monitor_$MONITOR_PORT.log 2>&1 &
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

from custom_modules.strategies.PPO.train_ppo import PPOTrainer
from custom_modules.strategies.PPO.monitor_client import TrainingMonitorClient

# Training configuration
api_base_url = "http://localhost:5010"
data_source = "$DATASET"
save_dir = "$SAVE_DIR"
num_iterations = $NUM_ITERATIONS
buffer_size = $BUFFER_SIZE
batch_size = $BATCH_SIZE
learning_rate = $LEARNING_RATE
device = "$DEVICE"
monitor_port = $MONITOR_PORT

# Connect to monitor
print("🔗 Connecting to dashboard...")
monitor = TrainingMonitorClient(f"http://localhost:{monitor_port}")
monitor.connect()

if not monitor.connected:
    print("⚠️  Dashboard not connected - training will continue without live updates")

# Initialize trainer
print("🤖 Initializing PPO trainer...")
trainer = PPOTrainer(
    api_base_url=api_base_url,
    data_source=data_source,
    save_dir=save_dir,
    monitor=monitor,
    buffer_size=buffer_size,
    batch_size=batch_size,
    learning_rate=learning_rate,
    device=device
)

# Start training
print(f"\\n🎯 Starting {num_iterations}-iteration PPO training...\\n")
print("="*70)

try:
    # PPO iteration-based training
    trainer.train(num_iterations=num_iterations)
    
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
$PYTHON_PATH /tmp/train_session.py

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
echo "To view training outputs:"
echo "  ls -lh $SAVE_DIR"
echo "  cat $SAVE_DIR/training_metrics.csv"
echo ""

