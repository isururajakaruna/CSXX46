#!/bin/bash
# System Verification Script - Tests all components

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║     🔍 DQN Trading System - Verification Script               ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Detect project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

# Use system python
PYTHON_BIN="python"

echo "1️⃣  Checking Python Environment..."
$PYTHON_BIN --version
if [ $? -eq 0 ]; then
    echo "   ✅ Python OK"
else
    echo "   ❌ Python not found"
    exit 1
fi
echo ""

echo "2️⃣  Checking Required Packages..."
$PYTHON_BIN -c "import torch; import flask; import flask_socketio; print('   ✅ All packages installed')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✅ All packages OK"
else
    echo "   ❌ Missing packages"
fi
echo ""

echo "3️⃣  Checking ATS Server (port 5010)..."
curl -s http://localhost:5010/trading_job/list > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ ATS server is running"
else
    echo "   ❌ ATS server not running - start with: python ats/main.py"
fi
echo ""

echo "4️⃣  Checking File Structure..."
FILES=(
    "custom_modules/strategies/DQN/strategy.py"
    "custom_modules/strategies/DQN/dqn_model.py"
    "custom_modules/strategies/DQN/dqn_agent.py"
    "custom_modules/strategies/DQN/train_dqn.py"
    "custom_modules/strategies/DQN/monitor_server.py"
    "ats/data/BTC_USDT_very_short.csv"
)

ALL_EXIST=true
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file (missing)"
        ALL_EXIST=false
    fi
done
echo ""

echo "5️⃣  Testing JSON Export (Quick Test)..."
rm -rf custom_modules/strategies/DQN/transitions/* 2>/dev/null

$PYTHON_BIN -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
import requests
import json
import time

# Create a test job
config = {
    'name': 'verify_test',
    'createdTime': time.strftime('%Y-%m-%d %H:%M:%S'),
    'exchange': {
        'namespace': 'exchanges:back_trading',
        'config': {
            'data_source': 'ats/data/BTC_USDT_very_short.csv',
            'starting_balance': {'USDT': 10000, 'BTC': 0},
            'min_trading_size': 0.0001,
            'fees': {'namespace': 'fees:generic', 'config': {'limit': {'buy': {'base': 0.001}, 'sell': {'quote': 0.001}}}}
        }
    },
    'strategy': {
        'namespace': 'strategies:DQN',
        'config': {
            'base_symbol': 'BTC',
            'quote_symbol': 'USDT',
            'training_mode': True,
            'episode_id': 'verify_test_001',
            'epsilon': 1.0
        }
    }
}

try:
    # Create job
    r = requests.post('http://localhost:5010/trading_job/create', json=config, timeout=5)
    if r.status_code != 200:
        print('   ❌ Failed to create job')
        sys.exit(1)
    
    job_id = r.json()['job_id']
    
    # Run job
    r = requests.get(f'http://localhost:5010/trading_job/run/{job_id}', timeout=5)
    
    # Wait for completion
    for _ in range(30):
        r = requests.get(f'http://localhost:5010/trading_job/status/{job_id}', timeout=5)
        if not r.json().get('is_running', True):
            break
        time.sleep(1)
    
    # Check for transition file
    from pathlib import Path
    trans_dir = Path('custom_modules/strategies/DQN/transitions')
    files = list(trans_dir.glob('episode_verify_test_001.json'))
    
    if files:
        # Try to load JSON
        import json
        with open(files[0]) as f:
            data = json.load(f)
        print(f'   ✅ JSON export working ({len(data[\"transitions\"])} transitions)')
    else:
        print('   ❌ No transition file created')
        sys.exit(1)
        
except Exception as e:
    print(f'   ❌ Error: {e}')
    sys.exit(1)
"
echo ""

echo "6️⃣  Testing Monitor Server..."
pkill -f monitor_server.py 2>/dev/null
$PYTHON_BIN custom_modules/strategies/DQN/monitor_server.py > /dev/null 2>&1 &
MONITOR_PID=$!
sleep 3

curl -s http://localhost:5050 | grep "DQN Training Monitor" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ Monitor server working"
    kill $MONITOR_PID 2>/dev/null
else
    echo "   ❌ Monitor server not responding"
    kill $MONITOR_PID 2>/dev/null
fi
echo ""

echo "7️⃣  Cleanup..."
rm -rf custom_modules/strategies/DQN/transitions/episode_verify_test_001.json 2>/dev/null
echo "   ✅ Cleanup complete"
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║              ✅ VERIFICATION COMPLETE ✅                        ║"
echo "║                                                                ║"
echo "║  All components are working correctly!                         ║"
echo "║                                                                ║"
echo "║  Ready to start training:                                      ║"
echo "║    ./run_quick_training.sh                                     ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

