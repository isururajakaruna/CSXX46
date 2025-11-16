#!/usr/bin/env python3
"""
Quick Real DQN Training - Uses actual RL strategy with minimal dataset
Perfect for testing the full pipeline quickly
"""

import sys
import os
import time
import requests

# Auto-detect project root (assumes this script is in project root or subfolder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR

# If this script is in a subdirectory (e.g., scripts/), walk up until we find 'strategies'
# Alternative: assume it's in project root for simplicity
if not os.path.exists(os.path.join(PROJECT_ROOT, "strategies")):
    # Try one level up
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    if not os.path.exists(os.path.join(PROJECT_ROOT, "strategies")):
        raise RuntimeError(
            "Could not locate project root. Expected 'strategies' directory."
        )

sys.path.insert(0, PROJECT_ROOT)

from strategies.DQN.train_dqn import DQNTrainer
from strategies.DQN.monitor_client import TrainingMonitorClient


def find_data_file(filename="BTC_USDT_very_short.csv"):
    """Find dataset using the same logic as train_interactive.sh"""
    candidate_dirs = [
        os.path.join(PROJECT_ROOT, "data"),
        os.path.join(PROJECT_ROOT, "..", "ats", "data"),
        os.path.join(PROJECT_ROOT, "..", "..", "ats", "data"),
        "/opt/ats/data",
    ]
    for d in candidate_dirs:
        full_path = os.path.join(d, filename)
        if os.path.isfile(full_path):
            return os.path.abspath(full_path)
    return None


def main():
    print("\n" + "=" * 70)
    print("🚀 Quick Real DQN Training")
    print("=" * 70)
    print("\nConfiguration:")
    print("  📊 Dataset: BTC_USDT_very_short.csv (smallest)")
    print("  🔢 Episodes: 2 (quick test)")
    print("  🧠 Model: DQN with experience replay")
    print("  📈 Monitor: http://localhost:5050")
    print("=" * 70 + "\n")

    # Configuration
    api_base_url = "http://localhost:5010"
    num_episodes = 2
    monitor_port = 5050
    save_dir = os.path.join(PROJECT_ROOT, "strategies", "DQN", "saved_models")

    # Locate dataset
    data_source = find_data_file("BTC_USDT_very_short.csv")
    if not data_source:
        print("❌ Could not find BTC_USDT_very_short.csv")
        print("   Please ensure the dataset exists in one of:")
        print(f"     - {os.path.join(PROJECT_ROOT, 'data')}")
        print(f"     - {os.path.join(PROJECT_ROOT, '..', 'ats', 'data')}")
        print(f"     - {os.path.join(PROJECT_ROOT, '..', '..', 'ats', 'data')}")
        return

    print(f"📁 Using dataset: {data_source}")

    # Check ATS server
    print("🔍 Checking ATS server...")
    try:
        response = requests.get(f"{api_base_url}/trading_job/list", timeout=5)
        if response.status_code == 200:
            print("✅ ATS server is running\n")
        else:
            print(f"⚠️  ATS server responded with status {response.status_code}")
            print("   Please make sure the server is running on port 5010\n")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to ATS server: {e}")
        print("\nPlease start the ATS server first:")
        print(f"  cd {PROJECT_ROOT}")
        print("  conda activate ats-nus-project  # or your env name")
        print("  python ats/main.py\n")
        return

    # Connect to monitor
    print("🌐 Connecting to training monitor...")
    monitor = TrainingMonitorClient(f"http://localhost:{monitor_port}")
    monitor.connect()

    if monitor.connected:
        print(f"✅ Connected to monitor at http://localhost:{monitor_port}")
    else:
        print(f"⚠️  Monitor not available - training will continue without dashboard")

    print(f"\n{'='*70}")
    print("📊 OPEN THIS IN YOUR BROWSER:")
    print(f"   👉 http://localhost:{monitor_port} 👈")
    print(f"{'='*70}\n")

    # Wait for user or auto-start
    if sys.stdin.isatty():
        try:
            input("Press ENTER when ready to start training (browser open)...")
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            return
    else:
        print("⏳ Auto-starting in 5 seconds...")
        time.sleep(5)

    # Initialize and train
    print("\n🤖 Initializing DQN trainer...")
    trainer = DQNTrainer(
        api_base_url=api_base_url,
        data_source=data_source,
        save_dir=save_dir,
        monitor=monitor,
    )

    print(f"\n🎯 Starting {num_episodes}-episode training run...\n")
    print("=" * 70)

    try:
        trainer.train(num_episodes=num_episodes)

        print("\n" + "=" * 70)
        print("✅ Training Complete!")
        print("=" * 70)
        print(f"\n📊 Dashboard: http://localhost:{monitor_port}")
        print(f"💾 Models saved in: {save_dir}")
        print("\n" + "=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\n⏸️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if monitor.connected:
            monitor.disconnect()

    print("\n💡 Monitor server still running at http://localhost:5050")
    print("   Kill with: pkill -f monitor_server.py\n")


if __name__ == "__main__":
    main()
