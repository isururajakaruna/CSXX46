#!/usr/bin/env python3
"""
Quick analysis of your training results
"""
import sys
import os
from pathlib import Path
import json

# Detect project root dynamically
script_dir = Path(__file__).parent
project_root = script_dir.parent  # Go up from custom_modules to project root

# Find the most recent training run
saved_models_dir = project_root / 'custom_modules' / 'strategies' / 'DQN' / 'saved_models'

# Ensure the directory exists
if not saved_models_dir.exists():
    print(f"❌ Saved models directory not found: {saved_models_dir}")
    print("   Please run a training session first!")
    sys.exit(1)

print("=" * 70)
print("📊 Training Results Analysis")
print("=" * 70)
print()

# List all training runs
runs = sorted([d for d in saved_models_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)

if not runs:
    print("❌ No training runs found!")
    sys.exit(1)

latest_run = runs[0]
print(f"📁 Analyzing latest run: {latest_run.name}")
print(f"   Location: {latest_run}")
print()

# Check what files exist
print("📂 Files generated:")
files = list(latest_run.glob('*'))
for f in sorted(files):
    size = f.stat().st_size
    if size < 1024:
        size_str = f"{size} B"
    elif size < 1024*1024:
        size_str = f"{size/1024:.1f} KB"
    else:
        size_str = f"{size/(1024*1024):.1f} MB"
    print(f"   ✅ {f.name:30s} ({size_str})")
print()

# Load and display metrics
metrics_file = latest_run / 'training_metrics.csv'
if metrics_file.exists():
    print("📈 Training Metrics:")
    with open(metrics_file) as f:
        lines = f.readlines()
        if len(lines) > 1:
            # Has data
            print(f"   Episodes completed: {len(lines) - 1}")
            print()
            print("   Last few episodes:")
            for line in lines[-6:]:
                print(f"   {line.strip()}")
        else:
            print("   ⚠️  No metrics recorded yet")
    print()
else:
    print("⚠️  training_metrics.csv not found")
    print()

# Load and display loss
loss_file = latest_run / 'training_loss.csv'
if loss_file.exists():
    print("📉 Training Loss:")
    with open(loss_file) as f:
        lines = f.readlines()
        if len(lines) > 1:
            print(f"   Training steps: {len(lines) - 1}")
            print()
            print("   Last few losses:")
            for line in lines[-6:]:
                print(f"   {line.strip()}")
        else:
            print("   ⚠️  No loss data recorded yet")
    print()
else:
    print("⚠️  training_loss.csv not found")
    print()

# Check for model file
model_file = latest_run / 'dqn_final.pth'
if model_file.exists():
    print("🤖 Model Status:")
    print(f"   ✅ Model saved: dqn_final.pth ({model_file.stat().st_size / 1024:.1f} KB)")
    
    # Try to load model info
    try:
        import torch
        checkpoint = torch.load(model_file, map_location='cpu')
        print(f"   📊 Model info:")
        for key in checkpoint.keys():
            if key not in ['model_state_dict', 'target_model_state_dict', 'optimizer_state_dict']:
                print(f"      {key}: {checkpoint[key]}")
    except Exception as e:
        print(f"   ⚠️  Could not load model details: {e}")
    print()
else:
    print("⚠️  No model file found")
    print()

# Load final stats if available
stats_files = list(latest_run.glob('training_stats_ep*.json'))
if stats_files:
    latest_stats = sorted(stats_files)[-1]
    print(f"📊 Final Episode Stats ({latest_stats.name}):")
    with open(latest_stats) as f:
        stats = json.load(f)
        for key, value in stats.items():
            if isinstance(value, list):
                print(f"   {key}: {len(value)} items")
            else:
                print(f"   {key}: {value}")
    print()

print("=" * 70)
print("✅ Analysis Complete!")
print()
print("📚 Next steps:")
print("   1. Read: POST_TRAINING_GUIDE.md")
print("   2. Evaluate your model performance")
print("   3. Train more episodes if needed")
print("   4. Deploy to testing/paper trading")
print()
print("🔄 Train again:")
print("   cd custom_scripts && ./train_interactive.sh")
print("=" * 70)


