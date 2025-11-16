# 🚀 How to Use - DQN Trading System

Complete guide to get started with the DQN reinforcement learning trading system built on ATS.

---

## ⚡ **Quick Start (5 Steps)**

### **Step 1: Get the Code & Setup**

**📍 Execute from**: Anywhere (then navigate to project root)

#### **Clone ATS from GitHub**
```bash
# Clone the official ATS repository
git clone https://github.com/ats-sys/ats.git
cd ats

# The custom DQN modules are in custom_modules/
```

**Repository**: https://github.com/ats-sys/ats

#### **Run Setup Script**

**📍 Execute from**: `<project-root>/`

```bash
# Run the setup script (creates conda environment with all dependencies)
./setup.sh           # Linux/Mac
# OR
.\setup.bat          # Windows
```

This will:
- ✅ Create a conda environment named `ats`
- ✅ Install all required dependencies
- ✅ Set up MongoDB connections
- ✅ Install optional AI packages for DQN strategy

**After setup completes:**
```bash
# Activate the environment
conda activate ats
```

**📚 Additional Setup Reference:**

For detailed setup instructions and environment configuration (including `.env` file setup, MongoDB configuration, and port settings), refer to the official ATS documentation:

👉 **[ATS Official Setup Guide](https://ats-doc.gitbook.io/v1/getting-started/quickstart)**

This guide covers:
- Environment variables configuration
- MongoDB connection settings
- Port configuration
- Alternative startup methods

---

### **Step 2: Activate Environment**

**📍 Execute from**: `<project-root>/`

```bash
# Make sure you're in the conda environment
conda activate ats
```

---

### **Step 3: Start ATS Server**

**📍 Execute from**: `<project-root>/` (the main `ats/` directory)

```bash
# From project root
./start.sh           # Linux/Mac
# OR
.\start.bat          # Windows
# OR
python ats/main.py   # Direct Python
```

This starts the ATS server on **port 5010**.

---

### **Step 4: Navigate to Custom Scripts**

**📍 Execute from**: `<project-root>/custom_modules/custom_scripts/`

```bash
# From project root, navigate to:
cd custom_modules/custom_scripts
```

---

### **Step 5: Run Interactive Training**

**📍 Execute from**: `<project-root>/custom_modules/custom_scripts/`

```bash
# You should be in: ats/custom_modules/custom_scripts/

# For DQN training:
./train_interactive_DQN.sh

# For PPO training:
./train_interactive_PPO.sh
```

**That's it!** 🎉

---

## 📁 **Project Structure & Paths**

```
ats/                                    ← PROJECT ROOT
├── start.sh / start.bat                ← Run from here
├── setup.sh / setup.bat                ← Run from here
├── README.md
├── requirements.txt
│
├── ats/                                ← Core ATS system
│   └── main.py                         ← Server entry point
│
└── custom_modules/                     ← All DQN custom code
    │
    ├── HOW_TO_USE.md                   ← This file
    │
    ├── custom_scripts/                 ⭐ TRAINING SCRIPTS HERE
    │   ├── train_interactive_DQN.sh    ← DQN training (run from this directory)
    │   ├── train_interactive_PPO.sh    ← PPO training (run from this directory)
    │   ├── train_real_quick.py         ← Quick test script
    │   └── verify_system.sh            ← System verification
    │
    ├── docs/                            ← Documentation
    │   ├── QUICKSTART.md
    │   ├── INTERACTIVE_TRAINING_GUIDE.md
    │   └── [10+ more guides]
    │
    └── strategies/DQN/          ← RL implementation
        ├── strategy.py
        ├── dqn_agent.py
        ├── train_dqn.py
        ├── monitor_server.py
        └── [+ more files]
```

---

## 🎯 **Common Commands (With Directories)**

### **Server Management**

**📍 Directory**: `<project-root>/`

```bash
# Start ATS server (from project root)
cd /path/to/ats                # Go to project root
./start.sh                     # Linux/Mac
# OR
.\start.bat                    # Windows

# Stop server
# Press Ctrl+C in the terminal where it's running
```

---

### **Training**

**📍 Directory**: `<project-root>/custom_modules/custom_scripts/`

```bash
# Navigate to custom_scripts
cd /path/to/ats                           # Go to project root
cd custom_modules/custom_scripts          # Enter scripts directory

# DQN interactive training (recommended)
./train_interactive_DQN.sh

# PPO interactive training
./train_interactive_PPO.sh

# Quick 2-episode DQN test
python train_real_quick.py
```

---

### **Inference (Run Trained Models)**

**📍 Directory**: `<project-root>/custom_modules/strategies/DQN/`

After training, use your models for trading:

#### **Via Web Interface (Recommended)**

```bash
# 1. Start ATS server
cd /path/to/ats
./start.sh

# 2. Edit inference config
cd custom_modules/strategies/DQN
cp inference_config_web.json my_inference.json

# Update model_path in my_inference.json:
# "model_path": "custom_modules/strategies/DQN/saved_models/run_YYYYMMDD_HHMMSS/dqn_final.pth"

# 3. Submit via web UI at http://localhost:3000
# OR via API:
curl -X POST http://localhost:5010/v1/jobs \
  -H "Content-Type: application/json" \
  -d @my_inference.json
```

**📚 Detailed Guides:**
- `custom_modules/strategies/DQN/WEB_INFERENCE_GUIDE.md` - Web interface inference
- `custom_modules/strategies/DQN/INFERENCE_GUIDE.md` - Complete inference guide

**Key Parameters:**
- `training_mode: false` - MUST be false for inference
- `model_path` - Relative path to your `.pth` file
- `epsilon: 0.0` - Greedy policy (no exploration)
- `namespace: "strategies:DQN"` - Points to custom DQN strategy

---

### **Dashboard Demo**

**📍 Directory**: `<project-root>/custom_modules/custom_scripts/`

```bash
# From project root
cd custom_modules/custom_scripts

# Test dashboard with simulated data
./run_dashboard_demo.sh
```

---

### **System Verification**

**📍 Directory**: `<project-root>/custom_modules/custom_scripts/`

```bash
# From project root
cd custom_modules/custom_scripts

# Verify all components
./verify_system.sh
```

---

## 📖 **Documentation Locations**

**📍 All documentation**: `<project-root>/custom_modules/docs/`

```bash
# From project root
cd custom_modules/docs

# View quick start
cat QUICKSTART.md

# View full training guide
cat INTERACTIVE_TRAINING_GUIDE.md

# View dashboard guide
cat DASHBOARD_TRAINING_GUIDE.md
```

### **Documentation Index**

| Document | Location | Purpose |
|----------|----------|---------|
| **QUICKSTART.md** | `custom_modules/docs/` | First-time user guide |
| **INTERACTIVE_TRAINING_GUIDE.md** | `custom_modules/docs/` | Complete training walkthrough |
| **DASHBOARD_TRAINING_GUIDE.md** | `custom_modules/docs/` | Dashboard features & troubleshooting |
| **WEB_INFERENCE_GUIDE.md** | `custom_modules/strategies/DQN/` | Run trained models via web UI |
| **INFERENCE_GUIDE.md** | `custom_modules/strategies/DQN/` | Complete inference documentation |
| **FINAL_PROJECT_SUMMARY.md** | `custom_modules/docs/` | Technical deep dive |
| **Scripts README** | `custom_modules/custom_scripts/` | Scripts documentation |

---

## 🔧 **Setup Instructions**

### **Option 1: DQN System Setup (Recommended for RL Training)**

**📍 Directory**: `<project-root>/custom_modules/`

This setup script is specifically configured for the DQN trading system with all required dependencies.

```bash
# Clone repository
git clone https://github.com/ats-sys/ats.git
cd ats/custom_modules

# Run DQN setup script (interactive)
./setup.sh

# Follow prompts to:
#   - Choose environment name (default: ats)
#   - Select Python version (default: 3.12)
#   - Install PyTorch (CPU or GPU)
#   - Install monitor dependencies

# Activate environment
conda activate ats  # Or your chosen name
```

**What it installs:**
- ✅ All 237 packages from `requirements.txt`
- ✅ PyTorch (CPU or GPU version)
- ✅ Real-time monitoring dashboard dependencies
- ✅ Complete DQN training environment

---

### **Option 2: Base ATS Setup**

**📍 Directory**: `<project-root>/`

For general ATS usage without DQN extensions:

```bash
# Clone repository
git clone https://github.com/ats-sys/ats.git
cd ats

# Run base setup
./setup.sh           # Linux/Mac
# OR
.\setup.bat          # Windows

# Activate environment
conda activate ats
```

---

### **Option 3: Manual Setup**

**📍 Directory**: `<project-root>/custom_modules/`

```bash
# Clone repository
git clone https://github.com/ats-sys/ats.git
cd ats

# Create conda environment
conda create -n ats python=3.12 -y

# Activate environment
conda activate ats

# Install all dependencies
pip install -r custom_modules/requirements.txt

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Install monitor dependencies
pip install -r custom_modules/strategies/DQN/requirements_monitor.txt
```

---

## 🎬 **Complete Example Session**

### **First Time Setup**

```bash
# 1. Clone repository
git clone https://github.com/ats-sys/ats.git
cd ats                          # PROJECT ROOT

# 2. Run setup
./setup.sh                      # Linux/Mac (or setup.bat for Windows)
# Follow prompts to install dependencies

# 3. Activate environment
conda activate ats
```

---

### **Terminal 1: Start Server**

```bash
# Navigate to project (if new terminal)
cd /path/to/ats                 # PROJECT ROOT

# Activate environment
conda activate ats

# Start ATS server
./start.sh
# Server starts on http://localhost:5010
```

### **Terminal 2: Run Training**

```bash
# Navigate to project (if new terminal)
cd /path/to/ats                 # PROJECT ROOT

# Activate environment (if new terminal)
conda activate ats

# Go to training scripts
cd custom_modules/custom_scripts

# Run DQN interactive training
./train_interactive_DQN.sh

# Answer prompts:
# Environment: [Enter]          # Use default (ats)
# Dataset: 1                    # BTC_USDT_very_short
# Episodes: 3                   # Quick test
# [Enter] for all others        # Use defaults

# Start training? Y

# Browser opens: http://localhost:5050
# Watch training in real-time!
```

---

## 💡 **Tips for Directory Navigation**

### **Always Know Where You Are**
```bash
pwd                             # Print current directory
```

### **Quick Navigation**
```bash
# From anywhere, go to project root
cd /path/to/ats

# Then navigate to scripts
cd custom_modules/custom_scripts

# Or combine
cd /path/to/ats/custom_modules/custom_scripts
```

### **Save Your Path**
```bash
# Add to ~/.bashrc or ~/.zshrc
export ATS_ROOT="/path/to/ats"

# Then use
cd $ATS_ROOT
cd $ATS_ROOT/custom_modules/custom_scripts
```

---

## 📊 **Results Location**

**📍 Results saved to**: `<project-root>/custom_modules/strategies/DQN/`

   ```bash
# View saved models
ls -lh custom_modules/strategies/DQN/saved_models/

# View training logs
cat custom_modules/strategies/DQN/training_logs/training_metrics.csv

# View loss progression
cat custom_modules/strategies/DQN/training_logs/training_loss.csv
```

---

## 🆘 **Troubleshooting**

### **"Command not found: ./train_interactive_DQN.sh"**

**Problem**: Wrong directory or permissions

   ```bash
# Check current directory
pwd
# Should be: /path/to/ats/custom_modules/custom_scripts

# If wrong directory
cd /path/to/ats/custom_modules/custom_scripts

# If permissions issue
chmod +x *.sh
```

---

### **"ATS server not running"**

**Problem**: Server not started on port 5010

   ```bash
# From project root
cd /path/to/ats
   ./start.sh
   ```

---

### **"Cannot import module"**

**Problem**: Wrong conda environment

```bash
# Check current environment
conda info --envs

# Activate correct environment
conda activate ats
```

---

### **"Monitor not connecting"**

**Problem**: Old monitor process running

```bash
# Kill old monitors
pkill -f monitor_server.py

# Restart training
cd /path/to/ats/custom_modules/custom_scripts
./train_interactive_DQN.sh
```

---

## 📋 **Path Quick Reference**

| Task | Navigate To | Command |
|------|-------------|---------|
| **Start server** | `<root>/` | `./start.sh` |
| **Run DQN training** | `<root>/custom_modules/custom_scripts/` | `./train_interactive_DQN.sh` |
| **Run PPO training** | `<root>/custom_modules/custom_scripts/` | `./train_interactive_PPO.sh` |
| **Read docs** | `<root>/custom_modules/docs/` | `cat QUICKSTART.md` |
| **View results** | `<root>/custom_modules/strategies/DQN/saved_models/` | `ls -lh` |
| **Edit strategy** | `<root>/custom_modules/strategies/DQN/` | `vim strategy.py` |

---

## 🌐 **Repository Information**

- **GitHub**: https://github.com/ats-sys/ats
- **Documentation**: https://ats-doc.gitbook.io/v1
- **License**: Apache-2.0
- **Contributors**: 
  - Isuru S. Rajakaruna
  - Sunanda Gamage
  - Kasun Imesha Wickramasinghe

---

## 🎓 **Learning Path**

### **Level 1: Beginner**
1. Clone repository from GitHub
2. Run `./setup.sh` to install dependencies
3. Read `custom_modules/docs/QUICKSTART.md`
4. Run `custom_modules/custom_scripts/train_interactive_DQN.sh` or `train_interactive_PPO.sh`
5. Complete first training with default settings

### **Level 2: Intermediate**
1. Read `custom_modules/docs/INTERACTIVE_TRAINING_GUIDE.md`
2. Experiment with different hyperparameters
3. Try different datasets
4. Understand dashboard metrics

### **Level 3: Advanced**
1. Read `custom_modules/docs/FINAL_PROJECT_SUMMARY.md`
2. Modify strategy code in `custom_modules/strategies/DQN/`
3. Create custom indicators
4. Optimize for your use case

---

## ✅ **Summary: Where to Execute What**

```bash
# PROJECT ROOT (ats/)
├─ ./start.sh                                    ← Run server
├─ ./setup.sh                                    ← Initial setup
│
└─ custom_modules/custom_scripts/
   ├─ ./train_interactive_DQN.sh                 ← DQN training ⭐
   ├─ ./train_interactive_PPO.sh                 ← PPO training ⭐
   ├─ python train_real_quick.py                 ← Quick test
   └─ ./verify_system.sh                         ← Verify
```

---

## 🚀 **Ready to Start?**

```bash
# 1. Clone
git clone https://github.com/ats-sys/ats.git
cd ats

# 2. Setup
./setup.sh
conda activate ats

# 3. Start Server (Terminal 1)
./start.sh

# 4. Train (Terminal 2)
cd custom_modules/custom_scripts
./train_interactive_DQN.sh  # or ./train_interactive_PPO.sh
```

**Good luck with your trading!** 📈

---

**Version**: 2.0  
**Last Updated**: October 22, 2025  
**Status**: ✅ Production Ready
**Repository**: https://github.com/ats-sys/ats
