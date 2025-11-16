# 🤖 Custom Modules - DQN Trading System

This directory contains all custom extensions and additions to the ATS (Automated Trading System), including the DQN reinforcement learning trading strategy and related tools.

---

## 🚀 **Quick Setup**

### **Option 1: Automated Setup (Recommended)**

```bash
# From custom_modules directory
./setup.sh
```

This interactive script will:
- ✅ Create a new conda environment (default: `ats`)
- ✅ Install all required dependencies
- ✅ Optionally install PyTorch (CPU or GPU version)
- ✅ Optionally install monitoring dashboard dependencies

---

### **Option 2: Manual Setup**

```bash
# Create conda environment
conda create -n ats python=3.12 -y

# Activate environment
conda activate ats

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Install monitor dependencies (optional)
pip install -r strategies/DQN/requirements_monitor.txt
```

---

## 📁 **Directory Structure**

```
custom_modules/
├── README.md                          ← This file
├── requirements.txt                   ← All Python dependencies (237 packages)
├── setup.sh                           ← Automated setup script
├── HOW_TO_USE.md                      ← Complete usage guide
│
├── CS5446_group_project/              🔒 Private repository (CS5446 project)
│   └── [Project-specific content]    ← Advanced RL implementations
│
├── fees/                              ← Custom fee structures
│   └── HOW_TO_IMPLEMENT.md
│
├── exchanges/                         ← Custom exchange connectors
│   └── HOW_TO_IMPLEMENT.md
│
├── indicators/                        ← Custom technical indicators
│   └── HOW_TO_IMPLEMENT.md
│
├── strategies/                        ← Custom trading strategies
│   └── DQN/                   ← DQN RL trading strategy
│       ├── strategy.py                ← Main strategy implementation
│       ├── dqn_agent.py               ← DQN agent
│       ├── dqn_model.py               ← Neural network models
│       ├── replay_buffer.py           ← Experience replay buffer
│       ├── monitor_server.py          ← Real-time monitoring server
│       ├── monitor_client.py          ← Monitor client for training
│       ├── train_dqn.py               ← Training logic
│       ├── DQN_README.md              ← DQN documentation
│       ├── requirements_monitor.txt   ← Monitor dependencies
│       ├── training_config.yaml       ← Training hyperparameters
│       └── transitions/               ← Runtime training data
│
├── custom_scripts/                    ← Training & testing scripts
│   ├── README.md
│   ├── train_interactive.sh          ⭐ Main entry point
│   ├── train_real_quick.py
│   ├── continuous_demo.py
│   ├── run_dashboard_demo.sh
│   └── verify_system.sh
│
└── docs/                              ← Documentation
    ├── QUICKSTART.md
    ├── INTERACTIVE_TRAINING_GUIDE.md
    ├── DASHBOARD_TRAINING_GUIDE.md
    ├── FINAL_PROJECT_SUMMARY.md
    └── [10+ more guides]
```

---

## 🔒 **CS5446 Group Project Repository**

This directory contains a private git repository for advanced RL implementations and research:

**Repository**: `https://github.com/isururajakaruna/CS5446_group_project.git`

### **For Team Members:**

#### **Initial Clone** (if not present):
```bash
cd custom_modules
git clone https://github.com/isururajakaruna/CS5446_group_project.git
```

#### **Update to Latest**:
```bash
cd custom_modules/CS5446_group_project
git pull origin main
```

#### **Push Changes**:
```bash
cd custom_modules/CS5446_group_project
git add .
git commit -m "Your commit message"
git push origin main
```

**Note**: This repository is private and requires authentication. Team members need appropriate access permissions.

---

## 📦 **Dependencies**

### **Core Dependencies** (in `requirements.txt`)

- **Python**: 3.12+
- **Flask**: 3.0.0+ (Web framework for ATS server and monitoring)
- **PyTorch**: Latest (Deep learning framework for DQN)
- **NumPy**: Latest (Numerical computing)
- **Pandas**: Latest (Data manipulation)
- **PyMongo**: 4.6.0+ (MongoDB connection)
- **Requests**: 2.31.0+ (HTTP client)
- **PyYAML**: 6.0.1+ (Configuration files)

### **Monitor Dependencies** (in `strategies/DQN/requirements_monitor.txt`)

- **Flask-SocketIO**: Real-time web communication
- **Eventlet**: Concurrent networking
- **Python-SocketIO**: Socket.IO client

**Total**: 237 packages (including all transitive dependencies)

---

## 🎯 **Usage**

### **1. Setup Environment**

```bash
# Navigate to custom_modules
cd custom_modules

# Run setup
./setup.sh

# Follow prompts to configure:
#   - Environment name
#   - Python version
#   - PyTorch installation (CPU/GPU)
#   - Monitor dependencies
```

---

### **2. Start ATS Server**

**📍 From project root** (`ats/`)

```bash
# Activate environment
conda activate ats

# Start server
cd ..  # Go to project root
./start.sh
```

Server runs on: **http://localhost:5010**

---

### **3. Run Training**

**📍 From** `custom_modules/custom_scripts/`

```bash
# Navigate to scripts
cd custom_modules/custom_scripts

# Run interactive training
./train_interactive.sh
```

Dashboard opens at: **http://localhost:5050**

---

## 📚 **Documentation**

| Document | Purpose |
|----------|---------|
| **HOW_TO_USE.md** | Complete usage guide with setup and training |
| **docs/QUICKSTART.md** | Fast 5-minute start guide |
| **docs/INTERACTIVE_TRAINING_GUIDE.md** | Detailed training walkthrough |
| **docs/DASHBOARD_TRAINING_GUIDE.md** | Dashboard features & troubleshooting |
| **docs/FINAL_PROJECT_SUMMARY.md** | Technical deep dive |
| **custom_scripts/README.md** | Training scripts documentation |

---

## 🔧 **Customization Guides**

| Type | Location | Purpose |
|------|----------|---------|
| **Fees** | `fees/HOW_TO_IMPLEMENT.md` | Implement custom fee structures |
| **Exchanges** | `exchanges/HOW_TO_IMPLEMENT.md` | Add new trading platforms |
| **Indicators** | `indicators/HOW_TO_IMPLEMENT.md` | Create technical indicators |
| **Strategies** | `strategies/DQN/` | Example: DQN RL strategy |

---

## 🎓 **Getting Started**

### **New Users (Complete Workflow)**

```bash
# 1. Setup environment (one time)
cd custom_modules
./setup.sh

# 2. Activate environment
conda activate ats

# 3. Read quick start guide
cat HOW_TO_USE.md

# 4. Start ATS server (Terminal 1)
cd ..
./start.sh

# 5. Run training (Terminal 2)
cd custom_modules/custom_scripts
./train_interactive.sh
```

---

### **Existing Users (Quick Start)**

```bash
# Activate environment
conda activate ats

# Start server (Terminal 1)
./start.sh

# Run training (Terminal 2)
cd custom_modules/custom_scripts
./train_interactive.sh
```

---

## 💡 **Key Features**

### **DQN Trading Strategy**

- 🧠 Deep Q-Learning with neural network
- 📊 State: 7 features (price, volume, indicators, wallet)
- 🎯 Actions: BUY, SELL, HOLD
- 💾 Experience replay buffer
- 📈 Real-time training monitoring
- 💰 Automatic model checkpointing

### **Real-time Monitoring**

- 📊 Live training loss charts
- 💰 Episode reward visualization
- 📉 Epsilon decay tracking
- 📝 Streaming logs
- 🌐 Web-based dashboard

### **Extensibility**

- 💰 Custom fee structures
- 🏦 Custom exchanges
- 📊 Custom indicators
- 🤖 Custom strategies

---

## 🔍 **Verification**

Check if everything is set up correctly:

```bash
cd custom_modules/custom_scripts
./verify_system.sh
```

This will check:
- ✅ Python environment
- ✅ Required packages
- ✅ ATS server connectivity
- ✅ File structure
- ✅ Monitor server

---

## 📊 **Requirements File Details**

**File**: `requirements.txt`
- **Size**: 4.7KB
- **Packages**: 237 (including all dependencies)
- **Source**: Exported from working `ats` conda environment
- **Purpose**: Ensures reproducible environment setup

To update requirements (after adding new packages):

```bash
conda activate ats
pip freeze > custom_modules/requirements.txt
```

---

## 🆘 **Troubleshooting**

### **"conda: command not found"**

Install Miniconda:
```bash
# Download from: https://docs.conda.io/en/latest/miniconda.html
# Or use homebrew (macOS):
brew install miniconda
```

### **"Environment already exists"**

Remove and recreate:
```bash
conda env remove -n ats
./setup.sh
```

### **"Package installation failed"**

Try updating pip:
```bash
conda activate ats
pip install --upgrade pip
pip install -r requirements.txt
```

### **PyTorch CUDA issues**

For GPU support, install CUDA-specific version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only (no GPU):
```bash
pip install torch torchvision torchaudio
```

---

## 🔗 **Additional Resources**

- **GitHub**: https://github.com/ats-sys/ats
- **Documentation**: https://ats-doc.gitbook.io/v1
- **Setup Guide**: https://ats-doc.gitbook.io/v1/getting-started/quickstart

---

## 🎯 **Quick Commands Reference**

```bash
# Setup
./setup.sh                              # Initial setup

# Environment
conda activate ats          # Activate
conda deactivate                        # Deactivate
conda env list                          # List environments

# Server
cd .. && ./start.sh                     # Start ATS server

# Training
cd custom_scripts && ./train_interactive.sh    # Interactive training
python train_real_quick.py                      # Quick 2-episode test
./run_dashboard_demo.sh                         # Test dashboard

# Verification
./verify_system.sh                      # Check setup
```

---

## 📝 **Notes**

- 🔒 **Security**: Never commit API keys or secrets
- 💾 **MongoDB**: Required for ATS (install separately)
- 🌐 **Ports**: ATS uses 5010, Monitor uses 5050
- 🐍 **Python**: Tested with Python 3.12
- 📦 **Updates**: Run `pip freeze` to update requirements.txt

---

**Version**: 1.0  
**Last Updated**: October 25, 2025  
**Status**: ✅ Production Ready

**Happy Trading!** 📈🤖

