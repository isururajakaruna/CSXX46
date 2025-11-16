#!/bin/bash

# Setup script for DQN Trading System (custom_modules)
# This script creates a conda environment with all dependencies for the RL trading system

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     🚀 DQN Trading System - Environment Setup                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ conda found: $(conda --version)"
echo ""

# Ask for environment name
read -p "Enter conda environment name (default: ats): " ENV_NAME
ENV_NAME=${ENV_NAME:-ats}

echo ""
echo "📦 Creating conda environment: $ENV_NAME"
echo ""

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "⚠️  Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove it and recreate? (y/N): " RECREATE
    if [[ $RECREATE =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        echo "❌ Setup cancelled. Please use a different environment name."
        exit 1
    fi
fi

# Ask for Python version
read -p "Python version (default: 3.12): " PYTHON_VERSION
PYTHON_VERSION=${PYTHON_VERSION:-3.12}

echo ""
echo "Creating conda environment with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

echo ""
echo "✅ Conda environment created successfully!"
echo ""

# Activate environment and install dependencies
echo "📥 Installing dependencies from requirements.txt..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "❌ Error: requirements.txt not found at $REQUIREMENTS_FILE"
    exit 1
fi

# Upgrade pip first to avoid dependency resolution issues
echo "Upgrading pip to latest version..."
conda run -n $ENV_NAME pip install --upgrade pip

# Fix known dependency conflicts
echo "Resolving known dependency conflicts..."
# Update aiohttp to a version compatible with charset-normalizer 3.x
conda run -n $ENV_NAME pip install "aiohttp>=3.9.0" --no-deps
# Update tzdata to be compatible with kombu 5.5.4
conda run -n $ENV_NAME pip install "tzdata>=2025.2" --no-deps

# Install packages using pip in the conda environment
echo "Installing packages from requirements.txt..."
echo "Note: Pip will resolve any remaining dependency conflicts automatically"
conda run -n $ENV_NAME pip install -r "$REQUIREMENTS_FILE" || {
    echo ""
    echo "⚠️  Some packages had conflicts. Attempting to resolve..."
    conda run -n $ENV_NAME pip install -r "$REQUIREMENTS_FILE" --upgrade
}

echo ""
echo "✅ All dependencies installed successfully!"
echo ""

# Ask about PyTorch (if not already in requirements)
echo "════════════════════════════════════════════════════════════════"
echo "🤖 PyTorch Installation"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "The DQN strategy requires PyTorch. Would you like to install it?"
echo ""
echo "Options:"
echo "  1) CPU only (faster download, works everywhere)"
echo "  2) GPU (CUDA) - if you have NVIDIA GPU"
echo "  3) Skip (already installed or will install manually)"
echo ""
read -p "Select option (1/2/3): " PYTORCH_OPTION

case $PYTORCH_OPTION in
    1)
        echo "Installing PyTorch (CPU version)..."
        conda run -n $ENV_NAME pip install torch torchvision torchaudio
        ;;
    2)
        echo "Installing PyTorch (CUDA version)..."
        conda run -n $ENV_NAME pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    3)
        echo "Skipping PyTorch installation"
        ;;
    *)
        echo "Invalid option. Skipping PyTorch installation"
        ;;
esac

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "📊 Monitor Dashboard Dependencies"
echo "════════════════════════════════════════════════════════════════"
echo ""
read -p "Install real-time monitoring dashboard dependencies? (Y/n): " INSTALL_MONITOR
INSTALL_MONITOR=${INSTALL_MONITOR:-Y}

if [[ $INSTALL_MONITOR =~ ^[Yy]$ ]]; then
    MONITOR_REQUIREMENTS="$SCRIPT_DIR/strategies/DQN/requirements_monitor.txt"
    if [ -f "$MONITOR_REQUIREMENTS" ]; then
        echo "Installing monitor dependencies..."
        conda run -n $ENV_NAME pip install -r "$MONITOR_REQUIREMENTS"
        echo "✅ Monitor dependencies installed!"
    else
        echo "⚠️  Monitor requirements file not found, skipping..."
    fi
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                 ✅ SETUP COMPLETE!                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 Environment Summary:"
echo "  Name:           $ENV_NAME"
echo "  Python:         $PYTHON_VERSION"
echo "  Location:       $(conda env list | grep "^$ENV_NAME " | awk '{print $2}')"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "  1. Activate the environment:"
echo "     conda activate $ENV_NAME"
echo ""
echo "  2. Start ATS server (from project root):"
echo "     cd .."
echo "     ./start.sh"
echo ""
echo "  3. Run training (in new terminal):"
echo "     cd custom_modules/custom_scripts"
echo "     ./train_interactive.sh"
echo ""
echo "📚 Documentation:"
echo "  • HOW_TO_USE.md - Complete usage guide"
echo "  • docs/ - All documentation"
echo "  • custom_scripts/README.md - Training scripts guide"
echo ""
echo "Happy Trading! 📈🤖"

