#!/bin/bash
# Installation script for Real-time Training Monitor

echo "======================================"
echo "🌐 Training Monitor Setup"
echo "======================================"
echo ""

# Check if conda environment is active
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "⚠️  No conda environment active!"
    echo "Please run: conda activate ats"
    exit 1
fi

echo "✓ Conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements_monitor.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "✅ Installation Complete!"
    echo "======================================"
    echo ""
    echo "📚 Next steps:"
    echo "  1. Start ATS server (in separate terminal):"
    echo "     python ats/main.py"
    echo ""
    echo "  2. Start training with monitor:"
    echo "     python train_dqn_with_monitor.py"
    echo ""
    echo "  3. Dashboard will open at:"
    echo "     http://localhost:5050"
    echo ""
    echo "📖 For more info, see:"
    echo "     MONITOR_GUIDE.md"
    echo ""
else
    echo ""
    echo "======================================"
    echo "❌ Installation Failed"
    echo "======================================"
    echo ""
    echo "Please check error messages above."
    exit 1
fi

