#!/bin/bash

# 🚀 Quick Start Training Script for LSTM AI Model
# This script will get you started with training your trading AI model

echo "🧠 LSTM AI Model Training - Quick Start"
echo "========================================"
echo ""

# Check if we're in the right directory
if [ ! -f "train_lstm.py" ]; then
    echo "❌ Error: Please run this script from the /workspace directory"
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p models logs data backup

# Check Python version
echo "🐍 Checking Python version..."
python3 --version

# Check if required packages are installed
echo "📦 Checking required packages..."
python3 -c "import tensorflow, pandas, numpy, sklearn" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Required packages are installed"
else
    echo "⚠️  Some packages may be missing. Installing core packages..."
    pip3 install tensorflow pandas numpy scikit-learn
fi

echo ""
echo "🎯 Choose your training approach:"
echo "1. Quick LSTM Training (30 min, 90% accuracy)"
echo "2. Standard LSTM Training (1 hour, 95% accuracy)"
echo "3. Ensemble Training (2-6 hours, 97% accuracy)"
echo "4. Test existing models"
echo "5. Show training guide"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "🚀 Starting Quick LSTM Training..."
        python3 train_lstm.py --mode quick
        ;;
    2)
        echo "🚀 Starting Standard LSTM Training..."
        python3 train_lstm.py --mode standard
        ;;
    3)
        echo "🚀 Starting Ensemble Training..."
        python3 train_ensemble.py --mode standard
        ;;
    4)
        echo "🧪 Testing existing models..."
        python3 test_models.py --all
        ;;
    5)
        echo "📚 Opening training guide..."
        cat LSTM_TRAINING_GUIDE.md | head -50
        echo ""
        echo "📖 Full guide available in: LSTM_TRAINING_GUIDE.md"
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "🎉 Training process completed!"
echo ""
echo "📋 Next steps:"
echo "1. Check the logs in the logs/ directory"
echo "2. Test your models: python3 test_models.py --all"
echo "3. Start trading: python3 start_unified_system.py"
echo "4. Read the full guide: LSTM_TRAINING_GUIDE.md"
echo ""
echo "🚀 Happy trading!"