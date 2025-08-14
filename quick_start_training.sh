#!/bin/bash

# 🚀 QUICK START TRAINING & VALIDATION SCRIPT
# Comprehensive AI/ML model training and validation for Pocket Option trading

echo "🚀 ULTIMATE AI TRADING SYSTEM - QUICK START TRAINING & VALIDATION"
echo "=================================================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python version: $python_version"

# Install required packages
echo ""
echo "📦 Installing required packages..."
pip3 install -r requirements.txt

# Create necessary directories
echo ""
echo "📁 Creating necessary directories..."
mkdir -p /workspace/logs
mkdir -p /workspace/data
mkdir -p /workspace/models
mkdir -p /workspace/backup

# Check if models already exist
if [ -f "/workspace/models/best_model.h5" ]; then
    echo ""
    echo "⚠️  Existing models found. Do you want to retrain?"
    echo "1. Quick validation only (skip training)"
    echo "2. Retrain all models"
    echo "3. Comprehensive training and validation"
    read -p "Enter your choice (1-3): " choice
    
    case $choice in
        1)
            echo "Running quick validation only..."
            python3 comprehensive_training_validation.py --mode quick --skip-training
            ;;
        2)
            echo "Retraining all models..."
            python3 comprehensive_training_validation.py --mode standard --skip-validation
            ;;
        3)
            echo "Running comprehensive training and validation..."
            python3 comprehensive_training_validation.py --mode standard
            ;;
        *)
            echo "Invalid choice. Running comprehensive training and validation..."
            python3 comprehensive_training_validation.py --mode standard
            ;;
    esac
else
    echo ""
    echo "🤖 No existing models found. Starting comprehensive training and validation..."
    echo ""
    echo "Available modes:"
    echo "1. Quick mode (7 days validation, 50 epochs)"
    echo "2. Standard mode (30 days validation, 100 epochs) - RECOMMENDED"
    echo "3. Intensive mode (90 days validation, 200 epochs)"
    echo ""
    read -p "Enter mode (1-3, default 2): " mode_choice
    
    case $mode_choice in
        1)
            echo "Running quick mode..."
            python3 comprehensive_training_validation.py --mode quick
            ;;
        2|"")
            echo "Running standard mode..."
            python3 comprehensive_training_validation.py --mode standard
            ;;
        3)
            echo "Running intensive mode..."
            python3 comprehensive_training_validation.py --mode intensive
            ;;
        *)
            echo "Invalid choice. Running standard mode..."
            python3 comprehensive_training_validation.py --mode standard
            ;;
    esac
fi

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 TRAINING & VALIDATION COMPLETED SUCCESSFULLY!"
    echo "=================================================================="
    echo ""
    echo "✅ Your AI/ML models are ready for live Pocket Option trading!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Start with small position sizes ($10-50)"
    echo "2. Monitor performance closely for the first week"
    echo "3. Gradually increase position sizes if performance is good"
    echo "4. Continue monitoring and retraining as needed"
    echo ""
    echo "🚀 To start trading:"
    echo "   python3 ultimate_telegram_bot_simple.py"
    echo ""
    echo "📊 To monitor performance:"
    echo "   python3 performance_tracker.py"
    echo ""
    echo "📈 To view logs:"
    echo "   tail -f /workspace/logs/trading_system.log"
    echo ""
else
    echo ""
    echo "❌ TRAINING & VALIDATION FAILED!"
    echo "=================================================================="
    echo ""
    echo "Please check the logs for errors:"
    echo "   tail -f /workspace/logs/comprehensive_training_validation_*.log"
    echo ""
    echo "Common issues and solutions:"
    echo "1. Insufficient memory: Reduce batch size or use fewer epochs"
    echo "2. Training errors: Check data quality and model configuration"
    echo "3. Validation failures: Models need more training or better data"
    echo ""
    echo "🔧 To retry with different settings:"
    echo "   python3 comprehensive_training_validation.py --mode quick"
    echo ""
fi

echo "=================================================================="
echo "📚 For more information, check the documentation:"
echo "   - README.md"
echo "   - PRODUCTION_READINESS.md"
echo "   - SYSTEM_ASSESSMENT_REPORT.md"
echo "=================================================================="