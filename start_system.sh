#!/bin/bash

# Unified Trading System Startup Script
# Ensures proper environment activation and system startup

cd /workspace

# Activate virtual environment
source venv/bin/activate

# Set Python path
export PYTHONPATH="/workspace:$PYTHONPATH"

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the trading system
echo "🚀 Starting Unified Trading System..."
echo "📅 $(date)"
echo "🔧 Virtual Environment: $(which python3)"
echo "📍 Working Directory: $(pwd)"

python3 run_trading_system.py > logs/system_output.log 2>&1 &

echo "✅ Trading System started in background"
echo "📊 Process ID: $!"
echo "📝 Logs: logs/system_output.log"
echo ""
echo "🤖 Your Telegram bot should now respond to commands!"
echo "📱 Send /start to your bot to begin trading"