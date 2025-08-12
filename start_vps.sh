#!/bin/bash
# VPS Startup Script for Binary Options Trading Bot
# This script ensures the bot runs 24/7 with automatic restarts

echo "🚀 Starting Binary Options Trading Bot on VPS..."

# Set working directory
cd /workspace

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH=/workspace
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging

# Function to start the bot
start_bot() {
    echo "📱 Starting Telegram Bot..."
    python start_bot.py
}

# Function to restart the bot
restart_bot() {
    echo "🔄 Restarting bot in 10 seconds..."
    sleep 10
    start_bot
}

# Main loop to keep the bot running
while true; do
    echo "🕐 $(date): Starting bot..."
    
    # Start the bot
    start_bot
    
    # If we get here, the bot has stopped
    echo "⚠️  Bot stopped at $(date)"
    
    # Check if we should exit (e.g., for maintenance)
    if [ -f "/workspace/stop_bot" ]; then
        echo "🛑 Stop signal detected. Exiting..."
        rm -f /workspace/stop_bot
        break
    fi
    
    # Restart the bot
    restart_bot
done

echo "👋 Bot startup script finished."