#!/bin/bash

# 🚀 Production Deployment Script for Digital Ocean VPS
# Binary Options Trading Bot with AI Models

echo "🚀 Binary Options Trading Bot - Production Deployment"
echo "======================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_error "Please run as root or with sudo"
    exit 1
fi

print_header "System Updates and Dependencies"

# Update system
print_status "Updating system packages..."
apt update && apt upgrade -y

# Install required system packages
print_status "Installing system dependencies..."
apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    curl \
    wget \
    htop \
    screen \
    supervisor \
    nginx \
    sqlite3 \
    pkg-config \
    libta-lib-dev \
    libssl-dev \
    libffi-dev

print_header "Python Environment Setup"

# Create trading user
print_status "Creating trading user..."
useradd -m -s /bin/bash trading || print_warning "User 'trading' already exists"

# Set up project directory
PROJECT_DIR="/home/trading/trading_bot"
print_status "Setting up project directory: $PROJECT_DIR"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Copy project files (assuming this script is run from the project directory)
print_status "Copying project files..."
cp -r /workspace/* $PROJECT_DIR/ 2>/dev/null || print_warning "Could not copy from /workspace, please copy files manually"

# Set ownership
chown -R trading:trading $PROJECT_DIR

print_header "Python Dependencies Installation"

# Switch to trading user for Python setup
sudo -u trading bash << 'EOF'
cd /home/trading/trading_bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install -r requirements_core.txt

# Install additional ML libraries
pip install xgboost lightgbm catboost optuna matplotlib seaborn plotly yfinance ccxt TA-Lib

print_status "Python dependencies installed successfully"
EOF

print_header "System Service Configuration"

# Create systemd service file
cat > /etc/systemd/system/trading-bot.service << 'EOF'
[Unit]
Description=AI Trading Bot
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
Restart=always
RestartSec=10
User=trading
Group=trading
WorkingDirectory=/home/trading/trading_bot
Environment=PATH=/home/trading/trading_bot/venv/bin:/usr/local/bin:/usr/bin:/bin
Environment=PYTHONPATH=/home/trading/trading_bot
ExecStart=/home/trading/trading_bot/venv/bin/python working_telegram_bot.py
StandardOutput=journal
StandardError=journal
SyslogIdentifier=trading-bot

[Install]
WantedBy=multi-user.target
EOF

# Create log rotation configuration
cat > /etc/logrotate.d/trading-bot << 'EOF'
/home/trading/trading_bot/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 trading trading
    postrotate
        systemctl reload trading-bot
    endscript
}
EOF

print_header "Firewall Configuration"

# Configure UFW firewall
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80
ufw allow 443
ufw --force enable

print_header "Monitoring Setup"

# Create monitoring script
cat > $PROJECT_DIR/monitor_system.py << 'EOF'
#!/usr/bin/env python3
"""System monitoring script for the trading bot"""

import psutil
import logging
import time
import os
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/trading/trading_bot/logs/monitor.log'),
        logging.StreamHandler()
    ]
)

def check_system_resources():
    """Check system resources"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    logging.info(f"CPU Usage: {cpu_percent}%")
    logging.info(f"Memory Usage: {memory.percent}%")
    logging.info(f"Disk Usage: {disk.percent}%")
    
    # Alert if resources are high
    if cpu_percent > 80:
        logging.warning(f"High CPU usage: {cpu_percent}%")
    if memory.percent > 80:
        logging.warning(f"High memory usage: {memory.percent}%")
    if disk.percent > 80:
        logging.warning(f"High disk usage: {disk.percent}%")

def check_trading_bot():
    """Check if trading bot is running"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'] and 'working_telegram_bot.py' in ' '.join(proc.info['cmdline']):
                logging.info(f"Trading bot is running (PID: {proc.info['pid']})")
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    logging.error("Trading bot is not running!")
    return False

if __name__ == "__main__":
    while True:
        check_system_resources()
        check_trading_bot()
        time.sleep(300)  # Check every 5 minutes
EOF

chmod +x $PROJECT_DIR/monitor_system.py
chown trading:trading $PROJECT_DIR/monitor_system.py

# Create monitoring service
cat > /etc/systemd/system/trading-monitor.service << 'EOF'
[Unit]
Description=Trading Bot Monitor
After=network.target

[Service]
Type=simple
Restart=always
RestartSec=30
User=trading
Group=trading
WorkingDirectory=/home/trading/trading_bot
Environment=PATH=/home/trading/trading_bot/venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/trading/trading_bot/venv/bin/python monitor_system.py
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

print_header "Backup Configuration"

# Create backup script
cat > $PROJECT_DIR/backup.sh << 'EOF'
#!/bin/bash
# Backup script for trading bot

BACKUP_DIR="/home/trading/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="trading_bot_backup_$DATE.tar.gz"

mkdir -p $BACKUP_DIR

# Create backup
tar -czf $BACKUP_DIR/$BACKUP_FILE \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    /home/trading/trading_bot/

# Keep only last 10 backups
cd $BACKUP_DIR
ls -t | tail -n +11 | xargs -r rm --

echo "Backup created: $BACKUP_FILE"
EOF

chmod +x $PROJECT_DIR/backup.sh
chown trading:trading $PROJECT_DIR/backup.sh

# Add to crontab for trading user
sudo -u trading bash -c 'echo "0 2 * * * /home/trading/trading_bot/backup.sh" | crontab -'

print_header "SSL Certificate Setup (Optional)"

print_warning "SSL certificate setup skipped. Configure manually if needed."

print_header "Final System Configuration"

# Reload systemd and enable services
systemctl daemon-reload
systemctl enable trading-bot
systemctl enable trading-monitor

# Create necessary directories
sudo -u trading mkdir -p $PROJECT_DIR/logs
sudo -u trading mkdir -p $PROJECT_DIR/models
sudo -u trading mkdir -p $PROJECT_DIR/data
sudo -u trading mkdir -p $PROJECT_DIR/backup

# Set proper permissions
chmod 755 $PROJECT_DIR
chmod -R 644 $PROJECT_DIR/*.py
chmod +x $PROJECT_DIR/*.sh

print_header "Deployment Summary"

print_status "✅ System packages installed"
print_status "✅ Python environment configured"
print_status "✅ Dependencies installed"
print_status "✅ System services created"
print_status "✅ Firewall configured"
print_status "✅ Monitoring setup completed"
print_status "✅ Backup system configured"

echo ""
print_header "Next Steps"

echo "1. Configure your credentials in config.py:"
echo "   - TELEGRAM_BOT_TOKEN"
echo "   - TELEGRAM_USER_ID"
echo "   - POCKET_OPTION_SSID"
echo ""
echo "2. Start the services:"
echo "   sudo systemctl start trading-bot"
echo "   sudo systemctl start trading-monitor"
echo ""
echo "3. Check service status:"
echo "   sudo systemctl status trading-bot"
echo "   sudo systemctl status trading-monitor"
echo ""
echo "4. View logs:"
echo "   sudo journalctl -u trading-bot -f"
echo "   sudo journalctl -u trading-monitor -f"
echo ""
echo "5. Test the bot by sending /start to your Telegram bot"

print_status "🎉 Deployment completed successfully!"
print_status "Your trading bot is ready for production use!"

echo ""
echo "📞 Support: Check README.md for troubleshooting"
echo "🔧 Configuration: Edit /home/trading/trading_bot/config.py"
echo "📊 Monitoring: Check /home/trading/trading_bot/logs/"
echo "🔄 Backups: Stored in /home/trading/backups/"