#!/bin/bash

# 🚀 ULTIMATE TRADING SYSTEM - Digital Ocean Auto-Deployment Script
# This script automates the deployment process on Digital Ocean

set -e  # Exit on any error

echo "🚀 ULTIMATE TRADING SYSTEM - Digital Ocean Deployment"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root. Please run as a regular user with sudo privileges."
   exit 1
fi

print_info "Starting deployment process..."

# Step 1: Update system
print_info "Step 1: Updating system packages..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget nano htop screen python3-full python3-pip
sudo apt install -y build-essential cmake redis-server sqlite3
print_status "System packages updated"

# Step 2: Install Python dependencies
print_info "Step 2: Setting up Python environment..."
python3 -m venv trading_env
source trading_env/bin/activate

# Install core packages
pip install --upgrade pip
pip install python-telegram-bot==20.8
pip install pandas numpy scikit-learn
pip install requests websocket-client aiohttp
pip install schedule plotly matplotlib seaborn
pip install psutil pytz joblib
pip install sqlalchemy beautifulsoup4 cryptography
pip install scipy textblob feedparser

print_status "Python environment configured"

# Step 3: Install TA-Lib
print_info "Step 3: Installing TA-Lib..."
cd /tmp
if [ ! -f "ta-lib-0.4.0-src.tar.gz" ]; then
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
fi
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make && sudo make install
sudo ldconfig

# Install Python TA-Lib wrapper
cd $HOME
source trading_env/bin/activate
pip install TA-Lib

print_status "TA-Lib installed successfully"

# Step 4: Create directories
print_info "Step 4: Creating system directories..."
mkdir -p logs data models backup
chmod 755 logs data models backup
print_status "Directories created"

# Step 5: Create systemd service
print_info "Step 5: Creating systemd service..."
sudo tee /etc/systemd/system/trading-bot.service > /dev/null <<EOF
[Unit]
Description=Ultimate Trading System Bot
After=network.target
Wants=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$PWD
Environment=PATH=$PWD/trading_env/bin
ExecStart=$PWD/trading_env/bin/python start_telegram_bot.py
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=10

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=trading-bot

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$PWD

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable trading-bot
print_status "Systemd service created"

# Step 6: Create backup script
print_info "Step 6: Setting up backup system..."
cat > backup_system.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="$HOME/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR/logs_$DATE

# Backup databases
cp -r data $BACKUP_DIR/data_$DATE 2>/dev/null || true

# Backup models
cp -r models $BACKUP_DIR/models_$DATE 2>/dev/null || true

# Backup logs (last 7 days)
find logs -name "*.log" -mtime -7 -exec cp {} $BACKUP_DIR/logs_$DATE/ \; 2>/dev/null || true

# Compress backup
cd $BACKUP_DIR
tar -czf trading_system_backup_$DATE.tar.gz *_$DATE 2>/dev/null
rm -rf *_$DATE

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete 2>/dev/null || true

echo "Backup completed: trading_system_backup_$DATE.tar.gz"
EOF

chmod +x backup_system.sh
print_status "Backup system configured"

# Step 7: Set up firewall
print_info "Step 7: Configuring firewall..."
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable
print_status "Firewall configured"

# Step 8: Install fail2ban
print_info "Step 8: Installing fail2ban..."
sudo apt install -y fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
print_status "Fail2ban installed"

# Step 9: Set up log rotation
print_info "Step 9: Configuring log rotation..."
sudo tee /etc/logrotate.d/trading-bot > /dev/null <<EOF
$PWD/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
}
EOF
print_status "Log rotation configured"

# Step 10: Create environment file template
print_info "Step 10: Creating environment configuration..."
cat > .env.template << 'EOF'
# Telegram Configuration
TELEGRAM_BOT_TOKEN=8226952507:AAGPhIvSNikHOkDFTUAZnjTKQzxR4m9yIAU
TELEGRAM_USER_ID=8093708320

# Pocket Option Configuration  
POCKET_OPTION_EMAIL=your_email@example.com
POCKET_OPTION_PASSWORD=your_secure_password
POCKET_OPTION_SSID=your_session_id

# Production Settings
ENVIRONMENT=production
DEBUG_MODE=false
LOG_LEVEL=INFO

# Risk Management
MAX_DAILY_LOSS_PERCENTAGE=5.0
MAX_RISK_PER_TRADE_PERCENTAGE=1.0
MIN_SIGNAL_ACCURACY=95.0
MIN_AI_CONFIDENCE=90.0
EOF

if [ ! -f ".env" ]; then
    cp .env.template .env
    chmod 600 .env
    print_warning "Please edit .env file with your actual credentials"
fi

print_status "Environment configuration created"

# Step 11: Start the service
print_info "Step 11: Starting trading bot service..."
sudo systemctl start trading-bot
sleep 3
sudo systemctl status trading-bot --no-pager

print_status "Trading bot service started"

echo ""
echo "🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!"
echo "======================================="
echo ""
print_info "Your Ultimate Trading System is now running on Digital Ocean!"
echo ""
print_info "Next Steps:"
echo "1. Edit .env file with your actual credentials: nano .env"
echo "2. Restart the service: sudo systemctl restart trading-bot"
echo "3. Check service status: sudo systemctl status trading-bot"
echo "4. View logs: sudo journalctl -u trading-bot -f"
echo "5. Test your Telegram bot with /start command"
echo ""
print_info "Management Commands:"
echo "• Start:   sudo systemctl start trading-bot"
echo "• Stop:    sudo systemctl stop trading-bot"
echo "• Restart: sudo systemctl restart trading-bot"
echo "• Status:  sudo systemctl status trading-bot"
echo "• Logs:    sudo journalctl -u trading-bot -f"
echo ""
print_info "Backup Command:"
echo "• Manual backup: ./backup_system.sh"
echo ""
print_status "Bot Token: 8226952507:AAGPhIvSNikHOkDFTUAZnjTKQzxR4m9yIAU"
print_status "User ID: 8093708320"
echo ""
print_info "Your bot is ready to receive commands! Send /start to begin."