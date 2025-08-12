# 🚀 DIGITAL OCEAN VPS DEPLOYMENT GUIDE
## Unified Trading System - 24/7 Real-Time Trading Setup

**Version:** 2.0.0  
**Target:** Digital Ocean Ubuntu VPS  
**Purpose:** 24/7 Automated Binary Options Trading

---

## 📋 PREREQUISITES

### 1. **Digital Ocean Account Setup**
- Create account at [digitalocean.com](https://digitalocean.com)
- Add payment method
- Generate SSH key pair on your local machine

### 2. **Local Machine Requirements**
- SSH client (Terminal/PuTTY)
- SCP/SFTP client (or rsync)
- Your trading system files

---

## 🖥️ STEP 1: CREATE DIGITAL OCEAN DROPLET

### **Droplet Configuration:**
```
Operating System: Ubuntu 24.04 LTS x64
Plan: Basic - $24/month (4GB RAM, 2 vCPUs, 80GB SSD)
Datacenter: Choose closest to your location
Authentication: SSH Key (recommended)
Hostname: trading-system-vps
```

### **Recommended Specifications:**
- **Minimum:** 2GB RAM, 1 vCPU, 50GB storage
- **Recommended:** 4GB RAM, 2 vCPUs, 80GB storage (for optimal performance)
- **Network:** 4TB transfer (sufficient for 24/7 trading)

### **Commands to Create via CLI (Optional):**
```bash
# Install doctl first: https://github.com/digitalocean/doctl
doctl compute droplet create trading-system-vps \
  --size s-2vcpu-4gb \
  --image ubuntu-24-04-x64 \
  --region nyc3 \
  --ssh-keys YOUR_SSH_KEY_ID
```

---

## 🔐 STEP 2: INITIAL VPS SETUP

### **Connect to Your VPS:**
```bash
ssh root@YOUR_VPS_IP
```

### **Update System:**
```bash
apt update && apt upgrade -y
apt install -y curl wget git htop nano screen tmux
```

### **Create Trading User:**
```bash
adduser trader
usermod -aG sudo trader
su - trader
```

### **Configure Firewall:**
```bash
sudo ufw allow ssh
sudo ufw allow 22
sudo ufw enable
```

---

## 📦 STEP 3: TRANSFER TRADING SYSTEM FILES

### **Method 1: Using SCP (from your local machine):**
```bash
# Create archive of your trading system
tar -czf trading-system.tar.gz /path/to/workspace/

# Transfer to VPS
scp trading-system.tar.gz trader@YOUR_VPS_IP:~/

# On VPS, extract files
ssh trader@YOUR_VPS_IP
tar -xzf trading-system.tar.gz
mv workspace trading-system
cd trading-system
```

### **Method 2: Using Git (if you have a repository):**
```bash
git clone https://github.com/yourusername/trading-system.git
cd trading-system
```

### **Method 3: Using rsync (recommended for large transfers):**
```bash
rsync -avz -e ssh /path/to/workspace/ trader@YOUR_VPS_IP:~/trading-system/
```

---

## 🐍 STEP 4: INSTALL DEPENDENCIES

### **Run the Automated Deployment Script:**
```bash
cd ~/trading-system
sudo chmod +x deploy_production.sh
sudo ./deploy_production.sh
```

### **Manual Installation (if script fails):**
```bash
# Install Python and pip
sudo apt install -y python3 python3-pip python3-venv python3-dev

# Install system dependencies for TA-Lib
sudo apt install -y build-essential wget libssl-dev libffi-dev

# Install TA-Lib
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ STEP 5: CONFIGURE ENVIRONMENT

### **Create Environment Variables File:**
```bash
nano .env
```

### **Add Configuration:**
```env
# Telegram Configuration
TELEGRAM_BOT_TOKEN=8226952507:AAGPhIvSNikHOkDFTUAZnjTKQzxR4m9yIAU
TELEGRAM_USER_ID=8093708320

# Pocket Option Configuration
POCKET_OPTION_SSID=42["auth",{"session":"a:4:{s:10:\"session_id\";s:32:\"8ddc70c84462c00f33c4e55cd07348c2\";s:10:\"ip_address\";s:14:\"102.88.110.242\";s:10:\"user_agent\";s:120:\"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.\";s:13:\"last_activity\";i:1750856786;}5273f506ca5eac602df49436664bca19","isDemo":0,"uid":74793694,"platform":2,"isFastHistory":true}]

# Trading Configuration
TRADING_MODE=production
MAX_DAILY_TRADES=20
MIN_ACCURACY=95.0
RISK_PER_TRADE=2.0

# Logging
LOG_LEVEL=INFO
```

### **Set Proper Permissions:**
```bash
chmod 600 .env
mkdir -p logs data models backup
chmod 755 logs data models backup
```

---

## 🚀 STEP 6: CREATE SYSTEMD SERVICE (24/7 Operation)

### **Create Service File:**
```bash
sudo nano /etc/systemd/system/trading-system.service
```

### **Service Configuration:**
```ini
[Unit]
Description=Unified Trading System
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=trader
Group=trader
WorkingDirectory=/home/trader/trading-system
Environment=PATH=/home/trader/trading-system/venv/bin
ExecStart=/home/trader/trading-system/venv/bin/python unified_trading_system.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### **Enable and Start Service:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable trading-system.service
sudo systemctl start trading-system.service
```

### **Check Service Status:**
```bash
sudo systemctl status trading-system.service
sudo journalctl -u trading-system.service -f
```

---

## 📱 STEP 7: TELEGRAM BOT SERVICE (Alternative Method)

### **Create Telegram Bot Service:**
```bash
sudo nano /etc/systemd/system/telegram-bot.service
```

### **Bot Service Configuration:**
```ini
[Unit]
Description=Trading Telegram Bot
After=network.target

[Service]
Type=simple
User=trader
Group=trader
WorkingDirectory=/home/trader/trading-system
Environment=PATH=/home/trader/trading-system/venv/bin
ExecStart=/home/trader/trading-system/venv/bin/python start_telegram_bot.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### **Enable Bot Service:**
```bash
sudo systemctl enable telegram-bot.service
sudo systemctl start telegram-bot.service
```

---

## 🔍 STEP 8: MONITORING AND MAINTENANCE

### **Setup Log Rotation:**
```bash
sudo nano /etc/logrotate.d/trading-system
```

```
/home/trader/trading-system/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 trader trader
}
```

### **Create Monitoring Script:**
```bash
nano ~/monitor.sh
```

```bash
#!/bin/bash
# Trading System Health Monitor

echo "=== Trading System Status ==="
date
echo ""

echo "=== Service Status ==="
sudo systemctl is-active trading-system.service
sudo systemctl is-active telegram-bot.service
echo ""

echo "=== System Resources ==="
free -h
df -h /
echo ""

echo "=== Recent Logs ==="
tail -n 10 ~/trading-system/logs/unified_system.log
echo ""

echo "=== Process Status ==="
pgrep -f "python.*trading" | wc -l
echo "Active Python processes"
```

```bash
chmod +x ~/monitor.sh
```

### **Setup Cron for Regular Monitoring:**
```bash
crontab -e
```

```
# Check system every 5 minutes
*/5 * * * * /home/trader/monitor.sh >> /home/trader/health_check.log 2>&1

# Daily backup at 2 AM
0 2 * * * tar -czf /home/trader/backup/trading-system-$(date +\%Y\%m\%d).tar.gz /home/trader/trading-system/
```

---

## 🛡️ STEP 9: SECURITY HARDENING

### **Configure SSH Security:**
```bash
sudo nano /etc/ssh/sshd_config
```

```
# Disable root login
PermitRootLogin no

# Change default port (optional)
Port 2222

# Disable password authentication
PasswordAuthentication no
PubkeyAuthentication yes
```

```bash
sudo systemctl restart ssh
```

### **Configure Firewall Rules:**
```bash
sudo ufw allow 2222/tcp  # SSH (if you changed port)
sudo ufw allow out 443   # HTTPS outbound
sudo ufw allow out 80    # HTTP outbound
sudo ufw deny in 22      # Block default SSH port
sudo ufw enable
```

### **Install Fail2Ban:**
```bash
sudo apt install fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

---

## 🚦 STEP 10: TESTING AND VALIDATION

### **Test System Components:**
```bash
cd ~/trading-system
source venv/bin/activate

# Test system verification
python verify_system.py

# Test configuration
python -c "from config import *; print('Config loaded successfully')"

# Test Telegram bot
python -c "
from telegram_bot import TradingBot
bot = TradingBot()
print('Bot initialized successfully')
"
```

### **Test Telegram Bot Commands:**
Send these commands to your bot:
- `/start` - Initialize bot
- `/status` - Check system status
- `/signal` - Request trading signal
- `/performance` - View performance metrics

---

## 📊 STEP 11: PRODUCTION MONITORING

### **Real-time Monitoring Commands:**
```bash
# View live logs
sudo journalctl -u trading-system.service -f

# Check system resources
htop

# Monitor network connections
netstat -tulpn | grep python

# Check disk usage
df -h

# View recent signals
tail -f logs/signal_engine.log

# Monitor Telegram bot
tail -f logs/telegram_bot.log
```

### **Performance Monitoring:**
```bash
# CPU and memory usage
ps aux | grep python

# Check database status
sqlite3 data/signals.db "SELECT COUNT(*) FROM signals;"

# Verify model files
ls -la models/

# Check latest trades
sqlite3 data/signals.db "SELECT * FROM signals ORDER BY timestamp DESC LIMIT 10;"
```

---

## ⚡ QUICK COMMANDS REFERENCE

### **Service Management:**
```bash
# Start/Stop/Restart services
sudo systemctl start trading-system.service
sudo systemctl stop trading-system.service
sudo systemctl restart trading-system.service

# View service logs
sudo journalctl -u trading-system.service --since "1 hour ago"

# Check service status
sudo systemctl status trading-system.service
```

### **Manual Operation:**
```bash
cd ~/trading-system
source venv/bin/activate

# Run unified system manually
python unified_trading_system.py

# Run only Telegram bot
python start_telegram_bot.py

# Generate manual signal
python -c "
from signal_engine import SignalEngine
engine = SignalEngine()
signal = engine.generate_signal()
print(signal)
"
```

---

## 🚨 TROUBLESHOOTING

### **Common Issues:**

1. **Service Won't Start:**
   ```bash
   sudo journalctl -u trading-system.service
   # Check for Python import errors or missing dependencies
   ```

2. **Telegram Bot Not Responding:**
   ```bash
   # Check bot token and network connectivity
   curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getMe"
   ```

3. **Model Loading Errors:**
   ```bash
   # Verify model files exist and are accessible
   ls -la models/
   # Test model loading
   python -c "from lstm_model import LSTMTradingModel; m=LSTMTradingModel(); m.load_model()"
   ```

4. **Memory Issues:**
   ```bash
   # Monitor memory usage
   free -h
   # Consider upgrading to larger droplet if needed
   ```

### **Emergency Commands:**
```bash
# Kill all trading processes
pkill -f "python.*trading"

# Restart all services
sudo systemctl restart trading-system.service telegram-bot.service

# Emergency stop
sudo systemctl stop trading-system.service
```

---

## ✅ DEPLOYMENT CHECKLIST

### **Pre-Deployment:**
- [ ] Digital Ocean droplet created and accessible
- [ ] SSH key configured
- [ ] Trading system files transferred
- [ ] Dependencies installed
- [ ] Environment variables configured

### **Post-Deployment:**
- [ ] Services running and enabled
- [ ] Telegram bot responding
- [ ] Signals generating correctly
- [ ] Logs writing properly
- [ ] Monitoring setup complete
- [ ] Security hardening applied

### **Validation:**
- [ ] Send `/start` to Telegram bot
- [ ] Request signal with `/signal`
- [ ] Check system status with `/status`
- [ ] Verify 24/7 operation
- [ ] Confirm accuracy targets met

---

**🚀 Your trading system is now deployed and ready for 24/7 operation on Digital Ocean VPS!**

For support, monitor the logs and use the troubleshooting section above.