# 🚀 Production Deployment Guide

## 📋 Pre-Deployment Checklist

### ✅ System Requirements
- **Python**: 3.8+ (tested with 3.13)
- **Memory**: Minimum 4GB RAM, Recommended 8GB+
- **Storage**: 10GB+ free space
- **Network**: Stable internet connection (10Mbps+)
- **OS**: Linux (Ubuntu 20.04+), Windows 10+, macOS 11+

### ✅ Dependencies Verification
```bash
# Run system validation
python3 validate_system.py

# Expected output: "ALL CHECKS PASSED! System is ready to run."
```

## 🔧 Production Setup

### 1. Environment Configuration

Create your production environment file:
```bash
cp .env.example .env
```

Edit `.env` with your actual credentials:
```bash
# Required: Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_actual_bot_token
TELEGRAM_USER_ID=your_actual_user_id

# Required: Pocket Option Configuration
POCKET_OPTION_EMAIL=your_pocket_option_email
POCKET_OPTION_PASSWORD=your_secure_password
POCKET_OPTION_SSID=your_session_id

# Production Settings
ENVIRONMENT=production
DEBUG_MODE=false
LOG_LEVEL=INFO

# Risk Management (Adjust based on your risk tolerance)
MAX_DAILY_LOSS_PERCENTAGE=5.0
MAX_RISK_PER_TRADE_PERCENTAGE=1.0
MIN_SIGNAL_ACCURACY=95.0
MIN_AI_CONFIDENCE=90.0
```

### 2. Security Configuration

```bash
# Set secure file permissions
chmod 600 .env
chmod 700 /workspace/data/
chmod 700 /workspace/logs/

# Optional: Create encrypted backup directory
mkdir -p /workspace/backup/encrypted
chmod 700 /workspace/backup/encrypted
```

### 3. Database Initialization

```bash
# Create production databases
python3 -c "
from config_manager import config_manager
from error_handler import global_error_handler
print('Production databases initialized')
"
```

## 🚀 Deployment Options

### Option 1: Standard Deployment (Recommended)

```bash
# Start the unified system in production mode
python3 unified_trading_system.py --mode hybrid

# Or start original system only
python3 unified_trading_system.py --mode original

# Or start institutional system only
python3 unified_trading_system.py --mode institutional
```

### Option 2: Background Service

Create a systemd service for auto-startup:

```bash
# Create service file
sudo nano /etc/systemd/system/trading-bot.service
```

Service configuration:
```ini
[Unit]
Description=AI Trading Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/workspace
Environment=PATH=/usr/bin:/usr/local/bin:/home/ubuntu/.local/bin
ExecStart=/usr/bin/python3 unified_trading_system.py --mode hybrid
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot

# Check status
sudo systemctl status trading-bot
```

### Option 3: Docker Deployment

Create Dockerfile:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN mkdir -p logs data models backup

EXPOSE 8080
CMD ["python3", "unified_trading_system.py", "--mode", "hybrid"]
```

Build and run:
```bash
docker build -t trading-bot .
docker run -d --name trading-bot -v $(pwd)/data:/app/data trading-bot
```

## 📊 Monitoring & Maintenance

### System Health Monitoring

```bash
# Real-time log monitoring
tail -f /workspace/logs/unified_system.log

# Check error statistics
python3 -c "
from error_handler import global_error_handler
stats = global_error_handler.get_error_statistics(24)
print(f'Errors in last 24h: {stats}')
"

# Performance monitoring
python3 -c "
from performance_tracker import PerformanceTracker
tracker = PerformanceTracker()
stats = tracker.get_daily_stats()
print(f'Daily performance: {stats}')
"
```

### Automated Backup Script

Create `/workspace/scripts/backup.sh`:
```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/workspace/backup"

# Create backup directory
mkdir -p $BACKUP_DIR/$DATE

# Backup databases
cp /workspace/data/*.db $BACKUP_DIR/$DATE/

# Backup logs (last 7 days)
find /workspace/logs -name "*.log" -mtime -7 -exec cp {} $BACKUP_DIR/$DATE/ \;

# Backup configuration (without sensitive data)
python3 -c "
from config_manager import config_manager
import json
config = config_manager.export_config(include_sensitive=False)
with open('$BACKUP_DIR/$DATE/config_backup.json', 'w') as f:
    json.dump(config, f, indent=2)
"

echo "Backup completed: $BACKUP_DIR/$DATE"
```

Make executable and run:
```bash
chmod +x /workspace/scripts/backup.sh
./scripts/backup.sh
```

### Performance Optimization

#### Memory Optimization
```bash
# Monitor memory usage
python3 -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
print(f'Available memory: {psutil.virtual_memory().available / 1024**3:.1f}GB')
"
```

#### CPU Optimization
```bash
# Set CPU affinity for better performance
taskset -c 0-3 python3 unified_trading_system.py
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Reinstall dependencies
pip3 install --user --break-system-packages -r requirements.txt

# Verify imports
python3 -c "import tensorflow, pandas, numpy; print('All imports successful')"
```

#### 2. API Connection Issues
```bash
# Test Telegram bot
python3 -c "
from telegram_bot import TradingBot
bot = TradingBot()
print('Telegram bot initialized successfully')
"

# Test Pocket Option connection
python3 -c "
from pocket_option_api import PocketOptionAPI
api = PocketOptionAPI()
print('Pocket Option API initialized')
"
```

#### 3. Database Issues
```bash
# Check database integrity
python3 -c "
import sqlite3
for db in ['signals.db', 'performance.db', 'monitoring.db', 'risk_management.db']:
    try:
        conn = sqlite3.connect(f'/workspace/data/{db}')
        conn.execute('PRAGMA integrity_check')
        print(f'{db}: OK')
        conn.close()
    except Exception as e:
        print(f'{db}: ERROR - {e}')
"
```

#### 4. Permission Issues
```bash
# Fix permissions
sudo chown -R $USER:$USER /workspace
chmod -R 755 /workspace
chmod 600 /workspace/.env
```

## 📈 Performance Tuning

### Signal Generation Optimization
```bash
# Adjust in .env file
MIN_SIGNAL_ACCURACY=95.0    # Higher = fewer but more accurate signals
MIN_AI_CONFIDENCE=90.0      # Higher = more conservative
MAX_SIGNALS_PER_DAY=15      # Lower = more selective
```

### Risk Management Optimization
```bash
# Conservative settings
MAX_DAILY_LOSS_PERCENTAGE=3.0
MAX_RISK_PER_TRADE_PERCENTAGE=0.5

# Aggressive settings (use with caution)
MAX_DAILY_LOSS_PERCENTAGE=10.0
MAX_RISK_PER_TRADE_PERCENTAGE=2.0
```

## 🔐 Security Best Practices

### 1. Credential Security
- Never commit `.env` to version control
- Use strong, unique passwords
- Rotate API keys regularly
- Enable 2FA on all accounts

### 2. Network Security
```bash
# Use firewall to restrict access
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow from your_ip_address
```

### 3. System Security
```bash
# Regular updates
sudo apt update && sudo apt upgrade -y

# Monitor system access
sudo tail -f /var/log/auth.log
```

## 📊 Production Monitoring

### Key Metrics to Monitor

1. **System Performance**
   - CPU usage < 80%
   - Memory usage < 90%
   - Disk space > 20% free

2. **Trading Performance**
   - Daily win rate > 75%
   - Signal accuracy > 95%
   - Risk per trade < 2%

3. **System Health**
   - Error rate < 1%
   - API response time < 2s
   - Uptime > 99%

### Alerting Setup

Create monitoring script `/workspace/scripts/monitor.py`:
```python
#!/usr/bin/env python3
import psutil
import sys
import time
from error_handler import global_error_handler

def check_system_health():
    # Check memory
    if psutil.virtual_memory().percent > 90:
        print("ALERT: High memory usage")
        return False
    
    # Check disk space
    if psutil.disk_usage('/').percent > 80:
        print("ALERT: Low disk space")
        return False
    
    # Check error rate
    stats = global_error_handler.get_error_statistics(1)
    if stats.get('total_errors', 0) > 10:
        print("ALERT: High error rate")
        return False
    
    return True

if __name__ == "__main__":
    if not check_system_health():
        sys.exit(1)
    print("System health: OK")
```

## 🎯 Production Checklist

Before going live:

- [ ] Environment variables configured
- [ ] All dependencies installed
- [ ] System validation passes
- [ ] Test mode runs successfully
- [ ] Backup system configured
- [ ] Monitoring setup complete
- [ ] Security measures in place
- [ ] Risk management settings reviewed
- [ ] Emergency stop procedures documented

## 🆘 Emergency Procedures

### Immediate Stop
```bash
# Stop the system immediately
pkill -f "unified_trading_system.py"

# Or if running as service
sudo systemctl stop trading-bot
```

### Emergency Backup
```bash
# Quick backup of critical data
tar -czf emergency_backup_$(date +%Y%m%d_%H%M%S).tar.gz \
    /workspace/data/ \
    /workspace/logs/ \
    /workspace/.env
```

### System Recovery
```bash
# Restart from backup
python3 unified_trading_system.py --mode hybrid --recovery

# Or restore from backup
tar -xzf emergency_backup_YYYYMMDD_HHMMSS.tar.gz -C /
```

---

## 📞 Support

For production support:
1. Check logs: `/workspace/logs/`
2. Review error database: `/workspace/data/errors.db`
3. Run diagnostics: `python3 validate_system.py`
4. Check system status: `python3 -c "from unified_trading_system import UnifiedTradingSystem; print('System OK')"`

**Remember**: Always test changes in a non-production environment first!