# VPS Deployment Guide for Binary Options Trading Bot

## 🚀 Overview
This guide will help you deploy the Binary Options Trading Bot on a VPS to run 24/7 with automatic restarts and monitoring.

## 📋 Prerequisites
- Ubuntu 20.04+ VPS with at least 2GB RAM and 20GB storage
- Root or sudo access
- Python 3.8+ (already installed on the system)

## 🔧 Installation Steps

### 1. Update System
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-venv python3-pip python3-dev build-essential
```

### 2. Clone/Upload Project
```bash
# If using git
git clone <your-repo-url> /workspace
cd /workspace

# Or upload files via SCP/SFTP to /workspace directory
```

### 3. Setup Virtual Environment
```bash
cd /workspace
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements_compatible.txt
pip install TA-Lib
```

### 4. Create Required Directories
```bash
mkdir -p /workspace/backup /workspace/models/trained /workspace/models/checkpoints
```

### 5. Train Initial AI Models
```bash
source venv/bin/activate
python train_lstm.py --mode quick
```

### 6. Setup Systemd Service
```bash
# Copy service file
sudo cp trading-bot.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable trading-bot.service

# Start the service
sudo systemctl start trading-bot.service
```

## 📱 Bot Configuration

### Telegram Bot Setup
1. Create a bot with @BotFather on Telegram
2. Get your bot token and user ID
3. Update `config.py` with your credentials:
   ```python
   TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
   TELEGRAM_USER_ID = "YOUR_USER_ID"
   ```

### Pocket Option Setup
1. Get your Pocket Option SSID from the browser
2. Update `config.py` with your credentials:
   ```python
   POCKET_OPTION_SSID = 'YOUR_SSID_HERE'
   ```

## 🚀 Starting the Bot

### Manual Start (for testing)
```bash
cd /workspace
source venv/bin/activate
python start_bot.py
```

### Service Start (for production)
```bash
sudo systemctl start trading-bot.service
sudo systemctl status trading-bot.service
```

## 📊 Monitoring and Management

### Check Bot Status
```bash
sudo systemctl status trading-bot.service
```

### View Logs
```bash
# System logs
sudo journalctl -u trading-bot.service -f

# Application logs
tail -f /workspace/logs/*.log
```

### Stop Bot
```bash
# Graceful stop
sudo systemctl stop trading-bot.service

# Emergency stop
touch /workspace/stop_bot
```

### Restart Bot
```bash
sudo systemctl restart trading-bot.service
```

## 🔄 Auto-Restart Features

The bot includes several auto-restart mechanisms:

1. **Systemd Service**: Automatically restarts if the process crashes
2. **Startup Script**: Handles application-level restarts
3. **Health Checks**: Built-in monitoring and recovery

## 📈 Performance Optimization

### Memory Management
- The bot uses ~500MB-1GB RAM
- LSTM models are loaded on-demand
- Automatic garbage collection

### CPU Usage
- AI predictions are CPU-intensive
- Consider using a VPS with good CPU performance
- Training can be done on a separate machine

## 🛡️ Security Considerations

1. **Firewall**: Only allow necessary ports
2. **User Permissions**: Run as non-root user
3. **API Keys**: Keep credentials secure
4. **Updates**: Regularly update dependencies

## 📝 Troubleshooting

### Common Issues

1. **Bot not responding**
   - Check Telegram bot token
   - Verify user ID authorization
   - Check logs for errors

2. **AI model errors**
   - Ensure models are trained
   - Check TensorFlow installation
   - Verify model file paths

3. **Memory issues**
   - Increase VPS RAM
   - Check for memory leaks in logs
   - Restart service

### Log Locations
- System logs: `/var/log/syslog`
- Application logs: `/workspace/logs/`
- Service logs: `sudo journalctl -u trading-bot.service`

## 🔧 Maintenance

### Regular Tasks
1. Monitor logs for errors
2. Check bot performance
3. Update dependencies monthly
4. Backup configuration and models

### Updates
```bash
cd /workspace
git pull origin main
source venv/bin/activate
pip install -r requirements_compatible.txt
sudo systemctl restart trading-bot.service
```

## 📞 Support

If you encounter issues:
1. Check the logs first
2. Verify configuration
3. Test components individually
4. Check system resources

## 🎯 Success Indicators

Your bot is successfully deployed when:
- ✅ Service starts without errors
- ✅ Telegram bot responds to /start command
- ✅ AI models load successfully
- ✅ Signal generation works
- ✅ Logs show normal operation

## 🚀 Next Steps

After successful deployment:
1. Test all bot commands
2. Monitor first few signals
3. Adjust risk parameters if needed
4. Set up monitoring alerts
5. Plan backup strategies

---

**Happy Trading! 🎉**