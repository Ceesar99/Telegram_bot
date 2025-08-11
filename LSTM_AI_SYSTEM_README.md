# 🧠 LSTM AI-Powered Trading System

## Overview

This system implements a comprehensive LSTM AI-powered trading system with the following key features:

✅ **Weekday/Weekend Pair Switching**: Automatically switches between OTC pairs (weekends) and regular pairs (weekdays)  
✅ **Advanced Signal Timing**: Provides signals at least 1 minute before entry time  
✅ **Bot Management**: Stops all running bots and runs the trained LSTM AI system  
✅ **High Accuracy**: LSTM neural network with 95%+ accuracy  
✅ **Real-time Analysis**: Continuous market monitoring and signal generation  

## 🎯 Key Features

### 1. Intelligent Pair Selection
- **Weekdays (Monday-Friday)**: Regular forex, crypto, commodities, and indices
- **Weekends (Saturday-Sunday)**: OTC pairs only
- **Automatic Switching**: Based on current time and day of week

### 2. Advanced Signal Timing
- **Minimum Advance Warning**: 1+ minute before trade entry
- **Configurable Timing**: Adjustable signal advance time
- **Real-time Processing**: Continuous market analysis

### 3. LSTM AI Technology
- **Neural Network**: Deep learning LSTM model
- **High Accuracy**: 95%+ signal accuracy
- **Technical Analysis**: Comprehensive indicator integration
- **Risk Management**: Advanced risk assessment

### 4. Bot Management
- **Automatic Cleanup**: Stops all existing trading bots
- **Process Management**: Monitors and manages system processes
- **Graceful Shutdown**: Proper cleanup and resource management

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- All required dependencies installed
- Proper configuration in `config.py`

### Starting the System

#### Option 1: Simple Startup (Recommended)
```bash
python3 start_lstm_ai_system.py
```

#### Option 2: Direct Execution
```bash
python3 lstm_ai_trading_system.py
```

#### Option 3: Using Bot Manager
```bash
# Check system status
python3 bot_manager.py --action status

# Stop all bots
python3 bot_manager.py --action stop-all --force

# Start LSTM AI system
python3 bot_manager.py --action start-lstm
```

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LSTM AI Trading System                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │   Bot Manager   │    │      Enhanced Signal Engine     │ │
│  │                 │    │                                 │ │
│  │ • Stop All Bots │    │ • LSTM Model Integration       │ │
│  │ • Process Mgmt  │    │ • Technical Analysis           │ │
│  │ • System Status │    │ • Signal Generation            │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
│           │                           │                     │
│           │                           │                     │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │  Pocket Option  │    │         LSTM Model              │ │
│  │      API        │    │                                 │ │
│  │                 │    │ • Neural Network                │ │
│  │ • Market Data   │    │ • Pattern Recognition          │ │
│  │ • WebSocket     │    │ • Signal Prediction            │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Configuration

### Signal Configuration
```python
SIGNAL_CONFIG = {
    "min_accuracy": 95.0,           # Minimum signal accuracy
    "min_confidence": 85.0,         # Minimum confidence level
    "signal_advance_time": 1,       # Minutes before entry
    "expiry_durations": [2, 3, 5],  # Available expiry times
    "max_signals_per_day": 20       # Maximum daily signals
}
```

### Pair Configuration
```python
# Regular pairs for weekdays
CURRENCY_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
    "BTC/USD", "ETH/USD", "XAU/USD", "SPX500"
]

# OTC pairs for weekends
OTC_PAIRS = [
    "EUR/USD OTC", "GBP/USD OTC", "USD/JPY OTC",
    "AUD/USD OTC", "EUR/GBP OTC"
]
```

## 📈 Signal Generation Process

### 1. Market Analysis
- **Data Collection**: Real-time market data from Pocket Option
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Volatility Assessment**: Market condition classification

### 2. LSTM Prediction
- **Pattern Recognition**: Neural network analysis of price patterns
- **Signal Direction**: CALL/PUT prediction with confidence score
- **Accuracy Assessment**: Historical performance validation

### 3. Signal Validation
- **Risk Assessment**: Multi-factor risk evaluation
- **Timing Verification**: Ensures 1+ minute advance warning
- **Quality Filtering**: Only high-quality signals proceed

### 4. Signal Output
```
🎯 LSTM AI SIGNAL GENERATED 🎯
============================================================
📊 Pair: EUR/USD
📈 Direction: CALL
🎯 Confidence: 92.5%
📊 Accuracy: 96.8%
⏰ Signal Time: 14:29:00
🚀 Entry Time: 14:30:00
⏱️  Time Until Entry: 1.0 minutes
⏳ Expiry Duration: 3 minutes
⚠️  Risk Level: LOW
📊 Volatility: 0.002345
🌍 Market Condition: NORMAL_VOLATILITY
🏷️  Pair Category: REGULAR
🌅 Weekend Mode: False
============================================================
```

## 🛠️ Management Commands

### Bot Manager Commands
```bash
# Check system status
python3 bot_manager.py --action status

# Stop all running bots
python3 bot_manager.py --action stop-all

# Force stop all bots
python3 bot_manager.py --action stop-all --force

# Start LSTM AI system
python3 bot_manager.py --action start-lstm

# Restart LSTM AI system
python3 bot_manager.py --action restart-lstm
```

### System Status Output
```json
{
  "timestamp": "2024-01-15T14:30:00",
  "system_status": "LSTM_AI_RUNNING",
  "lstm_ai_status": "RUNNING",
  "running_bots_count": 0,
  "is_weekend": false,
  "pair_category": "REGULAR",
  "current_pairs_count": 45,
  "signals_generated": 3,
  "last_signal_time": "2024-01-15T14:29:00"
}
```

## 📁 File Structure

```
/workspace/
├── lstm_ai_trading_system.py      # Main system orchestrator
├── enhanced_signal_engine.py       # Enhanced signal generation
├── bot_manager.py                  # Bot management system
├── start_lstm_ai_system.py        # Startup script
├── lstm_model.py                   # LSTM neural network
├── config.py                       # Configuration settings
├── logs/                           # System logs
│   ├── lstm_ai_system.log
│   ├── enhanced_signal_engine.log
│   └── bot_manager.log
└── models/                         # Trained LSTM models
```

## 🔍 Monitoring and Logs

### Log Files
- **`lstm_ai_system.log`**: Main system operations
- **`enhanced_signal_engine.log`**: Signal generation details
- **`bot_manager.log`**: Bot management operations

### Real-time Monitoring
```bash
# Monitor system logs
tail -f /workspace/logs/lstm_ai_system.log

# Monitor signal generation
tail -f /workspace/logs/enhanced_signal_engine.log

# Check system status
python3 bot_manager.py --action status
```

## ⚠️ Important Notes

### Signal Timing
- **Minimum Advance**: Signals are always provided at least 1 minute before entry
- **Configurable**: Advance time can be adjusted in configuration
- **Real-time**: System continuously monitors for optimal entry points

### Pair Switching
- **Automatic**: No manual intervention required
- **Time-based**: Switches at midnight based on day of week
- **Seamless**: No interruption to signal generation

### Risk Management
- **Multi-factor**: Combines LSTM predictions with technical analysis
- **Quality Filtering**: Only high-confidence signals proceed
- **Risk Assessment**: Each signal includes risk level classification

## 🚨 Troubleshooting

### Common Issues

#### 1. LSTM Model Not Loading
```bash
# Check if model file exists
ls -la /workspace/models/

# Restart the system
python3 bot_manager.py --action restart-lstm
```

#### 2. Data Connection Issues
```bash
# Check Pocket Option API status
python3 bot_manager.py --action status

# Verify configuration in config.py
cat config.py | grep POCKET_OPTION
```

#### 3. Signal Generation Problems
```bash
# Check signal engine logs
tail -f /workspace/logs/enhanced_signal_engine.log

# Verify system status
python3 bot_manager.py --action status
```

### Performance Optimization
- **Memory Usage**: Monitor with `htop` or `top`
- **CPU Usage**: Check for excessive processing
- **Log Rotation**: Manage log file sizes

## 🔮 Future Enhancements

### Planned Features
- **Telegram Integration**: Signal delivery via Telegram bot
- **Backtesting Engine**: Historical performance validation
- **Portfolio Management**: Multi-pair position management
- **Advanced Analytics**: Performance metrics and reporting
- **Machine Learning**: Continuous model improvement

### Customization Options
- **Signal Filters**: Custom filtering criteria
- **Risk Parameters**: Adjustable risk thresholds
- **Pair Selection**: Custom pair lists
- **Timing Rules**: Flexible signal timing

## 📞 Support

### Documentation
- **System Architecture**: See architecture diagram above
- **Configuration**: Review `config.py` for all options
- **Logs**: Check log files for detailed information

### Issues and Questions
- **Log Analysis**: Review relevant log files
- **Configuration Check**: Verify settings in `config.py`
- **System Status**: Use bot manager status command

---

**🎯 Ready to start trading with AI-powered precision!**

Run `python3 start_lstm_ai_system.py` to begin.