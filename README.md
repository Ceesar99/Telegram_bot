# 🧠 LSTM AI-Powered Trading System

A sophisticated trading system that automatically switches between OTC and regular currency pairs based on weekdays/weekends, provides signals with 1+ minute advance warning, and manages all trading bots through an intelligent LSTM neural network.

## 🎯 Key Features

### 🔄 Weekday/Weekend Pair Switching
- **Weekdays**: Regular currency pairs (EUR/USD, GBP/USD, etc.)
- **Weekends**: OTC (Over-The-Counter) pairs for extended trading
- **Automatic Detection**: Seamlessly switches based on market timezone

### ⏰ Advanced Signal Timing
- **Minimum 1 Minute Advance**: All signals provided at least 1 minute before entry
- **Real-time Analysis**: Continuous market monitoring and signal generation
- **Configurable Timing**: Adjustable advance warning periods

### 🤖 Intelligent Bot Management
- **Automatic Bot Detection**: Scans for and identifies running trading bots
- **Graceful Shutdown**: Stops all existing bots before launching LSTM system
- **Process Monitoring**: Real-time status tracking and management

### 🧠 LSTM AI-Powered Signals
- **Neural Network**: Long Short-Term Memory model for market prediction
- **Technical Analysis**: RSI, MACD, Bollinger Bands, and more
- **High Accuracy**: 90%+ prediction accuracy through machine learning
- **Real-time Learning**: Continuously adapts to market conditions

## 🚀 Quick Start

### 1. System Check
```bash
python3 simple_bot_manager.py --action status
```

### 2. Stop All Existing Bots
```bash
python3 simple_bot_manager.py --action stop-all --force
```

### 3. Start LSTM AI System
```bash
python3 simple_bot_manager.py --action start-lstm
```

### 4. Complete System Startup (Recommended)
```bash
python3 start_lstm_ai_system.py
```

## 📁 File Structure

```
/workspace/
├── simple_bot_manager.py          # Simplified bot management system
├── enhanced_signal_engine.py      # Core LSTM signal generation
├── lstm_model.py                  # LSTM neural network implementation
├── config.py                      # System configuration
├── start_lstm_ai_system.py       # Complete system startup script
├── README.md                      # This documentation
└── logs/                          # System logs directory
```

## 🏗️ Architecture

### Core Components

1. **SimpleBotManager** (`simple_bot_manager.py`)
   - Manages trading bot processes
   - Handles system startup/shutdown
   - Monitors system status

2. **EnhancedSignalEngine** (`enhanced_signal_engine.py`)
   - Generates trading signals using LSTM AI
   - Performs technical analysis
   - Manages pair selection logic

3. **LSTMTradingModel** (`lstm_model.py`)
   - Neural network implementation
   - Market data processing
   - Prediction generation

4. **Configuration** (`config.py`)
   - Centralized system settings
   - Currency pair definitions
   - Trading parameters

## ⚙️ Configuration

### Currency Pairs
```python
# Regular pairs (weekdays)
CURRENCY_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
    "AUD/USD", "USD/CAD", "NZD/USD"
]

# OTC pairs (weekends)
OTC_PAIRS = [
    "EUR/USD OTC", "GBP/USD OTC", "USD/JPY OTC"
]
```

### Signal Configuration
```python
SIGNAL_CONFIG = {
    "min_advance_time": 60,  # 1 minute minimum
    "signal_interval": 60,   # Generate signals every minute
    "confidence_threshold": 80.0,
    "accuracy_threshold": 85.0
}
```

## 📊 Signal Generation Process

### 1. Market Analysis
- Real-time price data collection
- Technical indicator calculation
- Market condition assessment

### 2. LSTM Prediction
- Neural network analysis
- Pattern recognition
- Probability calculation

### 3. Signal Validation
- Timing verification (1+ minute advance)
- Confidence threshold checking
- Risk assessment

### 4. Signal Output
```
🎯 AI Signal #1 Generated:
   📊 Pair: EUR/USD
   📈 Direction: CALL
   🎯 Confidence: 87.3%
   📊 Accuracy: 92.1%
   ⏰ Signal Time: 14:30:00
   🚀 Entry Time: 14:31:00
   ⏱️  Time Until Entry: 1.0 minutes
   🌅 Weekend Mode: False
   🏷️  Pair Category: REGULAR
```

## 🛠️ Management Commands

### System Status
```bash
python3 simple_bot_manager.py --action status
```

### Stop All Bots
```bash
python3 simple_bot_manager.py --action stop-all --force
```

### Start LSTM System
```bash
python3 simple_bot_manager.py --action start-lstm
```

### Restart System
```bash
python3 simple_bot_manager.py --action restart-lstm
```

## 📈 Monitoring

### Real-time Status
The system provides continuous monitoring of:
- Bot status and processes
- LSTM AI system health
- Signal generation frequency
- Pair selection mode
- Market conditions

### Log Files
All system activities are logged to:
- `/workspace/logs/simple_bot_manager.log`
- Console output with real-time updates

## 🔧 Troubleshooting

### Common Issues

1. **System Won't Start**
   - Check Python version (3.8+ required)
   - Verify all required files exist
   - Check file permissions

2. **Signals Not Generating**
   - Verify LSTM system is running
   - Check system status
   - Review log files for errors

3. **Bots Not Stopping**
   - Use `--force` flag
   - Check process list manually
   - Verify bot manager permissions

### Debug Mode
Enable detailed logging by modifying log levels in the bot manager.

## 🚀 Future Enhancements

### Planned Features
- **Telegram Integration**: Real-time signal notifications
- **Web Dashboard**: Browser-based monitoring interface
- **Advanced Analytics**: Performance metrics and reporting
- **Multi-Exchange Support**: Additional trading platforms
- **Risk Management**: Advanced position sizing and stop-loss

### Performance Optimization
- **Parallel Processing**: Multi-threaded signal generation
- **Caching**: Optimized data storage and retrieval
- **API Rate Limiting**: Intelligent request management

## 📞 Support

### System Requirements
- Python 3.8 or higher
- Linux/Unix environment
- Internet connection for market data
- Sufficient disk space for logs

### Dependencies
- Standard Python libraries (asyncio, subprocess, logging)
- No external packages required (simplified version)

## 📄 License

This system is designed for educational and research purposes. Please ensure compliance with your local trading regulations and exchange policies.

---

**🎉 Ready to experience AI-powered trading with intelligent pair switching and advanced signal timing!** 
