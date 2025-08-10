# 📱 Telegram Bot Commands Reference

## 🎯 Trading Commands

### `/start`
**Description**: Initialize the bot and show welcome message  
**Usage**: `/start`  
**Response**: Welcome message with feature overview and command list

### `/signal`
**Description**: Generate and receive an instant high-accuracy trading signal  
**Usage**: `/signal`  
**Response**: Complete trading signal with:
- Currency pair (with OTC support for weekends)
- Direction (BUY/SELL)
- Accuracy percentage (95%+ target)
- Time expiry (1-minute advance notice)
- AI confidence level
- Technical analysis details
- Entry price and risk level

### `/auto_on`
**Description**: Enable automatic signal generation  
**Usage**: `/auto_on`  
**Response**: Confirmation that automatic signals are now enabled  
**Note**: Bot will automatically send signals when high-confidence opportunities are detected

### `/auto_off`
**Description**: Disable automatic signal generation  
**Usage**: `/auto_off`  
**Response**: Confirmation that automatic signals are disabled  
**Note**: Use `/signal` for manual signal requests

### `/pairs`
**Description**: Show all available currency pairs  
**Usage**: `/pairs`  
**Response**: Categorized list of:
- Major forex pairs (EUR/USD, GBP/USD, etc.)
- Minor forex pairs
- Exotic pairs
- OTC pairs (for weekend trading)
- Crypto pairs (BTC/USD, ETH/USD, etc.)
- Commodities (Gold, Silver, Oil)
- Indices (SPX500, NASDAQ, etc.)

### `/market_status`
**Description**: Check current market conditions and trading environment  
**Usage**: `/market_status`  
**Response**: 
- Current time and market session
- Market state (open/closed)
- Overall volatility level
- Signal quality assessment
- Number of active pairs
- Risk level and recommendations

## 📊 Analysis Commands

### `/analyze [pair]`
**Description**: Get detailed technical analysis for a specific currency pair  
**Usage**: `/analyze GBP/USD` or `/analyze BTC/USD`  
**Response**: Comprehensive analysis including:
- Current price and 24h change
- Volatility assessment
- RSI, MACD, Bollinger Bands signals
- Stochastic oscillator reading
- Support and resistance levels
- Price position within range
- Trading recommendation with strength rating

### `/volatility [pair]`
**Description**: Check market volatility levels for a specific pair  
**Usage**: `/volatility EUR/USD`  
**Response**: Volatility metrics and classification (Low/Medium/High)

### `/support_resistance [pair]`
**Description**: Get support and resistance levels for a currency pair  
**Usage**: `/support_resistance GBP/USD`  
**Response**: Key support/resistance levels and current price position

### `/technical [pair]`
**Description**: View technical indicators overview for a pair  
**Usage**: `/technical USD/JPY`  
**Response**: Summary of all technical indicators and their signals

## 📈 Performance Commands

### `/stats`
**Description**: Show comprehensive trading statistics  
**Usage**: `/stats`  
**Response**: Detailed performance metrics including:
- Total signals generated
- Win/loss counts and win rate
- Performance by timeframe (today, week, month)
- Accuracy by expiry duration (2min, 3min, 5min)
- Best performing currency pairs
- Model accuracy and confidence levels
- Target achievement percentage

### `/performance`
**Description**: Get detailed performance report with charts  
**Usage**: `/performance`  
**Response**: 
- Overall win rate and signal accuracy
- Profit factor and Sharpe ratio
- Recent performance (last 30 days)
- Performance by timeframe
- AI model performance metrics
- Risk metrics (drawdown, VaR)
- Performance chart (if available)

### `/history`
**Description**: View signal history and results  
**Usage**: `/history`  
**Response**: Recent signal history with outcomes

### `/win_rate`
**Description**: Show current win rate metrics  
**Usage**: `/win_rate`  
**Response**: Current win rate statistics across different timeframes

## ⚙️ Settings Commands

### `/settings`
**Description**: Access bot configuration options  
**Usage**: `/settings`  
**Response**: Interactive menu with settings categories:
- Signal settings (accuracy thresholds, confidence levels)
- Risk settings (position sizing, loss limits)
- Time settings (trading hours, expiry times)
- Analysis settings (technical indicators, timeframes)
- Notification settings (alerts and messages)
- Backup settings (data backup and recovery)

### `/risk_settings`
**Description**: View and modify risk management parameters  
**Usage**: `/risk_settings`  
**Response**: Current risk management configuration including:
- Max risk per trade (2%)
- Max daily loss (10%)
- Min win rate (75%)
- Stop loss threshold
- Max concurrent trades

### `/alerts_on`
**Description**: Enable notification alerts  
**Usage**: `/alerts_on`  
**Response**: Confirmation that alerts are enabled

### `/alerts_off`
**Description**: Disable notification alerts  
**Usage**: `/alerts_off`  
**Response**: Confirmation that alerts are disabled

## 🔧 System Commands

### `/status`
**Description**: Check bot system health and status  
**Usage**: `/status`  
**Response**: Comprehensive system status including:
- Bot status (active/inactive)
- Auto signals status
- Daily signal count
- Last signal time
- System health (AI model, market data, database, API)
- Performance metrics (response time, memory, CPU usage)
- Configuration summary

### `/health`
**Description**: Perform comprehensive system health check  
**Usage**: `/health`  
**Response**: Detailed health assessment of all system components

### `/backup`
**Description**: Create system backup  
**Usage**: `/backup`  
**Response**: Confirmation of backup creation with file location

### `/restart`
**Description**: Restart bot services  
**Usage**: `/restart`  
**Response**: Confirmation of service restart
**Note**: May cause temporary interruption

## 📚 Help Commands

### `/help`
**Description**: Show comprehensive help guide  
**Usage**: `/help`  
**Response**: Detailed help information including:
- Quick start guide
- Main command overview
- Tips for best results
- Troubleshooting information
- Support resources

### `/commands`
**Description**: List all available commands  
**Usage**: `/commands`  
**Response**: Complete list of bot commands by category

### `/about`
**Description**: Information about the bot  
**Usage**: `/about`  
**Response**: Bot information including:
- Version and features
- AI model details
- Performance targets
- System capabilities

## 🎯 Interactive Features

### Inline Buttons
Many commands provide interactive buttons for:
- **Refresh Signal**: Generate a new signal
- **Analysis**: Get quick analysis of the signal pair
- **Chart**: View price chart (when available)
- **History**: Access signal history
- **Settings**: Navigate to specific settings

### Callback Actions
- Quick analysis requests
- Settings navigation
- Signal refreshing
- Performance chart generation

## 🔄 Automatic Features

### Auto Signal Generation
When enabled with `/auto_on`, the bot automatically:
- Monitors market conditions continuously
- Generates signals when optimal conditions are met
- Sends notifications 1 minute before trade execution
- Respects daily signal limits (max 20 per day)
- Maintains 5-minute minimum intervals between signals

### Signal Quality Filters
Automatic signals are only generated when:
- ✅ Accuracy ≥ 95%
- ✅ AI Confidence ≥ 85%
- ✅ Signal Strength ≥ 7/10
- ✅ Market volatility is optimal
- ✅ Risk conditions are met

## 💡 Usage Tips

### Best Practices
1. **Start with `/start`** to initialize the bot
2. **Use `/signal`** for your first manual signal
3. **Enable auto signals** with `/auto_on` for convenience
4. **Monitor performance** regularly with `/stats`
5. **Check system health** with `/status` if issues arise

### Signal Quality Indicators
- **Accuracy > 95%**: Highest quality signals
- **AI Confidence > 90%**: Strong algorithmic agreement
- **Strength 8-10**: Multiple indicators alignment
- **Low Volatility**: Optimal market conditions

### Weekend Trading
- Bot automatically switches to OTC pairs on weekends
- OTC pairs available: EUR/USD OTC, GBP/USD OTC, USD/JPY OTC, etc.
- Check with `/pairs` to see current available pairs

## 🚨 Important Notes

### Signal Timing
- Signals provide **1-minute advance notice**
- Enter trades within the specified time window
- Expiry times are calculated automatically (2, 3, or 5 minutes)

### Risk Management
- Follow recommended position sizes
- Respect daily loss limits
- Monitor risk scores
- Don't exceed concurrent trade limits

### System Requirements
- Stable internet connection required
- Real-time market data dependency
- AI model requires computation time

## 🆘 Troubleshooting

### Common Issues & Solutions

**Bot not responding:**
- Use `/status` to check system health
- Try `/restart` if needed
- Check internet connection

**No signals generated:**
- Market conditions may not be optimal
- Check `/market_status` for current conditions
- Ensure `/auto_on` is enabled for automatic signals

**Low accuracy:**
- Focus on signals with 95%+ accuracy only
- Avoid high volatility periods
- Follow recommended timeframes

### Support Commands
- `/status` - System diagnostics
- `/health` - Comprehensive health check
- `/help` - Detailed help guide

---

## 🚀 Quick Start Guide

1. **Initialize**: `/start`
2. **Get Signal**: `/signal`
3. **Enable Auto**: `/auto_on`
4. **Check Stats**: `/stats`
5. **Monitor System**: `/status`

**Ready to trade with 95%+ accuracy! 📈🤖**