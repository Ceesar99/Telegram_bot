import os
from datetime import datetime
import pytz

# Telegram Bot Configuration (load from environment)
import os as _os
TELEGRAM_BOT_TOKEN = _os.getenv("TELEGRAM_BOT_TOKEN", "1234567890:ABCdefGHIjklMNOpqrsTUVwxyz")  # Placeholder for testing
TELEGRAM_USER_ID = _os.getenv("TELEGRAM_USER_ID", "123456789")  # Placeholder for testing
TELEGRAM_CHANNEL_ID = _os.getenv("TELEGRAM_CHANNEL_ID", "")  # Optional channel for broadcasting

# Pocket Option Configuration (load from environment)
POCKET_OPTION_SSID = _os.getenv("POCKET_OPTION_SSID", "test_ssid_placeholder")  # Placeholder for testing
POCKET_OPTION_BASE_URL = _os.getenv("POCKET_OPTION_BASE_URL", "https://pocketoption.com")
POCKET_OPTION_WS_URL = _os.getenv("POCKET_OPTION_WS_URL", "wss://pocketoption.com/ws")

# Trading Configuration
CURRENCY_PAIRS = [
	# Major Pairs
	"EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD",
	# Minor Pairs
	"EUR/GBP", "EUR/JPY", "EUR/CHF", "EUR/AUD", "EUR/CAD", "EUR/NZD",
	"GBP/JPY", "GBP/CHF", "GBP/AUD", "GBP/CAD", "GBP/NZD",
	"CHF/JPY", "AUD/JPY", "CAD/JPY", "NZD/JPY",
	"AUD/CHF", "AUD/CAD", "AUD/NZD", "CAD/CHF", "NZD/CHF", "NZD/CAD",
	# Exotic Pairs
	"USD/TRY", "USD/ZAR", "USD/MXN", "USD/SGD", "USD/HKD", "USD/NOK", "USD/SEK",
	"EUR/TRY", "EUR/ZAR", "EUR/PLN", "EUR/CZK", "EUR/HUF",
	"GBP/TRY", "GBP/ZAR", "GBP/PLN",
	# Crypto Pairs
	"BTC/USD", "ETH/USD", "LTC/USD", "XRP/USD", "ADA/USD", "DOT/USD",
	# Commodities
	"XAU/USD", "XAG/USD", "OIL/USD", "GAS/USD",
	# Indices
	"SPX500", "NASDAQ", "DAX30", "FTSE100", "NIKKEI", "HANG_SENG"
]

# OTC Pairs for weekends
OTC_PAIRS = [
	"EUR/USD OTC", "GBP/USD OTC", "USD/JPY OTC", "AUD/USD OTC", "USD/CAD OTC",
	"EUR/GBP OTC", "GBP/JPY OTC", "EUR/JPY OTC", "AUD/JPY OTC", "NZD/USD OTC"
]

# LSTM Model Configuration
LSTM_CONFIG = {
	"sequence_length": 60,
	"features": 24,  # Updated to match actual feature count (23 features + 1 more)
	"lstm_units": [50, 50, 50],
	"dropout_rate": 0.2,
	"learning_rate": 0.001,
	"batch_size": 32,
	"epochs": 100,
	"validation_split": 0.2
}

# Technical Analysis Configuration
TECHNICAL_INDICATORS = {
	"RSI": {"period": 14, "overbought": 70, "oversold": 30},
	"MACD": {"fast": 12, "slow": 26, "signal": 9},
	"Bollinger_Bands": {"period": 20, "std": 2},
	"Stochastic": {"k_period": 14, "d_period": 3},
	"Williams_R": {"period": 14},
	"CCI": {"period": 20},
	"ADX": {"period": 14},
	"ATR": {"period": 14},
	"EMA": {"periods": [9, 21, 50, 200]},
	"SMA": {"periods": [10, 20, 50, 100]}
}

# Signal Configuration
SIGNAL_CONFIG = {
	"min_accuracy": 95.0,
	"min_confidence": 85.0,
	"expiry_durations": [2, 3, 5],  # minutes
	"signal_advance_time": 1,  # minutes before trade
	"max_signals_per_day": 20,
	"min_volatility_threshold": 0.001,
	"max_volatility_threshold": 0.01
}

# Risk Management
RISK_MANAGEMENT = {
	"max_risk_per_trade": 2.0,  # percentage
	"max_daily_loss": 10.0,  # percentage
	"min_win_rate": 75.0,  # percentage
	"stop_loss_threshold": 5.0,  # percentage
	"max_concurrent_trades": 3
}

# Database Configuration
DATABASE_CONFIG = {
	"signals_db": "/workspace/data/signals.db",
	"performance_db": "/workspace/data/performance.db",
	"models_dir": "/workspace/models/",
	"backup_dir": "/workspace/backup/"
}

# Time Zones
TIMEZONE = pytz.timezone('UTC')
MARKET_TIMEZONE = pytz.timezone('America/New_York')

# Logging Configuration
LOGGING_CONFIG = {
	"level": "INFO",
	"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
	"file": "/workspace/logs/trading_bot.log"
}

# Market Hours
MARKET_HOURS = {
	"forex_open": "17:00",  # Sunday 17:00 EST
	"forex_close": "17:00",  # Friday 17:00 EST
	"crypto_24_7": True
}

# Performance Targets
PERFORMANCE_TARGETS = {
	"daily_win_rate": 95.0,
	"weekly_win_rate": 92.0,
	"monthly_win_rate": 90.0,
	"max_drawdown": 5.0
}