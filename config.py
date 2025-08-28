import os
from datetime import datetime
import pytz

# Telegram Bot Configuration (load from environment)
import os as _os
TELEGRAM_BOT_TOKEN = _os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_USER_ID = _os.getenv("TELEGRAM_USER_ID", "")
TELEGRAM_CHANNEL_ID = _os.getenv("TELEGRAM_CHANNEL_ID", "")  # Optional channel for broadcasting

# Pocket Option Configuration (load from environment)
POCKET_OPTION_SSID = _os.getenv("POCKET_OPTION_SSID", "")
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

# LSTM Model Configuration - ANTI-OVERFITTING OPTIMIZED
LSTM_CONFIG = {
	"sequence_length": 60,
	"features": 24,  # Updated to match actual feature count
	"lstm_units": [64, 32, 16],  # REDUCED - Smaller network to prevent overfitting
	"dropout_rate": 0.5,  # INCREASED - More aggressive dropout
	"learning_rate": 0.0005,  # REDUCED - Slower learning for stability
	"batch_size": 128,  # INCREASED - Larger batches for stability
	"epochs": 50,  # REDUCED - Early stopping to prevent overfitting
	"validation_split": 0.3,  # INCREASED - More validation data
	"early_stopping_patience": 10,  # REDUCED - Stop earlier
	"reduce_lr_patience": 5,  # REDUCED - Faster LR reduction
	"temperature_calibration": True,
	"l1_regularization": 0.01,  # ADDED - L1 regularization
	"l2_regularization": 0.01,  # ADDED - L2 regularization
	"batch_normalization": True,  # ADDED - Batch normalization
	"gradient_clipping": 1.0  # ADDED - Gradient clipping
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

# Signal Configuration - REALISTIC FOR LIVE TRADING
SIGNAL_CONFIG = {
	"min_accuracy": 65.0,  # ACHIEVABLE - Reduced from unrealistic 95.0
	"min_confidence": 60.0,  # PRACTICAL - Reduced from unrealistic 85.0
	"expiry_durations": [2, 3, 5],  # minutes
	"signal_advance_time": 1,  # minutes before trade
	"max_signals_per_day": 15,  # Reduced for quality over quantity
	"min_volatility_threshold": 0.001,
	"max_volatility_threshold": 0.01,
	"ensemble_voting_threshold": 0.6,  # Added ensemble requirement
	"temperature_scaling": True,  # Enhanced calibration
	"cross_validation_required": True  # Require CV validation
}

# Risk Management - ENHANCED FOR MAXIMUM PROTECTION
RISK_MANAGEMENT = {
	"max_risk_per_trade": 2.0,  # percentage - SAFE LEVEL
	"max_daily_loss": 5.0,  # percentage - REDUCED from 10.0
	"max_drawdown_limit": 15.0,  # percentage - CRITICAL PROTECTION
	"min_win_rate": 60.0,  # percentage - REALISTIC TARGET
	"stop_loss_threshold": 2.0,  # percentage - REDUCED for protection
	"max_concurrent_trades": 3,
	"kelly_fraction": 0.25,  # Added Kelly Criterion
	"circuit_breaker_threshold": 10.0,  # Emergency halt at 10% rapid loss
	"atr_multiplier_stop": 2.0,  # ATR-based stop losses
	"atr_multiplier_profit": 3.0  # ATR-based take profits
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

# Performance Targets - REALISTIC FOR LIVE TRADING
PERFORMANCE_TARGETS = {
	"daily_win_rate": 65.0,  # ACHIEVABLE - Reduced from unrealistic 95.0
	"weekly_win_rate": 67.0,  # PRACTICAL - Reduced from unrealistic 92.0
	"monthly_win_rate": 70.0,  # REALISTIC - Reduced from unrealistic 90.0
	"max_drawdown": 15.0  # SAFE - Increased from overly optimistic 5.0
}