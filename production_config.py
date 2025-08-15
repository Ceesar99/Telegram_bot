#!/usr/bin/env python3
"""
🔧 PRODUCTION CONFIGURATION - REAL TRADING READY
Enhanced configuration for 100% real-time trading readiness
"""

import os
from datetime import datetime
import pytz

# =============================================================================
# CRITICAL: REPLACE ALL DEMO VALUES WITH REAL API KEYS
# =============================================================================

# 🔑 REAL-TIME DATA PROVIDERS (REQUIRED FOR 100% READINESS)
DATA_PROVIDERS = {
    'alpha_vantage': {
        'api_key': os.getenv("ALPHA_VANTAGE_KEY", "demo"),  # ⚠️ REPLACE WITH REAL KEY
        'calls_per_minute': 5,
        'priority': 1
    },
    'finnhub': {
        'api_key': os.getenv("FINNHUB_API_KEY", "demo"),   # ⚠️ REPLACE WITH REAL KEY
        'calls_per_minute': 60,
        'priority': 2
    },
    'twelve_data': {
        'api_key': os.getenv("TWELVE_DATA_KEY", "demo"),   # ⚠️ REPLACE WITH REAL KEY
        'calls_per_minute': 8,
        'priority': 3
    },
    'polygon': {
        'api_key': os.getenv("POLYGON_API_KEY", "demo"),   # ⚠️ REPLACE WITH REAL KEY
        'calls_per_minute': 5,
        'priority': 4
    },
    'yahoo_finance': {
        'enabled': True,
        'calls_per_minute': 120,
        'priority': 5  # Fallback
    }
}

# 📱 TELEGRAM BOT CONFIGURATION
TELEGRAM_CONFIG = {
    'bot_token': os.getenv("TELEGRAM_BOT_TOKEN", ""),     # ⚠️ REQUIRED
    'user_id': os.getenv("TELEGRAM_USER_ID", ""),        # ⚠️ REQUIRED
    'channel_id': os.getenv("TELEGRAM_CHANNEL_ID", ""),  # Optional
    'max_retries': 3,
    'timeout': 30,
    'rate_limit': 30  # messages per minute
}

# 🏦 BROKER INTEGRATION
BROKER_CONFIG = {
    'pocket_option': {
        'ssid': os.getenv("POCKET_OPTION_SSID", ""),      # ⚠️ REQUIRED FOR LIVE TRADING
        'base_url': "https://pocketoption.com",
        'ws_url': "wss://pocketoption.com/ws",
        'session_timeout': 3600,  # 1 hour
        'reconnect_attempts': 5
    },
    'iq_option': {
        'username': os.getenv("IQ_OPTION_USERNAME", ""),  # Optional alternative
        'password': os.getenv("IQ_OPTION_PASSWORD", ""),
        'enabled': False
    }
}

# 🤖 AI MODEL CONFIGURATION (PRODUCTION OPTIMIZED)
MODEL_CONFIG = {
    'lstm': {
        'model_path': '/workspace/models/production_lstm_optimized.h5',
        'sequence_length': 60,
        'features_count': 24,
        'confidence_threshold': 85.0,
        'retrain_threshold': 70.0,  # Retrain if accuracy drops below this
        'max_inference_time_ms': 100
    },
    'ensemble': {
        'enabled': True,
        'models': ['lstm', 'xgboost', 'transformer', 'random_forest', 'svm'],
        'voting_method': 'weighted',
        'confidence_threshold': 90.0,
        'max_inference_time_ms': 500
    },
    'transformer': {
        'enabled': True,
        'model_path': '/workspace/models/production_transformer.ts',
        'multi_timeframe': True,
        'confidence_threshold': 87.0,
        'max_inference_time_ms': 50
    }
}

# 📊 TRADING CONFIGURATION (PRODUCTION SETTINGS)
TRADING_CONFIG = {
    'mode': 'live',  # 'paper' or 'live'
    'pairs': [
        # Major Forex Pairs (High Liquidity)
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD",
        # Minor Pairs
        "EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/JPY", "EUR/CHF"
    ],
    'otc_pairs': [
        "EUR/USD OTC", "GBP/USD OTC", "USD/JPY OTC", "AUD/USD OTC"
    ],
    'timeframes': ['1m', '5m', '15m'],
    'signal_advance_time': 60,  # seconds before trade execution
    'max_signals_per_day': 50,
    'min_market_volatility': 0.001,
    'max_market_volatility': 0.05
}

# 🛡️ RISK MANAGEMENT (PRODUCTION READY)
RISK_CONFIG = {
    'max_risk_per_trade': 1.5,      # 1.5% max risk per trade
    'max_daily_loss': 5.0,          # 5% max daily loss
    'max_weekly_loss': 15.0,        # 15% max weekly loss
    'max_monthly_loss': 30.0,       # 30% max monthly loss
    'min_account_balance': 100.0,   # Minimum account balance
    'max_concurrent_trades': 3,
    'position_sizing_method': 'kelly',  # 'fixed', 'percentage', 'kelly'
    'stop_loss_percentage': 2.0,
    'take_profit_ratio': 2.0,       # Risk:Reward 1:2
    'max_correlation_exposure': 0.3  # Max 30% exposure to correlated pairs
}

# 📈 SIGNAL GENERATION (ENHANCED)
SIGNAL_CONFIG = {
    'min_accuracy_threshold': 90.0,    # Minimum 90% accuracy for live signals
    'min_confidence_threshold': 85.0,   # Minimum 85% confidence
    'consensus_threshold': 0.7,         # 70% model consensus required
    'expiry_times': [2, 3, 5],         # minutes
    'signal_types': ['BUY', 'SELL'],   # Remove 'HOLD' for binary options
    'market_condition_filter': True,   # Filter signals based on market conditions
    'volatility_filter': True,         # Filter based on volatility
    'correlation_filter': True,        # Avoid correlated signals
    'max_daily_signals': 25,           # Conservative limit
    'signal_cooling_period': 300       # 5 minutes between signals for same pair
}

# 🔄 DATA COLLECTION (REAL-TIME)
DATA_CONFIG = {
    'update_frequency': 1,              # Update every 1 second
    'historical_lookback': 1000,        # Data points for analysis
    'data_validation': True,            # Validate data quality
    'fallback_providers': 3,            # Number of fallback data sources
    'cache_timeout': 30,                # Cache data for 30 seconds
    'quality_threshold': 0.95,          # 95% data quality required
    'missing_data_threshold': 0.05      # Max 5% missing data allowed
}

# 📊 TECHNICAL INDICATORS (OPTIMIZED)
TECHNICAL_CONFIG = {
    'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
    'macd': {'fast': 12, 'slow': 26, 'signal': 9},
    'bollinger_bands': {'period': 20, 'std_dev': 2},
    'stochastic': {'k_period': 14, 'd_period': 3},
    'williams_r': {'period': 14},
    'cci': {'period': 20},
    'adx': {'period': 14, 'trend_threshold': 25},
    'atr': {'period': 14},
    'ema': {'periods': [9, 21, 50, 200]},
    'sma': {'periods': [10, 20, 50, 100]},
    'volume_profile': {'enabled': True, 'period': 20}
}

# 🗄️ DATABASE CONFIGURATION (PRODUCTION)
DATABASE_CONFIG = {
    'type': 'postgresql',  # Upgrade from SQLite for production
    'host': os.getenv("DB_HOST", "localhost"),
    'port': int(os.getenv("DB_PORT", "5432")),
    'database': os.getenv("DB_NAME", "trading_system"),
    'username': os.getenv("DB_USER", "trading"),
    'password': os.getenv("DB_PASSWORD", ""),
    'ssl_mode': 'require',
    'connection_pool_size': 10,
    'backup_enabled': True,
    'backup_frequency': 'hourly',
    'backup_retention_days': 30
}

# 📊 MONITORING & ALERTING
MONITORING_CONFIG = {
    'health_check_interval': 30,        # seconds
    'performance_tracking': True,
    'latency_threshold_ms': 1000,      # Alert if latency > 1s
    'memory_threshold_mb': 2048,       # Alert if memory > 2GB
    'cpu_threshold_percent': 80,       # Alert if CPU > 80%
    'disk_threshold_percent': 85,      # Alert if disk > 85%
    'alert_channels': ['telegram', 'email', 'webhook'],
    'log_level': 'INFO',
    'log_rotation_size_mb': 100,
    'log_retention_days': 30
}

# 🔐 SECURITY CONFIGURATION
SECURITY_CONFIG = {
    'encryption_enabled': True,
    'api_key_encryption': True,
    'session_timeout': 3600,
    'max_login_attempts': 3,
    'ip_whitelist': [],  # Add your IP addresses
    'rate_limiting': True,
    'audit_logging': True,
    'ssl_verification': True
}

# 🚀 PERFORMANCE OPTIMIZATION
PERFORMANCE_CONFIG = {
    'async_processing': True,
    'connection_pooling': True,
    'caching_enabled': True,
    'compression_enabled': True,
    'parallel_processing': True,
    'max_workers': 4,
    'memory_optimization': True,
    'gc_threshold': 1000
}

# ⏰ TIMEZONE CONFIGURATION
TIMEZONE_CONFIG = {
    'system_timezone': pytz.UTC,
    'market_timezone': pytz.timezone('America/New_York'),
    'user_timezone': pytz.timezone('UTC'),
    'auto_dst_adjustment': True
}

# 📁 FILE PATHS (PRODUCTION)
PATHS_CONFIG = {
    'models_dir': '/workspace/models/',
    'logs_dir': '/workspace/logs/',
    'data_dir': '/workspace/data/',
    'backup_dir': '/workspace/backups/',
    'temp_dir': '/tmp/trading_system/',
    'config_dir': '/workspace/config/'
}

# 🧪 TESTING CONFIGURATION
TESTING_CONFIG = {
    'paper_trading_enabled': True,
    'paper_trading_balance': 10000.0,
    'backtesting_enabled': True,
    'forward_testing_days': 7,
    'model_validation_enabled': True,
    'stress_testing_enabled': True,
    'performance_benchmarking': True
}

# =============================================================================
# PRODUCTION READINESS CHECKLIST
# =============================================================================

READINESS_CHECKLIST = {
    'data_providers_configured': False,    # ⚠️ Set to True after API keys added
    'telegram_bot_configured': False,      # ⚠️ Set to True after bot setup
    'broker_integration_active': False,    # ⚠️ Set to True after SSID validation
    'models_trained': False,               # ⚠️ Set to True after model training
    'risk_management_active': True,
    'monitoring_enabled': True,
    'security_enabled': True,
    'backup_system_active': True,
    'testing_completed': False             # ⚠️ Set to True after full testing
}

def validate_production_readiness() -> Dict[str, Any]:
    """Validate system readiness for production deployment"""
    
    issues = []
    warnings = []
    
    # Check critical configurations
    if not TELEGRAM_CONFIG['bot_token']:
        issues.append("Telegram bot token not configured")
    
    if not BROKER_CONFIG['pocket_option']['ssid']:
        issues.append("Pocket Option SSID not configured")
    
    if DATA_PROVIDERS['alpha_vantage']['api_key'] == 'demo':
        issues.append("Real data provider API keys not configured")
    
    if not all(READINESS_CHECKLIST.values()):
        warnings.append("Some readiness checklist items not completed")
    
    return {
        'ready': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'readiness_score': sum(READINESS_CHECKLIST.values()) / len(READINESS_CHECKLIST) * 100
    }

# Export main configurations
__all__ = [
    'DATA_PROVIDERS', 'TELEGRAM_CONFIG', 'BROKER_CONFIG', 'MODEL_CONFIG',
    'TRADING_CONFIG', 'RISK_CONFIG', 'SIGNAL_CONFIG', 'DATA_CONFIG',
    'TECHNICAL_CONFIG', 'DATABASE_CONFIG', 'MONITORING_CONFIG',
    'SECURITY_CONFIG', 'PERFORMANCE_CONFIG', 'validate_production_readiness'
]