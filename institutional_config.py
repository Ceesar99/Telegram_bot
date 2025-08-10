import os
from datetime import datetime
import pytz
from typing import Dict, List, Optional

# ==========================================
# INSTITUTIONAL-GRADE TRADING CONFIGURATION
# ==========================================

# Professional Data Providers
DATA_PROVIDERS = {
    'bloomberg': {
        'enabled': True,
        'priority': 1,
        'type': 'terminal',
        'api_key': os.getenv('BLOOMBERG_API_KEY', ''),
        'endpoint': 'https://api.bloomberg.com/v1',
        'timeout': 5,
        'rate_limit': 1000,  # requests per minute
        'data_types': ['market_data', 'news', 'analytics']
    },
    'refinitiv': {
        'enabled': True,
        'priority': 2,
        'type': 'eikon',
        'api_key': os.getenv('REFINITIV_API_KEY', ''),
        'endpoint': 'https://api.refinitiv.com/v1',
        'timeout': 5,
        'rate_limit': 500,
        'data_types': ['market_data', 'news', 'fundamentals']
    },
    'polygon': {
        'enabled': True,
        'priority': 3,
        'type': 'rest',
        'api_key': os.getenv('POLYGON_API_KEY', 'demo_key'),
        'endpoint': 'https://api.polygon.io/v2',
        'timeout': 3,
        'rate_limit': 5,  # per minute for free tier
        'data_types': ['market_data', 'options', 'forex']
    },
    'alpha_vantage': {
        'enabled': True,
        'priority': 4,
        'type': 'rest',
        'api_key': os.getenv('ALPHA_VANTAGE_API_KEY', 'demo'),
        'endpoint': 'https://www.alphavantage.co/query',
        'timeout': 10,
        'rate_limit': 5,  # per minute for free tier
        'data_types': ['market_data', 'fundamentals', 'technical']
    },
    'iex_cloud': {
        'enabled': True,
        'priority': 5,
        'type': 'rest',
        'api_key': os.getenv('IEX_CLOUD_API_KEY', 'Tpk_demo'),
        'endpoint': 'https://cloud.iexapis.com/stable',
        'timeout': 5,
        'rate_limit': 100,
        'data_types': ['market_data', 'news', 'social_sentiment']
    },
    'quandl': {
        'enabled': True,
        'priority': 6,
        'type': 'rest',
        'api_key': os.getenv('QUANDL_API_KEY', ''),
        'endpoint': 'https://www.quandl.com/api/v3',
        'timeout': 10,
        'rate_limit': 300,
        'data_types': ['economic', 'alternative', 'futures']
    }
}

# FIX Protocol Configuration
FIX_CONFIG = {
    'enabled': False,  # Enable when connecting to institutional brokers
    'version': 'FIX.4.4',
    'session_configs': {
        'primary': {
            'host': os.getenv('FIX_HOST', 'localhost'),
            'port': int(os.getenv('FIX_PORT', '8080')),
            'sender_comp_id': os.getenv('FIX_SENDER_ID', 'TRADING_BOT'),
            'target_comp_id': os.getenv('FIX_TARGET_ID', 'BROKER'),
            'username': os.getenv('FIX_USERNAME', ''),
            'password': os.getenv('FIX_PASSWORD', ''),
            'heartbeat_interval': 30
        }
    }
}

# Smart Order Routing Configuration
SOR_CONFIG = {
    'enabled': True,
    'algorithms': {
        'twap': {
            'enabled': True,
            'min_duration_seconds': 60,
            'max_duration_seconds': 3600,
            'slice_size_percent': 10
        },
        'vwap': {
            'enabled': True,
            'lookback_minutes': 20,
            'participation_rate': 0.1,
            'max_volume_percent': 25
        },
        'implementation_shortfall': {
            'enabled': True,
            'risk_aversion': 0.5,
            'max_participation': 0.3
        }
    },
    'venue_preferences': {
        'primary_exchange': 0.6,
        'dark_pools': 0.3,
        'ecn': 0.1
    }
}

# Advanced Risk Management
INSTITUTIONAL_RISK = {
    'portfolio_level': {
        'max_portfolio_var': 0.02,  # 2% daily VaR
        'max_sector_concentration': 0.25,  # 25% max in any sector
        'max_single_position': 0.05,  # 5% max single position
        'correlation_limit': 0.7,  # Max correlation between positions
        'leverage_limit': 3.0  # Max portfolio leverage
    },
    'var_models': {
        'historical_simulation': {
            'enabled': True,
            'lookback_days': 252,
            'confidence_level': 0.95
        },
        'monte_carlo': {
            'enabled': True,
            'simulations': 10000,
            'confidence_level': 0.95
        },
        'parametric': {
            'enabled': True,
            'distribution': 'normal',
            'confidence_level': 0.95
        }
    },
    'stress_testing': {
        'enabled': True,
        'scenarios': {
            '2008_crisis': {'equity_shock': -0.4, 'volatility_shock': 2.0},
            '2020_covid': {'equity_shock': -0.35, 'volatility_shock': 3.0},
            'interest_rate_shock': {'rate_shock': 0.02, 'duration_impact': True},
            'currency_crisis': {'fx_shock': 0.15, 'correlation_breakdown': True}
        }
    }
}

# Market Data Quality Standards
DATA_QUALITY = {
    'latency_requirements': {
        'market_data': 50,  # milliseconds
        'news': 1000,  # milliseconds
        'execution': 10   # milliseconds
    },
    'validation_rules': {
        'price_movement_threshold': 0.05,  # 5% max price jump
        'volume_spike_threshold': 10.0,    # 10x normal volume
        'bid_ask_spread_threshold': 0.01,  # 1% max spread
        'timestamp_tolerance': 1000        # milliseconds
    },
    'redundancy': {
        'min_sources': 2,
        'max_divergence': 0.001,  # 0.1% max price divergence
        'failover_time': 5000     # milliseconds
    }
}

# System Monitoring Configuration
MONITORING_CONFIG = {
    'metrics': {
        'system_health': {
            'cpu_threshold': 80,      # percent
            'memory_threshold': 85,   # percent
            'disk_threshold': 90,     # percent
            'network_latency': 100    # milliseconds
        },
        'trading_metrics': {
            'signal_generation_time': 1000,  # milliseconds
            'execution_time': 500,           # milliseconds
            'data_staleness': 5000          # milliseconds
        }
    },
    'alerting': {
        'email': {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': 587,
            'username': os.getenv('EMAIL_USERNAME', ''),
            'password': os.getenv('EMAIL_PASSWORD', ''),
            'recipients': ['trader@example.com', 'risk@example.com']
        },
        'slack': {
            'webhook_url': os.getenv('SLACK_WEBHOOK', ''),
            'channel': '#trading-alerts'
        },
        'pagerduty': {
            'api_key': os.getenv('PAGERDUTY_API_KEY', ''),
            'service_key': os.getenv('PAGERDUTY_SERVICE_KEY', '')
        }
    }
}

# High-Frequency Trading Configuration
HFT_CONFIG = {
    'enabled': False,  # Enable for HFT strategies
    'tick_data': {
        'enabled': True,
        'compression': 'lz4',
        'storage': 'memory_mapped'
    },
    'colocation': {
        'enabled': False,
        'datacenter': 'NY4',  # Equinix NY4
        'latency_target': 0.1  # microseconds
    },
    'hardware_timestamping': {
        'enabled': False,
        'precision': 'nanosecond'
    }
}

# Machine Learning Model Configuration
ML_MODELS = {
    'ensemble': {
        'models': ['lstm', 'xgboost', 'transformer', 'random_forest', 'svm'],
        'meta_learner': 'stacking',
        'cross_validation': 'time_series_split',
        'hyperparameter_optimization': 'optuna'
    },
    'online_learning': {
        'enabled': True,
        'update_frequency': '1hour',
        'batch_size': 1000,
        'learning_rate_decay': 0.95
    },
    'feature_selection': {
        'method': 'recursive_feature_elimination',
        'scoring': 'information_ratio',
        'cross_validation': 5
    }
}

# Alternative Data Sources
ALTERNATIVE_DATA = {
    'news_sentiment': {
        'ravenpack': {
            'enabled': False,
            'api_key': os.getenv('RAVENPACK_API_KEY', ''),
            'priority': 1
        },
        'refinitiv_sentiment': {
            'enabled': True,
            'priority': 2
        }
    },
    'social_media': {
        'twitter': {
            'enabled': False,
            'api_key': os.getenv('TWITTER_API_KEY', ''),
            'bearer_token': os.getenv('TWITTER_BEARER_TOKEN', '')
        },
        'reddit': {
            'enabled': False,
            'client_id': os.getenv('REDDIT_CLIENT_ID', ''),
            'client_secret': os.getenv('REDDIT_CLIENT_SECRET', '')
        }
    },
    'satellite_data': {
        'enabled': False,
        'provider': 'spaceknow',
        'api_key': os.getenv('SATELLITE_API_KEY', '')
    },
    'economic_calendars': {
        'trading_economics': {
            'enabled': True,
            'api_key': os.getenv('TRADING_ECONOMICS_API_KEY', '')
        },
        'forex_factory': {
            'enabled': True,
            'scraping': True
        }
    }
}

# Database Configuration for Institutional Scale
DATABASE_CONFIG_INSTITUTIONAL = {
    'time_series': {
        'influxdb': {
            'host': os.getenv('INFLUXDB_HOST', 'localhost'),
            'port': 8086,
            'database': 'trading_data',
            'username': os.getenv('INFLUXDB_USER', ''),
            'password': os.getenv('INFLUXDB_PASSWORD', ''),
            'retention_policy': '30d'
        }
    },
    'cache': {
        'redis': {
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': 6379,
            'db': 0,
            'password': os.getenv('REDIS_PASSWORD', ''),
            'ttl': 300  # seconds
        }
    },
    'analytics': {
        'clickhouse': {
            'host': os.getenv('CLICKHOUSE_HOST', 'localhost'),
            'port': 9000,
            'database': 'analytics',
            'username': os.getenv('CLICKHOUSE_USER', ''),
            'password': os.getenv('CLICKHOUSE_PASSWORD', '')
        }
    }
}

# Compliance and Regulatory
COMPLIANCE_CONFIG = {
    'mifid_ii': {
        'enabled': False,
        'best_execution': True,
        'transaction_reporting': True
    },
    'dodd_frank': {
        'enabled': False,
        'swap_reporting': True,
        'position_limits': True
    },
    'trade_reporting': {
        'enabled': True,
        'real_time': True,
        'format': 'FIX',
        'destinations': ['internal_compliance', 'regulator']
    }
}

# Performance Targets (Institutional Grade)
INSTITUTIONAL_PERFORMANCE_TARGETS = {
    'accuracy': {
        'target': 0.96,  # 96% target accuracy
        'minimum': 0.93,  # 93% minimum acceptable
        'measurement_period': '30d'
    },
    'sharpe_ratio': {
        'target': 2.5,
        'minimum': 1.5
    },
    'max_drawdown': {
        'target': 0.02,  # 2%
        'maximum': 0.05  # 5%
    },
    'information_ratio': {
        'target': 1.5,
        'minimum': 1.0
    },
    'execution_metrics': {
        'slippage_target': 0.0005,  # 0.05%
        'fill_rate_target': 0.98,   # 98%
        'latency_target': 50        # milliseconds
    }
}

# Time Zones for Global Trading
GLOBAL_TIMEZONES = {
    'new_york': pytz.timezone('America/New_York'),
    'london': pytz.timezone('Europe/London'),
    'tokyo': pytz.timezone('Asia/Tokyo'),
    'hong_kong': pytz.timezone('Asia/Hong_Kong'),
    'sydney': pytz.timezone('Australia/Sydney'),
    'utc': pytz.UTC
}

# Market Sessions
MARKET_SESSIONS = {
    'forex': {
        'sydney': {'open': '21:00', 'close': '06:00', 'timezone': 'UTC'},
        'tokyo': {'open': '00:00', 'close': '09:00', 'timezone': 'UTC'},
        'london': {'open': '08:00', 'close': '17:00', 'timezone': 'UTC'},
        'new_york': {'open': '13:00', 'close': '22:00', 'timezone': 'UTC'}
    },
    'overlap_sessions': {
        'tokyo_london': {'start': '08:00', 'end': '09:00'},
        'london_new_york': {'start': '13:00', 'end': '17:00'}
    }
}