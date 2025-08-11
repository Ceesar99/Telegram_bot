#!/usr/bin/env python3
"""
Secure Configuration Manager
Handles configuration loading from environment variables and config files
with proper security practices for sensitive data.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json
from cryptography.fernet import Fernet
from dotenv import load_dotenv

class ConfigManager:
    def __init__(self, config_file: str = "config.py", env_file: str = ".env"):
        self.logger = logging.getLogger('ConfigManager')
        self.config_file = config_file
        self.env_file = env_file
        self._config = {}
        self._encrypted_keys = ['POCKET_OPTION_PASSWORD', 'DATABASE_ENCRYPTION_KEY']
        
        # Load configuration
        self._load_environment()
        self._load_config_file()
        self._validate_config()
    
    def _load_environment(self):
        """Load environment variables from .env file"""
        env_path = Path(self.env_file)
        if env_path.exists():
            load_dotenv(env_path)
            self.logger.info(f"Loaded environment from {self.env_file}")
        else:
            self.logger.warning(f"No {self.env_file} file found, using system environment")
    
    def _load_config_file(self):
        """Load configuration from config.py file"""
        try:
            # Import the config module dynamically
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", self.config_file)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # Extract all uppercase variables
            for attr in dir(config_module):
                if attr.isupper():
                    self._config[attr] = getattr(config_module, attr)
                    
            self.logger.info(f"Loaded configuration from {self.config_file}")
        except Exception as e:
            self.logger.error(f"Failed to load config file: {e}")
            raise
    
    def get(self, key: str, default: Any = None, use_env: bool = True) -> Any:
        """
        Get configuration value with environment variable override
        
        Args:
            key: Configuration key
            default: Default value if key not found
            use_env: Whether to check environment variables first
        """
        if use_env and key in os.environ:
            value = os.environ[key]
            # Try to parse as JSON for complex types
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                return value
        
        return self._config.get(key, default)
    
    def get_secure(self, key: str, default: Any = None) -> Any:
        """Get sensitive configuration value with decryption if needed"""
        value = self.get(key, default)
        
        # If this is an encrypted key and we have encryption support
        if key in self._encrypted_keys and isinstance(value, str):
            try:
                # Simple encryption check - in production use proper key management
                if value.startswith('gAAAAAB'):  # Fernet token prefix
                    encryption_key = self.get('ENCRYPTION_KEY')
                    if encryption_key:
                        fernet = Fernet(encryption_key.encode())
                        return fernet.decrypt(value.encode()).decode()
            except Exception as e:
                self.logger.warning(f"Failed to decrypt {key}: {e}")
        
        return value
    
    def _validate_config(self):
        """Validate critical configuration values"""
        required_keys = [
            'TELEGRAM_BOT_TOKEN',
            'TELEGRAM_USER_ID'
        ]
        
        missing_keys = []
        for key in required_keys:
            if not self.get(key):
                missing_keys.append(key)
        
        if missing_keys:
            self.logger.error(f"Missing required configuration: {missing_keys}")
            # Don't raise in validation to allow for graceful handling
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.get('ENVIRONMENT', 'development').lower() == 'production'
    
    def is_debug(self) -> bool:
        """Check if debug mode is enabled"""
        debug = self.get('DEBUG_MODE', 'false').lower()
        return debug in ['true', '1', 'yes', 'on']
    
    def get_database_config(self) -> Dict[str, str]:
        """Get database configuration"""
        return {
            'signals_db': self.get('SIGNALS_DB_PATH', '/workspace/data/signals.db'),
            'performance_db': self.get('PERFORMANCE_DB_PATH', '/workspace/data/performance.db'),
            'risk_management_db': self.get('RISK_DB_PATH', '/workspace/data/risk_management.db'),
            'monitoring_db': self.get('MONITORING_DB_PATH', '/workspace/data/monitoring.db')
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        level = self.get('LOG_LEVEL', 'INFO').upper()
        return {
            'level': getattr(logging, level, logging.INFO),
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': self.get('LOG_FILE', '/workspace/logs/trading_bot.log'),
            'max_bytes': int(self.get('LOG_MAX_BYTES', 10485760)),  # 10MB
            'backup_count': int(self.get('LOG_BACKUP_COUNT', 5))
        }
    
    def get_risk_config(self) -> Dict[str, float]:
        """Get risk management configuration"""
        return {
            'max_daily_loss': float(self.get('MAX_DAILY_LOSS_PERCENTAGE', 10.0)),
            'max_risk_per_trade': float(self.get('MAX_RISK_PER_TRADE_PERCENTAGE', 2.0)),
            'min_win_rate': float(self.get('MIN_WIN_RATE', 75.0)),
            'max_concurrent_trades': int(self.get('MAX_CONCURRENT_TRADES', 3)),
            'stop_loss_threshold': float(self.get('STOP_LOSS_THRESHOLD', 5.0))
        }
    
    def get_signal_config(self) -> Dict[str, Any]:
        """Get signal generation configuration"""
        return {
            'min_accuracy': float(self.get('MIN_SIGNAL_ACCURACY', 95.0)),
            'min_confidence': float(self.get('MIN_AI_CONFIDENCE', 85.0)),
            'expiry_durations': self.get('EXPIRY_DURATIONS', [2, 3, 5]),
            'signal_advance_time': int(self.get('SIGNAL_ADVANCE_TIME', 1)),
            'max_signals_per_day': int(self.get('MAX_SIGNALS_PER_DAY', 20))
        }
    
    def mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data for logging"""
        sensitive_keys = [
            'TELEGRAM_BOT_TOKEN', 'POCKET_OPTION_SSID', 'POCKET_OPTION_PASSWORD',
            'API_KEY', 'SECRET', 'PASSWORD', 'TOKEN'
        ]
        
        masked_data = {}
        for key, value in data.items():
            if any(sensitive in key.upper() for sensitive in sensitive_keys):
                if isinstance(value, str) and len(value) > 8:
                    masked_data[key] = f"{value[:4]}***{value[-4:]}"
                else:
                    masked_data[key] = "***"
            else:
                masked_data[key] = value
        
        return masked_data
    
    def export_config(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Export configuration for debugging (with optional sensitive data masking)"""
        config_export = {}
        
        # Add environment variables
        for key, value in os.environ.items():
            if key.startswith(('TELEGRAM_', 'POCKET_', 'LOG_', 'DATABASE_', 'MAX_', 'MIN_')):
                config_export[key] = value
        
        # Add config file values
        config_export.update(self._config)
        
        if not include_sensitive:
            config_export = self.mask_sensitive_data(config_export)
        
        return config_export

# Global configuration instance
config_manager = ConfigManager()

# Convenience functions for backward compatibility
def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value"""
    return config_manager.get(key, default)

def get_secure_config(key: str, default: Any = None) -> Any:
    """Get secure configuration value"""
    return config_manager.get_secure(key, default)

def is_production() -> bool:
    """Check if running in production"""
    return config_manager.is_production()

def is_debug() -> bool:
    """Check if debug mode is enabled"""
    return config_manager.is_debug()