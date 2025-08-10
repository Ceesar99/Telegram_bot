#!/usr/bin/env python3
"""
Basic Test Script for Trading Bot Components
This script tests individual components without complex asyncio operations
"""

import sys
import os
import logging
import numpy as np
import pandas as pd

# Add the workspace to Python path
sys.path.append('/workspace')

def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported"""
    logger = logging.getLogger('TestImports')
    
    required_modules = [
        'tensorflow',
        'pandas', 
        'numpy',
        'sklearn',
        'telegram',
        'requests',
        'websocket',
        'matplotlib',
        'seaborn',
        'talib',
        'psutil'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✅ {module} imported successfully")
        except ImportError as e:
            logger.error(f"❌ {module} import failed: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_config():
    """Test if configuration can be loaded"""
    logger = logging.getLogger('TestConfig')
    
    try:
        from config import TELEGRAM_USER_ID, POCKET_OPTION_BASE_URL
        logger.info("✅ Configuration loaded successfully")
        logger.info(f"   Telegram User ID: {TELEGRAM_USER_ID}")
        logger.info(f"   Pocket Option Base URL: {POCKET_OPTION_BASE_URL}")
        return True
    except Exception as e:
        logger.error(f"❌ Configuration loading failed: {e}")
        return False

def test_model_loading():
    """Test if LSTM model can be loaded"""
    logger = logging.getLogger('TestModel')
    
    try:
        from lstm_model import LSTMTradingModel
        
        model = LSTMTradingModel()
        logger.info("✅ LSTM model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ LSTM model initialization failed: {e}")
        return False

def test_telegram_bot():
    """Test if Telegram bot can be created"""
    logger = logging.getLogger('TestTelegramBot')
    
    try:
        from telegram_bot import TradingBot
        
        bot = TradingBot()
        logger.info("✅ Telegram bot created successfully")
        logger.info(f"   Bot status: {bot.bot_status}")
        return True
    except Exception as e:
        logger.error(f"❌ Telegram bot creation failed: {e}")
        return False

def test_data_manager():
    """Test if data manager can be created"""
    logger = logging.getLogger('TestDataManager')
    
    try:
        from data_manager import DataManager
        
        dm = DataManager()
        logger.info("✅ Data manager created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Data manager creation failed: {e}")
        return False

def test_pocket_api():
    """Test if Pocket Option API can be created"""
    logger = logging.getLogger('TestPocketAPI')
    
    try:
        from pocket_option_api import PocketOptionAPI
        
        api = PocketOptionAPI()
        logger.info("✅ Pocket Option API created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Pocket Option API creation failed: {e}")
        return False

def run_all_tests():
    """Run all basic tests"""
    logger = setup_logging()
    
    logger.info("🧪 Starting Basic Component Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_config),
        ("LSTM Model", test_model_loading),
        ("Telegram Bot", test_telegram_bot),
        ("Data Manager", test_data_manager),
        ("Pocket API", test_pocket_api)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Testing: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Basic components are working.")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Some components may have issues.")
        return False

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error during testing: {e}")
        sys.exit(1)