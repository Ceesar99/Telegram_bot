#!/usr/bin/env python3
"""
Environment Setup and Validation Script
"""

import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('EnvironmentSetup')

def create_directories():
    """Create required directory structure"""
    logger.info("Creating directory structure...")
    
    directories = [
        '/workspace/models',
        '/workspace/logs',
        '/workspace/data',
        '/workspace/backup',
        '/workspace/logs/lstm',
        '/workspace/logs/ensemble',
        '/workspace/logs/transformer',
        '/workspace/logs/reinforcement_learning',
        '/workspace/models/cache',
        '/workspace/features/cache'
    ]
    
    created_count = 0
    for dir_path in directories:
        try:
            os.makedirs(dir_path, exist_ok=True)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logger.info(f"✅ Created directory: {dir_path}")
                created_count += 1
            else:
                logger.info(f"✅ Directory exists: {dir_path}")
        except Exception as e:
            logger.error(f"❌ Failed to create directory {dir_path}: {e}")
    
    logger.info(f"Directory setup complete. {created_count} new directories created.")
    return True

def create_log_files():
    """Create log files for different components"""
    logger.info("Creating log files...")
    
    log_files = [
        '/workspace/logs/lstm_model.log',
        '/workspace/logs/ensemble_training.log',
        '/workspace/logs/transformer.log',
        '/workspace/logs/reinforcement_learning.log',
        '/workspace/logs/trading_system.log',
        '/workspace/logs/backtesting.log'
    ]
    
    created_count = 0
    for log_file in log_files:
        try:
            if not os.path.exists(log_file):
                with open(log_file, 'w') as f:
                    f.write(f"# Log file created at {datetime.now().isoformat()}\n")
                logger.info(f"✅ Created log file: {log_file}")
                created_count += 1
            else:
                logger.info(f"✅ Log file exists: {log_file}")
        except Exception as e:
            logger.error(f"❌ Failed to create log file {log_file}: {e}")
    
    logger.info(f"Log file setup complete. {created_count} new log files created.")
    return True

def check_environment_variables():
    """Check if environment variables are set"""
    logger.info("Checking environment variables...")
    
    required_vars = [
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_USER_ID',
        'POCKET_OPTION_SSID'
    ]
    
    optional_vars = [
        'TELEGRAM_CHANNEL_ID',
        'POCKET_OPTION_BASE_URL',
        'POCKET_OPTION_WS_URL',
        'LOG_LEVEL',
        'ENVIRONMENT'
    ]
    
    missing_required = []
    missing_optional = []
    
    for var in required_vars:
        if os.getenv(var):
            logger.info(f"✅ {var}: Set")
        else:
            missing_required.append(var)
            logger.warning(f"⚠️  {var}: Not set (REQUIRED)")
    
    for var in optional_vars:
        if os.getenv(var):
            logger.info(f"✅ {var}: Set")
        else:
            missing_optional.append(var)
            logger.info(f"ℹ️  {var}: Not set (optional)")
    
    if missing_required:
        logger.warning(f"⚠️  Missing required environment variables: {', '.join(missing_required)}")
        logger.info("💡 Create a .env file based on .env.example and set these variables")
        return False
    else:
        logger.info("✅ All required environment variables are set")
        return True

def create_env_file():
    """Create .env file if it doesn't exist"""
    logger.info("Checking .env file...")
    
    env_file = '/workspace/.env'
    env_example = '/workspace/.env.example'
    
    if os.path.exists(env_file):
        logger.info("✅ .env file already exists")
        return True
    
    if not os.path.exists(env_example):
        logger.error("❌ .env.example file not found")
        return False
    
    try:
        # Copy .env.example to .env
        with open(env_example, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        logger.info("✅ Created .env file from .env.example")
        logger.info("💡 Edit .env file with your actual values")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create .env file: {e}")
        return False

def validate_config_files():
    """Validate that key configuration files exist"""
    logger.info("Validating configuration files...")
    
    config_files = [
        '/workspace/config.py',
        '/workspace/requirements.txt',
        '/workspace/requirements-constraints.txt'
    ]
    
    missing_files = []
    for config_file in config_files:
        if os.path.exists(config_file):
            logger.info(f"✅ Config file exists: {config_file}")
        else:
            missing_files.append(config_file)
            logger.warning(f"⚠️  Config file missing: {config_file}")
    
    if missing_files:
        logger.warning(f"⚠️  Missing config files: {len(missing_files)}")
        return False
    else:
        logger.info("✅ All configuration files present")
        return True

def check_python_modules():
    """Check if key Python modules are available"""
    logger.info("Checking Python module availability...")
    
    # Basic modules that should be available
    basic_modules = ['os', 'sys', 'json', 'datetime', 'logging']
    
    # ML modules that might not be available
    ml_modules = ['numpy', 'pandas', 'tensorflow', 'torch', 'sklearn']
    
    missing_basic = []
    missing_ml = []
    
    for module in basic_modules:
        try:
            __import__(module)
            logger.info(f"✅ {module}: Available")
        except ImportError:
            missing_basic.append(module)
            logger.error(f"❌ {module}: Missing (CRITICAL)")
    
    for module in ml_modules:
        try:
            __import__(module)
            logger.info(f"✅ {module}: Available")
        except ImportError:
            missing_ml.append(module)
            logger.warning(f"⚠️  {module}: Missing (will need installation)")
    
    if missing_basic:
        logger.error(f"❌ Critical modules missing: {', '.join(missing_basic)}")
        return False
    
    if missing_ml:
        logger.warning(f"⚠️  ML modules missing: {', '.join(missing_ml)}")
        logger.info("💡 Install missing modules with: pip install <module_name>")
    
    return True

def main():
    """Run environment setup and validation"""
    logger.info("🚀 Starting Environment Setup and Validation...")
    
    setup_tasks = [
        ("Directory Structure", create_directories),
        ("Log Files", create_log_files),
        ("Configuration Files", validate_config_files),
        ("Python Modules", check_python_modules),
        ("Environment File", create_env_file),
        ("Environment Variables", check_environment_variables)
    ]
    
    passed = 0
    total = len(setup_tasks)
    
    for task_name, task_func in setup_tasks:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {task_name}")
        logger.info(f"{'='*50}")
        
        if task_func():
            passed += 1
            logger.info(f"✅ {task_name} PASSED")
        else:
            logger.error(f"❌ {task_name} FAILED")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("ENVIRONMENT SETUP SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 Environment setup completed successfully!")
        logger.info("✅ System is ready for development and testing")
    else:
        logger.warning(f"⚠️  {total-passed} task(s) failed. Check logs above.")
    
    # Next steps
    logger.info("\n📋 Next Steps:")
    logger.info("1. Edit .env file with your actual credentials")
    logger.info("2. Install missing Python modules if any")
    logger.info("3. Run tests to validate functionality")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)