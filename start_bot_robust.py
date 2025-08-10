#!/usr/bin/env python3
"""
Robust Trading Bot Startup Script

This script handles the startup process for the binary options trading bot,
including environment setup, dependency checks, graceful error handling,
and fallback data generation when APIs fail.
"""

import os
import sys
import subprocess
import time
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Add the workspace to Python path
sys.path.append('/workspace')

def setup_logging():
    """Setup comprehensive logging for startup"""
    # Create logs directory if it doesn't exist
    os.makedirs('/workspace/logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/workspace/logs/startup.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    package_mappings = {
        'tensorflow': 'tensorflow',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'python-telegram-bot': 'telegram',
        'requests': 'requests',
        'websocket-client': 'websocket',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'talib': 'talib',
        'psutil': 'psutil',
        'yfinance': 'yfinance',
        'aiohttp': 'aiohttp',
        'asyncio': 'asyncio'
    }
    
    missing_packages = []
    
    for package, import_name in package_mappings.items():
        try:
            __import__(import_name)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("\n🔧 Installing dependencies...")
    try:
        # Try to install from requirements files
        requirements_files = ['requirements.txt', 'requirements_core.txt']
        
        for req_file in requirements_files:
            if os.path.exists(req_file):
                print(f"Installing from {req_file}...")
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', '-r', req_file
                ], check=True, capture_output=True, text=True)
                
                print(f"✅ Dependencies installed successfully from {req_file}")
                return True
        
        # If no requirements file found, install core packages
        core_packages = [
            'pandas', 'numpy', 'requests', 'python-telegram-bot',
            'websocket-client', 'aiohttp', 'yfinance'
        ]
        
        for package in core_packages:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', package
            ], check=True, capture_output=True)
        
        print("✅ Core dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print(f"Error output: {e.stderr}")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = [
        '/workspace/logs',
        '/workspace/data', 
        '/workspace/models',
        '/workspace/backup',
        '/workspace/execution',
        '/workspace/monitoring',
        '/workspace/portfolio'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory: {directory}")

def check_configuration():
    """Check if configuration is properly set"""
    try:
        from config import TELEGRAM_BOT_TOKEN, TELEGRAM_USER_ID, POCKET_OPTION_SSID
        
        issues = []
        
        if TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN":
            print("✅ Telegram Bot Token: Configured")
        else:
            issues.append("Telegram Bot Token not configured")
        
        if TELEGRAM_USER_ID and TELEGRAM_USER_ID != "YOUR_USER_ID":
            print("✅ Telegram User ID: Configured")
        else:
            issues.append("Telegram User ID not configured")
        
        if POCKET_OPTION_SSID and len(POCKET_OPTION_SSID) > 10:
            print("✅ Pocket Option SSID: Configured")
        else:
            issues.append("Pocket Option SSID not configured")
        
        if issues:
            print(f"\n⚠️  Configuration issues found:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Configuration import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Configuration check error: {e}")
        return False

def test_api_connectivity():
    """Test API connectivity and provide fallback options"""
    print("\n🔍 Testing API connectivity...")
    
    try:
        # Test PocketOption API
        from pocket_option_api import PocketOptionAPI
        po_api = PocketOptionAPI()
        
        # Test with a simple symbol
        test_symbol = "EUR/USD"
        price_data = po_api.get_current_price(test_symbol)
        
        if price_data:
            print("✅ PocketOption API: Working")
        else:
            print("⚠️  PocketOption API: Using fallback data generation")
            
    except Exception as e:
        print(f"⚠️  PocketOption API: Error - {e}")
        print("   Will use fallback data generation")
    
    try:
        # Test Yahoo Finance
        import yfinance as yf
        ticker = yf.Ticker("EURUSD=X")
        data = ticker.history(period="1d")
        
        if not data.empty:
            print("✅ Yahoo Finance API: Working")
        else:
            print("⚠️  Yahoo Finance API: Limited data")
            
    except Exception as e:
        print(f"⚠️  Yahoo Finance API: Error - {e}")
        print("   Will use fallback data generation")
    
    print("✅ Fallback data generation: Available")

def create_demo_data_if_needed():
    """Create demo data if no real data is available"""
    try:
        data_dir = "/workspace/data"
        demo_file = os.path.join(data_dir, "demo_eurusd_data.csv")
        
        if not os.path.exists(demo_file):
            print("\n📊 Creating demo data...")
            from demo_mode import create_demo_data
            create_demo_data()
            print("✅ Demo data created")
        else:
            print("✅ Demo data already exists")
            
    except Exception as e:
        print(f"⚠️  Demo data creation failed: {e}")

def print_startup_info():
    """Print startup information"""
    print("\n" + "="*60)
    print("🤖 BINARY OPTIONS TRADING BOT - ROBUST STARTUP")
    print("="*60)
    print(f"📅 Startup time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python version: {sys.version.split()[0]}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🔧 Environment: {'Production' if os.getenv('ENVIRONMENT') == 'prod' else 'Development'}")
    print("="*60)

def print_ready_message():
    """Print ready message"""
    print("\n" + "🎯"*20)
    print("🚀 TRADING BOT IS READY TO START!")
    print("🎯"*20)
    print("\n📱 Telegram bot will be available for commands")
    print("📊 Signal generation will start automatically")
    print("🔄 Fallback data generation is active")
    print("📈 Performance tracking is enabled")
    print("\n💡 Use /help in Telegram for available commands")
    print("📋 Check logs in /workspace/logs for detailed information")

async def run_bot():
    """Run the trading bot with error handling"""
    try:
        print("\n🚀 Starting trading bot...")
        
        # Import and run the main bot
        from main import TradingBotSystem
        
        bot_system = TradingBotSystem()
        
        # Run the bot
        await bot_system.run()
        
    except KeyboardInterrupt:
        print("\n⚠️  Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Bot startup failed: {e}")
        print("📋 Check the logs for detailed error information")
        return False
    
    return True

def main():
    """Main startup function"""
    logger = setup_logging()
    
    try:
        print_startup_info()
        
        # Check Python version
        if not check_python_version():
            return False
        
        # Setup directories
        setup_directories()
        
        # Check dependencies
        missing_packages = check_dependencies()
        if missing_packages:
            print(f"\n📦 Installing {len(missing_packages)} missing packages...")
            if not install_dependencies():
                print("❌ Failed to install dependencies")
                return False
        
        # Check configuration
        if not check_configuration():
            print("\n⚠️  Configuration issues detected")
            print("   Bot may not function properly")
            print("   Please check config.py file")
        
        # Test API connectivity
        test_api_connectivity()
        
        # Create demo data if needed
        create_demo_data_if_needed()
        
        print_ready_message()
        
        # Start the bot
        return asyncio.run(run_bot())
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        print(f"\n❌ Startup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Trading bot startup failed")
        print("📋 Check the logs for detailed error information")
        sys.exit(1)
    else:
        print("\n✅ Trading bot startup completed successfully")