#!/usr/bin/env python3
"""
🚀 QUICK START GUIDE - AUTOMATED SETUP
This script automates everything I can do for you
"""

import os
import sys
import subprocess
import requests
import json
from datetime import datetime

class QuickStartGuide:
    def __init__(self):
        self.steps_completed = []
        self.steps_pending = []
        
    def check_system_requirements(self):
        """Check if system has required dependencies"""
        print("🔍 Checking system requirements...")
        
        required_packages = [
            'pandas', 'numpy', 'tensorflow', 'scikit-learn', 
            'yfinance', 'talib', 'aiohttp', 'python-telegram-bot'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"✅ {package} - installed")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package} - missing")
        
        if missing_packages:
            print(f"\n⚠️ Installing missing packages: {', '.join(missing_packages)}")
            for package in missing_packages:
                subprocess.run([sys.executable, '-m', 'pip', 'install', package])
        
        self.steps_completed.append("System requirements checked")
        return len(missing_packages) == 0
    
    def setup_directory_structure(self):
        """Create necessary directories"""
        print("\n📁 Setting up directory structure...")
        
        directories = [
            '/workspace/models',
            '/workspace/logs', 
            '/workspace/data',
            '/workspace/backups',
            '/workspace/config'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created: {directory}")
        
        self.steps_completed.append("Directory structure created")
    
    def create_environment_template(self):
        """Create environment variables template"""
        print("\n🔧 Creating environment template...")
        
        env_template = """
# =============================================================================
# TRADING SYSTEM ENVIRONMENT VARIABLES
# =============================================================================

# 🔑 DATA PROVIDER API KEYS (REPLACE WITH REAL KEYS)
export ALPHA_VANTAGE_KEY="demo"          # Get from: https://www.alphavantage.co/
export FINNHUB_API_KEY="demo"            # Get from: https://finnhub.io/
export TWELVE_DATA_KEY="demo"            # Get from: https://twelvedata.com/
export POLYGON_API_KEY="demo"            # Get from: https://polygon.io/

# 📱 TELEGRAM BOT (REPLACE WITH YOUR BOT)
export TELEGRAM_BOT_TOKEN=""             # Get from: @BotFather
export TELEGRAM_USER_ID=""               # Your Telegram user ID

# 🏦 BROKER INTEGRATION (REPLACE WITH REAL SSID)
export POCKET_OPTION_SSID=""             # Get from Pocket Option browser session

# 🗄️ DATABASE CONFIGURATION
export DB_HOST="localhost"
export DB_PORT="5432"
export DB_NAME="trading_system"
export DB_USER="trading"
export DB_PASSWORD=""

# 🔐 SECURITY
export ENCRYPTION_KEY=""
export API_SECRET_KEY=""

# =============================================================================
# INSTRUCTIONS:
# 1. Copy this file to .env
# 2. Replace all "demo" and empty values with real credentials
# 3. Run: source .env
# =============================================================================
"""
        
        with open('/workspace/.env.template', 'w') as f:
            f.write(env_template)
        
        print("✅ Environment template created: .env.template")
        print("⚠️ ACTION REQUIRED: Copy to .env and add your real API keys")
        
        self.steps_pending.append("Set up real API keys in .env file")
    
    def validate_existing_files(self):
        """Check which files are already created"""
        print("\n📋 Validating existing system files...")
        
        required_files = {
            'enhanced_data_collector.py': 'Real-time data collection',
            'production_config.py': 'Production configuration',
            'production_model_trainer.py': 'Advanced model training',
            'production_trading_system.py': 'Complete trading system',
            'production_risk_manager.py': 'Risk management',
            'lstm_model.py': 'LSTM model',
            'ensemble_models.py': 'Ensemble models',
            'pocket_option_api.py': 'Broker API'
        }
        
        files_ready = 0
        for filename, description in required_files.items():
            filepath = f'/workspace/{filename}'
            if os.path.exists(filepath):
                print(f"✅ {filename} - {description}")
                files_ready += 1
            else:
                print(f"❌ {filename} - {description}")
        
        print(f"\n📊 System files ready: {files_ready}/{len(required_files)}")
        
        if files_ready == len(required_files):
            self.steps_completed.append("All system files present")
        else:
            self.steps_pending.append(f"Create missing {len(required_files) - files_ready} system files")
    
    def create_quick_test_script(self):
        """Create a script to test system readiness"""
        print("\n🧪 Creating quick test script...")
        
        test_script = '''#!/usr/bin/env python3
"""
🧪 QUICK SYSTEM TEST
Test all components before going live
"""

import asyncio
import sys
import os

async def test_data_collection():
    """Test real-time data collection"""
    try:
        from enhanced_data_collector import RealTimeDataCollector
        
        collector = RealTimeDataCollector()
        
        # Test data collection
        data = await collector.get_real_time_data('EUR/USD', '1m')
        
        if data is not None and len(data) > 0:
            print("✅ Data collection working")
            
            # Test technical indicators
            indicators = collector.calculate_real_technical_indicators(data)
            if indicators:
                print("✅ Technical indicators working")
                return True
            else:
                print("❌ Technical indicators failed")
                return False
        else:
            print("❌ Data collection failed")
            return False
            
    except Exception as e:
        print(f"❌ Data collection error: {e}")
        return False

async def test_model_loading():
    """Test AI model loading"""
    try:
        from lstm_model import LSTMTradingModel
        
        model = LSTMTradingModel()
        
        # Try to load existing model
        if os.path.exists('/workspace/models/best_model.h5'):
            loaded = model.load_model('/workspace/models/best_model.h5')
            if loaded:
                print("✅ LSTM model loading working")
                return True
            else:
                print("⚠️ LSTM model exists but failed to load")
                return False
        else:
            print("⚠️ No trained model found - need to train models")
            return False
            
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_configuration():
    """Test configuration setup"""
    try:
        from production_config import validate_production_readiness
        
        result = validate_production_readiness()
        
        print(f"📊 Production readiness: {result['readiness_score']:.1f}%")
        
        if result['issues']:
            print("❌ Critical issues:")
            for issue in result['issues']:
                print(f"  - {issue}")
        
        if result['warnings']:
            print("⚠️ Warnings:")
            for warning in result['warnings']:
                print(f"  - {warning}")
        
        return result['readiness_score'] > 70
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 ULTIMATE TRADING SYSTEM - QUICK TEST")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Data Collection", test_data_collection),
        ("Model Loading", test_model_loading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\\n🧪 Testing {test_name}...")
        
        if asyncio.iscoroutinefunction(test_func):
            result = await test_func()
        else:
            result = test_func()
        
        results.append((test_name, result))
    
    print("\\n" + "=" * 50)
    print("📋 TEST RESULTS:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\\n📊 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 SYSTEM READY FOR NEXT PHASE!")
    else:
        print("⚠️ SYSTEM NEEDS ATTENTION BEFORE PROCEEDING")
    
    return passed == len(tests)

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        with open('/workspace/quick_test.py', 'w') as f:
            f.write(test_script)
        
        os.chmod('/workspace/quick_test.py', 0o755)
        print("✅ Quick test script created: quick_test.py")
        
        self.steps_completed.append("Quick test script created")
    
    def create_deployment_checklist(self):
        """Create a deployment checklist"""
        print("\n📋 Creating deployment checklist...")
        
        checklist = """
# 🚀 DEPLOYMENT CHECKLIST

## Phase 1: Prerequisites ✅ (I can help with all of these)
- [ ] System requirements installed
- [ ] Directory structure created  
- [ ] All Python files present
- [ ] Quick test script ready

## Phase 2: Your Action Items ⚠️ (Only you can do these)
- [ ] **Get Alpha Vantage API key** ($12.99/month): https://www.alphavantage.co/
- [ ] **Get Finnhub API key** (Free tier available): https://finnhub.io/
- [ ] **Get Twelve Data API key** ($8/month): https://twelvedata.com/  
- [ ] **Get Polygon API key** ($9/month): https://polygon.io/
- [ ] **Create Telegram bot** (Free): Message @BotFather
- [ ] **Get your Telegram User ID** (Free): Message @userinfobot
- [ ] **Set up Pocket Option account** (Your choice): https://pocketoption.com/
- [ ] **Create .env file** with real API keys
- [ ] **Run quick_test.py** to validate setup

## Phase 3: Training & Testing ⚡ (We work together)
- [ ] **Run model training** (I provide scripts, you run them)
- [ ] **Validate model accuracy** (Target: 85%+)
- [ ] **Paper trading test** (7 days minimum)
- [ ] **System integration test**

## Phase 4: Production Deployment 🚀 (Your infrastructure)
- [ ] **Get VPS/server** (DigitalOcean, AWS, etc.)
- [ ] **Deploy system to server**
- [ ] **Configure monitoring**
- [ ] **Start live trading** (conservative)

## Estimated Costs:
- **Data APIs**: ~$30-50/month
- **VPS Server**: ~$10-20/month  
- **Total**: ~$40-70/month

## Estimated Timeline:
- **Phase 1**: 1 day (mostly automated)
- **Phase 2**: 2-3 days (API signups and setup)
- **Phase 3**: 1-2 weeks (training and testing)
- **Phase 4**: 3-5 days (deployment)

**Total**: 2-4 weeks to live trading
"""
        
        with open('/workspace/deployment_checklist.md', 'w') as f:
            f.write(checklist)
        
        print("✅ Deployment checklist created: deployment_checklist.md")
        self.steps_completed.append("Deployment checklist created")
    
    def create_user_action_script(self):
        """Create a script with user actions"""
        print("\n👤 Creating user action guide...")
        
        user_script = '''#!/bin/bash
# 👤 USER ACTION SCRIPT
# These are the things only YOU can do

echo "🚀 ULTIMATE TRADING SYSTEM - USER ACTIONS"
echo "========================================"

echo ""
echo "📋 STEP 1: Get API Keys (REQUIRED)"
echo "=================================="
echo "1. Alpha Vantage: https://www.alphavantage.co/"
echo "   - Click 'Get Free API Key'"
echo "   - Basic plan: $12.99/month"
echo ""
echo "2. Finnhub: https://finnhub.io/"
echo "   - Register for free account"
echo "   - Free tier: 60 calls/minute"
echo ""
echo "3. Twelve Data: https://twelvedata.com/"
echo "   - Basic plan: $8/month"
echo "   - 800 requests/day"
echo ""
echo "4. Polygon: https://polygon.io/"
echo "   - Starter plan: $9/month"
echo "   - Real-time data access"

echo ""
echo "📱 STEP 2: Telegram Bot Setup"
echo "============================="
echo "1. Message @BotFather on Telegram"
echo "2. Send: /newbot"
echo "3. Follow instructions to create bot"
echo "4. Save the bot token"
echo "5. Message @userinfobot to get your user ID"

echo ""
echo "🏦 STEP 3: Broker Account"
echo "========================"
echo "1. Sign up at Pocket Option: https://pocketoption.com/"
echo "2. Verify your account"
echo "3. Make a deposit (start small - $100-500)"
echo "4. Get SSID from browser developer tools"

echo ""
echo "⚙️ STEP 4: Configuration"
echo "======================="
echo "1. Copy .env.template to .env"
echo "2. Edit .env with your real API keys"
echo "3. Run: source .env"
echo "4. Test with: python3 quick_test.py"

echo ""
echo "🧪 STEP 5: Testing"
echo "=================="
echo "1. Run model training: python3 production_model_trainer.py"
echo "2. Paper trading: python3 production_trading_system.py --mode paper"
echo "3. Monitor for 7 days minimum"

echo ""
echo "🚀 STEP 6: Go Live"
echo "=================="
echo "1. Get VPS (DigitalOcean, AWS, Vultr)"
echo "2. Deploy system to server"
echo "3. Start with minimum position sizes"
echo "4. Monitor continuously"

echo ""
echo "💰 ESTIMATED COSTS:"
echo "==================="
echo "- Data APIs: $30-50/month"
echo "- VPS Server: $10-20/month"
echo "- Broker deposit: $100-500 (your choice)"
echo "- Total monthly: $40-70"

echo ""
echo "⏰ ESTIMATED TIMELINE:"
echo "======================"
echo "- API setup: 1-2 days"
echo "- Model training: 3-5 days" 
echo "- Paper trading: 7 days"
echo "- Deployment: 2-3 days"
echo "- Total: 2-3 weeks"

echo ""
echo "🆘 NEED HELP?"
echo "============="
echo "- Read: ULTIMATE_READINESS_ROADMAP.md"
echo "- Run: python3 quick_test.py"
echo "- Check logs in /workspace/logs/"
'''
        
        with open('/workspace/user_actions.sh', 'w') as f:
            f.write(user_script)
        
        os.chmod('/workspace/user_actions.sh', 0o755)
        print("✅ User action guide created: user_actions.sh")
        self.steps_completed.append("User action guide created")
    
    def run_setup(self):
        """Run the complete setup process"""
        print("🚀 ULTIMATE TRADING SYSTEM - QUICK START")
        print("=" * 60)
        
        # Run all setup steps
        self.check_system_requirements()
        self.setup_directory_structure()
        self.create_environment_template()
        self.validate_existing_files()
        self.create_quick_test_script()
        self.create_deployment_checklist()
        self.create_user_action_script()
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 SETUP SUMMARY")
        print("=" * 60)
        
        print("\n✅ COMPLETED BY AI:")
        for step in self.steps_completed:
            print(f"  ✅ {step}")
        
        print("\n⚠️ REQUIRES YOUR ACTION:")
        for step in self.steps_pending:
            print(f"  ⚠️ {step}")
        
        print("\n🎯 NEXT STEPS:")
        print("1. Run: ./user_actions.sh (for detailed instructions)")
        print("2. Get your API keys (30 minutes)")
        print("3. Set up .env file (5 minutes)")  
        print("4. Run: python3 quick_test.py (validation)")
        print("5. Follow ULTIMATE_READINESS_ROADMAP.md")
        
        print("\n🤝 COLLABORATION MODEL:")
        print("- I handle: All code, architecture, optimization")
        print("- You handle: API keys, accounts, infrastructure, deployment")
        
        print("\n💰 INVESTMENT NEEDED:")
        print("- Time: 2-4 weeks")
        print("- Money: $40-70/month + initial broker deposit")
        print("- Effort: ~2-3 hours/day during setup")
        
        print("\n🎉 RESULT:")
        print("- 100% automated trading system")
        print("- 24/7 operation") 
        print("- 85%+ accuracy target")
        print("- Professional risk management")

if __name__ == "__main__":
    guide = QuickStartGuide()
    guide.run_setup()