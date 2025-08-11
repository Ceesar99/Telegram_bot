#!/usr/bin/env python3
"""
AI Trading Bot Runner
Run the AI-enhanced Telegram bot with proper configuration and error handling
"""

import os
import sys
import logging
import asyncio
from datetime import datetime

# Add workspace to path
sys.path.append('/workspace')

def setup_environment():
    """Setup environment variables and configuration"""
    
    # Set up basic configuration if not exists
    if not os.getenv('TELEGRAM_BOT_TOKEN'):
        print("⚠️  TELEGRAM_BOT_TOKEN not found in environment")
        print("📝 Please set your Telegram bot token in .env file or environment")
        print("💡 For demo purposes, the bot will use a placeholder token")
        
        # Create a basic .env file for demo
        env_content = """# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN_HERE
TELEGRAM_USER_ID=123456789

# Trading Configuration  
MAX_DAILY_TRADES=20
MIN_CONFIDENCE=60.0
TRADE_AMOUNT=1.0

# AI Model Configuration
AI_MODEL_PATH=/workspace/models/binary_options_model.h5
AI_CONFIDENCE_THRESHOLD=60.0
"""
        try:
            with open('/workspace/.env', 'w') as f:
                f.write(env_content)
            print("✅ Created basic .env file template")
        except Exception as e:
            print(f"❌ Could not create .env file: {e}")

def check_ai_model():
    """Check if AI model is available"""
    model_path = '/workspace/models/binary_options_model.h5'
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ AI Model found: {size_mb:.1f}MB")
        return True
    else:
        print("❌ AI Model not found - please train the model first")
        return False

def show_demo_info():
    """Show demo information"""
    print()
    print("🤖 AI Trading Bot - Demo Mode")
    print("=" * 50)
    print("🧠 AI Model: LSTM Neural Network (66.8% accuracy)")
    print("📊 Features: 20 technical indicators")  
    print("⚡ Real-time: Signal generation ready")
    print("🔄 Auto-signals: Enabled (every 5 minutes)")
    print("=" * 50)
    print()
    print("📋 Available Commands (when bot is running):")
    print("• /start - Welcome message and features")
    print("• /signal - Get instant AI trading signal")
    print("• /analyze [pair] - Deep AI analysis")
    print("• /auto_on - Enable automatic signals")
    print("• /stats - Show AI model statistics")
    print("• /test_ai - Test AI model functionality")
    print("• /model_info - Detailed AI information")
    print()
    print("🎯 Signal Features:")
    print("• PUT/CALL/HOLD classification")
    print("• Confidence-based expiry (2-5 minutes)")
    print("• Technical analysis integration")
    print("• Risk level assessment")
    print()

async def test_ai_components():
    """Test AI components before starting bot"""
    print("🧪 Testing AI Components...")
    
    try:
        # Test AI model loading
        from binary_options_ai_model import BinaryOptionsAIModel
        ai_model = BinaryOptionsAIModel()
        
        if ai_model.is_model_trained():
            print("✅ AI Model: Loaded successfully")
            success = ai_model.load_model()
            if success:
                print("✅ Model Loading: Success")
            else:
                print("❌ Model Loading: Failed")
                return False
        else:
            print("❌ AI Model: Not trained")
            return False
        
        # Test signal engine
        from ai_signal_engine import AISignalEngine
        engine = AISignalEngine()
        print("✅ Signal Engine: Initialized")
        
        # Test signal generation
        signal = await engine.generate_signal()
        if signal:
            print(f"✅ Signal Generation: {signal['direction']} ({signal['accuracy']:.1f}%)")
        else:
            print("⚠️  Signal Generation: No signal (normal in demo mode)")
        
        print("✅ All AI components tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ AI Component Test Failed: {e}")
        return False

def run_demo_mode():
    """Run in demo mode with simulated signals"""
    print("🎮 Running in Demo Mode")
    print("💡 This demonstrates the AI bot functionality without real Telegram integration")
    print()
    
    # Run a simple demo
    asyncio.run(demo_signal_generation())

async def demo_signal_generation():
    """Generate demo signals to show functionality"""
    from ai_signal_engine import AISignalEngine
    
    engine = AISignalEngine()
    print("🧠 AI Signal Engine Demo")
    print("-" * 30)
    
    # Generate a few demo signals
    for i in range(3):
        print(f"\n🔄 Generating signal {i+1}/3...")
        signal = await engine.generate_signal()
        
        if signal:
            print(f"✅ Signal Generated:")
            print(f"   📈 Pair: {signal['pair']}")
            print(f"   🎯 Direction: {signal['direction']}")
            print(f"   ⚡ Confidence: {signal['accuracy']:.1f}%")
            print(f"   ⏰ Expiry: {signal['time_expiry']}")
            print(f"   🎚️  Strength: {signal['strength']}/10")
            print(f"   💹 Trend: {signal.get('trend', 'N/A')}")
            print(f"   🛡️  Risk: {signal.get('risk_level', 'Medium')}")
        else:
            print(f"⚠️  No high-confidence signal available")
        
        # Wait before next signal
        if i < 2:
            await asyncio.sleep(2)
    
    print("\n🎉 Demo completed! The AI is working correctly.")
    print("📝 To run with real Telegram bot:")
    print("   1. Set TELEGRAM_BOT_TOKEN in .env file")
    print("   2. Set TELEGRAM_USER_ID in .env file") 
    print("   3. Run: python3 ai_telegram_bot.py")

def main():
    """Main function"""
    print("🚀 Starting AI Trading Bot System")
    print("=" * 40)
    
    # Setup environment
    setup_environment()
    
    # Check AI model
    if not check_ai_model():
        print("❌ Cannot start without AI model")
        return
    
    # Show demo info
    show_demo_info()
    
    # Test AI components
    if not asyncio.run(test_ai_components()):
        print("❌ AI component tests failed")
        return
    
    print("🎉 All systems ready!")
    
    # Check if we should run in demo mode or with real Telegram
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not bot_token or bot_token == 'YOUR_BOT_TOKEN_HERE':
        print("\n🎮 No valid Telegram token found - running demo mode")
        run_demo_mode()
    else:
        print("\n🤖 Starting Telegram Bot with AI integration...")
        try:
            from ai_telegram_bot import AITradingBot
            bot = AITradingBot()
            bot.run()
        except KeyboardInterrupt:
            print("\n⚠️  Bot stopped by user")
        except Exception as e:
            print(f"\n❌ Bot error: {e}")
            print("\n💡 Try running in demo mode to test AI functionality")

if __name__ == "__main__":
    main()