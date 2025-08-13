#!/usr/bin/env python3
"""
🚀 ULTIMATE TRADING SYSTEM - DIRECT BOT RUNNER
Version: 1.0.0 - Direct Execution

This script runs the Ultimate Trading System Telegram bot directly
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
sys.path.append('/workspace')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """🚀 Main entry point"""
    logger = logging.getLogger('UltimateBotRunner')
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  🏆 ULTIMATE TRADING SYSTEM - TELEGRAM BOT 🏆               ║
║                                                              ║
║  📱 Professional Trading Interface                          ║
║  🤖 Intelligent Signal Generation                           ║
║  ⚡ Real-time Market Analysis                               ║
║  🔒 Institutional-Grade Security                            ║
║                                                              ║
║  Status: 🟢 STARTING...                                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Import and run the simplified bot
        from simple_telegram_interface import SimpleTelegramBot
        
        logger.info("🚀 Initializing Ultimate Trading System...")
        bot = SimpleTelegramBot()
        
        logger.info("🔥 Starting Telegram Bot...")
        logger.info("📱 Bot is now ready to respond to commands!")
        logger.info("💬 Send /start to begin trading")
        
        # Run the bot
        asyncio.run(bot.run())
        
    except KeyboardInterrupt:
        logger.info("🛑 Bot shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Bot error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()