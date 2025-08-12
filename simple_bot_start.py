#!/usr/bin/env python3
"""
Simple Telegram Bot Startup
Direct bot initialization without complex launcher
"""

import sys
import asyncio
import logging
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/workspace')

from telegram_bot import TradingBot
from config import TELEGRAM_BOT_TOKEN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/logs/simple_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Start the Telegram bot directly"""
    try:
        logger.info("🚀 Starting Simple Telegram Bot...")
        
        # Initialize bot
        bot = TradingBot()
        
        # Get the application
        app = bot.build_application()
        
        logger.info("✅ Bot initialized successfully")
        logger.info(f"🤖 Bot Token: {TELEGRAM_BOT_TOKEN[:10]}...")
        logger.info("📱 Bot is ready to receive commands!")
        
        # Start polling
        await app.initialize()
        await app.start()
        await app.updater.start_polling()
        
        logger.info("✅ Bot is now running and polling for messages...")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Stopping bot...")
            await app.updater.stop()
            await app.stop()
            await app.shutdown()
            
    except Exception as e:
        logger.error(f"❌ Bot startup failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())