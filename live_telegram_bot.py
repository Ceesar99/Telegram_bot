#!/usr/bin/env python3
"""
🤖 LIVE TELEGRAM TRADING BOT
Continuous operation with full command response capability
"""

import asyncio
import logging
import signal
import sys
import os
from datetime import datetime
import json

# Add project root to path
sys.path.append('/workspace')

from working_telegram_bot import WorkingTradingBot
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_USER_ID

class LiveTelegramBot:
    """Live Telegram bot with continuous operation and command response"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger('LiveTelegramBot')
        self.is_running = False
        self.shutdown_requested = False
        self.start_time = None
        self.bot = None
        self.application = None
        self.stats = {
            'commands_processed': 0,
            'messages_sent': 0,
            'uptime_start': None,
            'last_command': None
        }
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        try:
            os.makedirs('/workspace/logs', exist_ok=True)
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('/workspace/logs/live_telegram_bot.log'),
                    logging.StreamHandler(sys.stdout)
                ]
            )
            
            # Reduce httpx logging noise
            logging.getLogger('httpx').setLevel(logging.WARNING)
            
        except Exception as e:
            print(f"Error setting up logging: {e}")
            sys.exit(1)
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def print_startup_banner(self):
        """Print detailed startup banner"""
        banner = f"""
🤖 LIVE TELEGRAM TRADING BOT - STARTING UP
════════════════════════════════════════════════════════════════════
📱 Bot Token: {TELEGRAM_BOT_TOKEN[:20]}...
👤 Authorized User: {TELEGRAM_USER_ID}
🕐 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🌍 Mode: CONTINUOUS OPERATION
📡 Status: CONNECTING TO TELEGRAM...
════════════════════════════════════════════════════════════════════
"""
        print(banner)
        self.logger.info("Live Telegram Trading Bot starting up...")
    
    async def initialize_bot(self):
        """Initialize the Telegram bot with enhanced monitoring"""
        try:
            self.logger.info("🤖 Initializing Telegram Bot...")
            
            # Create bot instance
            self.bot = WorkingTradingBot()
            self.logger.info("✅ Bot instance created")
            
            # Build application
            self.application = self.bot.build_application()
            self.logger.info("✅ Application built with all handlers")
            
            # Initialize application
            await self.application.initialize()
            self.logger.info("✅ Application initialized")
            
            # Start application
            await self.application.start()
            self.logger.info("✅ Application started")
            
            # Test bot connection
            bot_info = await self.application.bot.get_me()
            self.logger.info(f"✅ Bot connected: @{bot_info.username} ({bot_info.first_name})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Bot initialization failed: {e}")
            return False
    
    async def start_polling(self):
        """Start polling for messages with detailed logging"""
        try:
            self.logger.info("🔄 Starting message polling...")
            
            # Start polling
            await self.application.updater.start_polling(
                poll_interval=1.0,  # Check every second
                timeout=30,         # 30 second timeout
                bootstrap_retries=5
            )
            
            self.logger.info("✅ Polling started successfully")
            self.logger.info("📱 Bot is now LIVE and ready to receive commands!")
            self.logger.info(f"🎯 Send /start to @{(await self.application.bot.get_me()).username} to test!")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Polling failed to start: {e}")
            return False
    
    async def monitor_bot_health(self):
        """Monitor bot health and log activity"""
        health_check_count = 0
        
        while self.is_running and not self.shutdown_requested:
            try:
                health_check_count += 1
                
                # Check if updater is still running
                if self.application and self.application.updater.running:
                    uptime = datetime.now() - self.start_time
                    self.logger.info(f"💚 Bot Health Check #{health_check_count} - Status: HEALTHY")
                    self.logger.info(f"⏰ Uptime: {uptime}")
                    self.logger.info(f"📊 Commands Processed: {self.stats['commands_processed']}")
                    self.logger.info(f"📤 Messages Sent: {self.stats['messages_sent']}")
                    
                    # Log last activity
                    if self.stats['last_command']:
                        self.logger.info(f"🎯 Last Command: {self.stats['last_command']}")
                    
                else:
                    self.logger.warning("⚠️ Bot updater is not running!")
                
                # Wait 30 seconds before next health check
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(10)
    
    async def run_bot(self):
        """Main bot operation"""
        try:
            self.print_startup_banner()
            
            # Validate configuration
            if not TELEGRAM_BOT_TOKEN or len(TELEGRAM_BOT_TOKEN) < 10:
                self.logger.error("❌ Invalid Telegram Bot Token")
                return False
            
            if not TELEGRAM_USER_ID or not str(TELEGRAM_USER_ID).isdigit():
                self.logger.error("❌ Invalid Telegram User ID")
                return False
            
            self.logger.info("✅ Configuration validated")
            
            # Initialize bot
            if not await self.initialize_bot():
                return False
            
            # Start polling
            if not await self.start_polling():
                return False
            
            # Mark as running
            self.is_running = True
            self.start_time = datetime.now()
            self.stats['uptime_start'] = self.start_time
            
            self.logger.info("🚀 BOT IS NOW LIVE AND OPERATIONAL!")
            self.logger.info("=" * 60)
            self.logger.info("📱 TELEGRAM BOT COMMANDS TO TEST:")
            self.logger.info("   /start - Initialize bot and get welcome message")
            self.logger.info("   /signal - Get AI-powered trading signal")
            self.logger.info("   /status - Check bot and system status")
            self.logger.info("   /help - Show all available commands")
            self.logger.info("   /test - Run bot functionality test")
            self.logger.info("=" * 60)
            
            # Start health monitoring
            monitor_task = asyncio.create_task(self.monitor_bot_health())
            
            # Wait for shutdown signal
            while self.is_running and not self.shutdown_requested:
                await asyncio.sleep(1)
            
            # Cancel monitoring
            monitor_task.cancel()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Bot operation error: {e}")
            return False
    
    async def shutdown_bot(self):
        """Graceful bot shutdown"""
        try:
            self.logger.info("🛑 Shutting down bot...")
            
            if self.application:
                # Stop updater
                if self.application.updater.running:
                    await self.application.updater.stop()
                    self.logger.info("✅ Updater stopped")
                
                # Stop application
                await self.application.stop()
                self.logger.info("✅ Application stopped")
                
                # Shutdown application
                await self.application.shutdown()
                self.logger.info("✅ Application shutdown complete")
            
            # Final statistics
            if self.start_time:
                uptime = datetime.now() - self.start_time
                self.logger.info(f"📊 Final Statistics:")
                self.logger.info(f"   Total Uptime: {uptime}")
                self.logger.info(f"   Commands Processed: {self.stats['commands_processed']}")
                self.logger.info(f"   Messages Sent: {self.stats['messages_sent']}")
            
            self.logger.info("✅ Bot shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

async def main():
    """Main function"""
    bot = LiveTelegramBot()
    
    try:
        success = await bot.run_bot()
        if success:
            print("\n🎉 Bot ran successfully!")
        else:
            print("\n❌ Bot failed to run")
            return False
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False
    finally:
        await bot.shutdown_bot()
    
    return True

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        sys.exit(1)