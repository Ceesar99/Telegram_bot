#!/usr/bin/env python3
"""
🚀 Unified Trading System Launcher
Real-time 24/7 Trading with Telegram Bot Integration

This script starts the complete trading system including:
- Telegram Bot for signal delivery
- Signal generation engine with AI/ML models
- Real-time market data processing
- Performance tracking and risk management
"""

import asyncio
import logging
import sys
import os
import signal
import threading
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/workspace')

# Import core components
from telegram_bot import TradingBot
from signal_engine import SignalEngine
from pocket_option_api import PocketOptionAPI
from performance_tracker import PerformanceTracker
from config import LOGGING_CONFIG

class TradingSystemLauncher:
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger('TradingSystemLauncher')
        self.running = False
        self.telegram_bot = None
        self.signal_engine = None
        self.pocket_api = None
        self.performance_tracker = None
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=getattr(logging, LOGGING_CONFIG.get('level', 'INFO')),
            format=LOGGING_CONFIG.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.FileHandler('/workspace/logs/trading_system.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
    async def initialize_components(self):
        """Initialize all trading system components"""
        try:
            self.logger.info("🚀 Initializing Unified Trading System...")
            
            # Initialize Pocket Option API
            self.logger.info("📡 Initializing Pocket Option API...")
            self.pocket_api = PocketOptionAPI()
            
            # Initialize Performance Tracker
            self.logger.info("📊 Initializing Performance Tracker...")
            self.performance_tracker = PerformanceTracker()
            
            # Initialize Signal Engine
            self.logger.info("🎯 Initializing Signal Engine...")
            self.signal_engine = SignalEngine()
            
            # Initialize Telegram Bot
            self.logger.info("🤖 Initializing Telegram Bot...")
            self.telegram_bot = TradingBot()
            
            self.logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize components: {e}")
            return False
    
    async def start_telegram_bot(self):
        """Start the Telegram bot"""
        try:
            self.logger.info("🤖 Starting Telegram Bot...")
            
            # Build the application
            application = self.telegram_bot.build_application()
            
            # Start the bot
            await application.initialize()
            await application.start()
            await application.updater.start_polling()
            
            self.logger.info("✅ Telegram Bot started successfully")
            
            # Keep the bot running
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"❌ Telegram Bot error: {e}")
        finally:
            if hasattr(self.telegram_bot, 'app') and self.telegram_bot.app:
                await self.telegram_bot.app.updater.stop()
                await self.telegram_bot.app.stop()
                await self.telegram_bot.app.shutdown()
    
    async def signal_generation_loop(self):
        """Main signal generation loop"""
        try:
            self.logger.info("🎯 Starting signal generation loop...")
            
            while self.running:
                try:
                    # Generate signal
                    signal_data = await self.signal_engine.generate_signal()
                    
                    if signal_data and signal_data.get('accuracy', 0) >= 95.0:
                        self.logger.info(f"✅ High accuracy signal generated: {signal_data.get('pair')} - {signal_data.get('direction')}")
                        
                        # Send signal via Telegram bot if available
                        if self.telegram_bot and hasattr(self.telegram_bot, 'send_signal_to_users'):
                            await self.telegram_bot.send_signal_to_users(signal_data)
                    
                    # Wait before next signal generation
                    await asyncio.sleep(60)  # Check every minute
                    
                except Exception as e:
                    self.logger.error(f"Error in signal generation loop: {e}")
                    await asyncio.sleep(10)  # Wait before retry
                    
        except Exception as e:
            self.logger.error(f"❌ Signal generation loop failed: {e}")
    
    async def performance_monitoring_loop(self):
        """Monitor system performance"""
        try:
            self.logger.info("📊 Starting performance monitoring...")
            
            while self.running:
                try:
                    # Update performance metrics
                    if self.performance_tracker:
                        # Log current performance
                        stats = self.performance_tracker.get_daily_stats()
                        if stats:
                            self.logger.info(f"📈 Daily Stats - Win Rate: {stats.get('win_rate', 0):.1f}%")
                    
                    # Wait 5 minutes between performance checks
                    await asyncio.sleep(300)
                    
                except Exception as e:
                    self.logger.error(f"Error in performance monitoring: {e}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            self.logger.error(f"❌ Performance monitoring failed: {e}")
    
    async def run(self):
        """Main execution method"""
        try:
            self.logger.info("🚀 UNIFIED TRADING SYSTEM STARTUP")
            self.logger.info("=" * 50)
            
            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            # Initialize components
            if not await self.initialize_components():
                self.logger.error("❌ Failed to initialize, exiting...")
                return
            
            self.running = True
            
            # Display startup banner
            print("\n" + "=" * 60)
            print("🚀 UNIFIED TRADING SYSTEM - ACTIVE")
            print("=" * 60)
            print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("🤖 Telegram Bot: ACTIVE")
            print("🎯 Signal Engine: ACTIVE") 
            print("📡 Pocket Option API: CONNECTED")
            print("📊 Performance Tracking: ACTIVE")
            print("=" * 60)
            print("\n✅ System is ready for 24/7 trading!")
            print("📱 Send /start to your Telegram bot to begin")
            print("\n💡 To stop the system: Ctrl+C")
            print("-" * 60)
            
            # Start all components concurrently
            tasks = await asyncio.gather(
                self.start_telegram_bot(),
                self.signal_generation_loop(),
                self.performance_monitoring_loop(),
                return_exceptions=True
            )
            
            # Log any exceptions from tasks
            for i, task in enumerate(tasks):
                if isinstance(task, Exception):
                    self.logger.error(f"Task {i} failed: {task}")
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"❌ System error: {e}")
        finally:
            self.logger.info("🔄 Shutting down trading system...")
            self.running = False
            
            # Cleanup
            if self.signal_engine:
                self.signal_engine.cleanup()
            if self.pocket_api:
                self.pocket_api.disconnect()
                
            self.logger.info("✅ Trading system shutdown complete")

def main():
    """Main entry point"""
    print("🚀 Starting Unified Trading System...")
    
    # Ensure required directories exist
    os.makedirs('/workspace/logs', exist_ok=True)
    os.makedirs('/workspace/data', exist_ok=True)
    os.makedirs('/workspace/backup', exist_ok=True)
    
    # Create and run the launcher
    launcher = TradingSystemLauncher()
    
    try:
        # Run the system
        asyncio.run(launcher.run())
    except Exception as e:
        print(f"❌ Failed to start trading system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()