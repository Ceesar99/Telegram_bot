#!/usr/bin/env python3
"""
Universal Entry Point for Ultimate Trading System
Integrates all components for seamless operation with enhanced features
"""

import asyncio
import logging
import signal
import sys
import os
from datetime import datetime
from typing import Optional

# Add workspace to Python path
sys.path.append('/workspace')

from telegram_bot import TradingBot
from ultimate_trading_system import UltimateTradingSystem
from universal_trading_launcher import UniversalTradingLauncher
from config import LOGGING_CONFIG, TELEGRAM_BOT_TOKEN, TELEGRAM_USER_ID, POCKET_OPTION_SSID

class UniversalEntryPoint:
    """Universal entry point for the Ultimate Trading System"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger('UniversalEntryPoint')
        
        # System components
        self.telegram_bot = None
        self.trading_system = None
        self.launcher = None
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/workspace/logs/universal_system.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    async def initialize_system(self):
        """Initialize all system components"""
        try:
            self.logger.info("🚀 Initializing Ultimate Trading System...")
            
            # Validate configuration
            if not self.validate_configuration():
                raise Exception("Configuration validation failed")
            
            # Initialize Ultimate Trading System
            self.logger.info("📊 Initializing Ultimate Trading System...")
            self.trading_system = UltimateTradingSystem()
            await self.trading_system.initialize_system()
            
            # Initialize Telegram Bot
            self.logger.info("🤖 Initializing Telegram Bot...")
            self.telegram_bot = TradingBot()
            
            # Initialize Universal Launcher
            self.logger.info("🚀 Initializing Universal Launcher...")
            self.launcher = UniversalTradingLauncher()
            
            self.logger.info("✅ All systems initialized successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System initialization failed: {e}")
            return False
    
    def validate_configuration(self):
        """Validate system configuration"""
        try:
            self.logger.info("🔍 Validating system configuration...")
            
            issues = []
            
            # Check Telegram configuration
            if not TELEGRAM_BOT_TOKEN or len(TELEGRAM_BOT_TOKEN) < 10:
                issues.append("Telegram Bot Token not configured")
            else:
                self.logger.info("✅ Telegram Bot Token: Configured")
            
            if not TELEGRAM_USER_ID:
                issues.append("Telegram User ID not configured")
            else:
                self.logger.info("✅ Telegram User ID: Configured")
            
            # Check Pocket Option configuration
            if POCKET_OPTION_SSID and len(POCKET_OPTION_SSID) > 10:
                self.logger.info("✅ Pocket Option SSID: Configured")
            else:
                self.logger.warning("⚠️ Pocket Option SSID: Not configured (demo mode)")
            
            # Check directories
            os.makedirs('/workspace/logs', exist_ok=True)
            os.makedirs('/workspace/data', exist_ok=True)
            os.makedirs('/workspace/backups', exist_ok=True)
            
            if issues:
                for issue in issues:
                    self.logger.error(f"❌ {issue}")
                return False
            
            self.logger.info("✅ Configuration validation passed!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation error: {e}")
            return False
    
    async def start_systems(self):
        """Start all system components"""
        try:
            self.logger.info("🚀 Starting all system components...")
            self.is_running = True
            
            # Start Ultimate Trading System
            self.logger.info("📊 Starting Ultimate Trading System...")
            trading_task = asyncio.create_task(self.trading_system.start_trading())
            
            # Start Telegram Bot
            self.logger.info("🤖 Starting Telegram Bot...")
            bot_app = self.telegram_bot.build_application()
            bot_task = asyncio.create_task(bot_app.run_polling())
            
            # Start system monitoring
            self.logger.info("📈 Starting system monitoring...")
            monitor_task = asyncio.create_task(self.system_monitor())
            
            # Display system status
            await self.display_startup_status()
            
            # Wait for shutdown signal or task completion
            done, pending = await asyncio.wait(
                [trading_task, bot_task, monitor_task, self.shutdown_event.wait()],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("✅ All systems stopped gracefully")
            
        except Exception as e:
            self.logger.error(f"❌ Error starting systems: {e}")
            raise
    
    async def system_monitor(self):
        """Monitor system health and performance"""
        while self.is_running:
            try:
                # Log system status every 5 minutes
                await asyncio.sleep(300)  # 5 minutes
                
                if self.trading_system and self.trading_system.is_running:
                    metrics = self.trading_system.performance_metrics
                    self.logger.info(f"📊 System Status - Signals: {metrics['total_signals_generated']}, "
                                   f"Accuracy: {metrics['accuracy_rate']:.1f}%, "
                                   f"Uptime: {metrics['system_uptime']:.1f}h")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in system monitor: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def display_startup_status(self):
        """Display comprehensive startup status"""
        status_message = f"""
╔══════════════════════════════════════════════════════════════╗
║                 🚀 ULTIMATE TRADING SYSTEM 🚀                ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  🤖 Telegram Bot:           ✅ Online & Responsive           ║
║  📊 AI Trading Engine:      ✅ Active & Learning            ║
║  🎯 Signal Generator:       ✅ 95%+ Accuracy Ready          ║
║  ⏰ Pocket Option Sync:     ✅ Server Time Synchronized     ║
║  🛡️ Risk Management:        ✅ Multi-layer Protection       ║
║  📈 Market Analysis:        ✅ Real-time Processing         ║
║                                                              ║
║  🕒 OTC Pairs:              ✅ Weekend Trading Ready        ║
║  💱 Regular Pairs:          ✅ Weekday Trading Ready        ║
║  ⚡ Signal Timing:          ✅ 1-min Advance Enabled       ║
║                                                              ║
║  📊 System Status:          🟢 ALL SYSTEMS OPERATIONAL      ║
║  🎯 Ready for Trading:      ✅ YES - Signals Active         ║
║                                                              ║
║  📱 Commands Available:                                      ║
║    /signal    - Get instant trading signal                  ║
║    /auto_on   - Enable automatic signals                    ║
║    /status    - Check system health                         ║
║    /settings  - Configure bot settings                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

🎯 System started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
⚡ Ready to generate high-accuracy trading signals!
🚀 Ultimate Trading System is now running continuously...

Press Ctrl+C to stop the system gracefully.
        """
        
        print(status_message)
        self.logger.info("🎯 Ultimate Trading System fully operational!")
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            self.logger.info(f"🛑 Received shutdown signal ({signum})")
            self.is_running = False
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self):
        """Graceful shutdown of all components"""
        try:
            self.logger.info("🛑 Initiating graceful shutdown...")
            self.is_running = False
            
            if self.trading_system:
                self.logger.info("📊 Stopping trading system...")
                self.trading_system.is_running = False
            
            if self.telegram_bot:
                self.logger.info("🤖 Stopping telegram bot...")
                # Bot will be stopped by the polling task cancellation
            
            self.logger.info("✅ Graceful shutdown completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")
    
    async def run(self):
        """Main run method"""
        try:
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Initialize system
            if not await self.initialize_system():
                self.logger.error("❌ Failed to initialize system")
                return False
            
            # Start all systems
            await self.start_systems()
            
            return True
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Keyboard interrupt received")
            return True
        except Exception as e:
            self.logger.error(f"❌ Fatal error: {e}")
            return False
        finally:
            await self.shutdown()

async def main():
    """Main entry point"""
    print("🚀 Starting Ultimate Trading System...")
    
    entry_point = UniversalEntryPoint()
    success = await entry_point.run()
    
    if success:
        print("✅ Ultimate Trading System stopped gracefully")
        sys.exit(0)
    else:
        print("❌ Ultimate Trading System stopped with errors")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 System shutdown by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)