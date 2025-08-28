#!/usr/bin/env python3
"""
🚀 LAUNCH SCRIPT - ULTIMATE TRADING SYSTEM
Properly launches the complete system with all components
"""

import sys
import os
import asyncio
import logging
import signal
import time
from pathlib import Path

# Add project root to path
sys.path.append('/workspace')

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ultimate_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class SystemLauncher:
    """Launches and manages the Ultimate Trading System"""
    
    def __init__(self):
        self.running = False
        self.components = {}
        
    async def launch_system(self):
        """Launch the complete Ultimate Trading System"""
        print("🏆 ULTIMATE TRADING SYSTEM - LAUNCHING")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        try:
            # Step 1: Initialize core components
            print("🔧 Initializing core components...")
            await self._initialize_core_components()
            
            # Step 2: Launch Telegram Bot
            print("🤖 Launching Telegram Bot...")
            await self._launch_telegram_bot()
            
            # Step 3: Launch Trading System
            print("📊 Launching Trading System...")
            await self._launch_trading_system()
            
            # Step 4: Start monitoring
            print("📈 Starting system monitoring...")
            await self._start_monitoring()
            
            print("✅ SYSTEM LAUNCHED SUCCESSFULLY!")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("🎯 Your Ultimate Trading System is now operational!")
            print("📱 Use your Telegram bot to interact with the system")
            print("📊 Check /status for system health")
            print("🚀 Ready to generate trading signals!")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            # Keep system running
            self.running = True
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Failed to launch system: {e}")
            print(f"❌ SYSTEM LAUNCH FAILED: {e}")
            sys.exit(1)
    
    async def _initialize_core_components(self):
        """Initialize core system components"""
        try:
            # Import and test core components
            from ultimate_trading_system import UltimateTradingSystem
            from ultimate_telegram_bot import UltimateTradingBot
            
            print("✅ Core components imported successfully")
            
            # Test component initialization
            print("🧪 Testing component initialization...")
            
            # Test trading system
            trading_system = UltimateTradingSystem()
            await trading_system.initialize_system()
            self.components['trading_system'] = trading_system
            print("✅ Trading system initialized")
            
            # Test Telegram bot
            bot = UltimateTradingBot()
            # Telegram bot doesn't need explicit initialization
            self.components['telegram_bot'] = bot
            print("✅ Telegram bot initialized")
            
        except Exception as e:
            logger.error(f"Core component initialization failed: {e}")
            raise
    
    async def _launch_telegram_bot(self):
        """Launch the Telegram bot"""
        try:
            bot = self.components['telegram_bot']
            
            # Start the bot in background
            bot_task = asyncio.create_task(bot.run_continuous())
            self.components['bot_task'] = bot_task
            
            print("✅ Telegram bot launched successfully")
            
        except Exception as e:
            logger.error(f"Telegram bot launch failed: {e}")
            raise
    
    async def _launch_trading_system(self):
        """Launch the trading system"""
        try:
            trading_system = self.components['trading_system']
            
            # Start trading system in background
            trading_task = asyncio.create_task(trading_system.start_trading())
            self.components['trading_task'] = trading_task
            
            print("✅ Trading system launched successfully")
            
        except Exception as e:
            logger.error(f"Trading system launch failed: {e}")
            raise
    
    async def _start_monitoring(self):
        """Start system monitoring"""
        try:
            # Start monitoring task
            monitor_task = asyncio.create_task(self._monitor_system())
            self.components['monitor_task'] = monitor_task
            
            print("✅ System monitoring started")
            
        except Exception as e:
            logger.error(f"Monitoring start failed: {e}")
            raise
    
    async def _monitor_system(self):
        """Monitor system health"""
        while self.running:
            try:
                # Check component health
                for name, component in self.components.items():
                    if hasattr(component, 'is_healthy'):
                        if not component.is_healthy():
                            logger.warning(f"Component {name} health check failed")
                
                # Log system status
                logger.info("System health check completed")
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(10)
    
    def shutdown(self):
        """Gracefully shutdown the system"""
        print("\n🔄 Shutting down Ultimate Trading System...")
        self.running = False
        
        # Cancel all tasks
        for name, task in self.components.items():
            if isinstance(task, asyncio.Task):
                task.cancel()
        
        print("✅ System shutdown complete")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"\n📡 Received signal {signum}, shutting down...")
    if hasattr(signal_handler, 'launcher'):
        signal_handler.launcher.shutdown()
    sys.exit(0)

async def main():
    """Main entry point"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and launch system
    launcher = SystemLauncher()
    signal_handler.launcher = launcher
    
    try:
        await launcher.launch_system()
    except KeyboardInterrupt:
        print("\n🔄 Keyboard interrupt received")
        launcher.shutdown()
    except Exception as e:
        logger.error(f"System error: {e}")
        launcher.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    print("🚀 Starting Ultimate Trading System Launcher...")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Run the system
    asyncio.run(main())