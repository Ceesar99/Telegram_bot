#!/usr/bin/env python3
"""
Unified Trading System Startup Script
Handles all system components and ensures proper initialization
"""

import asyncio
import logging
import sys
import time
import signal
import os
from pathlib import Path

# Add workspace to Python path
sys.path.insert(0, '/workspace')

from config import *
from signal_engine import SignalEngine
from telegram_bot import TradingBot
from performance_tracker import PerformanceTracker
from risk_manager import RiskManager
from data_manager import DataManager

class SystemManager:
    def __init__(self):
        self.logger = self._setup_logger()
        self.components = {}
        self.running = False
        self.shutdown_event = asyncio.Event()
        
    def _setup_logger(self):
        """Setup system logger"""
        logger = logging.getLogger('SystemManager')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        os.makedirs('/workspace/logs', exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler('/workspace/logs/system_manager.log')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    async def initialize_components(self):
        """Initialize all system components"""
        try:
            self.logger.info("🚀 Initializing Unified Trading System...")
            
            # Initialize data manager
            self.logger.info("📊 Initializing Data Manager...")
            self.components['data_manager'] = DataManager()
            
            # Initialize performance tracker
            self.logger.info("📈 Initializing Performance Tracker...")
            self.components['performance_tracker'] = PerformanceTracker()
            
            # Initialize risk manager
            self.logger.info("⚠️ Initializing Risk Manager...")
            self.components['risk_manager'] = RiskManager()
            
            # Initialize signal engine
            self.logger.info("🧠 Initializing Signal Engine...")
            self.components['signal_engine'] = SignalEngine()
            
            # Wait for signal engine to initialize
            await asyncio.sleep(5)
            
            # Initialize Telegram bot
            self.logger.info("🤖 Initializing Telegram Bot...")
            self.components['telegram_bot'] = TradingBot()
            
            self.logger.info("✅ All components initialized successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize components: {e}")
            return False
    
    async def start_components(self):
        """Start all system components"""
        try:
            self.logger.info("🚀 Starting system components...")
            
            # Start Telegram bot in background
            if 'telegram_bot' in self.components:
                bot_task = asyncio.create_task(self._run_telegram_bot())
                self.components['bot_task'] = bot_task
                self.logger.info("✅ Telegram bot started")
            
            # Start signal generation loop
            signal_task = asyncio.create_task(self._signal_generation_loop())
            self.components['signal_task'] = signal_task
            self.logger.info("✅ Signal generation started")
            
            # Start monitoring loop
            monitor_task = asyncio.create_task(self._monitoring_loop())
            self.components['monitor_task'] = monitor_task
            self.logger.info("✅ System monitoring started")
            
            self.running = True
            self.logger.info("🎉 All components started successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start components: {e}")
            return False
    
    async def _run_telegram_bot(self):
        """Run Telegram bot in background"""
        try:
            bot = self.components['telegram_bot']
            await bot.start_bot()
        except Exception as e:
            self.logger.error(f"Telegram bot error: {e}")
    
    async def _signal_generation_loop(self):
        """Main signal generation loop"""
        while self.running and not self.shutdown_event.is_set():
            try:
                if 'signal_engine' in self.components:
                    signal_engine = self.components['signal_engine']
                    
                    # Generate signal every 5 minutes
                    signal = await signal_engine.generate_signal()
                    if signal:
                        self.logger.info(f"📊 Generated signal: {signal.get('symbol', 'Unknown')} - {signal.get('direction', 'Unknown')}")
                    
                    await asyncio.sleep(300)  # 5 minutes
                else:
                    await asyncio.sleep(60)
                    
            except Exception as e:
                self.logger.error(f"Signal generation error: {e}")
                await asyncio.sleep(60)
    
    async def _monitoring_loop(self):
        """System monitoring loop"""
        while self.running and not self.shutdown_event.is_set():
            try:
                # Check component health
                health_status = await self._check_system_health()
                
                if not health_status['overall_healthy']:
                    self.logger.warning("⚠️ System health issues detected")
                    for component, status in health_status['components'].items():
                        if not status['healthy']:
                            self.logger.warning(f"  {component}: {status['issue']}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _check_system_health(self):
        """Check overall system health"""
        health_status = {
            'overall_healthy': True,
            'components': {},
            'timestamp': time.time()
        }
        
        try:
            # Check signal engine
            if 'signal_engine' in self.components:
                signal_engine = self.components['signal_engine']
                health_status['components']['signal_engine'] = {
                    'healthy': signal_engine.is_model_loaded() and signal_engine.is_data_connected(),
                    'model_loaded': signal_engine.is_model_loaded(),
                    'data_connected': signal_engine.is_data_connected(),
                    'issue': None
                }
                
                if not health_status['components']['signal_engine']['healthy']:
                    health_status['overall_healthy'] = False
                    if not signal_engine.is_model_loaded():
                        health_status['components']['signal_engine']['issue'] = "Model not loaded"
                    elif not signal_engine.is_data_connected():
                        health_status['components']['signal_engine']['issue'] = "Data not connected"
            
            # Check performance tracker
            if 'performance_tracker' in self.components:
                performance_tracker = self.components['performance_tracker']
                db_ok = performance_tracker.test_connection()
                health_status['components']['performance_tracker'] = {
                    'healthy': db_ok,
                    'database_ok': db_ok,
                    'issue': None if db_ok else "Database connection failed"
                }
                
                if not db_ok:
                    health_status['overall_healthy'] = False
            
            # Check data manager
            if 'data_manager' in self.components:
                data_manager = self.components['data_manager']
                symbols_available = len(data_manager.get_available_symbols()) > 0
                health_status['components']['data_manager'] = {
                    'healthy': symbols_available,
                    'symbols_available': symbols_available,
                    'issue': None if symbols_available else "No symbols available"
                }
                
                if not symbols_available:
                    health_status['overall_healthy'] = False
                    
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            health_status['overall_healthy'] = False
        
        return health_status
    
    async def shutdown(self):
        """Gracefully shutdown the system"""
        self.logger.info("🛑 Shutting down system...")
        self.running = False
        self.shutdown_event.set()
        
        # Cancel all tasks
        for task_name, task in self.components.items():
            if task_name.endswith('_task') and hasattr(task, 'cancel'):
                task.cancel()
                self.logger.info(f"Cancelled task: {task_name}")
        
        # Cleanup components
        for component_name, component in self.components.items():
            if hasattr(component, 'cleanup'):
                try:
                    component.cleanup()
                    self.logger.info(f"Cleaned up component: {component_name}")
                except Exception as e:
                    self.logger.error(f"Error cleaning up {component_name}: {e}")
        
        self.logger.info("✅ System shutdown complete")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.shutdown())
    
    async def run(self):
        """Main system run loop"""
        try:
            # Setup signal handlers
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            # Initialize components
            if not await self.initialize_components():
                self.logger.error("❌ Component initialization failed")
                return False
            
            # Start components
            if not await self.start_components():
                self.logger.error("❌ Component startup failed")
                return False
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System error: {e}")
            return False

async def main():
    """Main entry point"""
    system_manager = SystemManager()
    
    try:
        success = await system_manager.run()
        if success:
            print("✅ System shutdown completed successfully")
        else:
            print("❌ System encountered errors")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
        await system_manager.shutdown()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        await system_manager.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    # Run the system
    asyncio.run(main())