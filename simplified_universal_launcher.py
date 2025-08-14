#!/usr/bin/env python3
"""
🚀 SIMPLIFIED UNIVERSAL LAUNCHER FOR ADVANCED TRADING SYSTEMS
Launches the most advanced trading system available with proper error handling
"""

import os
import sys
import asyncio
import logging
import signal
import threading
import time
import subprocess
import psutil
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/workspace')

class AdvancedTradingLauncher:
    """Simplified launcher for the most advanced trading systems"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.is_running = False
        self.shutdown_requested = False
        self.processes = []
        
        # Available systems in order of sophistication
        self.available_systems = [
            {
                'name': 'Ultimate AI Trading Bot',
                'module': 'ultimate_ai_trading_bot',
                'class': None,
                'type': 'telegram_bot',
                'advanced_features': ['AI_ML', 'Real_Time_Signals', 'Risk_Management']
            },
            {
                'name': 'Working Telegram Bot',
                'module': 'working_telegram_bot',
                'class': 'WorkingTradingBot',
                'type': 'telegram_bot',
                'advanced_features': ['Real_Time_Signals', 'Performance_Tracking']
            },
            {
                'name': 'Enhanced Signal Engine',
                'module': 'enhanced_signal_engine',
                'class': 'EnhancedSignalEngine',
                'type': 'signal_engine',
                'advanced_features': ['Enhanced_Analysis', 'Multiple_Models']
            },
            {
                'name': 'Signal Engine',
                'module': 'signal_engine',
                'class': 'SignalEngine',
                'type': 'signal_engine',
                'advanced_features': ['Basic_Signals']
            }
        ]
        
    def _setup_logger(self):
        """Setup comprehensive logging"""
        logger = logging.getLogger('AdvancedTradingLauncher')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        handler = logging.FileHandler('/workspace/logs/advanced_launcher.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def validate_system_dependencies(self) -> bool:
        """Validate that essential dependencies are available"""
        try:
            essential_deps = [
                'numpy', 'pandas', 'requests', 'asyncio', 'logging',
                'datetime', 'json', 'sqlite3'
            ]
            
            for dep in essential_deps:
                try:
                    if dep == 'asyncio':
                        import asyncio
                    elif dep == 'logging':
                        import logging
                    elif dep == 'datetime':
                        import datetime
                    elif dep == 'json':
                        import json
                    elif dep == 'sqlite3':
                        import sqlite3
                    else:
                        __import__(dep)
                    self.logger.info(f"✅ {dep}")
                except ImportError:
                    self.logger.warning(f"⚠️ {dep} - Missing but continuing")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Dependency validation failed: {e}")
            return False
    
    def find_most_advanced_system(self) -> Optional[Dict[str, Any]]:
        """Find the most advanced system that can be loaded"""
        self.logger.info("🔍 Scanning for most advanced trading system...")
        
        for system in self.available_systems:
            try:
                self.logger.info(f"Testing: {system['name']}")
                
                # Try to import the module
                module = __import__(system['module'])
                
                # Verify the file exists and is accessible
                module_path = f"/workspace/{system['module']}.py"
                if os.path.exists(module_path):
                    self.logger.info(f"✅ {system['name']} - Available and Advanced")
                    return system
                else:
                    self.logger.warning(f"⚠️ {system['name']} - Module file not found")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ {system['name']} - Cannot import: {e}")
                continue
        
        self.logger.error("❌ No advanced trading systems could be loaded")
        return None
    
    async def launch_telegram_bot_system(self, system: Dict[str, Any]) -> bool:
        """Launch a Telegram bot based system"""
        try:
            self.logger.info(f"🚀 Launching {system['name']} as Telegram bot...")
            
            # Launch the bot as a subprocess for better control
            cmd = [
                'python3', f"/workspace/{system['module']}.py"
            ]
            
            # Set environment variables
            env = os.environ.copy()
            env['LD_LIBRARY_PATH'] = '/usr/local/lib:' + env.get('LD_LIBRARY_PATH', '')
            env['PYTHONPATH'] = '/workspace:' + env.get('PYTHONPATH', '')
            
            process = subprocess.Popen(
                cmd,
                cwd='/workspace',
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes.append(process)
            self.logger.info(f"✅ {system['name']} launched with PID: {process.pid}")
            
            # Monitor the process
            await self._monitor_process(process, system['name'])
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to launch {system['name']}: {e}")
            return False
    
    async def launch_signal_engine_system(self, system: Dict[str, Any]) -> bool:
        """Launch a signal engine based system"""
        try:
            self.logger.info(f"🚀 Launching {system['name']} as signal engine...")
            
            # Import and start the signal engine
            module = __import__(system['module'])
            if system['class']:
                signal_class = getattr(module, system['class'])
                signal_engine = signal_class()
                
                # Start the engine if it has a start method
                if hasattr(signal_engine, 'start'):
                    await signal_engine.start()
                elif hasattr(signal_engine, 'initialize'):
                    signal_engine.initialize()
                
                self.logger.info(f"✅ {system['name']} signal engine started")
                return True
            else:
                self.logger.error(f"No class specified for {system['name']}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to launch {system['name']}: {e}")
            return False
    
    async def _monitor_process(self, process: subprocess.Popen, name: str):
        """Monitor a launched process"""
        try:
            while process.poll() is None and not self.shutdown_requested:
                await asyncio.sleep(5)
                
                # Check if process is still healthy
                try:
                    proc = psutil.Process(process.pid)
                    cpu_percent = proc.cpu_percent()
                    memory_info = proc.memory_info()
                    
                    self.logger.info(
                        f"📊 {name} - CPU: {cpu_percent:.1f}% | "
                        f"Memory: {memory_info.rss / 1024 / 1024:.1f}MB"
                    )
                    
                except psutil.NoSuchProcess:
                    self.logger.warning(f"⚠️ {name} process no longer exists")
                    break
                    
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                if stdout:
                    self.logger.info(f"{name} stdout: {stdout}")
                if stderr:
                    self.logger.error(f"{name} stderr: {stderr}")
                    
        except Exception as e:
            self.logger.error(f"Error monitoring {name}: {e}")
    
    async def launch_most_advanced_system(self) -> bool:
        """Launch the most advanced system available"""
        try:
            # Validate dependencies first
            if not self.validate_system_dependencies():
                self.logger.error("❌ Critical dependencies missing")
                return False
            
            # Find the most advanced system
            system = self.find_most_advanced_system()
            if not system:
                return False
            
            self.logger.info(f"🎯 Selected: {system['name']}")
            self.logger.info(f"Advanced Features: {', '.join(system['advanced_features'])}")
            
            # Launch based on system type
            if system['type'] == 'telegram_bot':
                return await self.launch_telegram_bot_system(system)
            elif system['type'] == 'signal_engine':
                return await self.launch_signal_engine_system(system)
            else:
                self.logger.error(f"Unknown system type: {system['type']}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to launch advanced system: {e}")
            return False
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_requested = True
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self):
        """Graceful shutdown of all processes"""
        self.logger.info("🛑 Initiating graceful shutdown...")
        
        for process in self.processes:
            try:
                if process.poll() is None:  # Process is still running
                    self.logger.info(f"Terminating process {process.pid}")
                    process.terminate()
                    
                    # Wait for graceful termination
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.logger.warning(f"Force killing process {process.pid}")
                        process.kill()
                        
            except Exception as e:
                self.logger.error(f"Error terminating process: {e}")
        
        self.logger.info("✅ Shutdown complete")
    
    async def run(self) -> bool:
        """Main execution function"""
        try:
            self.setup_signal_handlers()
            self.is_running = True
            
            self.logger.info("🚀 Advanced Trading System Launcher Starting...")
            
            # Launch the most advanced system
            success = await self.launch_most_advanced_system()
            
            if success:
                self.logger.info("✅ Advanced trading system launched successfully!")
                self.logger.info("📱 Your Telegram bot should now be responding to commands")
                self.logger.info("🔄 System is running in background...")
                
                # Keep running until shutdown requested
                while self.is_running and not self.shutdown_requested:
                    await asyncio.sleep(1)
                    
            else:
                self.logger.error("❌ Failed to launch any advanced trading system")
                return False
                
        except Exception as e:
            self.logger.error(f"Critical error in main execution: {e}")
            return False
        finally:
            await self.shutdown()
            
        return success

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Advanced Trading System Launcher')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    launcher = AdvancedTradingLauncher()
    
    if args.debug:
        launcher.logger.setLevel(logging.DEBUG)
    
    print("🚀 ADVANCED TRADING SYSTEM LAUNCHER")
    print("=" * 50)
    print("Launching the most advanced trading system available...")
    print()
    
    success = await launcher.run()
    
    if success:
        print("\n✅ SUCCESS: Advanced trading system launched!")
        print("📱 Check your Telegram bot for responses")
    else:
        print("\n❌ FAILED: Could not launch advanced trading system")
        
    return 0 if success else 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        sys.exit(1)