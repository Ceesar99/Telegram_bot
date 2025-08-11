#!/usr/bin/env python3
"""
Bot Management System

This system manages the lifecycle of all trading bots and ensures
proper switching between different trading modes including the
LSTM AI-powered signal system.
"""

import os
import sys
import psutil
import signal
import subprocess
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import asyncio
import json

class BotManager:
    def __init__(self):
        self.logger = self._setup_logger()
        self.running_bots = {}
        self.bot_processes = {}
        self.system_status = "STOPPED"
        
    def _setup_logger(self):
        """Setup logging for bot manager"""
        logger = logging.getLogger('BotManager')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        os.makedirs('/workspace/logs', exist_ok=True)
        
        handler = logging.FileHandler('/workspace/logs/bot_manager.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Also log to console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def scan_running_bots(self) -> Dict[str, Dict]:
        """Scan for all running trading bot processes"""
        try:
            running_bots = {}
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                try:
                    # Check if this is a trading bot process
                    if self._is_trading_bot_process(proc):
                        bot_info = {
                            'pid': proc.pid,
                            'name': proc.name(),
                            'cmdline': ' '.join(proc.cmdline()),
                            'create_time': datetime.fromtimestamp(proc.create_time()),
                            'status': 'RUNNING',
                            'memory_mb': proc.memory_info().rss / 1024 / 1024,
                            'cpu_percent': proc.cpu_percent()
                        }
                        running_bots[proc.pid] = bot_info
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            self.running_bots = running_bots
            self.logger.info(f"Found {len(running_bots)} running trading bot processes")
            return running_bots
            
        except Exception as e:
            self.logger.error(f"Error scanning running bots: {e}")
            return {}
    
    def _is_trading_bot_process(self, proc) -> bool:
        """Check if a process is a trading bot"""
        try:
            cmdline = ' '.join(proc.cmdline()).lower()
            
            # Check for common trading bot process names and patterns
            bot_indicators = [
                'trading_bot', 'signal_engine', 'telegram_bot', 'main.py',
                'start_bot.py', 'unified_trading_system', 'lstm_model',
                'pocket_option', 'binary_options', 'trading_system'
            ]
            
            return any(indicator in cmdline for indicator in bot_indicators)
            
        except Exception:
            return False
    
    def stop_all_bots(self, force: bool = False) -> Dict[str, bool]:
        """Stop all running trading bots"""
        try:
            self.logger.info("🛑 Stopping all running trading bots...")
            
            # First scan for running bots
            running_bots = self.scan_running_bots()
            if not running_bots:
                self.logger.info("✅ No running bots found")
                return {}
            
            stop_results = {}
            
            for pid, bot_info in running_bots.items():
                try:
                    self.logger.info(f"Stopping bot {bot_info['name']} (PID: {pid})")
                    
                    if force:
                        # Force kill
                        os.kill(pid, signal.SIGKILL)
                        self.logger.info(f"Force killed bot {pid}")
                    else:
                        # Graceful shutdown
                        os.kill(pid, signal.SIGTERM)
                        
                        # Wait for graceful shutdown
                        time.sleep(2)
                        
                        # Check if still running
                        if psutil.pid_exists(pid):
                            os.kill(pid, signal.SIGKILL)
                            self.logger.info(f"Force killed bot {pid} after graceful shutdown attempt")
                        else:
                            self.logger.info(f"Gracefully stopped bot {pid}")
                    
                    stop_results[pid] = True
                    
                except ProcessLookupError:
                    self.logger.info(f"Bot {pid} already stopped")
                    stop_results[pid] = True
                except Exception as e:
                    self.logger.error(f"Error stopping bot {pid}: {e}")
                    stop_results[pid] = False
            
            # Wait a bit for processes to fully terminate
            time.sleep(3)
            
            # Verify all bots are stopped
            final_scan = self.scan_running_bots()
            if not final_scan:
                self.logger.info("✅ All trading bots successfully stopped")
                self.system_status = "STOPPED"
            else:
                self.logger.warning(f"⚠️  {len(final_scan)} bots still running after stop attempt")
            
            return stop_results
            
        except Exception as e:
            self.logger.error(f"Error stopping all bots: {e}")
            return {}
    
    def start_lstm_ai_system(self) -> bool:
        """Start the LSTM AI-powered signal system"""
        try:
            self.logger.info("🚀 Starting LSTM AI-powered signal system...")
            
            # Check if any bots are still running
            running_bots = self.scan_running_bots()
            if running_bots:
                self.logger.warning(f"⚠️  {len(running_bots)} bots still running, stopping them first...")
                self.stop_all_bots(force=True)
            
            # Start the LSTM AI system
            success = self._launch_lstm_system()
            
            if success:
                self.logger.info("✅ LSTM AI-powered signal system started successfully")
                self.system_status = "LSTM_AI_RUNNING"
                return True
            else:
                self.logger.error("❌ Failed to start LSTM AI system")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting LSTM AI system: {e}")
            return False
    
    def _launch_lstm_system(self) -> bool:
        """Launch the LSTM AI trading system"""
        try:
            # Start the enhanced signal engine with LSTM AI
            cmd = [
                sys.executable, 
                '-c',
                '''
import asyncio
import sys
sys.path.append("/workspace")
from enhanced_signal_engine import EnhancedSignalEngine
from bot_manager import BotManager

async def main():
    try:
        # Initialize the enhanced signal engine
        signal_engine = EnhancedSignalEngine()
        
        # Wait for initialization
        await asyncio.sleep(5)
        
        # Start generating signals
        while True:
            try:
                signal = await signal_engine.generate_enhanced_signal()
                if signal:
                    print(f"🎯 AI Signal Generated: {signal.pair} {signal.direction}")
                    print(f"   Confidence: {signal.confidence:.1f}%")
                    print(f"   Accuracy: {signal.accuracy:.1f}%")
                    print(f"   Entry Time: {signal.entry_time.strftime('%H:%M:%S')}")
                    print(f"   Signal Time: {signal.signal_time.strftime('%H:%M:%S')}")
                    print(f"   Risk Level: {signal.risk_level}")
                    print(f"   Pair Category: {signal.pair_category}")
                    print(f"   Weekend Mode: {signal.is_weekend}")
                    print("─" * 50)
                
                # Wait before next signal generation
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"Error generating signal: {e}")
                await asyncio.sleep(30)
                
    except KeyboardInterrupt:
        print("\\n🛑 LSTM AI system stopped by user")
    except Exception as e:
        print(f"Error in LSTM AI system: {e}")

if __name__ == "__main__":
    asyncio.run(main())
                '''
            ]
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd='/workspace'
            )
            
            # Store process info
            self.bot_processes['lstm_ai'] = {
                'process': process,
                'pid': process.pid,
                'start_time': datetime.now(),
                'type': 'LSTM_AI'
            }
            
            # Wait a bit to see if it starts successfully
            time.sleep(3)
            
            # Check if process is still running
            if process.poll() is None:
                self.logger.info(f"LSTM AI system started with PID: {process.pid}")
                return True
            else:
                # Process failed to start
                stdout, stderr = process.communicate()
                self.logger.error(f"LSTM AI system failed to start")
                self.logger.error(f"STDOUT: {stdout}")
                self.logger.error(f"STDERR: {stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error launching LSTM system: {e}")
            return False
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        try:
            # Scan for running bots
            running_bots = self.scan_running_bots()
            
            # Get LSTM system status
            lstm_status = "STOPPED"
            if 'lstm_ai' in self.bot_processes:
                process_info = self.bot_processes['lstm_ai']
                if process_info['process'].poll() is None:
                    lstm_status = "RUNNING"
                else:
                    lstm_status = "STOPPED"
            
            status = {
                "timestamp": datetime.now().isoformat(),
                "system_status": self.system_status,
                "lstm_ai_status": lstm_status,
                "running_bots_count": len(running_bots),
                "running_bots": running_bots,
                "bot_processes": {
                    name: {
                        "pid": info["pid"],
                        "start_time": info["start_time"].isoformat(),
                        "type": info["type"],
                        "status": "RUNNING" if info["process"].poll() is None else "STOPPED"
                    }
                    for name, info in self.bot_processes.items()
                }
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
    
    def restart_lstm_system(self) -> bool:
        """Restart the LSTM AI system"""
        try:
            self.logger.info("🔄 Restarting LSTM AI system...")
            
            # Stop current LSTM system
            if 'lstm_ai' in self.bot_processes:
                try:
                    self.bot_processes['lstm_ai']['process'].terminate()
                    time.sleep(2)
                    if self.bot_processes['lstm_ai']['process'].poll() is None:
                        self.bot_processes['lstm_ai']['process'].kill()
                except Exception as e:
                    self.logger.error(f"Error stopping LSTM system: {e}")
            
            # Start new LSTM system
            return self.start_lstm_ai_system()
            
        except Exception as e:
            self.logger.error(f"Error restarting LSTM system: {e}")
            return False
    
    def cleanup(self):
        """Cleanup all resources"""
        try:
            self.logger.info("🧹 Cleaning up bot manager...")
            
            # Stop all bots
            self.stop_all_bots(force=True)
            
            # Stop LSTM system
            if 'lstm_ai' in self.bot_processes:
                try:
                    self.bot_processes['lstm_ai']['process'].terminate()
                    time.sleep(2)
                    if self.bot_processes['lstm_ai']['process'].poll() is None:
                        self.bot_processes['lstm_ai']['process'].kill()
                except Exception as e:
                    self.logger.error(f"Error stopping LSTM system during cleanup: {e}")
            
            self.logger.info("✅ Bot manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bot Management System')
    parser.add_argument('--action', choices=['status', 'stop-all', 'start-lstm', 'restart-lstm'], 
                       default='status', help='Action to perform')
    parser.add_argument('--force', action='store_true', help='Force stop bots')
    
    args = parser.parse_args()
    
    bot_manager = BotManager()
    
    try:
        if args.action == 'status':
            status = bot_manager.get_system_status()
            print(json.dumps(status, indent=2))
            
        elif args.action == 'stop-all':
            results = bot_manager.stop_all_bots(force=args.force)
            print(f"Stopped {len(results)} bots")
            
        elif args.action == 'start-lstm':
            success = bot_manager.start_lstm_ai_system()
            if success:
                print("✅ LSTM AI system started successfully")
            else:
                print("❌ Failed to start LSTM AI system")
                
        elif args.action == 'restart-lstm':
            success = bot_manager.restart_lstm_system()
            if success:
                print("✅ LSTM AI system restarted successfully")
            else:
                print("❌ Failed to restart LSTM AI system")
                
    except KeyboardInterrupt:
        print("\n🛑 Operation interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        bot_manager.cleanup()

if __name__ == "__main__":
    main()