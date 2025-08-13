#!/usr/bin/env python3
"""
🚀 ULTIMATE AI TRADING SYSTEM STARTUP SCRIPT
Robust startup and monitoring system that keeps all components running
"""

import os
import sys
import time
import subprocess
import signal
import threading
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/logs/system_startup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('UltimateStartup')

class UltimateSystemLauncher:
    """🎯 Ultimate System Launcher and Monitor"""
    
    def __init__(self):
        self.workspace = '/workspace'
        self.processes = {}
        self.running = True
        
        # System components to launch
        self.components = {
            'telegram_bot': {
                'script': 'working_telegram_bot.py',
                'description': 'Telegram Bot Interface',
                'critical': True
            },
            'trading_system': {
                'script': 'ultimate_ai_trading_bot.py', 
                'description': 'AI Trading System',
                'critical': True
            },
            'monitor': {
                'script': 'monitor_system.py --continuous',
                'description': 'System Monitor',
                'critical': False
            }
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🛑 Received signal {signum}, shutting down...")
        self.running = False
        self.shutdown_all()
        
    def start_component(self, name, config):
        """Start a system component"""
        try:
            script_path = os.path.join(self.workspace, config['script'])
            if not os.path.exists(script_path.split()[0]):
                logger.error(f"❌ Script not found: {script_path}")
                return None
                
            logger.info(f"🚀 Starting {config['description']}...")
            
            # Start the process
            process = subprocess.Popen(
                ['python3'] + config['script'].split(),
                cwd=self.workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            self.processes[name] = {
                'process': process,
                'config': config,
                'start_time': datetime.now(),
                'restart_count': 0
            }
            
            logger.info(f"✅ {config['description']} started (PID: {process.pid})")
            return process
            
        except Exception as e:
            logger.error(f"❌ Failed to start {name}: {e}")
            return None
    
    def check_process(self, name):
        """Check if a process is still running"""
        if name not in self.processes:
            return False
            
        process_info = self.processes[name]
        process = process_info['process']
        
        if process.poll() is None:
            return True  # Still running
        else:
            # Process has died
            logger.warning(f"⚠️  {process_info['config']['description']} has stopped")
            return False
    
    def restart_component(self, name):
        """Restart a failed component"""
        if name not in self.processes:
            return False
            
        process_info = self.processes[name]
        config = process_info['config']
        
        # Clean up old process
        try:
            if process_info['process'].poll() is None:
                os.killpg(os.getpgid(process_info['process'].pid), signal.SIGTERM)
        except:
            pass
            
        # Increment restart counter
        process_info['restart_count'] += 1
        
        if process_info['restart_count'] > 5:
            logger.error(f"❌ {config['description']} failed too many times, giving up")
            return False
            
        logger.info(f"🔄 Restarting {config['description']} (attempt {process_info['restart_count']})")
        
        # Wait a bit before restarting
        time.sleep(5)
        
        # Start new process
        new_process = self.start_component(name, config)
        return new_process is not None
    
    def monitor_processes(self):
        """Monitor all processes and restart if needed"""
        logger.info("🔍 Starting process monitoring...")
        
        while self.running:
            try:
                for name in list(self.processes.keys()):
                    if not self.check_process(name):
                        config = self.processes[name]['config']
                        if config['critical']:
                            logger.warning(f"🚨 Critical component {name} failed, restarting...")
                            self.restart_component(name)
                        else:
                            logger.info(f"ℹ️  Non-critical component {name} stopped")
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"❌ Monitor error: {e}")
                time.sleep(5)
    
    def show_status(self):
        """Show current system status"""
        logger.info("📊 SYSTEM STATUS:")
        for name, info in self.processes.items():
            config = info['config']
            is_running = self.check_process(name)
            status = "🟢 RUNNING" if is_running else "🔴 STOPPED"
            uptime = (datetime.now() - info['start_time']).total_seconds() / 60
            
            logger.info(f"  {status} {config['description']} (PID: {info['process'].pid}, Uptime: {uptime:.1f}m, Restarts: {info['restart_count']})")
    
    def shutdown_all(self):
        """Shutdown all processes gracefully"""
        logger.info("🛑 Shutting down all processes...")
        
        for name, info in self.processes.items():
            try:
                process = info['process']
                if process.poll() is None:
                    logger.info(f"🔄 Stopping {info['config']['description']}...")
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"⚠️  Force killing {name}")
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        
            except Exception as e:
                logger.error(f"❌ Error stopping {name}: {e}")
        
        logger.info("✅ Shutdown complete")
    
    def start_all(self):
        """Start all system components"""
        logger.info("🚀 ULTIMATE AI TRADING SYSTEM - STARTUP")
        logger.info("="*60)
        
        # Start all components
        for name, config in self.components.items():
            self.start_component(name, config)
            time.sleep(2)  # Stagger startup
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
        monitor_thread.start()
        
        logger.info("✅ All components started")
        
        try:
            # Main loop
            while self.running:
                self.show_status()
                time.sleep(30)  # Status update every 30 seconds
                
        except KeyboardInterrupt:
            logger.info("🛑 Received keyboard interrupt")
        finally:
            self.shutdown_all()

def main():
    """Main function"""
    # Change to workspace directory
    os.chdir('/workspace')
    
    # Create launcher
    launcher = UltimateSystemLauncher()
    
    # Start everything
    launcher.start_all()

if __name__ == "__main__":
    main()