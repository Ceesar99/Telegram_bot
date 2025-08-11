#!/usr/bin/env python3
"""
Telegram Bot Daemon Runner
Runs the telegram bot as a daemon with monitoring and restart capabilities
"""
import os
import sys
import time
import signal
import subprocess
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/logs/bot_daemon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TelegramBotDaemon:
    def __init__(self):
        self.running = True
        self.bot_process = None
        self.restart_count = 0
        self.max_restarts = 100  # Allow many restarts
        self.start_time = datetime.now()
        self.pid_file = '/tmp/telegram_bot_daemon.pid'
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        if self.bot_process:
            self.bot_process.terminate()
    
    def start_bot(self):
        """Start the telegram bot process"""
        try:
            logger.info("Starting telegram bot process...")
            self.bot_process = subprocess.Popen([
                sys.executable, '/workspace/working_telegram_bot.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logger.info(f"✅ Bot started with PID: {self.bot_process.pid}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start bot: {e}")
            return False
    
    def check_bot_health(self):
        """Check if bot process is healthy"""
        if not self.bot_process:
            return False
        
        poll_result = self.bot_process.poll()
        if poll_result is None:
            # Process is still running
            return True
        else:
            # Process has ended
            stdout, stderr = self.bot_process.communicate()
            logger.warning(f"Bot process ended with code {poll_result}")
            if stderr:
                logger.error(f"Bot stderr: {stderr.decode()}")
            return False
    
    def restart_bot(self):
        """Restart the bot process"""
        self.restart_count += 1
        logger.info(f"🔄 Restarting bot (attempt {self.restart_count})...")
        
        if self.bot_process:
            try:
                self.bot_process.terminate()
                self.bot_process.wait(timeout=10)
            except:
                self.bot_process.kill()
        
        time.sleep(5)  # Wait before restart
        return self.start_bot()
    
    def save_pid(self):
        """Save daemon PID to file"""
        try:
            with open(self.pid_file, 'w') as f:
                f.write(str(os.getpid()))
        except Exception as e:
            logger.warning(f"Could not save PID: {e}")
    
    def run(self):
        """Main daemon loop"""
        logger.info("🛡️ Starting Telegram Bot Daemon")
        logger.info(f"📅 Start time: {self.start_time}")
        logger.info("⏰ Target: 24/7 Continuous Operation")
        
        self.save_pid()
        
        # Start initial bot
        if not self.start_bot():
            logger.error("❌ Failed to start initial bot process")
            return
        
        # Main monitoring loop
        while self.running and self.restart_count < self.max_restarts:
            try:
                # Check bot health every 30 seconds
                time.sleep(30)
                
                if not self.check_bot_health():
                    logger.warning("❌ Bot health check failed")
                    if self.running:  # Only restart if we're supposed to be running
                        if not self.restart_bot():
                            logger.error("❌ Failed to restart bot")
                            break
                else:
                    # Bot is healthy
                    uptime = datetime.now() - self.start_time
                    if uptime.total_seconds() % 3600 < 30:  # Log every hour
                        logger.info(f"✅ Bot healthy - Uptime: {uptime.days}d {uptime.seconds//3600}h")
                
            except KeyboardInterrupt:
                logger.info("🛑 Received keyboard interrupt")
                break
            except Exception as e:
                logger.error(f"❌ Daemon error: {e}")
                time.sleep(60)  # Wait before continuing
        
        # Cleanup
        if self.bot_process:
            logger.info("🔄 Stopping bot process...")
            self.bot_process.terminate()
            try:
                self.bot_process.wait(timeout=10)
            except:
                self.bot_process.kill()
        
        # Remove PID file
        try:
            os.remove(self.pid_file)
        except:
            pass
        
        final_uptime = datetime.now() - self.start_time
        logger.info(f"🔚 Daemon stopped - Total uptime: {final_uptime.days}d {final_uptime.seconds//3600}h")

def check_if_running():
    """Check if daemon is already running"""
    try:
        if os.path.exists('/tmp/telegram_bot_daemon.pid'):
            with open('/tmp/telegram_bot_daemon.pid', 'r') as f:
                pid = int(f.read().strip())
            # Check if process exists
            os.kill(pid, 0)  # Doesn't kill, just checks if exists
            return pid
    except (OSError, ValueError):
        # Process doesn't exist or PID file is invalid
        try:
            os.remove('/tmp/telegram_bot_daemon.pid')
        except:
            pass
    return None

def main():
    """Main entry point"""
    print("🛡️ Telegram Bot Daemon Manager")
    print("=" * 40)
    
    # Check if already running
    existing_pid = check_if_running()
    if existing_pid:
        print(f"⚠️  Daemon already running with PID: {existing_pid}")
        print("💡 Use 'python3 stop_daemon.py' to stop it first")
        return
    
    print("🚀 Starting 24/7 Telegram Bot Daemon...")
    print(f"📅 Start time: {datetime.now()}")
    print("⏰ Runtime: Indefinite (until manually stopped)")
    print("🔄 Auto-restart: Enabled")
    print("💡 The bot will run continuously in the background")
    print("=" * 40)
    
    # Run daemon
    daemon = TelegramBotDaemon()
    try:
        daemon.run()
    except Exception as e:
        logger.error(f"❌ Daemon failed: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()