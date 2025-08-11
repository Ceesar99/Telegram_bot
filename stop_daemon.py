#!/usr/bin/env python3
"""
Stop Telegram Bot Daemon
"""
import os
import signal
import time

def stop_daemon():
    """Stop the running daemon"""
    print("🛑 Stopping Telegram Bot Daemon...")
    
    try:
        if os.path.exists('/tmp/telegram_bot_daemon.pid'):
            with open('/tmp/telegram_bot_daemon.pid', 'r') as f:
                pid = int(f.read().strip())
            
            print(f"📍 Found daemon PID: {pid}")
            
            # Send termination signal
            os.kill(pid, signal.SIGTERM)
            print("⏳ Waiting for graceful shutdown...")
            
            # Wait for process to stop
            for i in range(10):
                try:
                    os.kill(pid, 0)  # Check if still exists
                    time.sleep(1)
                except OSError:
                    break
            
            # Force kill if still running
            try:
                os.kill(pid, signal.SIGKILL)
                print("🔥 Force stopped daemon")
            except OSError:
                print("✅ Daemon stopped gracefully")
            
            # Remove PID file
            os.remove('/tmp/telegram_bot_daemon.pid')
            print("🗑️ Cleaned up PID file")
            
        else:
            print("⚠️ No daemon PID file found")
            
    except Exception as e:
        print(f"❌ Error stopping daemon: {e}")

if __name__ == "__main__":
    stop_daemon()