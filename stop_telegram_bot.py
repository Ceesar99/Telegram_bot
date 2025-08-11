#!/usr/bin/env python3
"""
Telegram Bot Stop Script
Simple script to stop the running telegram bot
"""
import os
import signal
import subprocess

def stop_bot():
    """Stop the telegram bot"""
    print("🛑 Stopping Telegram Trading Bot...")
    
    try:
        # Try to get PID from file
        if os.path.exists('/tmp/telegram_bot.pid'):
            with open('/tmp/telegram_bot.pid', 'r') as f:
                pid = int(f.read().strip())
            
            try:
                # Kill the process
                os.kill(pid, signal.SIGTERM)
                print(f"✅ Bot stopped (PID: {pid})")
                os.remove('/tmp/telegram_bot.pid')
                return True
            except ProcessLookupError:
                print("⚠️  Process not found (already stopped)")
                os.remove('/tmp/telegram_bot.pid')
        
        # Fallback: kill by process name
        result = subprocess.run(['pkill', '-f', 'working_telegram_bot.py'], 
                              capture_output=True)
        
        if result.returncode == 0:
            print("✅ Bot stopped successfully!")
            return True
        else:
            print("⚠️  No running bot found")
            return False
            
    except Exception as e:
        print(f"❌ Error stopping bot: {e}")
        return False

def main():
    """Main function"""
    print("🤖 Telegram Trading Bot Stopper")
    print("=" * 40)
    
    success = stop_bot()
    
    if success:
        print("\n🎉 Bot stopped successfully!")
        print("💡 Use 'python3 start_telegram_bot.py' to start it again")
    else:
        print("\n⚠️  Bot was not running or failed to stop")

if __name__ == "__main__":
    main()