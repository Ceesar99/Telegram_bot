#!/usr/bin/env python3
"""
Telegram Bot Startup Script
Simple script to start the working telegram bot
"""
import os
import sys
import subprocess
import signal
import time

def check_bot_running():
    """Check if bot is already running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'working_telegram_bot.py'], 
                              capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except:
        return False

def start_bot():
    """Start the telegram bot"""
    print("🚀 Starting Telegram Trading Bot...")
    
    if check_bot_running():
        print("⚠️  Bot is already running!")
        print("💡 Use 'python3 stop_telegram_bot.py' to stop it first")
        return False
    
    try:
        # Start the bot in background
        process = subprocess.Popen([
            sys.executable, 'working_telegram_bot.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment to check if it started successfully
        time.sleep(3)
        
        if process.poll() is None:  # Still running
            print("✅ Telegram bot started successfully!")
            print(f"📱 Bot process ID: {process.pid}")
            print("🤖 Bot username: @CEESARBOT")
            print("📱 Send /start to the bot in Telegram to test!")
            print("💡 Use 'python3 stop_telegram_bot.py' to stop the bot")
            
            # Save PID for stopping later
            with open('/tmp/telegram_bot.pid', 'w') as f:
                f.write(str(process.pid))
            
            return True
        else:
            # Bot failed to start
            stdout, stderr = process.communicate()
            print("❌ Failed to start bot!")
            print(f"Error: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        return False

def main():
    """Main function"""
    print("🤖 Telegram Trading Bot Starter")
    print("=" * 40)
    
    # Change to workspace directory
    os.chdir('/workspace')
    
    # Start the bot
    success = start_bot()
    
    if success:
        print("\n🎉 SUCCESS!")
        print("📱 Your Telegram bot is now running and ready to use!")
        print("\n📋 Available Commands in Telegram:")
        print("   /start - Start the bot")
        print("   /signal - Get trading signal")
        print("   /status - Bot status")
        print("   /help - Show all commands")
    else:
        print("\n❌ FAILED!")
        print("💡 Check the error messages above")

if __name__ == "__main__":
    main()