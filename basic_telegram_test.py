#!/usr/bin/env python3
"""
Basic Telegram Bot Test
Very simple test to verify the bot token works
"""
import asyncio
from telegram import Bot
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_USER_ID

async def test_bot():
    """Test basic bot functionality"""
    print(f"🤖 Testing bot with token: {TELEGRAM_BOT_TOKEN[:10]}...")
    print(f"👤 Authorized user ID: {TELEGRAM_USER_ID}")
    
    try:
        # Create bot instance
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        
        # Get bot info
        print("⏳ Getting bot info...")
        bot_info = await bot.get_me()
        
        print(f"✅ Bot connected successfully!")
        print(f"📱 Bot username: @{bot_info.username}")
        print(f"🆔 Bot ID: {bot_info.id}")
        print(f"👤 Bot name: {bot_info.first_name}")
        
        # Send test message to authorized user
        print(f"📤 Sending test message to user {TELEGRAM_USER_ID}...")
        
        test_message = """🤖 **Bot Connection Test - SUCCESS!** 🤖

✅ **Your Telegram bot is working correctly!**

🎯 **Bot Details:**
• Username: @{username}
• Bot ID: {bot_id}
• Status: Online and responding

📱 **Next Steps:**
1. The bot token is valid
2. Bot can send messages  
3. Ready for command integration

🚀 **Your bot is fully operational!**""".format(
            username=bot_info.username,
            bot_id=bot_info.id
        )
        
        try:
            await bot.send_message(
                chat_id=int(TELEGRAM_USER_ID), 
                text=test_message,
                parse_mode='Markdown'
            )
            print("✅ Test message sent successfully!")
            print(f"📱 Check your Telegram chat for the test message")
            
        except Exception as e:
            print(f"⚠️  Could not send message (user might need to start bot first): {e}")
            print("💡 Go to Telegram and send /start to your bot first")
        
        return True
        
    except Exception as e:
        print(f"❌ Bot test failed: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Starting Telegram Bot Test...")
    
    try:
        result = asyncio.run(test_bot())
        if result:
            print("\n🎉 SUCCESS: Your Telegram bot is working!")
            print("💡 You can now use the bot commands")
        else:
            print("\n❌ FAILED: Bot test unsuccessful")
            
    except Exception as e:
        print(f"\n❌ Error during test: {e}")

if __name__ == "__main__":
    main()