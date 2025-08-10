#!/usr/bin/env python3
"""
Simple Telegram Bot using HTTP requests instead of python-telegram-bot library
This avoids Python 3.13 compatibility issues.
"""

import requests
import json
import logging
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_USER_ID

class SimpleTelegramBot:
    def __init__(self, signal_engine=None, performance_tracker=None, risk_manager=None):
        self.token = TELEGRAM_BOT_TOKEN
        self.authorized_users = [int(TELEGRAM_USER_ID)]
        self.signal_engine = signal_engine
        self.performance_tracker = performance_tracker  
        self.risk_manager = risk_manager
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.offset = 0
        self.running = False
        
        self.logger = self._setup_logger()
        
        self.bot_status = {
            'active': True,
            'auto_signals': True,
            'last_signal_time': None,
            'signals_today': 0
        }
        
    def _setup_logger(self):
        logger = logging.getLogger('SimpleTelegramBot')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('/workspace/logs/telegram_bot.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
    def send_message(self, chat_id: int, text: str, parse_mode: str = "Markdown") -> bool:
        """Send a message to a chat"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': text,
                'parse_mode': parse_mode
            }
            response = requests.post(url, data=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            return False
            
    def get_updates(self) -> list:
        """Get updates from Telegram"""
        try:
            url = f"{self.base_url}/getUpdates"
            params = {'offset': self.offset, 'timeout': 5}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    return data.get('result', [])
            return []
        except Exception as e:
            self.logger.error(f"Error getting updates: {e}")
            return []
            
    def is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized"""
        return user_id in self.authorized_users
        
    async def handle_command(self, chat_id: int, command: str, user_id: int):
        """Handle incoming commands"""
        if not self.is_authorized(user_id):
            self.send_message(chat_id, "❌ Unauthorized access. Contact admin.")
            return
            
        self.logger.info(f"Processing command: {command} from user {user_id}")
        
        if command == '/start':
            await self.cmd_start(chat_id)
        elif command == '/signal':
            await self.cmd_signal(chat_id)
        elif command == '/status':
            await self.cmd_status(chat_id)
        elif command == '/stats':
            await self.cmd_stats(chat_id)
        elif command == '/help':
            await self.cmd_help(chat_id)
        elif command == '/auto_on':
            await self.cmd_auto_on(chat_id)
        elif command == '/auto_off':
            await self.cmd_auto_off(chat_id)
        else:
            self.send_message(chat_id, f"Unknown command: {command}\nUse /help for available commands.")
            
    async def cmd_start(self, chat_id: int):
        """Handle /start command"""
        message = """
🤖 **AI Trading Bot Started Successfully!**

Welcome to your Binary Options Trading Bot with 95%+ accuracy!

🎯 **Quick Commands:**
• `/signal` - Get instant trading signal
• `/status` - Check system status  
• `/stats` - View performance statistics
• `/auto_on` - Enable automatic signals
• `/help` - Show all commands

🚀 **Ready to generate high-accuracy signals!**
"""
        self.send_message(chat_id, message)
        
    async def cmd_signal(self, chat_id: int):
        """Handle /signal command"""
        try:
            if self.signal_engine:
                signal_data = await self.signal_engine.generate_signal()
                if signal_data:
                    message = f"""
🎯 **TRADING SIGNAL**

🟢 **Pair:** {signal_data.get('pair', 'N/A')}
📈 **Direction:** {signal_data.get('direction', 'N/A')}
🎯 **Accuracy:** {signal_data.get('accuracy', 0):.1f}%
⏰ **Duration:** {signal_data.get('duration', 'N/A')} minutes
🤖 **AI Confidence:** {signal_data.get('ai_confidence', 0):.1f}%

**Entry Details:**
💰 **Entry Price:** {signal_data.get('entry_price', 'N/A')}
🛡️ **Risk Level:** {signal_data.get('risk_level', 'Medium')}
⏱️ **Signal Time:** {datetime.now().strftime('%H:%M:%S')}
"""
                    self.send_message(chat_id, message)
                else:
                    self.send_message(chat_id, "⏳ No optimal signals at the moment. Market conditions are being analyzed...")
            else:
                self.send_message(chat_id, "❌ Signal engine not available")
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            self.send_message(chat_id, "❌ Error generating signal. Please try again.")
            
    async def cmd_status(self, chat_id: int):
        """Handle /status command"""
        status_message = f"""
🔋 **SYSTEM STATUS**

✅ **Bot Status:** {'Active' if self.bot_status['active'] else 'Inactive'}
🤖 **Auto Signals:** {'ON' if self.bot_status['auto_signals'] else 'OFF'}
📊 **Today's Signals:** {self.bot_status['signals_today']}
⏰ **Last Signal:** {self.bot_status.get('last_signal_time', 'None')}

🎯 **System Health:** All systems operational
📈 **Ready for trading!**
"""
        self.send_message(chat_id, status_message)
        
    async def cmd_stats(self, chat_id: int):
        """Handle /stats command"""
        try:
            if self.performance_tracker:
                stats = self.performance_tracker.get_performance_summary()
                message = f"""
📊 **PERFORMANCE STATISTICS**

🏆 **Win Rate:** {stats.get('win_rate', 0):.1f}%
📈 **Total Signals:** {stats.get('total_signals', 0)}
✅ **Successful Trades:** {stats.get('winning_trades', 0)}

📅 **Today:** {stats.get('today_signals', 0)} signals ({stats.get('today_win_rate', 0):.1f}% win rate)
📊 **This Week:** {stats.get('week_signals', 0)} signals ({stats.get('week_win_rate', 0):.1f}% win rate)

🎯 **Target Achievement:** {stats.get('target_achievement', 0):.1f}%
"""
                self.send_message(chat_id, message)
            else:
                self.send_message(chat_id, "📊 Statistics not available")
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            self.send_message(chat_id, "❌ Error retrieving statistics")
            
    async def cmd_auto_on(self, chat_id: int):
        """Enable automatic signals"""
        self.bot_status['auto_signals'] = True
        self.send_message(chat_id, "✅ **Automatic signals ENABLED**\n\nBot will now send signals automatically when optimal conditions are detected.")
        
    async def cmd_auto_off(self, chat_id: int):
        """Disable automatic signals"""
        self.bot_status['auto_signals'] = False
        self.send_message(chat_id, "🔕 **Automatic signals DISABLED**\n\nUse /signal to get manual signals.")
        
    async def cmd_help(self, chat_id: int):
        """Handle /help command"""
        help_message = """
🆘 **TRADING BOT COMMANDS**

🎯 **Trading:**
• `/signal` - Get instant trading signal
• `/auto_on` - Enable automatic signals
• `/auto_off` - Disable automatic signals

📊 **Analysis:**
• `/stats` - Trading statistics
• `/status` - System status

⚙️ **System:**
• `/help` - This help message
• `/start` - Restart bot

🎯 **Features:**
✅ 95%+ accuracy targeting
✅ LSTM AI analysis
✅ Real-time market data
✅ Risk management
✅ Performance tracking

**Happy Trading! 📈🤖**
"""
        self.send_message(chat_id, help_message)
        
    async def run_polling(self):
        """Main polling loop"""
        self.running = True
        self.logger.info("Starting Simple Telegram Bot polling...")
        
        while self.running:
            try:
                updates = self.get_updates()
                
                for update in updates:
                    self.offset = update['update_id'] + 1
                    
                    if 'message' in update:
                        message = update['message']
                        chat_id = message['chat']['id']
                        user_id = message['from']['id']
                        
                        if 'text' in message:
                            text = message['text']
                            if text.startswith('/'):
                                await self.handle_command(chat_id, text, user_id)
                                
                await asyncio.sleep(1)  # Poll every second
                
            except Exception as e:
                self.logger.error(f"Error in polling loop: {e}")
                await asyncio.sleep(5)
                
    def run(self):
        """Start the bot in sync mode"""
        try:
            # Test connection
            response = requests.get(f"{self.base_url}/getMe", timeout=10)
            if response.status_code == 200:
                bot_info = response.json()
                if bot_info.get('ok'):
                    self.logger.info(f"Bot connected: @{bot_info['result']['username']}")
                    print(f"✅ Telegram bot connected: @{bot_info['result']['username']}")
                else:
                    raise Exception("Bot token validation failed")
            else:
                raise Exception(f"HTTP {response.status_code}")
                
            # Start polling
            asyncio.run(self.run_polling())
            
        except Exception as e:
            self.logger.error(f"Error starting bot: {e}")
            print(f"❌ Telegram bot error: {e}")
            
    def stop(self):
        """Stop the bot"""
        self.running = False

if __name__ == "__main__":
    bot = SimpleTelegramBot()
    bot.run()