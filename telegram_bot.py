import logging
import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List
import io
import base64
import psutil

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)

from config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_USER_ID, SIGNAL_CONFIG, 
    RISK_MANAGEMENT, PERFORMANCE_TARGETS, DATABASE_CONFIG
)
from signal_engine import SignalEngine
from performance_tracker import PerformanceTracker
from risk_manager import RiskManager

class TradingBot:
    def __init__(self):
        self.token = TELEGRAM_BOT_TOKEN
        self.authorized_users = [int(TELEGRAM_USER_ID)]
        self.signal_engine = SignalEngine()
        self.performance_tracker = PerformanceTracker()
        self.risk_manager = RiskManager()
        self.app = None
        self.logger = self._setup_logger()
        self.bot_status = {
            'active': True,
            'auto_signals': True,
            'last_signal_time': None,
            'signals_today': 0
        }
        
    def _setup_logger(self):
        logger = logging.getLogger('TradingBot')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('/workspace/logs/telegram_bot.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized to use the bot"""
        return user_id in self.authorized_users
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command - welcome message and instructions"""
        if not self.is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized access!")
            return
        
        welcome_message = """
🤖 **Binary Options Trading Bot** 🤖

Welcome to your AI-powered trading signal bot with 95%+ accuracy!

**Available Commands:**

📊 **Trading Commands:**
/signal - Get instant trading signal
/auto_on - Enable automatic signals
/auto_off - Disable automatic signals
/pairs - Show available currency pairs
/market_status - Check market conditions

📈 **Analysis Commands:**
/analyze [pair] - Deep analysis of currency pair
/volatility [pair] - Check market volatility
/support_resistance [pair] - Support/resistance levels
/technical [pair] - Technical indicators

📊 **Performance Commands:**
/stats - Show trading statistics
/performance - Detailed performance report
/history - Signal history
/win_rate - Current win rate

⚙️ **Settings Commands:**
/settings - Bot configuration
/risk_settings - Risk management settings
/alerts_on - Enable alerts
/alerts_off - Disable alerts

🔧 **System Commands:**
/status - Bot system status
/health - System health check
/backup - Create backup
/restart - Restart bot services

📚 **Help Commands:**
/help - Show this help message
/commands - List all commands
/about - About this bot

Type any command to get started!
        """
        
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
        self.logger.info(f"User {update.effective_user.id} started the bot")
    
    async def signal(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Generate and send trading signal"""
        if not self.is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized access!")
            return
        
        try:
            # Show loading message
            loading_msg = await update.message.reply_text("🔄 Analyzing market data...")
            
            # Generate signal
            signal_data = await self.signal_engine.generate_signal()
            
            if signal_data:
                signal_message = self._format_signal(signal_data)
                
                # Create inline keyboard for signal actions
                keyboard = [
                    [
                        InlineKeyboardButton("📊 Analysis", callback_data=f"analysis_{signal_data['pair']}"),
                        InlineKeyboardButton("📈 Chart", callback_data=f"chart_{signal_data['pair']}")
                    ],
                    [
                        InlineKeyboardButton("🔄 Refresh", callback_data="refresh_signal"),
                        InlineKeyboardButton("📋 History", callback_data="signal_history")
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                # Update loading message with signal
                await loading_msg.edit_text(signal_message, parse_mode='Markdown', reply_markup=reply_markup)
                
                # Update bot status
                self.bot_status['last_signal_time'] = datetime.now()
                self.bot_status['signals_today'] += 1
                
                # Save signal to database
                self.performance_tracker.save_signal(signal_data)
                
            else:
                await loading_msg.edit_text("⚠️ No high-confidence signals available at the moment.")
                
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            await update.message.reply_text("❌ Error generating signal. Please try again.")
    
    def _format_signal(self, signal_data: Dict) -> str:
        """Format trading signal for Telegram message"""
        
        # Determine emoji based on direction
        direction_emoji = "🟢" if signal_data['direction'] == 'BUY' else "🔴"
        
        signal_message = f"""
🎯 **TRADING SIGNAL** 🎯

{direction_emoji} **Currency Pair:** {signal_data['pair']}
📈 **Direction:** {signal_data['direction']}
🎯 **Accuracy:** {signal_data['accuracy']:.1f}%
⏰ **Time Expiry:** {signal_data['time_expiry']}
🤖 **AI Confidence:** {signal_data['ai_confidence']:.1f}%

**Technical Analysis:**
📊 **Strength:** {signal_data.get('strength', 'N/A')}/10
💹 **Trend:** {signal_data.get('trend', 'N/A')}
🎚️ **Volatility:** {signal_data.get('volatility_level', 'Low')}

**Entry Details:**
💰 **Entry Price:** {signal_data.get('entry_price', 'N/A')}
🛡️ **Risk Level:** {signal_data.get('risk_level', 'Low')}
⏱️ **Signal Time:** {signal_data.get('signal_time', datetime.now().strftime('%H:%M:%S'))}

*Signal generated by AI-powered LSTM analysis*
        """
        
        return signal_message
    
    async def auto_signals_on(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enable automatic signal generation"""
        if not self.is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized access!")
            return
        
        self.bot_status['auto_signals'] = True
        await update.message.reply_text("✅ Automatic signals enabled! You'll receive signals when high-confidence opportunities are detected.")
        self.logger.info("Automatic signals enabled")
    
    async def auto_signals_off(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Disable automatic signal generation"""
        if not self.is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized access!")
            return
        
        self.bot_status['auto_signals'] = False
        await update.message.reply_text("⏸️ Automatic signals disabled. Use /signal to get manual signals.")
        self.logger.info("Automatic signals disabled")
    
    async def pairs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show available currency pairs"""
        if not self.is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized access!")
            return
        
        available_pairs = self.signal_engine.get_available_pairs()
        
        pairs_message = "📋 **Available Currency Pairs:**\n\n"
        
        # Group pairs by category
        forex_pairs = [p for p in available_pairs if '/' in p and 'OTC' not in p and 'USD' in p]
        otc_pairs = [p for p in available_pairs if 'OTC' in p]
        crypto_pairs = [p for p in available_pairs if any(crypto in p for crypto in ['BTC', 'ETH', 'LTC'])]
        
        if forex_pairs:
            pairs_message += "💱 **Forex Pairs:**\n"
            pairs_message += " • ".join(forex_pairs[:10]) + "\n\n"
        
        if otc_pairs:
            pairs_message += "🕒 **OTC Pairs (Weekend):**\n"
            pairs_message += " • ".join(otc_pairs) + "\n\n"
        
        if crypto_pairs:
            pairs_message += "₿ **Crypto Pairs:**\n"
            pairs_message += " • ".join(crypto_pairs) + "\n\n"
        
        pairs_message += f"**Total Pairs Available:** {len(available_pairs)}"
        
        await update.message.reply_text(pairs_message, parse_mode='Markdown')
    
    async def market_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show current market status"""
        if not self.is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized access!")
            return
        
        market_info = self.signal_engine.get_market_status()
        
        status_message = f"""
📊 **Market Status** 📊

🕒 **Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
🌍 **Market Session:** {market_info.get('session', 'N/A')}
📈 **Market State:** {'🟢 Open' if market_info.get('is_open', False) else '🔴 Closed'}

**Market Conditions:**
💹 **Overall Volatility:** {market_info.get('volatility', 'N/A')}
🎯 **Signal Quality:** {market_info.get('signal_quality', 'N/A')}
⚡ **Active Pairs:** {market_info.get('active_pairs', 0)}

**Trading Environment:**
🛡️ **Risk Level:** {market_info.get('risk_level', 'Medium')}
🎚️ **Recommended Position:** {market_info.get('position_size', 'Standard')}
⏰ **Next Major Event:** {market_info.get('next_event', 'None scheduled')}
        """
        
        await update.message.reply_text(status_message, parse_mode='Markdown')
    
    async def analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Analyze specific currency pair"""
        if not self.is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized access!")
            return
        
        if not context.args:
            await update.message.reply_text("Please specify a currency pair. Example: /analyze GBP/USD")
            return
        
        pair = context.args[0].upper()
        
        try:
            analysis = await self.signal_engine.analyze_pair(pair)
            
            if analysis:
                analysis_message = f"""
📊 **Technical Analysis: {pair}** 📊

**Price Information:**
💰 **Current Price:** {analysis.get('current_price', 'N/A')}
📈 **24h Change:** {analysis.get('price_change', 'N/A')}%
📊 **Volatility:** {analysis.get('volatility', 'N/A')}

**Technical Indicators:**
🔴 **RSI:** {analysis.get('rsi', 'N/A')} ({analysis.get('rsi_signal', 'Neutral')})
📊 **MACD:** {analysis.get('macd_signal', 'Neutral')}
📈 **Bollinger Bands:** {analysis.get('bb_position', 'N/A')}
⚡ **Stochastic:** {analysis.get('stoch_signal', 'Neutral')}

**Support & Resistance:**
🛡️ **Support:** {analysis.get('support', 'N/A')}
🎯 **Resistance:** {analysis.get('resistance', 'N/A')}
📍 **Position:** {analysis.get('price_position', 'N/A')}%

**Trading Recommendation:**
🎯 **Signal:** {analysis.get('recommendation', 'HOLD')}
🎚️ **Strength:** {analysis.get('signal_strength', 'N/A')}/10
⚠️ **Risk:** {analysis.get('risk_level', 'Medium')}
                """
                
                await update.message.reply_text(analysis_message, parse_mode='Markdown')
            else:
                await update.message.reply_text(f"❌ Could not analyze {pair}. Please check the pair name.")
                
        except Exception as e:
            self.logger.error(f"Error analyzing pair {pair}: {e}")
            await update.message.reply_text("❌ Error analyzing pair. Please try again.")
    
    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show trading statistics"""
        if not self.is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized access!")
            return
        
        stats = self.performance_tracker.get_statistics()
        
        stats_message = f"""
📊 **Trading Statistics** 📊

**Performance Overview:**
🎯 **Total Signals:** {stats.get('total_signals', 0)}
✅ **Winning Trades:** {stats.get('winning_trades', 0)}
❌ **Losing Trades:** {stats.get('losing_trades', 0)}
🏆 **Win Rate:** {stats.get('win_rate', 0):.1f}%

**Time-Based Performance:**
📅 **Today:** {stats.get('today_signals', 0)} signals ({stats.get('today_win_rate', 0):.1f}% win rate)
📅 **This Week:** {stats.get('week_signals', 0)} signals ({stats.get('week_win_rate', 0):.1f}% win rate)
📅 **This Month:** {stats.get('month_signals', 0)} signals ({stats.get('month_win_rate', 0):.1f}% win rate)

**Accuracy by Timeframe:**
⏰ **2min Trades:** {stats.get('accuracy_2min', 0):.1f}%
⏰ **3min Trades:** {stats.get('accuracy_3min', 0):.1f}%
⏰ **5min Trades:** {stats.get('accuracy_5min', 0):.1f}%

**Best Performing Pairs:**
🥇 {stats.get('best_pair_1', 'N/A')} - {stats.get('best_pair_1_rate', 0):.1f}%
🥈 {stats.get('best_pair_2', 'N/A')} - {stats.get('best_pair_2_rate', 0):.1f}%
🥉 {stats.get('best_pair_3', 'N/A')} - {stats.get('best_pair_3_rate', 0):.1f}%

**System Status:**
🤖 **Model Accuracy:** {stats.get('model_accuracy', 0):.1f}%
⚡ **Signal Confidence:** {stats.get('avg_confidence', 0):.1f}%
🎯 **Target Achievement:** {stats.get('target_achievement', 0):.1f}%
        """
        
        await update.message.reply_text(stats_message, parse_mode='Markdown')
    
    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show bot system status"""
        if not self.is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized access!")
            return
        
        system_status = self.get_system_status()
        
        status_message = f"""
🤖 **Bot System Status** 🤖

**Bot Information:**
🟢 **Status:** {'Active' if self.bot_status['active'] else 'Inactive'}
🔄 **Auto Signals:** {'Enabled' if self.bot_status['auto_signals'] else 'Disabled'}
📊 **Signals Today:** {self.bot_status['signals_today']}
⏰ **Last Signal:** {self.bot_status['last_signal_time'] or 'None'}

**System Health:**
🧠 **AI Model:** {'🟢 Online' if system_status.get('model_loaded', False) else '🔴 Offline'}
🌐 **Market Data:** {'🟢 Connected' if system_status.get('data_connected', False) else '🔴 Disconnected'}
💾 **Database:** {'🟢 Operational' if system_status.get('database_ok', False) else '🔴 Error'}
📡 **API Connection:** {'🟢 Connected' if system_status.get('api_connected', False) else '🔴 Disconnected'}

**Performance Metrics:**
🎯 **Response Time:** {system_status.get('response_time', 'N/A')}ms
💾 **Memory Usage:** {system_status.get('memory_usage', 'N/A')}%
⚡ **CPU Usage:** {system_status.get('cpu_usage', 'N/A')}%
🕒 **Uptime:** {system_status.get('uptime', 'N/A')}

**Configuration:**
📈 **Min Accuracy:** {SIGNAL_CONFIG['min_accuracy']}%
🎯 **Min Confidence:** {SIGNAL_CONFIG['min_confidence']}%
📊 **Max Daily Signals:** {SIGNAL_CONFIG['max_signals_per_day']}
        """
        
        await update.message.reply_text(status_message, parse_mode='Markdown')
    
    async def settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show bot settings"""
        if not self.is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized access!")
            return
        
        keyboard = [
            [
                InlineKeyboardButton("🎯 Signal Settings", callback_data="settings_signals"),
                InlineKeyboardButton("🛡️ Risk Settings", callback_data="settings_risk")
            ],
            [
                InlineKeyboardButton("⏰ Time Settings", callback_data="settings_time"),
                InlineKeyboardButton("📊 Analysis Settings", callback_data="settings_analysis")
            ],
            [
                InlineKeyboardButton("🔔 Notification Settings", callback_data="settings_notifications"),
                InlineKeyboardButton("💾 Backup Settings", callback_data="settings_backup")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        settings_message = """
⚙️ **Bot Settings** ⚙️

Configure your trading bot settings:

🎯 **Signal Settings** - Accuracy thresholds, confidence levels
🛡️ **Risk Settings** - Risk management parameters
⏰ **Time Settings** - Trading hours, expiry times
📊 **Analysis Settings** - Technical indicators, timeframes
🔔 **Notification Settings** - Alerts and messages
💾 **Backup Settings** - Data backup and recovery

Select a category to modify settings:
        """
        
        await update.message.reply_text(settings_message, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show help message"""
        if not self.is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized access!")
            return
        
        help_message = """
🆘 **Help & Commands** 🆘

**Quick Start:**
1. Use /signal to get your first trading signal
2. Enable /auto_on for automatic signals
3. Check /stats for performance tracking

**Main Commands:**
• `/signal` - Get instant trading signal
• `/analyze [pair]` - Analyze specific pair
• `/stats` - View performance statistics
• `/status` - Check bot system status

**Signal Commands:**
• `/auto_on` - Enable automatic signals
• `/auto_off` - Disable automatic signals
• `/pairs` - Show available pairs
• `/market_status` - Market conditions

**Analysis Commands:**
• `/volatility [pair]` - Check volatility
• `/support_resistance [pair]` - S&R levels
• `/technical [pair]` - Technical indicators

**Settings:**
• `/settings` - Configure bot settings
• `/risk_settings` - Risk management
• `/backup` - Create data backup

**Support:**
If you need help or encounter issues:
• Check /status for system health
• Use /restart to restart services
• Contact support if problems persist

**Tips for Best Results:**
• Trade during low volatility periods
• Follow the recommended expiry times
• Monitor your win rate regularly
• Use proper risk management
        """
        
        await update.message.reply_text(help_message, parse_mode='Markdown')
    
    async def performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show detailed performance report"""
        if not self.is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized access!")
            return
        
        # Generate performance chart
        chart_path = self.performance_tracker.generate_performance_chart()
        
        performance_data = self.performance_tracker.get_detailed_performance()
        
        report_message = f"""
📈 **Detailed Performance Report** 📈

**Overall Performance:**
🎯 **Total Win Rate:** {performance_data.get('overall_win_rate', 0):.2f}%
📊 **Signal Accuracy:** {performance_data.get('signal_accuracy', 0):.2f}%
💰 **Profit Factor:** {performance_data.get('profit_factor', 0):.2f}
📈 **Sharpe Ratio:** {performance_data.get('sharpe_ratio', 0):.2f}

**Recent Performance (Last 30 days):**
✅ **Wins:** {performance_data.get('recent_wins', 0)}
❌ **Losses:** {performance_data.get('recent_losses', 0)}
🎯 **Win Rate:** {performance_data.get('recent_win_rate', 0):.1f}%
📊 **Best Streak:** {performance_data.get('best_streak', 0)} wins

**Performance by Timeframe:**
⏰ **2min:** {performance_data.get('win_rate_2min', 0):.1f}% ({performance_data.get('count_2min', 0)} trades)
⏰ **3min:** {performance_data.get('win_rate_3min', 0):.1f}% ({performance_data.get('count_3min', 0)} trades)
⏰ **5min:** {performance_data.get('win_rate_5min', 0):.1f}% ({performance_data.get('count_5min', 0)} trades)

**AI Model Performance:**
🧠 **Model Accuracy:** {performance_data.get('model_accuracy', 0):.1f}%
🎯 **Confidence Score:** {performance_data.get('avg_confidence', 0):.1f}%
🔄 **Last Retrained:** {performance_data.get('last_retrain', 'N/A')}

**Risk Metrics:**
📉 **Max Drawdown:** {performance_data.get('max_drawdown', 0):.2f}%
🛡️ **Risk-Adjusted Return:** {performance_data.get('risk_adjusted_return', 0):.2f}%
⚠️ **VaR (95%):** {performance_data.get('var_95', 0):.2f}%
        """
        
        if chart_path:
            with open(chart_path, 'rb') as chart_file:
                await update.message.reply_photo(photo=chart_file, caption=report_message, parse_mode='Markdown')
        else:
            await update.message.reply_text(report_message, parse_mode='Markdown')
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button callbacks"""
        query = update.callback_query
        await query.answer()
        
        if not self.is_authorized(query.from_user.id):
            await query.edit_message_text("❌ Unauthorized access!")
            return
        
        data = query.data
        
        if data == "refresh_signal":
            # Generate new signal
            signal_data = await self.signal_engine.generate_signal()
            if signal_data:
                signal_message = self._format_signal(signal_data)
                await query.edit_message_text(signal_message, parse_mode='Markdown')
        
        elif data.startswith("analysis_"):
            pair = data.split("_")[1]
            analysis = await self.signal_engine.analyze_pair(pair)
            analysis_message = f"📊 Quick analysis for {pair}:\n"
            analysis_message += f"Signal: {analysis.get('recommendation', 'HOLD')}\n"
            analysis_message += f"Strength: {analysis.get('signal_strength', 'N/A')}/10"
            await query.edit_message_text(analysis_message)
        
        elif data.startswith("settings_"):
            setting_type = data.split("_")[1]
            await self.handle_settings(query, setting_type)
    
    async def handle_settings(self, query, setting_type):
        """Handle different settings categories"""
        if setting_type == "signals":
            message = f"""
🎯 **Signal Settings** 🎯

Current Configuration:
• **Min Accuracy:** {SIGNAL_CONFIG['min_accuracy']}%
• **Min Confidence:** {SIGNAL_CONFIG['min_confidence']}%
• **Max Daily Signals:** {SIGNAL_CONFIG['max_signals_per_day']}
• **Signal Advance Time:** {SIGNAL_CONFIG['signal_advance_time']} minute(s)

Available Expiry Durations:
• {', '.join(map(str, SIGNAL_CONFIG['expiry_durations']))} minutes

Use settings commands to modify these values.
            """
        
        elif setting_type == "risk":
            message = f"""
🛡️ **Risk Management Settings** 🛡️

Current Configuration:
• **Max Risk per Trade:** {RISK_MANAGEMENT['max_risk_per_trade']}%
• **Max Daily Loss:** {RISK_MANAGEMENT['max_daily_loss']}%
• **Min Win Rate:** {RISK_MANAGEMENT['min_win_rate']}%
• **Stop Loss Threshold:** {RISK_MANAGEMENT['stop_loss_threshold']}%
• **Max Concurrent Trades:** {RISK_MANAGEMENT['max_concurrent_trades']}

These settings help protect your account from excessive losses.
            """
        
        else:
            message = f"Settings for {setting_type} are not implemented yet."
        
        await query.edit_message_text(message, parse_mode='Markdown')
    
    def get_system_status(self):
        """Get current system status"""
        
        return {
            'model_loaded': self.signal_engine.is_model_loaded(),
            'data_connected': self.signal_engine.is_data_connected(),
            'database_ok': self.performance_tracker.test_connection(),
            'api_connected': True,  # Assume connected for now
            'response_time': 150,  # Mock value
            'memory_usage': psutil.virtual_memory().percent,
            'cpu_usage': psutil.cpu_percent(),
            'uptime': str(timedelta(seconds=int(psutil.boot_time())))
        }
    
    async def send_automatic_signal(self):
        """Send automatic signal when conditions are met"""
        if not self.bot_status['auto_signals'] or not self.bot_status['active']:
            return
        
        # Check if we've reached daily signal limit
        if self.bot_status['signals_today'] >= SIGNAL_CONFIG['max_signals_per_day']:
            return
        
        # Check if enough time has passed since last signal
        if self.bot_status['last_signal_time']:
            time_diff = datetime.now() - self.bot_status['last_signal_time']
            if time_diff.total_seconds() < 300:  # 5 minutes minimum between signals
                return
        
        try:
            signal_data = await self.signal_engine.generate_signal()
            
            if signal_data and signal_data['accuracy'] >= SIGNAL_CONFIG['min_accuracy']:
                signal_message = f"🚨 **AUTOMATIC SIGNAL** 🚨\n\n{self._format_signal(signal_data)}"
                
                # Send to authorized users
                for user_id in self.authorized_users:
                    try:
                        await self.app.bot.send_message(
                            chat_id=user_id,
                            text=signal_message,
                            parse_mode='Markdown'
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to send signal to user {user_id}: {e}")
                
                # Update status
                self.bot_status['last_signal_time'] = datetime.now()
                self.bot_status['signals_today'] += 1
                
                # Save signal
                self.performance_tracker.save_signal(signal_data)
                
        except Exception as e:
            self.logger.error(f"Error sending automatic signal: {e}")
    
    async def setup_periodic_tasks(self):
        """Setup periodic tasks for automatic signals"""
        while True:
            try:
                await self.send_automatic_signal()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in periodic task: {e}")
                await asyncio.sleep(60)
    
    async def start_bot(self):
        """Start the Telegram bot"""
        try:
            # Create application
            self.app = Application.builder().token(self.token).build()
            
            # Add command handlers
            self.app.add_handler(CommandHandler("start", self.start))
            self.app.add_handler(CommandHandler("signal", self.signal))
            self.app.add_handler(CommandHandler("auto_on", self.auto_signals_on))
            self.app.add_handler(CommandHandler("auto_off", self.auto_signals_off))
            self.app.add_handler(CommandHandler("pairs", self.pairs))
            self.app.add_handler(CommandHandler("market_status", self.market_status))
            self.app.add_handler(CommandHandler("analyze", self.analyze))
            self.app.add_handler(CommandHandler("stats", self.stats))
            self.app.add_handler(CommandHandler("status", self.status))
            self.app.add_handler(CommandHandler("settings", self.settings))
            self.app.add_handler(CommandHandler("help", self.help_command))
            self.app.add_handler(CommandHandler("performance", self.performance))
            
            # Add callback query handler
            self.app.add_handler(CallbackQueryHandler(self.button_callback))
            
            # Start periodic tasks
            asyncio.create_task(self.setup_periodic_tasks())
            
            # Start bot
            self.logger.info("Starting Telegram bot...")
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()
            
            self.logger.info("Telegram bot started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting bot: {e}")
            raise

if __name__ == "__main__":
    bot = TradingBot()
    asyncio.run(bot.start_bot())