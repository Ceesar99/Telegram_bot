#!/usr/bin/env python3
"""
AI Trading Bot for Binary Options - Enhanced Version
Integrates advanced AI models with Pocket Option platform for automated trading
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import time
from typing import Dict, List, Optional

# Add workspace to path
sys.path.append('/workspace')

# Import our components
from binary_options_ai_model import BinaryOptionsAIModel
from pocket_option_enhanced_api import EnhancedPocketOptionAPI, TradingSignal, PocketOptionDataCollector
from config_manager import config_manager
from error_handler import global_error_handler, ErrorSeverity

class AITradingBot:
    def __init__(self, demo_mode=True):
        self.logger = logging.getLogger('AITradingBot')
        self.demo_mode = demo_mode
        
        # Initialize components
        self.ai_model = BinaryOptionsAIModel()
        self.api = EnhancedPocketOptionAPI(demo_mode=demo_mode)
        self.data_collector = PocketOptionDataCollector(self.api)
        
        # Trading configuration
        self.config = {
            'min_confidence': 60.0,  # Minimum confidence to place trade
            'max_daily_trades': 20,
            'max_daily_loss': 50.0,  # USD
            'trade_amount': 1.0,     # USD per trade
            'trading_pairs': ['EURUSD_OTC', 'GBPUSD_OTC', 'USDJPY_OTC', 'AUDUSD_OTC'],
            'data_history_minutes': 120,  # How much historical data to use
            'signal_cooldown': 300,   # 5 minutes between signals for same pair
        }
        
        # Trading state
        self.active_trades = {}
        self.daily_trades = 0
        self.daily_profit_loss = 0.0
        self.last_signals = {}  # Track last signal time per pair
        self.is_trading = False
        self.session_start = datetime.now()
        
        # Load AI model
        self._load_ai_model()
    
    def _load_ai_model(self):
        """Load the pre-trained AI model"""
        try:
            if self.ai_model.is_model_trained():
                success = self.ai_model.load_model()
                if success:
                    self.logger.info("✅ AI model loaded successfully")
                else:
                    self.logger.error("❌ Failed to load AI model")
            else:
                self.logger.warning("⚠️ No pre-trained model found, training new model...")
                # Train model with sample data
                from binary_options_ai_model import create_realistic_market_data
                training_data = create_realistic_market_data(5000)
                self.ai_model.train(training_data)
                self.logger.info("✅ AI model trained and ready")
                
        except Exception as e:
            global_error_handler.handle_error(e, "AITradingBot", ErrorSeverity.HIGH)
    
    async def connect(self, ssid=None, email=None, password=None):
        """Connect to Pocket Option platform"""
        try:
            success = await self.api.connect(email=email, password=password, ssid=ssid)
            if success:
                self.logger.info("✅ Connected to Pocket Option platform")
                
                # Set up event callbacks
                self.api.register_callback('trade_result', self._handle_trade_result)
                self.api.register_callback('balance_update', self._handle_balance_update)
                
                return True
            else:
                self.logger.error("❌ Failed to connect to Pocket Option")
                return False
                
        except Exception as e:
            global_error_handler.handle_error(e, "APIConnection", ErrorSeverity.CRITICAL)
            return False
    
    async def start_trading(self):
        """Start the automated trading process"""
        try:
            if not self.api.is_connected:
                self.logger.error("❌ Not connected to platform. Call connect() first.")
                return
            
            self.is_trading = True
            self.session_start = datetime.now()
            self.daily_trades = 0
            self.daily_profit_loss = 0.0
            
            self.logger.info("🚀 Starting AI trading bot...")
            self.logger.info(f"📊 Trading pairs: {self.config['trading_pairs']}")
            self.logger.info(f"💰 Trade amount: ${self.config['trade_amount']}")
            self.logger.info(f"🎯 Min confidence: {self.config['min_confidence']}%")
            
            # Start the main trading loop
            await self._main_trading_loop()
            
        except Exception as e:
            global_error_handler.handle_error(e, "TradingLoop", ErrorSeverity.CRITICAL)
            self.is_trading = False
    
    async def _main_trading_loop(self):
        """Main trading loop"""
        while self.is_trading:
            try:
                # Check trading limits
                if not self._check_trading_limits():
                    self.logger.info("📊 Trading limits reached, pausing...")
                    await asyncio.sleep(60)  # Wait 1 minute before checking again
                    continue
                
                # Generate signals for each trading pair
                for pair in self.config['trading_pairs']:
                    try:
                        await self._process_trading_pair(pair)
                        await asyncio.sleep(2)  # Small delay between pairs
                    except Exception as e:
                        self.logger.error(f"Error processing {pair}: {e}")
                
                # Wait before next iteration
                await asyncio.sleep(30)  # 30 seconds between full cycles
                
            except Exception as e:
                global_error_handler.handle_error(e, "MainLoop", ErrorSeverity.HIGH)
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _process_trading_pair(self, pair: str):
        """Process a single trading pair for signals"""
        try:
            # Check cooldown
            if self._is_in_cooldown(pair):
                return
            
            # Get historical data
            historical_data = await self.api.get_candle_data(pair, timeframe='1m', count=self.config['data_history_minutes'])
            
            if historical_data.empty or len(historical_data) < 60:
                self.logger.debug(f"Insufficient data for {pair}")
                return
            
            # Generate AI signal
            signal = self.ai_model.predict_signal(historical_data)
            
            if signal['signal'] in ['CALL', 'PUT'] and signal['confidence'] >= self.config['min_confidence']:
                self.logger.info(f"🎯 Strong signal for {pair}: {signal}")
                
                # Create trading signal
                trading_signal = TradingSignal(
                    asset=pair,
                    direction='call' if signal['direction'] == 'CALL' else 'put',
                    amount=self.config['trade_amount'],
                    expiry_time=signal['expiry_minutes'],
                    confidence=signal['confidence'],
                    timestamp=datetime.now()
                )
                
                # Place trade
                success = await self._place_trade(trading_signal)
                if success:
                    self.last_signals[pair] = time.time()
                    self.daily_trades += 1
                    
        except Exception as e:
            self.logger.error(f"Error processing pair {pair}: {e}")
    
    def _check_trading_limits(self) -> bool:
        """Check if trading limits are exceeded"""
        # Check daily trade limit
        if self.daily_trades >= self.config['max_daily_trades']:
            return False
        
        # Check daily loss limit
        if self.daily_profit_loss <= -self.config['max_daily_loss']:
            return False
        
        # Check market hours (for forex pairs)
        current_time = datetime.now()
        current_hour = current_time.hour
        
        # Avoid trading during low volatility hours (22:00 - 06:00 UTC)
        if current_hour >= 22 or current_hour <= 6:
            return False
        
        return True
    
    def _is_in_cooldown(self, pair: str) -> bool:
        """Check if pair is in cooldown period"""
        if pair not in self.last_signals:
            return False
        
        time_since_last = time.time() - self.last_signals[pair]
        return time_since_last < self.config['signal_cooldown']
    
    async def _place_trade(self, signal: TradingSignal) -> bool:
        """Place a trade based on signal"""
        try:
            # Check balance
            balance = self.api.get_balance()
            if balance < signal.amount:
                self.logger.warning(f"⚠️ Insufficient balance: ${balance:.2f}")
                return False
            
            # Place trade
            trade_result = await self.api.place_trade(signal)
            
            if trade_result:
                self.logger.info(f"✅ Trade placed: {signal.asset} {signal.direction} ${signal.amount} - Confidence: {signal.confidence:.1f}%")
                
                # Track trade
                self.active_trades[trade_result['trade_id']] = {
                    'signal': signal,
                    'trade_info': trade_result,
                    'placed_at': datetime.now()
                }
                
                return True
            else:
                self.logger.error("❌ Failed to place trade")
                return False
                
        except Exception as e:
            global_error_handler.handle_error(e, "TradePlacement", ErrorSeverity.HIGH)
            return False
    
    async def _handle_trade_result(self, trade_result):
        """Handle trade completion"""
        try:
            trade_id = trade_result['trade_id']
            result = trade_result['result']
            payout = trade_result.get('payout', 0)
            
            if trade_id in self.active_trades:
                trade_info = self.active_trades[trade_id]
                signal = trade_info['signal']
                
                if result == 'win':
                    profit = payout - signal.amount
                    self.daily_profit_loss += profit
                    self.logger.info(f"🎉 WIN: {signal.asset} +${profit:.2f} (Total: ${self.daily_profit_loss:.2f})")
                else:
                    loss = -signal.amount
                    self.daily_profit_loss += loss
                    self.logger.info(f"💸 LOSS: {signal.asset} ${loss:.2f} (Total: ${self.daily_profit_loss:.2f})")
                
                # Log detailed result
                self._log_trade_result(trade_info, result, profit if result == 'win' else loss)
                
                # Remove from active trades
                del self.active_trades[trade_id]
                
        except Exception as e:
            self.logger.error(f"Error handling trade result: {e}")
    
    def _log_trade_result(self, trade_info, result, profit_loss):
        """Log detailed trade result"""
        signal = trade_info['signal']
        trade_data = {
            'timestamp': datetime.now().isoformat(),
            'asset': signal.asset,
            'direction': signal.direction,
            'amount': signal.amount,
            'confidence': signal.confidence,
            'expiry_minutes': signal.expiry_time,
            'result': result,
            'profit_loss': profit_loss,
            'balance_after': self.api.get_balance(),
            'daily_total': self.daily_profit_loss
        }
        
        # Save to file
        try:
            with open('/workspace/logs/trade_results.json', 'a') as f:
                f.write(json.dumps(trade_data) + '\n')
        except Exception as e:
            self.logger.error(f"Error saving trade result: {e}")
    
    async def _handle_balance_update(self, balance_data):
        """Handle balance updates"""
        new_balance = balance_data['new_balance']
        old_balance = balance_data['old_balance']
        change = new_balance - old_balance
        
        if abs(change) > 0.01:  # Only log significant changes
            self.logger.info(f"💰 Balance: ${old_balance:.2f} → ${new_balance:.2f} ({change:+.2f})")
    
    def get_trading_statistics(self) -> Dict:
        """Get current trading statistics"""
        session_duration = datetime.now() - self.session_start
        
        return {
            'session_duration_hours': session_duration.total_seconds() / 3600,
            'daily_trades': self.daily_trades,
            'daily_profit_loss': self.daily_profit_loss,
            'current_balance': self.api.get_balance(),
            'active_trades_count': len(self.active_trades),
            'trading_pairs': self.config['trading_pairs'],
            'is_trading': self.is_trading
        }
    
    def stop_trading(self):
        """Stop the trading bot"""
        self.is_trading = False
        self.logger.info("🛑 Trading bot stopped")
        
        # Print session summary
        stats = self.get_trading_statistics()
        self.logger.info(f"📊 Session Summary:")
        self.logger.info(f"   Duration: {stats['session_duration_hours']:.1f} hours")
        self.logger.info(f"   Trades: {stats['daily_trades']}")
        self.logger.info(f"   P&L: ${stats['daily_profit_loss']:.2f}")
        self.logger.info(f"   Balance: ${stats['current_balance']:.2f}")
    
    async def disconnect(self):
        """Disconnect from platform"""
        self.stop_trading()
        await self.api.disconnect()

async def main():
    """Main function to run the AI trading bot"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create trading bot
    bot = AITradingBot(demo_mode=True)  # Start in demo mode for safety
    
    # Connect to platform
    # In production, use real credentials:
    # success = await bot.connect(email="your_email", password="your_password")
    # Or use SSID from config:
    # success = await bot.connect(ssid=config_manager.get('POCKET_OPTION_SSID'))
    
    # For demo purposes, we'll simulate connection
    print("🤖 AI Trading Bot for Binary Options")
    print("=" * 50)
    print("Demo Mode: ON")
    print("AI Model: ✅ Loaded")
    print("Platform: Pocket Option")
    print("=" * 50)
    
    # In production, uncomment this to start actual trading:
    # if success:
    #     try:
    #         await bot.start_trading()
    #     except KeyboardInterrupt:
    #         print("\n⚠️ Interrupted by user")
    #     finally:
    #         await bot.disconnect()
    # else:
    #     print("❌ Failed to connect to platform")
    
    # Demo information
    print("\n📋 To run in production:")
    print("1. Set up your Pocket Option credentials in .env file")
    print("2. Update config.py with your trading parameters")
    print("3. Set demo_mode=False")
    print("4. Uncomment the trading loop above")
    print("\n🚨 DISCLAIMER: Trading binary options involves significant risk")

if __name__ == "__main__":
    asyncio.run(main())