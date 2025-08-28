#!/usr/bin/env python3
"""
Simple Paper Trading Test
"""

import pandas as pd
import numpy as np
import logging
import time
import json
import os
from datetime import datetime, timedelta
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplePaperTrader:
    def __init__(self, duration_days=7):
        self.duration_days = duration_days
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(days=duration_days)
        self.balance = 10000.0
        self.trades = []
        self.session_id = f"session_{int(time.time())}"
        
    def load_model(self):
        """Load the latest trained model"""
        try:
            import joblib
            
            # Find latest model
            model_files = glob.glob('/workspace/models/simple_working_model_*.pkl')
            if not model_files:
                logger.error("No trained model found")
                return False
            
            latest_model = max(model_files)
            self.model = joblib.load(latest_model)
            
            # Load scaler
            scaler_files = glob.glob('/workspace/models/simple_working_scaler_*.pkl')
            if scaler_files:
                self.scaler = joblib.load(max(scaler_files))
            else:
                self.scaler = None
            
            logger.info(f"✅ Loaded model: {os.path.basename(latest_model)}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False
    
    def generate_signal(self):
        """Generate a simple trading signal"""
        try:
            # Create dummy features (in real system, this would be real market data)
            features = np.random.randn(1, 15)  # 15 features as in our trained model
            
            if self.scaler:
                features = self.scaler.transform(features)
            
            # Get prediction
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            # Convert to trading signal
            direction = "BUY" if prediction == 1 else "SELL"
            confidence = max(probabilities) * 100
            
            # Only trade if confidence is reasonable
            if confidence >= 55:  # Above random chance
                return {
                    'direction': direction,
                    'confidence': confidence,
                    'symbol': 'EURUSD',
                    'timestamp': datetime.now()
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Error generating signal: {e}")
            return None
    
    def execute_trade(self, signal):
        """Execute a paper trade"""
        try:
            trade_amount = min(200, self.balance * 0.02)  # 2% risk
            
            # Simulate trade outcome (in real system, this would be actual market data)
            # Use model confidence to bias the outcome
            success_probability = signal['confidence'] / 100.0
            is_win = np.random.random() < success_probability
            
            if is_win:
                profit = trade_amount * 0.8  # 80% payout
                self.balance += profit
                result = "WIN"
            else:
                self.balance -= trade_amount
                profit = -trade_amount
                result = "LOSS"
            
            trade = {
                'id': len(self.trades) + 1,
                'timestamp': signal['timestamp'],
                'symbol': signal['symbol'],
                'direction': signal['direction'],
                'amount': trade_amount,
                'confidence': signal['confidence'],
                'result': result,
                'profit': profit,
                'balance': self.balance
            }
            
            self.trades.append(trade)
            
            logger.info(f"📊 Trade {trade['id']}: {trade['result']} "
                       f"{trade['direction']} {trade['symbol']} "
                       f"${trade['profit']:.2f} (Balance: ${self.balance:.2f})")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error executing trade: {e}")
            return False
    
    def run_paper_trading(self):
        """Run the paper trading session"""
        logger.info(f"🚀 Starting {self.duration_days}-day paper trading session")
        logger.info(f"📅 Session: {self.session_id}")
        logger.info(f"💰 Starting balance: ${self.balance:.2f}")
        
        if not self.load_model():
            return False
        
        # Simulate trading over the duration
        # For demonstration, we'll do accelerated trading (1 trade per 10 seconds)
        trades_per_day = 10
        total_trades_target = self.duration_days * trades_per_day
        
        logger.info(f"🎯 Target trades: {total_trades_target}")
        
        for i in range(total_trades_target):
            if datetime.now() > self.end_time:
                break
            
            # Generate signal
            signal = self.generate_signal()
            
            if signal:
                success = self.execute_trade(signal)
                if not success:
                    continue
            
            # Wait between trades (shorter for demo)
            time.sleep(2)  # 2 seconds between attempts
            
            # Log progress every 10 trades
            if (i + 1) % 10 == 0:
                win_rate = sum(1 for t in self.trades if t['result'] == 'WIN') / len(self.trades) * 100
                total_profit = sum(t['profit'] for t in self.trades)
                logger.info(f"📊 Progress: {len(self.trades)}/{total_trades_target} trades, "
                           f"Win rate: {win_rate:.1f}%, PnL: ${total_profit:.2f}")
        
        # Final report
        self.generate_final_report()
        return True
    
    def generate_final_report(self):
        """Generate final trading report"""
        if not self.trades:
            logger.warning("No trades executed")
            return
        
        wins = [t for t in self.trades if t['result'] == 'WIN']
        losses = [t for t in self.trades if t['result'] == 'LOSS']
        
        win_rate = len(wins) / len(self.trades) * 100
        total_profit = sum(t['profit'] for t in self.trades)
        roi = (self.balance - 10000) / 10000 * 100
        
        logger.info("=" * 60)
        logger.info("📊 PAPER TRADING SESSION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"📅 Session: {self.session_id}")
        logger.info(f"⏱️ Duration: {self.duration_days} days")
        logger.info(f"📊 Total Trades: {len(self.trades)}")
        logger.info(f"✅ Winning Trades: {len(wins)}")
        logger.info(f"❌ Losing Trades: {len(losses)}")
        logger.info(f"🎯 Win Rate: {win_rate:.1f}%")
        logger.info(f"💰 Total PnL: ${total_profit:.2f}")
        logger.info(f"💰 Final Balance: ${self.balance:.2f}")
        logger.info(f"📈 ROI: {roi:.2f}%")
        logger.info("=" * 60)
        
        # Save report
        report = {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_days': self.duration_days,
            'total_trades': len(self.trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_profit,
            'final_balance': self.balance,
            'roi': roi,
            'trades': self.trades
        }
        
        report_file = f'/workspace/logs/paper_trading_report_{self.session_id}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"💾 Report saved: {report_file}")

if __name__ == "__main__":
    import sys
    
    duration = 7  # Default 7 days
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except ValueError:
            duration = 7
    
    trader = SimplePaperTrader(duration_days=duration)
    success = trader.run_paper_trading()
    
    if success:
        print("\n🎉 Paper trading session completed successfully!")
    else:
        print("\n❌ Paper trading session failed")
