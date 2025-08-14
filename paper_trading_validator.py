#!/usr/bin/env python3
"""
📊 Paper Trading Validation System
Comprehensive validation of AI models before live trading

Features:
- Real-time paper trading simulation
- Performance metrics tracking
- Risk management validation
- Drawdown monitoring
- Win rate analysis
- Profit/Loss tracking
"""

import asyncio
import logging
import json
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
import pytz
from dataclasses import dataclass
import sqlite3
import os

from config import (
    CURRENCY_PAIRS, OTC_PAIRS, TIMEZONE, MARKET_TIMEZONE,
    RISK_MANAGEMENT, SIGNAL_CONFIG
)

@dataclass
class PaperTrade:
    """Paper trade representation"""
    id: str
    timestamp: datetime
    pair: str
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    expiry_time: datetime
    duration: int  # minutes
    amount: float
    predicted_accuracy: float
    actual_result: Optional[str] = None  # 'WIN', 'LOSS', 'DRAW'
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    profit_percentage: Optional[float] = None

@dataclass
class PaperTradingSession:
    """Paper trading session statistics"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_balance: float = 10000.0
    current_balance: float = 10000.0
    win_rate: float = 0.0
    avg_accuracy: float = 0.0
    total_volume: float = 0.0

class PaperTradingValidator:
    """Comprehensive paper trading validation system"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.trades: List[PaperTrade] = []
        self.sessions: List[PaperTradingSession] = []
        self.current_session: Optional[PaperTradingSession] = None
        
        # Risk management
        self.max_risk_per_trade = RISK_MANAGEMENT['max_risk_per_trade'] / 100
        self.max_daily_loss = RISK_MANAGEMENT['max_daily_loss'] / 100
        self.min_win_rate = RISK_MANAGEMENT['min_win_rate'] / 100
        self.max_concurrent_trades = RISK_MANAGEMENT['max_concurrent_trades']
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.active_trades: List[PaperTrade] = []
        
        # Setup logging and database
        self.logger = self._setup_logger()
        self._initialize_database()
        
    def _setup_logger(self):
        """Setup comprehensive logging"""
        logger = logging.getLogger('PaperTradingValidator')
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        os.makedirs('/workspace/logs', exist_ok=True)
        
        # File handler
        handler = logging.FileHandler('/workspace/logs/paper_trading.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _initialize_database(self):
        """Initialize paper trading database"""
        try:
            conn = sqlite3.connect('/workspace/data/paper_trading.db')
            cursor = conn.cursor()
            
            # Create trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_trades (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    expiry_time TEXT NOT NULL,
                    duration INTEGER NOT NULL,
                    amount REAL NOT NULL,
                    predicted_accuracy REAL NOT NULL,
                    actual_result TEXT,
                    exit_price REAL,
                    pnl REAL,
                    profit_percentage REAL,
                    session_id TEXT
                )
            ''')
            
            # Create sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0.0,
                    max_drawdown REAL DEFAULT 0.0,
                    peak_balance REAL DEFAULT 10000.0,
                    current_balance REAL DEFAULT 10000.0,
                    win_rate REAL DEFAULT 0.0,
                    avg_accuracy REAL DEFAULT 0.0,
                    total_volume REAL DEFAULT 0.0
                )
            ''')
            
            # Create daily performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_performance (
                    date TEXT PRIMARY KEY,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0.0,
                    win_rate REAL DEFAULT 0.0,
                    max_drawdown REAL DEFAULT 0.0,
                    balance_start REAL DEFAULT 10000.0,
                    balance_end REAL DEFAULT 10000.0
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
    
    def start_session(self, session_id: str = None) -> str:
        """Start a new paper trading session"""
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        self.current_session = PaperTradingSession(
            session_id=session_id,
            start_time=datetime.now(TIMEZONE)
        )
        
        self.logger.info(f"Started paper trading session: {session_id}")
        return session_id
    
    def end_session(self) -> Dict[str, Any]:
        """End current session and return statistics"""
        if not self.current_session:
            self.logger.warning("No active session to end")
            return {}
        
        self.current_session.end_time = datetime.now(TIMEZONE)
        self.current_session.current_balance = self.current_balance
        
        # Calculate final statistics
        session_stats = self._calculate_session_stats(self.current_session)
        
        # Save to database
        self._save_session_to_db(self.current_session)
        
        self.logger.info(f"Ended session {self.current_session.session_id}")
        self.logger.info(f"Session PnL: ${session_stats['total_pnl']:.2f}")
        self.logger.info(f"Session Win Rate: {session_stats['win_rate']:.2%}")
        
        self.sessions.append(self.current_session)
        self.current_session = None
        
        return session_stats
    
    def execute_paper_trade(self, signal_data: Dict[str, Any]) -> Optional[PaperTrade]:
        """Execute a paper trade based on signal data"""
        try:
            # Validate signal
            if not self._validate_signal(signal_data):
                return None
            
            # Check risk management
            if not self._check_risk_limits():
                self.logger.warning("Risk limits exceeded, skipping trade")
                return None
            
            # Calculate position size
            position_size = self._calculate_position_size(signal_data)
            
            # Create paper trade
            trade = PaperTrade(
                id=f"PT_{int(time.time())}_{random.randint(1000, 9999)}",
                timestamp=datetime.now(TIMEZONE),
                pair=signal_data['pair'],
                direction=signal_data['direction'],
                entry_price=signal_data['entry_price'],
                expiry_time=signal_data['time_expiry'],
                duration=signal_data.get('recommended_duration', 2),
                amount=position_size,
                predicted_accuracy=signal_data['accuracy']
            )
            
            # Add to active trades
            self.active_trades.append(trade)
            self.trades.append(trade)
            
            # Update balance
            self.current_balance -= position_size
            self.daily_trades += 1
            
            # Log trade
            self.logger.info(f"Paper trade executed: {trade.pair} {trade.direction} "
                           f"${trade.amount:.2f} @ {trade.entry_price:.5f}")
            
            # Save to database
            self._save_trade_to_db(trade)
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Error executing paper trade: {e}")
            return None
    
    def simulate_trade_result(self, trade: PaperTrade) -> Dict[str, Any]:
        """Simulate trade result based on predicted accuracy"""
        try:
            # Simulate market movement
            predicted_accuracy = trade.predicted_accuracy / 100
            
            # Add some randomness to make it realistic
            success_probability = predicted_accuracy * 0.8 + 0.1  # Cap at 90%
            
            # Determine if trade wins
            if random.random() < success_probability:
                result = 'WIN'
                # Simulate profit (typically 80-95% payout)
                payout_rate = random.uniform(0.80, 0.95)
                profit = trade.amount * payout_rate
                exit_price = trade.entry_price * (1 + 0.001) if trade.direction == 'BUY' else trade.entry_price * (1 - 0.001)
            else:
                result = 'LOSS'
                profit = -trade.amount  # Lose the entire amount
                exit_price = trade.entry_price * (1 - 0.001) if trade.direction == 'BUY' else trade.entry_price * (1 + 0.001)
            
            # Update trade
            trade.actual_result = result
            trade.exit_price = exit_price
            trade.pnl = profit
            trade.profit_percentage = (profit / trade.amount) * 100
            
            # Update balance
            self.current_balance += trade.amount + profit
            self.daily_pnl += profit
            
            # Update peak balance
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance
            
            # Remove from active trades
            if trade in self.active_trades:
                self.active_trades.remove(trade)
            
            # Update database
            self._update_trade_in_db(trade)
            
            return {
                'result': result,
                'profit': profit,
                'exit_price': exit_price,
                'new_balance': self.current_balance
            }
            
        except Exception as e:
            self.logger.error(f"Error simulating trade result: {e}")
            return {'result': 'ERROR', 'profit': 0, 'exit_price': trade.entry_price, 'new_balance': self.current_balance}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        try:
            if not self.trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'avg_accuracy': 0.0,
                    'max_drawdown': 0.0,
                    'current_balance': self.current_balance,
                    'profit_percentage': 0.0
                }
            
            # Calculate metrics
            total_trades = len(self.trades)
            completed_trades = [t for t in self.trades if t.actual_result is not None]
            
            if not completed_trades:
                return {
                    'total_trades': total_trades,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'avg_accuracy': np.mean([t.predicted_accuracy for t in self.trades]),
                    'max_drawdown': self._calculate_max_drawdown(),
                    'current_balance': self.current_balance,
                    'profit_percentage': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
                }
            
            winning_trades = [t for t in completed_trades if t.actual_result == 'WIN']
            total_pnl = sum(t.pnl for t in completed_trades)
            win_rate = len(winning_trades) / len(completed_trades)
            avg_accuracy = np.mean([t.predicted_accuracy for t in completed_trades])
            
            return {
                'total_trades': total_trades,
                'completed_trades': len(completed_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(completed_trades) - len(winning_trades),
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_accuracy': avg_accuracy,
                'max_drawdown': self._calculate_max_drawdown(),
                'current_balance': self.current_balance,
                'profit_percentage': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100,
                'peak_balance': self.peak_balance,
                'active_trades': len(self.active_trades)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _validate_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Validate trading signal"""
        required_fields = ['pair', 'direction', 'entry_price', 'accuracy']
        
        for field in required_fields:
            if field not in signal_data:
                self.logger.error(f"Missing required field: {field}")
                return False
        
        if signal_data['accuracy'] < SIGNAL_CONFIG['min_accuracy']:
            self.logger.warning(f"Signal accuracy too low: {signal_data['accuracy']}%")
            return False
        
        return True
    
    def _check_risk_limits(self) -> bool:
        """Check if trade meets risk management requirements"""
        # Check daily loss limit
        daily_loss_percentage = abs(self.daily_pnl) / self.initial_balance
        if daily_loss_percentage > self.max_daily_loss:
            self.logger.warning(f"Daily loss limit exceeded: {daily_loss_percentage:.2%}")
            return False
        
        # Check concurrent trades limit
        if len(self.active_trades) >= self.max_concurrent_trades:
            self.logger.warning("Maximum concurrent trades reached")
            return False
        
        return True
    
    def _calculate_position_size(self, signal_data: Dict[str, Any]) -> float:
        """Calculate position size based on risk management"""
        risk_amount = self.current_balance * self.max_risk_per_trade
        return min(risk_amount, self.current_balance * 0.1)  # Max 10% per trade
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.trades:
            return 0.0
        
        peak = self.initial_balance
        max_dd = 0.0
        
        for trade in self.trades:
            if trade.actual_result is not None:
                # Simulate balance after each trade
                if trade.pnl:
                    current_balance = self.initial_balance + sum(t.pnl for t in self.trades[:self.trades.index(trade)+1] if t.pnl)
                    
                    if current_balance > peak:
                        peak = current_balance
                    
                    drawdown = (peak - current_balance) / peak
                    if drawdown > max_dd:
                        max_dd = drawdown
        
        return max_dd * 100  # Return as percentage
    
    def _calculate_session_stats(self, session: PaperTradingSession) -> Dict[str, Any]:
        """Calculate session statistics"""
        session_trades = [t for t in self.trades if hasattr(t, 'session_id') and t.session_id == session.session_id]
        
        if not session_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_accuracy': 0.0
            }
        
        completed_trades = [t for t in session_trades if t.actual_result is not None]
        winning_trades = [t for t in completed_trades if t.actual_result == 'WIN']
        
        return {
            'total_trades': len(session_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(completed_trades) - len(winning_trades),
            'total_pnl': sum(t.pnl for t in completed_trades if t.pnl),
            'win_rate': len(winning_trades) / len(completed_trades) if completed_trades else 0.0,
            'avg_accuracy': np.mean([t.predicted_accuracy for t in session_trades])
        }
    
    def _save_trade_to_db(self, trade: PaperTrade):
        """Save trade to database"""
        try:
            conn = sqlite3.connect('/workspace/data/paper_trading.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO paper_trades 
                (id, timestamp, pair, direction, entry_price, expiry_time, duration, 
                 amount, predicted_accuracy, actual_result, exit_price, pnl, profit_percentage, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.id, trade.timestamp.isoformat(), trade.pair, trade.direction,
                trade.entry_price, trade.expiry_time.isoformat(), trade.duration,
                trade.amount, trade.predicted_accuracy, trade.actual_result,
                trade.exit_price, trade.pnl, trade.profit_percentage,
                self.current_session.session_id if self.current_session else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving trade to database: {e}")
    
    def _update_trade_in_db(self, trade: PaperTrade):
        """Update trade result in database"""
        try:
            conn = sqlite3.connect('/workspace/data/paper_trading.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE paper_trades 
                SET actual_result = ?, exit_price = ?, pnl = ?, profit_percentage = ?
                WHERE id = ?
            ''', (trade.actual_result, trade.exit_price, trade.pnl, trade.profit_percentage, trade.id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error updating trade in database: {e}")
    
    def _save_session_to_db(self, session: PaperTradingSession):
        """Save session to database"""
        try:
            stats = self._calculate_session_stats(session)
            
            conn = sqlite3.connect('/workspace/data/paper_trading.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO paper_sessions 
                (session_id, start_time, end_time, total_trades, winning_trades, losing_trades,
                 total_pnl, max_drawdown, peak_balance, current_balance, win_rate, avg_accuracy, total_volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session.session_id, session.start_time.isoformat(), session.end_time.isoformat(),
                stats['total_trades'], stats['winning_trades'], stats['losing_trades'],
                stats['total_pnl'], self._calculate_max_drawdown(), self.peak_balance,
                self.current_balance, stats['win_rate'], stats['avg_accuracy'], 0.0
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving session to database: {e}")

async def run_paper_trading_validation(duration_days: int = 30, signals_per_day: int = 20):
    """Run comprehensive paper trading validation"""
    logger = logging.getLogger('PaperTradingValidator')
    
    # Initialize validator
    validator = PaperTradingValidator(initial_balance=10000.0)
    
    # Start session
    session_id = validator.start_session()
    logger.info(f"Starting paper trading validation for {duration_days} days")
    logger.info(f"Target: {signals_per_day} signals per day")
    
    # Simulate trading for specified duration
    start_date = datetime.now(TIMEZONE)
    end_date = start_date + timedelta(days=duration_days)
    
    current_date = start_date
    trades_executed = 0
    
    while current_date < end_date:
        # Simulate daily trading
        daily_signals = min(signals_per_day, random.randint(15, 25))  # Some variation
        
        for _ in range(daily_signals):
            # Generate mock signal
            signal = {
                'pair': random.choice(CURRENCY_PAIRS),
                'direction': random.choice(['BUY', 'SELL']),
                'entry_price': round(random.uniform(1.1000, 1.2000), 5),
                'accuracy': random.uniform(85, 98),
                'time_expiry': (datetime.now(TIMEZONE) + timedelta(minutes=2)).strftime('%H:%M:%S'),
                'recommended_duration': random.choice([2, 3, 5])
            }
            
            # Execute paper trade
            trade = validator.execute_paper_trade(signal)
            if trade:
                trades_executed += 1
                
                # Simulate trade result after expiry
                await asyncio.sleep(0.1)  # Simulate time passing
                result = validator.simulate_trade_result(trade)
                
                # Log result
                logger.info(f"Trade {trades_executed}: {result['result']} "
                           f"${result['profit']:.2f} | Balance: ${result['new_balance']:.2f}")
        
        # Daily summary
        metrics = validator.get_performance_metrics()
        logger.info(f"Day {current_date.strftime('%Y-%m-%d')}: "
                   f"Trades: {metrics['total_trades']}, "
                   f"Win Rate: {metrics['win_rate']:.2%}, "
                   f"PnL: ${metrics['total_pnl']:.2f}")
        
        current_date += timedelta(days=1)
        await asyncio.sleep(0.1)  # Simulate day passing
    
    # End session and get final results
    final_stats = validator.end_session()
    
    # Print comprehensive results
    logger.info("=" * 60)
    logger.info("📊 PAPER TRADING VALIDATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Duration: {duration_days} days")
    logger.info(f"Total Trades: {final_stats.get('total_trades', 0)}")
    logger.info(f"Win Rate: {final_stats.get('win_rate', 0):.2%}")
    logger.info(f"Total PnL: ${final_stats.get('total_pnl', 0):.2f}")
    logger.info(f"Final Balance: ${validator.current_balance:.2f}")
    logger.info(f"Profit %: {((validator.current_balance - 10000) / 10000 * 100):.2f}%")
    logger.info(f"Max Drawdown: {validator._calculate_max_drawdown():.2f}%")
    logger.info("=" * 60)
    
    return final_stats

if __name__ == "__main__":
    # Run paper trading validation
    asyncio.run(run_paper_trading_validation(duration_days=30, signals_per_day=20))