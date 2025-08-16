import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import sqlite3
import json
import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TradeAction(Enum):
    ALLOW = "allow"
    REDUCE = "reduce"
    BLOCK = "block"
    CLOSE_ALL = "close_all"

@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    current_drawdown: float
    max_drawdown: float
    daily_pnl: float
    win_rate: float
    sharpe_ratio: float
    volatility: float
    var_95: float  # Value at Risk 95%
    expected_shortfall: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    profit_factor: float
    risk_score: float

@dataclass
class PositionSizing:
    """Position sizing recommendation"""
    recommended_size: float
    max_size: float
    risk_per_trade: float
    stop_loss: float
    take_profit: float
    confidence_adjustment: float
    volatility_adjustment: float

class EnhancedRiskManager:
    """
    🛡️ Enhanced Risk Management System
    
    Features:
    - Dynamic Position Sizing based on Kelly Criterion
    - Real-time Drawdown Protection
    - Volatility-Adjusted Risk Controls
    - Multi-timeframe Risk Assessment
    - Emergency Stop Mechanisms
    - Performance-based Risk Scaling
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.logger = self._setup_logger()
        
        # Risk state tracking
        self.account_balance = self.config['initial_balance']
        self.peak_balance = self.account_balance
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_trades = 0
        
        # Trade tracking
        self.open_positions = {}
        self.trade_history = []
        self.daily_trades_history = []
        
        # Risk metrics
        self.risk_metrics = None
        self.last_risk_update = None
        
        # Emergency controls
        self.trading_halted = False
        self.halt_reason = None
        self.halt_timestamp = None
        
        # Initialize database
        self._initialize_database()
        
        # Load historical data for calculations
        self._load_historical_performance()
        
    def _get_default_config(self) -> Dict:
        """Enhanced risk management configuration"""
        return {
            # Account Settings
            'initial_balance': 10000,  # Starting balance
            'base_currency': 'USD',
            
            # Core Risk Limits
            'max_risk_per_trade': 0.02,  # 2% max risk per trade
            'max_daily_risk': 0.05,  # 5% max daily risk
            'max_drawdown_limit': 0.15,  # 15% maximum drawdown
            'daily_loss_limit': 0.05,  # 5% daily loss limit
            
            # Position Sizing
            'kelly_fraction': 0.25,  # Kelly criterion fraction
            'min_position_size': 0.001,  # 0.1% minimum
            'max_position_size': 0.05,  # 5% maximum
            'volatility_lookback': 20,  # Days for volatility calculation
            
            # Trade Limits
            'max_daily_trades': 10,  # Maximum trades per day
            'max_concurrent_trades': 3,  # Maximum open positions
            'min_time_between_trades': 300,  # 5 minutes minimum
            
            # Risk Scaling
            'win_rate_threshold': 0.60,  # 60% minimum win rate
            'sharpe_threshold': 1.5,  # Minimum Sharpe ratio
            'consecutive_loss_limit': 5,  # Max consecutive losses
            
            # Emergency Controls
            'circuit_breaker_threshold': 0.10,  # 10% rapid loss triggers halt
            'volatility_spike_threshold': 3.0,  # 3x normal volatility
            'margin_call_threshold': 0.20,  # 20% margin requirement
            
            # ATR-based Stops
            'atr_multiplier_stop': 2.0,  # Stop loss = 2x ATR
            'atr_multiplier_profit': 3.0,  # Take profit = 3x ATR
            'atr_lookback_period': 14,  # ATR calculation period
            
            # Risk Assessment
            'risk_update_interval': 300,  # Update every 5 minutes
            'performance_review_trades': 50,  # Review after N trades
            
            # Database
            'database_path': '/workspace/data/enhanced_risk.db'
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('EnhancedRiskManager')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler('/workspace/logs/enhanced_risk_manager.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)  # Only show warnings/errors in console
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _initialize_database(self):
        """Initialize enhanced risk database"""
        try:
            conn = sqlite3.connect(self.config['database_path'])
            cursor = conn.cursor()
            
            # Trade history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    pnl REAL,
                    risk_amount REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    holding_time INTEGER,
                    win_loss TEXT,
                    confidence REAL,
                    volatility REAL,
                    drawdown_at_entry REAL
                )
            ''')
            
            # Risk metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    account_balance REAL,
                    current_drawdown REAL,
                    max_drawdown REAL,
                    daily_pnl REAL,
                    win_rate REAL,
                    sharpe_ratio REAL,
                    volatility REAL,
                    var_95 REAL,
                    expected_shortfall REAL,
                    total_trades INTEGER,
                    risk_score REAL,
                    risk_level TEXT
                )
            ''')
            
            # Risk events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT NOT NULL,
                    action_taken TEXT,
                    metrics_snapshot TEXT
                )
            ''')
            
            # Daily performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_performance (
                    date DATE PRIMARY KEY,
                    starting_balance REAL,
                    ending_balance REAL,
                    daily_pnl REAL,
                    daily_return REAL,
                    trades_count INTEGER,
                    win_rate REAL,
                    max_drawdown REAL,
                    volatility REAL,
                    sharpe_ratio REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    def _load_historical_performance(self):
        """Load historical performance for risk calculations"""
        try:
            conn = sqlite3.connect(self.config['database_path'])
            
            # Load recent trade history
            df = pd.read_sql_query('''
                SELECT * FROM trade_history 
                ORDER BY timestamp DESC 
                LIMIT 1000
            ''', conn)
            
            if len(df) > 0:
                self.trade_history = df.to_dict('records')
                self.total_trades = len(self.trade_history)
                self.winning_trades = len(df[df['win_loss'] == 'WIN'])
                
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"Could not load historical performance: {e}")
    
    def assess_trade_risk(self, 
                         symbol: str,
                         signal_data: Dict,
                         market_data: Dict) -> Tuple[TradeAction, PositionSizing]:
        """
        Comprehensive trade risk assessment
        """
        try:
            self.logger.info(f"🔍 Assessing risk for {symbol}")
            
            # Update current risk metrics
            self._update_risk_metrics()
            
            # Check emergency conditions first
            emergency_action = self._check_emergency_conditions()
            if emergency_action != TradeAction.ALLOW:
                return emergency_action, None
            
            # Check basic trading limits
            basic_check = self._check_basic_limits()
            if basic_check != TradeAction.ALLOW:
                return basic_check, None
            
            # Calculate position sizing
            position_sizing = self._calculate_position_sizing(
                symbol, signal_data, market_data
            )
            
            # Risk-adjusted final decision
            final_action = self._make_final_risk_decision(
                symbol, signal_data, position_sizing
            )
            
            # Log the decision
            self._log_risk_decision(symbol, signal_data, final_action, position_sizing)
            
            return final_action, position_sizing
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed for {symbol}: {e}")
            return TradeAction.BLOCK, None
    
    def _check_emergency_conditions(self) -> TradeAction:
        """Check for emergency halt conditions"""
        
        # Check if trading is already halted
        if self.trading_halted:
            time_since_halt = (datetime.now() - self.halt_timestamp).seconds
            if time_since_halt < 3600:  # 1 hour cooldown
                return TradeAction.BLOCK
            else:
                self._resume_trading()
        
        # Check drawdown limit
        if self.current_drawdown >= self.config['max_drawdown_limit']:
            self._halt_trading("Maximum drawdown exceeded", RiskLevel.CRITICAL)
            return TradeAction.CLOSE_ALL
        
        # Check daily loss limit
        daily_loss_pct = abs(self.daily_pnl) / self.account_balance
        if self.daily_pnl < 0 and daily_loss_pct >= self.config['daily_loss_limit']:
            self._halt_trading("Daily loss limit exceeded", RiskLevel.CRITICAL)
            return TradeAction.CLOSE_ALL
        
        # Check circuit breaker (rapid loss)
        if self._check_circuit_breaker():
            self._halt_trading("Circuit breaker triggered", RiskLevel.CRITICAL)
            return TradeAction.CLOSE_ALL
        
        return TradeAction.ALLOW
    
    def _check_basic_limits(self) -> TradeAction:
        """Check basic trading limits"""
        
        # Daily trade limit
        if self.daily_trades >= self.config['max_daily_trades']:
            self.logger.warning("Daily trade limit exceeded")
            return TradeAction.BLOCK
        
        # Concurrent positions limit
        if len(self.open_positions) >= self.config['max_concurrent_trades']:
            self.logger.warning("Maximum concurrent positions reached")
            return TradeAction.BLOCK
        
        # Time between trades
        if self._check_time_between_trades():
            self.logger.warning("Minimum time between trades not met")
            return TradeAction.BLOCK
        
        return TradeAction.ALLOW
    
    def _calculate_position_sizing(self, 
                                 symbol: str,
                                 signal_data: Dict,
                                 market_data: Dict) -> PositionSizing:
        """
        Calculate optimal position sizing using multiple methods
        """
        
        # Get signal parameters
        confidence = signal_data.get('confidence', 0.5)
        accuracy = signal_data.get('accuracy', 0.5)
        direction = signal_data.get('direction', 'BUY')
        
        # Get market data
        current_price = market_data.get('close', 0)
        volatility = self._calculate_volatility(market_data)
        atr = self._calculate_atr(market_data)
        
        # Base risk amount (percentage of account)
        base_risk = self.config['max_risk_per_trade']
        
        # Adjust for performance
        performance_multiplier = self._get_performance_multiplier()
        
        # Adjust for confidence
        confidence_multiplier = confidence  # Scale by signal confidence
        
        # Adjust for volatility
        volatility_multiplier = min(1.0, 1.0 / (1.0 + volatility))
        
        # Calculate Kelly fraction if we have historical data
        kelly_fraction = self._calculate_kelly_fraction()
        
        # Final risk per trade
        adjusted_risk = (
            base_risk * 
            performance_multiplier * 
            confidence_multiplier * 
            volatility_multiplier
        )
        
        # Apply Kelly constraint
        if kelly_fraction > 0:
            adjusted_risk = min(adjusted_risk, kelly_fraction * self.config['kelly_fraction'])
        
        # Ensure within bounds
        adjusted_risk = max(self.config['min_position_size'], 
                          min(adjusted_risk, self.config['max_position_size']))
        
        # Calculate position size in currency
        risk_amount = self.account_balance * adjusted_risk
        
        # Calculate stop loss and take profit using ATR
        if direction.upper() == 'BUY':
            stop_loss = current_price - (atr * self.config['atr_multiplier_stop'])
            take_profit = current_price + (atr * self.config['atr_multiplier_profit'])
        else:
            stop_loss = current_price + (atr * self.config['atr_multiplier_stop'])
            take_profit = current_price - (atr * self.config['atr_multiplier_profit'])
        
        # Calculate position size based on stop loss
        if direction.upper() == 'BUY':
            stop_distance = current_price - stop_loss
        else:
            stop_distance = stop_loss - current_price
        
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
        else:
            position_size = risk_amount / (current_price * 0.02)  # 2% fallback
        
        # Apply maximum position size constraint
        max_position_value = self.account_balance * self.config['max_position_size']
        max_position_size = max_position_value / current_price
        position_size = min(position_size, max_position_size)
        
        return PositionSizing(
            recommended_size=position_size,
            max_size=max_position_size,
            risk_per_trade=adjusted_risk,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence_adjustment=confidence_multiplier,
            volatility_adjustment=volatility_multiplier
        )
    
    def _calculate_volatility(self, market_data: Dict) -> float:
        """Calculate market volatility"""
        try:
            if 'price_history' in market_data:
                prices = market_data['price_history']
                returns = np.diff(np.log(prices))
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                return volatility
            else:
                return 0.02  # Default 2% volatility
        except:
            return 0.02
    
    def _calculate_atr(self, market_data: Dict) -> float:
        """Calculate Average True Range"""
        try:
            if all(k in market_data for k in ['high', 'low', 'close', 'prev_close']):
                high = market_data['high']
                low = market_data['low']
                close = market_data['close']
                prev_close = market_data.get('prev_close', close)
                
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                atr = max(tr1, tr2, tr3)
                
                return atr
            else:
                # Fallback: use current price * 1%
                return market_data.get('close', 1) * 0.01
        except:
            return market_data.get('close', 1) * 0.01
    
    def _calculate_kelly_fraction(self) -> float:
        """Calculate Kelly Criterion fraction"""
        if len(self.trade_history) < 20:
            return 0.0
        
        try:
            wins = [t for t in self.trade_history if t['win_loss'] == 'WIN']
            losses = [t for t in self.trade_history if t['win_loss'] == 'LOSS']
            
            if not wins or not losses:
                return 0.0
            
            win_rate = len(wins) / len(self.trade_history)
            avg_win = np.mean([w['pnl'] for w in wins])
            avg_loss = abs(np.mean([l['pnl'] for l in losses]))
            
            if avg_loss == 0:
                return 0.0
            
            win_loss_ratio = avg_win / avg_loss
            kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio
            
            return max(0, kelly_fraction)
            
        except Exception as e:
            self.logger.error(f"Kelly calculation failed: {e}")
            return 0.0
    
    def _get_performance_multiplier(self) -> float:
        """Get performance-based risk multiplier"""
        if len(self.trade_history) < 10:
            return 0.5  # Conservative for new systems
        
        # Calculate recent win rate
        recent_trades = self.trade_history[-50:]  # Last 50 trades
        recent_wins = [t for t in recent_trades if t['win_loss'] == 'WIN']
        win_rate = len(recent_wins) / len(recent_trades)
        
        # Performance multipliers
        if win_rate >= 0.70:
            return 1.2  # Increase risk for good performance
        elif win_rate >= 0.60:
            return 1.0  # Normal risk
        elif win_rate >= 0.50:
            return 0.8  # Reduce risk
        else:
            return 0.5  # Significantly reduce risk
    
    def _make_final_risk_decision(self, 
                                symbol: str,
                                signal_data: Dict,
                                position_sizing: PositionSizing) -> TradeAction:
        """Make final risk-adjusted decision"""
        
        # Check minimum position size
        if position_sizing.recommended_size < self.config['min_position_size']:
            return TradeAction.BLOCK
        
        # Check if risk is too high for signal quality
        signal_strength = signal_data.get('strength', 5)
        confidence = signal_data.get('confidence', 0.5)
        
        # Require higher confidence for larger positions
        min_confidence = 0.7 if position_sizing.risk_per_trade > 0.015 else 0.6
        
        if confidence < min_confidence:
            return TradeAction.REDUCE
        
        # Check consecutive losses
        if self._check_consecutive_losses():
            return TradeAction.REDUCE
        
        # All checks passed
        return TradeAction.ALLOW
    
    def _check_consecutive_losses(self) -> bool:
        """Check for consecutive loss streak"""
        if len(self.trade_history) < self.config['consecutive_loss_limit']:
            return False
        
        recent_trades = self.trade_history[-self.config['consecutive_loss_limit']:]
        consecutive_losses = all(t['win_loss'] == 'LOSS' for t in recent_trades)
        
        return consecutive_losses
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker should trigger"""
        # Check for rapid loss in short time period
        one_hour_ago = datetime.now() - timedelta(hours=1)
        
        recent_trades = [
            t for t in self.trade_history 
            if datetime.fromisoformat(t['timestamp']) > one_hour_ago
        ]
        
        if len(recent_trades) >= 5:  # At least 5 trades
            total_pnl = sum(t['pnl'] for t in recent_trades)
            loss_pct = abs(total_pnl) / self.account_balance
            
            if total_pnl < 0 and loss_pct >= self.config['circuit_breaker_threshold']:
                return True
        
        return False
    
    def _check_time_between_trades(self) -> bool:
        """Check minimum time between trades"""
        if not self.trade_history:
            return False
        
        last_trade_time = datetime.fromisoformat(self.trade_history[-1]['timestamp'])
        time_diff = (datetime.now() - last_trade_time).seconds
        
        return time_diff < self.config['min_time_between_trades']
    
    def _update_risk_metrics(self):
        """Update current risk metrics"""
        try:
            # Calculate current drawdown
            self.current_drawdown = (self.peak_balance - self.account_balance) / self.peak_balance
            
            # Update peak balance
            if self.account_balance > self.peak_balance:
                self.peak_balance = self.account_balance
            
            # Update max drawdown
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
            
            # Calculate comprehensive risk metrics
            self.risk_metrics = self._calculate_comprehensive_metrics()
            self.last_risk_update = datetime.now()
            
            # Store in database
            self._store_risk_metrics()
            
        except Exception as e:
            self.logger.error(f"Risk metrics update failed: {e}")
    
    def _calculate_comprehensive_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        if len(self.trade_history) < 2:
            return RiskMetrics(
                current_drawdown=self.current_drawdown,
                max_drawdown=self.max_drawdown,
                daily_pnl=self.daily_pnl,
                win_rate=0.0,
                sharpe_ratio=0.0,
                volatility=0.0,
                var_95=0.0,
                expected_shortfall=0.0,
                total_trades=self.total_trades,
                winning_trades=self.winning_trades,
                losing_trades=self.total_trades - self.winning_trades,
                average_win=0.0,
                average_loss=0.0,
                profit_factor=0.0,
                risk_score=5.0  # Neutral risk score
            )
        
        # Calculate metrics from trade history
        pnls = [t['pnl'] for t in self.trade_history]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        win_rate = len(wins) / len(pnls) if pnls else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        
        # Sharpe ratio
        returns = np.array(pnls) / self.account_balance
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # VaR and Expected Shortfall
        var_95 = np.percentile(pnls, 5) if len(pnls) >= 20 else 0
        es_trades = [p for p in pnls if p <= var_95]
        expected_shortfall = np.mean(es_trades) if es_trades else 0
        
        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 1
        profit_factor = total_wins / total_losses
        
        # Risk score (1-10, where 1 is lowest risk, 10 is highest)
        risk_score = self._calculate_risk_score(win_rate, sharpe_ratio, self.current_drawdown)
        
        return RiskMetrics(
            current_drawdown=self.current_drawdown,
            max_drawdown=self.max_drawdown,
            daily_pnl=self.daily_pnl,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            volatility=np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0,
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            total_trades=len(pnls),
            winning_trades=len(wins),
            losing_trades=len(losses),
            average_win=avg_win,
            average_loss=avg_loss,
            profit_factor=profit_factor,
            risk_score=risk_score
        )
    
    def _calculate_risk_score(self, win_rate: float, sharpe_ratio: float, drawdown: float) -> float:
        """Calculate overall risk score (1-10)"""
        
        # Base score is 5 (neutral)
        score = 5.0
        
        # Adjust for win rate
        if win_rate >= 0.70:
            score -= 2.0
        elif win_rate >= 0.60:
            score -= 1.0
        elif win_rate <= 0.40:
            score += 2.0
        elif win_rate <= 0.50:
            score += 1.0
        
        # Adjust for Sharpe ratio
        if sharpe_ratio >= 2.0:
            score -= 1.5
        elif sharpe_ratio >= 1.5:
            score -= 1.0
        elif sharpe_ratio <= 0.5:
            score += 2.0
        elif sharpe_ratio <= 1.0:
            score += 1.0
        
        # Adjust for drawdown
        if drawdown >= 0.15:
            score += 3.0
        elif drawdown >= 0.10:
            score += 2.0
        elif drawdown >= 0.05:
            score += 1.0
        elif drawdown <= 0.02:
            score -= 1.0
        
        # Ensure score is within bounds
        return max(1.0, min(10.0, score))
    
    def _halt_trading(self, reason: str, severity: RiskLevel):
        """Halt trading with specified reason"""
        self.trading_halted = True
        self.halt_reason = reason
        self.halt_timestamp = datetime.now()
        
        self.logger.critical(f"🚨 TRADING HALTED: {reason}")
        
        # Log risk event
        self._log_risk_event("TRADING_HALT", severity, reason, "HALT_TRADING")
    
    def _resume_trading(self):
        """Resume trading after halt"""
        self.trading_halted = False
        self.halt_reason = None
        self.halt_timestamp = None
        
        self.logger.info("✅ Trading resumed")
        self._log_risk_event("TRADING_RESUME", RiskLevel.MEDIUM, "Trading resumed", "RESUME_TRADING")
    
    def record_trade(self, trade_result: Dict):
        """Record completed trade for risk tracking"""
        try:
            # Update account balance
            pnl = trade_result.get('pnl', 0)
            self.account_balance += pnl
            self.daily_pnl += pnl
            self.total_trades += 1
            self.daily_trades += 1
            
            if pnl > 0:
                self.winning_trades += 1
            
            # Store in trade history
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': trade_result.get('symbol', ''),
                'action': trade_result.get('action', ''),
                'size': trade_result.get('size', 0),
                'entry_price': trade_result.get('entry_price', 0),
                'exit_price': trade_result.get('exit_price', 0),
                'pnl': pnl,
                'risk_amount': trade_result.get('risk_amount', 0),
                'stop_loss': trade_result.get('stop_loss', 0),
                'take_profit': trade_result.get('take_profit', 0),
                'holding_time': trade_result.get('holding_time', 0),
                'win_loss': 'WIN' if pnl > 0 else 'LOSS',
                'confidence': trade_result.get('confidence', 0),
                'volatility': trade_result.get('volatility', 0),
                'drawdown_at_entry': self.current_drawdown
            }
            
            self.trade_history.append(trade_record)
            
            # Keep only recent trades in memory
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
            
            # Store in database
            self._store_trade_record(trade_record)
            
            self.logger.info(f"📊 Trade recorded: {trade_result.get('symbol')} PnL: {pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"Failed to record trade: {e}")
    
    def get_risk_status(self) -> Dict:
        """Get current risk status summary"""
        self._update_risk_metrics()
        
        if not self.risk_metrics:
            return {"status": "UNKNOWN", "message": "Risk metrics not available"}
        
        # Determine risk level
        if self.risk_metrics.risk_score <= 3:
            risk_level = RiskLevel.LOW
            status = "HEALTHY"
        elif self.risk_metrics.risk_score <= 5:
            risk_level = RiskLevel.MEDIUM
            status = "MODERATE"
        elif self.risk_metrics.risk_score <= 7:
            risk_level = RiskLevel.HIGH
            status = "ELEVATED"
        else:
            risk_level = RiskLevel.CRITICAL
            status = "CRITICAL"
        
        return {
            "status": status,
            "risk_level": risk_level.value,
            "risk_score": self.risk_metrics.risk_score,
            "account_balance": self.account_balance,
            "current_drawdown": self.current_drawdown * 100,  # Percentage
            "max_drawdown": self.max_drawdown * 100,
            "daily_pnl": self.daily_pnl,
            "win_rate": self.risk_metrics.win_rate * 100,
            "sharpe_ratio": self.risk_metrics.sharpe_ratio,
            "total_trades": self.risk_metrics.total_trades,
            "daily_trades": self.daily_trades,
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason
        }
    
    def _store_trade_record(self, trade_record: Dict):
        """Store trade record in database"""
        try:
            conn = sqlite3.connect(self.config['database_path'])
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trade_history (
                    timestamp, symbol, action, size, entry_price, exit_price,
                    pnl, risk_amount, stop_loss, take_profit, holding_time,
                    win_loss, confidence, volatility, drawdown_at_entry
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_record['timestamp'], trade_record['symbol'], trade_record['action'],
                trade_record['size'], trade_record['entry_price'], trade_record['exit_price'],
                trade_record['pnl'], trade_record['risk_amount'], trade_record['stop_loss'],
                trade_record['take_profit'], trade_record['holding_time'], trade_record['win_loss'],
                trade_record['confidence'], trade_record['volatility'], trade_record['drawdown_at_entry']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store trade record: {e}")
    
    def _store_risk_metrics(self):
        """Store current risk metrics in database"""
        try:
            if not self.risk_metrics:
                return
            
            conn = sqlite3.connect(self.config['database_path'])
            cursor = conn.cursor()
            
            # Determine risk level
            if self.risk_metrics.risk_score <= 3:
                risk_level = "LOW"
            elif self.risk_metrics.risk_score <= 5:
                risk_level = "MEDIUM"
            elif self.risk_metrics.risk_score <= 7:
                risk_level = "HIGH"
            else:
                risk_level = "CRITICAL"
            
            cursor.execute('''
                INSERT INTO risk_metrics (
                    account_balance, current_drawdown, max_drawdown, daily_pnl,
                    win_rate, sharpe_ratio, volatility, var_95, expected_shortfall,
                    total_trades, risk_score, risk_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.account_balance, self.current_drawdown, self.max_drawdown,
                self.daily_pnl, self.risk_metrics.win_rate, self.risk_metrics.sharpe_ratio,
                self.risk_metrics.volatility, self.risk_metrics.var_95,
                self.risk_metrics.expected_shortfall, self.risk_metrics.total_trades,
                self.risk_metrics.risk_score, risk_level
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store risk metrics: {e}")
    
    def _log_risk_event(self, event_type: str, severity: RiskLevel, 
                       description: str, action_taken: str):
        """Log risk event to database"""
        try:
            conn = sqlite3.connect(self.config['database_path'])
            cursor = conn.cursor()
            
            metrics_snapshot = json.dumps({
                'account_balance': self.account_balance,
                'current_drawdown': self.current_drawdown,
                'daily_pnl': self.daily_pnl,
                'total_trades': self.total_trades
            })
            
            cursor.execute('''
                INSERT INTO risk_events (
                    event_type, severity, description, action_taken, metrics_snapshot
                ) VALUES (?, ?, ?, ?, ?)
            ''', (event_type, severity.value, description, action_taken, metrics_snapshot))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to log risk event: {e}")
    
    def _log_risk_decision(self, symbol: str, signal_data: Dict, 
                          action: TradeAction, position_sizing: PositionSizing):
        """Log risk management decision"""
        
        decision_log = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'signal_confidence': signal_data.get('confidence', 0),
            'signal_strength': signal_data.get('strength', 0),
            'action': action.value,
            'recommended_size': position_sizing.recommended_size if position_sizing else 0,
            'risk_per_trade': position_sizing.risk_per_trade if position_sizing else 0,
            'account_balance': self.account_balance,
            'current_drawdown': self.current_drawdown,
            'risk_score': self.risk_metrics.risk_score if self.risk_metrics else 5.0
        }
        
        self.logger.info(f"Risk Decision: {symbol} -> {action.value} "
                        f"(Risk: {decision_log['risk_per_trade']:.2%})")
    
    def reset_daily_counters(self):
        """Reset daily counters (call at start of new trading day)"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.logger.info("📅 Daily counters reset")


def main():
    """Test the enhanced risk manager"""
    
    # Initialize risk manager
    rm = EnhancedRiskManager()
    
    # Test signal
    test_signal = {
        'symbol': 'EUR/USD',
        'direction': 'BUY',
        'confidence': 0.85,
        'strength': 8,
        'accuracy': 0.75
    }
    
    # Test market data
    test_market = {
        'close': 1.1050,
        'high': 1.1060,
        'low': 1.1040,
        'prev_close': 1.1045,
        'volume': 1000000
    }
    
    # Assess trade
    action, sizing = rm.assess_trade_risk('EUR/USD', test_signal, test_market)
    
    print("\n" + "="*60)
    print("🛡️ ENHANCED RISK MANAGER TEST")
    print("="*60)
    print(f"Trade Action: {action.value}")
    
    if sizing:
        print(f"Recommended Size: {sizing.recommended_size:.4f}")
        print(f"Risk per Trade: {sizing.risk_per_trade:.2%}")
        print(f"Stop Loss: {sizing.stop_loss:.4f}")
        print(f"Take Profit: {sizing.take_profit:.4f}")
    
    # Get risk status
    status = rm.get_risk_status()
    print(f"\nRisk Status: {status['status']}")
    print(f"Risk Score: {status['risk_score']:.1f}/10")
    print(f"Account Balance: ${status['account_balance']:.2f}")
    print(f"Current Drawdown: {status['current_drawdown']:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()