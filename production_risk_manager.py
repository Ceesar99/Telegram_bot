#!/usr/bin/env python3
"""
🛡️ PRODUCTION RISK MANAGER - LIVE TRADING READY
Advanced risk management system with real-time monitoring and circuit breakers
"""

import asyncio
import logging
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import threading
import warnings
warnings.filterwarnings('ignore')

from production_config import RISK_CONFIG, TRADING_CONFIG, MONITORING_CONFIG

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    POSITION_LIMIT = "position_limit"
    LOSS_LIMIT = "loss_limit"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    DRAWDOWN = "drawdown"
    MODEL_PERFORMANCE = "model_performance"
    SYSTEM_ERROR = "system_error"

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    current_exposure: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    expected_shortfall: float = 0.0
    correlation_risk: float = 0.0
    volatility_risk: float = 0.0
    model_accuracy: float = 0.0
    active_trades: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class RiskAlert:
    """Risk alert data structure"""
    alert_type: AlertType
    risk_level: RiskLevel
    message: str
    value: float
    threshold: float
    symbol: Optional[str] = None
    trade_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    action_required: str = "monitor"

class ProductionRiskManager:
    """
    Production-ready risk management system with real-time monitoring
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.risk_metrics = RiskMetrics()
        self.active_trades = {}
        self.position_history = []
        self.pnl_history = []
        self.alerts = []
        
        # Circuit breakers
        self.circuit_breakers = {
            'trading_enabled': True,
            'new_positions_enabled': True,
            'high_risk_pairs_blocked': [],
            'emergency_stop': False
        }
        
        # Risk tracking
        self.account_balance = 10000.0  # Initialize with default
        self.peak_balance = 10000.0
        self.daily_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Performance tracking windows
        self.returns_window = []
        self.max_window_size = 1000
        
        # Risk monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Initialize database
        self._initialize_database()
        
        self.logger.info("Production Risk Manager initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up comprehensive logging"""
        logger = logging.getLogger('ProductionRiskManager')
        logger.setLevel(logging.INFO)
        
        # File handler
        handler = logging.FileHandler('/workspace/logs/production_risk.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _initialize_database(self):
        """Initialize risk management database tables"""
        try:
            conn = sqlite3.connect('/workspace/data/production_risk.db')
            cursor = conn.cursor()
            
            # Risk metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    current_exposure REAL,
                    daily_pnl REAL,
                    weekly_pnl REAL,
                    monthly_pnl REAL,
                    max_drawdown REAL,
                    current_drawdown REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    sharpe_ratio REAL,
                    var_95 REAL,
                    expected_shortfall REAL,
                    correlation_risk REAL,
                    volatility_risk REAL,
                    model_accuracy REAL,
                    active_trades INTEGER,
                    account_balance REAL
                )
            ''')
            
            # Risk alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    value REAL,
                    threshold REAL,
                    symbol TEXT,
                    trade_id TEXT,
                    action_required TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Trade risk assessments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    trade_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    position_size REAL,
                    risk_score REAL,
                    correlation_risk REAL,
                    volatility_risk REAL,
                    model_confidence REAL,
                    approved BOOLEAN,
                    rejection_reason TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing risk database: {e}")
    
    def update_account_balance(self, balance: float):
        """Update account balance and recalculate metrics"""
        self.account_balance = balance
        
        # Update peak balance
        if balance > self.peak_balance:
            self.peak_balance = balance
        
        # Calculate current drawdown
        current_drawdown = (self.peak_balance - balance) / self.peak_balance
        self.risk_metrics.current_drawdown = current_drawdown
        
        # Update max drawdown
        if current_drawdown > self.risk_metrics.max_drawdown:
            self.risk_metrics.max_drawdown = current_drawdown
        
        # Add to returns history
        if len(self.pnl_history) > 0:
            daily_return = (balance - self.pnl_history[-1]) / self.pnl_history[-1]
            self.returns_window.append(daily_return)
            
            # Maintain window size
            if len(self.returns_window) > self.max_window_size:
                self.returns_window.pop(0)
        
        self.pnl_history.append(balance)
        
        # Check risk thresholds
        self._check_balance_thresholds()
    
    def assess_trade_risk(self, symbol: str, signal: str, confidence: float, 
                         market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive trade risk assessment"""
        
        try:
            # Initialize assessment
            assessment = {
                'approved': False,
                'position_size': 0.0,
                'risk_score': 0.0,
                'rejection_reason': '',
                'risk_metrics': {}
            }
            
            # 1. Check circuit breakers
            if not self.circuit_breakers['trading_enabled']:
                assessment['rejection_reason'] = 'Trading disabled by circuit breaker'
                return assessment
            
            if not self.circuit_breakers['new_positions_enabled']:
                assessment['rejection_reason'] = 'New positions disabled by circuit breaker'
                return assessment
            
            if symbol in self.circuit_breakers['high_risk_pairs_blocked']:
                assessment['rejection_reason'] = f'Symbol {symbol} blocked due to high risk'
                return assessment
            
            if self.circuit_breakers['emergency_stop']:
                assessment['rejection_reason'] = 'Emergency stop activated'
                return assessment
            
            # 2. Check position limits
            if len(self.active_trades) >= RISK_CONFIG['max_concurrent_trades']:
                assessment['rejection_reason'] = 'Maximum concurrent trades reached'
                return assessment
            
            # 3. Check daily loss limits
            daily_loss_pct = abs(self.risk_metrics.daily_pnl) / self.account_balance * 100
            if daily_loss_pct >= RISK_CONFIG['max_daily_loss']:
                assessment['rejection_reason'] = f'Daily loss limit reached: {daily_loss_pct:.2f}%'
                self._trigger_alert(AlertType.LOSS_LIMIT, RiskLevel.CRITICAL, 
                                  f"Daily loss limit reached: {daily_loss_pct:.2f}%", 
                                  daily_loss_pct, RISK_CONFIG['max_daily_loss'])
                return assessment
            
            # 4. Check weekly/monthly limits
            weekly_loss_pct = abs(self.risk_metrics.weekly_pnl) / self.account_balance * 100
            if weekly_loss_pct >= RISK_CONFIG['max_weekly_loss']:
                assessment['rejection_reason'] = f'Weekly loss limit reached: {weekly_loss_pct:.2f}%'
                return assessment
            
            monthly_loss_pct = abs(self.risk_metrics.monthly_pnl) / self.account_balance * 100
            if monthly_loss_pct >= RISK_CONFIG['max_monthly_loss']:
                assessment['rejection_reason'] = f'Monthly loss limit reached: {monthly_loss_pct:.2f}%'
                return assessment
            
            # 5. Check minimum confidence threshold
            if confidence < 85.0:  # Production minimum
                assessment['rejection_reason'] = f'Signal confidence too low: {confidence:.1f}%'
                return assessment
            
            # 6. Calculate position size using Kelly Criterion
            position_size = self._calculate_optimal_position_size(symbol, confidence, market_data)
            
            if position_size <= 0:
                assessment['rejection_reason'] = 'Position size calculation failed'
                return assessment
            
            # 7. Check correlation risk
            correlation_risk = self._calculate_correlation_risk(symbol)
            if correlation_risk > RISK_CONFIG['max_correlation_exposure']:
                assessment['rejection_reason'] = f'Correlation risk too high: {correlation_risk:.2f}'
                return assessment
            
            # 8. Check volatility risk
            volatility_risk = self._calculate_volatility_risk(market_data)
            if volatility_risk > 0.8:  # High volatility threshold
                assessment['rejection_reason'] = f'Market volatility too high: {volatility_risk:.2f}'
                return assessment
            
            # 9. Check model performance
            if self.risk_metrics.model_accuracy < 75.0:  # Minimum accuracy
                assessment['rejection_reason'] = f'Model accuracy too low: {self.risk_metrics.model_accuracy:.1f}%'
                return assessment
            
            # 10. Calculate overall risk score
            risk_score = self._calculate_risk_score(confidence, correlation_risk, volatility_risk)
            
            # 11. Final approval
            if risk_score <= 0.7:  # Risk score threshold
                assessment.update({
                    'approved': True,
                    'position_size': position_size,
                    'risk_score': risk_score,
                    'risk_metrics': {
                        'correlation_risk': correlation_risk,
                        'volatility_risk': volatility_risk,
                        'confidence': confidence,
                        'position_size_pct': position_size / self.account_balance * 100
                    }
                })
            else:
                assessment['rejection_reason'] = f'Risk score too high: {risk_score:.3f}'
            
            # Log assessment
            self._log_trade_assessment(symbol, signal, assessment)
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error in trade risk assessment: {e}")
            return {
                'approved': False,
                'rejection_reason': f'Risk assessment error: {str(e)}',
                'position_size': 0.0,
                'risk_score': 1.0
            }
    
    def _calculate_optimal_position_size(self, symbol: str, confidence: float, 
                                       market_data: Dict[str, Any]) -> float:
        """Calculate optimal position size using Kelly Criterion and risk constraints"""
        
        try:
            # Base position size (percentage of account)
            base_risk_pct = RISK_CONFIG['max_risk_per_trade']
            
            # Adjust based on confidence
            confidence_factor = confidence / 100.0
            
            # Adjust based on current performance
            performance_factor = 1.0
            if self.risk_metrics.win_rate > 0:
                performance_factor = min(1.5, self.risk_metrics.win_rate / 50.0)
            
            # Adjust based on current drawdown
            drawdown_factor = 1.0 - (self.risk_metrics.current_drawdown * 2)
            drawdown_factor = max(0.3, drawdown_factor)  # Minimum 30% of normal size
            
            # Kelly Criterion approximation
            if self.risk_metrics.win_rate > 0 and self.risk_metrics.profit_factor > 0:
                win_prob = self.risk_metrics.win_rate / 100.0
                avg_win = self.risk_metrics.profit_factor
                avg_loss = 1.0
                
                kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
                kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
            else:
                kelly_fraction = base_risk_pct / 100.0
            
            # Calculate final position size
            adjusted_risk_pct = (base_risk_pct * confidence_factor * 
                               performance_factor * drawdown_factor * kelly_fraction)
            
            # Apply absolute limits
            adjusted_risk_pct = max(0.5, min(RISK_CONFIG['max_risk_per_trade'], adjusted_risk_pct))
            
            # Convert to dollar amount
            position_size = self.account_balance * (adjusted_risk_pct / 100.0)
            
            # Minimum position size
            position_size = max(10.0, position_size)  # Minimum $10
            
            self.logger.debug(f"Position size calculation for {symbol}: ${position_size:.2f} "
                            f"({adjusted_risk_pct:.2f}% of account)")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def _calculate_correlation_risk(self, symbol: str) -> float:
        """Calculate correlation risk with existing positions"""
        
        if not self.active_trades:
            return 0.0
        
        try:
            # Simple correlation based on currency pairs
            base_currency = symbol.split('/')[0] if '/' in symbol else symbol[:3]
            quote_currency = symbol.split('/')[1] if '/' in symbol else symbol[3:]
            
            correlated_exposure = 0.0
            total_exposure = 0.0
            
            for trade in self.active_trades.values():
                trade_symbol = trade.get('symbol', '')
                trade_size = trade.get('position_size', 0)
                
                total_exposure += trade_size
                
                # Check for currency correlation
                if (base_currency in trade_symbol or quote_currency in trade_symbol):
                    correlated_exposure += trade_size
            
            correlation_risk = correlated_exposure / max(1, total_exposure)
            
            return min(1.0, correlation_risk)
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation risk: {e}")
            return 0.5  # Conservative default
    
    def _calculate_volatility_risk(self, market_data: Dict[str, Any]) -> float:
        """Calculate market volatility risk"""
        
        try:
            # Get volatility indicators from market data
            rsi = market_data.get('rsi', {}).get('value', 50)
            atr_ratio = market_data.get('atr_ratio', 0.01)
            
            # Normalize RSI extreme values (higher = more volatile)
            rsi_volatility = 0.0
            if rsi > 70 or rsi < 30:
                rsi_volatility = min(1.0, abs(rsi - 50) / 20.0)
            
            # ATR-based volatility (higher ATR = more volatile)
            atr_volatility = min(1.0, atr_ratio * 100)
            
            # Combined volatility risk
            volatility_risk = (rsi_volatility + atr_volatility) / 2.0
            
            return volatility_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility risk: {e}")
            return 0.3  # Conservative default
    
    def _calculate_risk_score(self, confidence: float, correlation_risk: float, 
                            volatility_risk: float) -> float:
        """Calculate overall risk score (0-1, lower is better)"""
        
        # Normalize confidence (lower confidence = higher risk)
        confidence_risk = (100 - confidence) / 100.0
        
        # Weight the different risk components
        weights = {
            'confidence': 0.4,
            'correlation': 0.3,
            'volatility': 0.3
        }
        
        risk_score = (confidence_risk * weights['confidence'] +
                     correlation_risk * weights['correlation'] +
                     volatility_risk * weights['volatility'])
        
        return min(1.0, risk_score)
    
    def _check_balance_thresholds(self):
        """Check balance-based risk thresholds"""
        
        # Daily loss check
        daily_loss_pct = abs(self.risk_metrics.daily_pnl) / self.account_balance * 100
        
        if daily_loss_pct >= RISK_CONFIG['max_daily_loss'] * 0.8:  # 80% of limit
            self._trigger_alert(AlertType.LOSS_LIMIT, RiskLevel.HIGH,
                              f"Approaching daily loss limit: {daily_loss_pct:.2f}%",
                              daily_loss_pct, RISK_CONFIG['max_daily_loss'])
        
        # Drawdown check
        if self.risk_metrics.current_drawdown >= 0.15:  # 15% drawdown
            self._trigger_alert(AlertType.DRAWDOWN, RiskLevel.CRITICAL,
                              f"High drawdown detected: {self.risk_metrics.current_drawdown*100:.1f}%",
                              self.risk_metrics.current_drawdown * 100, 15.0)
            
            # Activate defensive measures
            self.circuit_breakers['new_positions_enabled'] = False
    
    def register_trade(self, trade_id: str, trade_data: Dict[str, Any]):
        """Register a new trade for risk monitoring"""
        
        try:
            self.active_trades[trade_id] = {
                'trade_id': trade_id,
                'symbol': trade_data.get('symbol', ''),
                'signal': trade_data.get('signal', ''),
                'position_size': trade_data.get('position_size', 0),
                'entry_price': trade_data.get('entry_price', 0),
                'entry_time': trade_data.get('entry_time', datetime.now(timezone.utc)),
                'confidence': trade_data.get('confidence', 0),
                'risk_score': trade_data.get('risk_score', 0),
                'status': 'active'
            }
            
            # Update exposure
            self.risk_metrics.current_exposure += trade_data.get('position_size', 0)
            self.risk_metrics.active_trades = len(self.active_trades)
            
            self.daily_trades += 1
            
            self.logger.info(f"Trade registered: {trade_id} - {trade_data.get('symbol')} "
                           f"{trade_data.get('signal')} ${trade_data.get('position_size', 0):.2f}")
            
        except Exception as e:
            self.logger.error(f"Error registering trade: {e}")
    
    def close_trade(self, trade_id: str, result: Dict[str, Any]):
        """Close a trade and update risk metrics"""
        
        try:
            if trade_id not in self.active_trades:
                self.logger.warning(f"Trade {trade_id} not found in active trades")
                return
            
            trade = self.active_trades[trade_id]
            
            # Calculate P&L
            entry_price = trade['entry_price']
            exit_price = result.get('exit_price', entry_price)
            position_size = trade['position_size']
            
            # Determine P&L based on signal direction
            if trade['signal'] == 'BUY':
                pnl = position_size * (exit_price - entry_price) / entry_price
            elif trade['signal'] == 'SELL':
                pnl = position_size * (entry_price - exit_price) / entry_price
            else:
                pnl = 0
            
            # Update metrics
            self.risk_metrics.daily_pnl += pnl
            self.risk_metrics.current_exposure -= position_size
            self.risk_metrics.active_trades = len(self.active_trades) - 1
            
            # Update win/loss statistics
            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Update win rate
            total_completed = self.winning_trades + self.losing_trades
            if total_completed > 0:
                self.risk_metrics.win_rate = (self.winning_trades / total_completed) * 100
            
            # Update profit factor
            total_wins = sum([r for r in self.returns_window if r > 0])
            total_losses = abs(sum([r for r in self.returns_window if r < 0]))
            if total_losses > 0:
                self.risk_metrics.profit_factor = total_wins / total_losses
            
            # Remove from active trades
            del self.active_trades[trade_id]
            
            self.logger.info(f"Trade closed: {trade_id} - P&L: ${pnl:.2f}")
            
            # Update account balance
            new_balance = self.account_balance + pnl
            self.update_account_balance(new_balance)
            
        except Exception as e:
            self.logger.error(f"Error closing trade: {e}")
    
    def _trigger_alert(self, alert_type: AlertType, risk_level: RiskLevel, 
                      message: str, value: float, threshold: float, 
                      symbol: str = None, trade_id: str = None):
        """Trigger a risk alert"""
        
        alert = RiskAlert(
            alert_type=alert_type,
            risk_level=risk_level,
            message=message,
            value=value,
            threshold=threshold,
            symbol=symbol,
            trade_id=trade_id,
            action_required="immediate_action" if risk_level == RiskLevel.CRITICAL else "monitor"
        )
        
        self.alerts.append(alert)
        
        # Log alert
        self.logger.warning(f"RISK ALERT [{risk_level.value.upper()}]: {message}")
        
        # Take automatic action for critical alerts
        if risk_level == RiskLevel.CRITICAL:
            self._handle_critical_alert(alert)
        
        # Save to database
        self._save_alert_to_db(alert)
    
    def _handle_critical_alert(self, alert: RiskAlert):
        """Handle critical risk alerts with automatic actions"""
        
        if alert.alert_type == AlertType.LOSS_LIMIT:
            # Disable new positions
            self.circuit_breakers['new_positions_enabled'] = False
            self.logger.critical("New positions disabled due to loss limit breach")
        
        elif alert.alert_type == AlertType.DRAWDOWN:
            # Enable conservative mode
            self.circuit_breakers['new_positions_enabled'] = False
            self.logger.critical("Conservative mode activated due to high drawdown")
        
        elif alert.alert_type == AlertType.MODEL_PERFORMANCE:
            # Block trading temporarily
            self.circuit_breakers['trading_enabled'] = False
            self.logger.critical("Trading disabled due to poor model performance")
        
        elif alert.alert_type == AlertType.SYSTEM_ERROR:
            # Emergency stop
            self.circuit_breakers['emergency_stop'] = True
            self.logger.critical("Emergency stop activated due to system error")
    
    def _save_alert_to_db(self, alert: RiskAlert):
        """Save alert to database"""
        try:
            conn = sqlite3.connect('/workspace/data/production_risk.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO risk_alerts 
                (timestamp, alert_type, risk_level, message, value, threshold, 
                 symbol, trade_id, action_required)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.timestamp.isoformat(),
                alert.alert_type.value,
                alert.risk_level.value,
                alert.message,
                alert.value,
                alert.threshold,
                alert.symbol,
                alert.trade_id,
                alert.action_required
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving alert to database: {e}")
    
    def _log_trade_assessment(self, symbol: str, signal: str, assessment: Dict[str, Any]):
        """Log trade assessment to database"""
        try:
            conn = sqlite3.connect('/workspace/data/production_risk.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trade_assessments 
                (timestamp, trade_id, symbol, signal, position_size, risk_score,
                 correlation_risk, volatility_risk, model_confidence, approved, rejection_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(timezone.utc).isoformat(),
                f"assessment_{int(time.time())}",
                symbol,
                signal,
                assessment.get('position_size', 0),
                assessment.get('risk_score', 0),
                assessment.get('risk_metrics', {}).get('correlation_risk', 0),
                assessment.get('risk_metrics', {}).get('volatility_risk', 0),
                assessment.get('risk_metrics', {}).get('confidence', 0),
                assessment.get('approved', False),
                assessment.get('rejection_reason', '')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error logging trade assessment: {e}")
    
    def start_monitoring(self):
        """Start continuous risk monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Risk monitoring started")
    
    def stop_monitoring(self):
        """Stop risk monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Risk monitoring stopped")
    
    def _monitoring_loop(self):
        """Continuous risk monitoring loop"""
        while self.monitoring_active:
            try:
                # Update risk metrics
                self._update_risk_metrics()
                
                # Check for risk threshold breaches
                self._check_risk_thresholds()
                
                # Save metrics to database
                self._save_metrics_to_db()
                
                # Sleep for monitoring interval
                time.sleep(MONITORING_CONFIG['health_check_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in risk monitoring loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _update_risk_metrics(self):
        """Update all risk metrics"""
        try:
            # Calculate Value at Risk (95%)
            if len(self.returns_window) >= 20:
                returns_array = np.array(self.returns_window)
                self.risk_metrics.var_95 = np.percentile(returns_array, 5) * self.account_balance
                
                # Expected Shortfall (average of worst 5% returns)
                worst_returns = returns_array[returns_array <= np.percentile(returns_array, 5)]
                if len(worst_returns) > 0:
                    self.risk_metrics.expected_shortfall = np.mean(worst_returns) * self.account_balance
            
            # Calculate Sharpe Ratio
            if len(self.returns_window) >= 30:
                returns_array = np.array(self.returns_window)
                if np.std(returns_array) > 0:
                    self.risk_metrics.sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
            
            # Update correlation and volatility risks
            self.risk_metrics.correlation_risk = self._calculate_portfolio_correlation_risk()
            self.risk_metrics.volatility_risk = self._calculate_portfolio_volatility_risk()
            
            # Update timestamp
            self.risk_metrics.timestamp = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {e}")
    
    def _calculate_portfolio_correlation_risk(self) -> float:
        """Calculate overall portfolio correlation risk"""
        if not self.active_trades:
            return 0.0
        
        # Simple correlation risk based on currency concentration
        currency_exposure = {}
        total_exposure = 0.0
        
        for trade in self.active_trades.values():
            symbol = trade.get('symbol', '')
            size = trade.get('position_size', 0)
            total_exposure += size
            
            if '/' in symbol:
                base, quote = symbol.split('/')
                currency_exposure[base] = currency_exposure.get(base, 0) + size
                currency_exposure[quote] = currency_exposure.get(quote, 0) + size
        
        if total_exposure == 0:
            return 0.0
        
        # Calculate concentration risk
        max_concentration = max(currency_exposure.values()) / total_exposure if currency_exposure else 0
        return min(1.0, max_concentration)
    
    def _calculate_portfolio_volatility_risk(self) -> float:
        """Calculate overall portfolio volatility risk"""
        if len(self.returns_window) < 10:
            return 0.0
        
        recent_returns = self.returns_window[-10:]
        volatility = np.std(recent_returns)
        
        # Normalize volatility (higher = more risk)
        normalized_volatility = min(1.0, volatility * 100)
        return normalized_volatility
    
    def _check_risk_thresholds(self):
        """Check all risk thresholds and trigger alerts"""
        
        # Check VaR
        if abs(self.risk_metrics.var_95) > self.account_balance * 0.05:  # 5% VaR limit
            self._trigger_alert(AlertType.VOLATILITY, RiskLevel.HIGH,
                              f"High VaR detected: ${abs(self.risk_metrics.var_95):.2f}",
                              abs(self.risk_metrics.var_95), self.account_balance * 0.05)
        
        # Check correlation risk
        if self.risk_metrics.correlation_risk > 0.7:
            self._trigger_alert(AlertType.CORRELATION, RiskLevel.MEDIUM,
                              f"High correlation risk: {self.risk_metrics.correlation_risk:.2f}",
                              self.risk_metrics.correlation_risk, 0.7)
        
        # Check volatility risk
        if self.risk_metrics.volatility_risk > 0.8:
            self._trigger_alert(AlertType.VOLATILITY, RiskLevel.HIGH,
                              f"High volatility risk: {self.risk_metrics.volatility_risk:.2f}",
                              self.risk_metrics.volatility_risk, 0.8)
    
    def _save_metrics_to_db(self):
        """Save current risk metrics to database"""
        try:
            conn = sqlite3.connect('/workspace/data/production_risk.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO risk_metrics 
                (timestamp, current_exposure, daily_pnl, weekly_pnl, monthly_pnl,
                 max_drawdown, current_drawdown, win_rate, profit_factor, sharpe_ratio,
                 var_95, expected_shortfall, correlation_risk, volatility_risk,
                 model_accuracy, active_trades, account_balance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.risk_metrics.timestamp.isoformat(),
                self.risk_metrics.current_exposure,
                self.risk_metrics.daily_pnl,
                self.risk_metrics.weekly_pnl,
                self.risk_metrics.monthly_pnl,
                self.risk_metrics.max_drawdown,
                self.risk_metrics.current_drawdown,
                self.risk_metrics.win_rate,
                self.risk_metrics.profit_factor,
                self.risk_metrics.sharpe_ratio,
                self.risk_metrics.var_95,
                self.risk_metrics.expected_shortfall,
                self.risk_metrics.correlation_risk,
                self.risk_metrics.volatility_risk,
                self.risk_metrics.model_accuracy,
                self.risk_metrics.active_trades,
                self.account_balance
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving metrics to database: {e}")
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get comprehensive risk status"""
        return {
            'risk_metrics': {
                'current_exposure': self.risk_metrics.current_exposure,
                'daily_pnl': self.risk_metrics.daily_pnl,
                'current_drawdown': self.risk_metrics.current_drawdown * 100,
                'win_rate': self.risk_metrics.win_rate,
                'sharpe_ratio': self.risk_metrics.sharpe_ratio,
                'var_95': self.risk_metrics.var_95,
                'correlation_risk': self.risk_metrics.correlation_risk,
                'volatility_risk': self.risk_metrics.volatility_risk,
                'active_trades': self.risk_metrics.active_trades
            },
            'circuit_breakers': self.circuit_breakers,
            'account_balance': self.account_balance,
            'peak_balance': self.peak_balance,
            'recent_alerts': [
                {
                    'type': alert.alert_type.value,
                    'level': alert.risk_level.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in self.alerts[-10:]  # Last 10 alerts
            ],
            'monitoring_active': self.monitoring_active,
            'last_update': datetime.now(timezone.utc).isoformat()
        }
    
    def reset_circuit_breakers(self, authorization_code: str = None):
        """Reset circuit breakers (requires authorization for production)"""
        
        # In production, require authorization code
        if authorization_code != "RISK_RESET_2024":
            self.logger.warning("Unauthorized circuit breaker reset attempt")
            return False
        
        self.circuit_breakers['trading_enabled'] = True
        self.circuit_breakers['new_positions_enabled'] = True
        self.circuit_breakers['high_risk_pairs_blocked'] = []
        self.circuit_breakers['emergency_stop'] = False
        
        self.logger.info("Circuit breakers reset")
        return True

# Example usage and testing
if __name__ == "__main__":
    # Initialize risk manager
    risk_manager = ProductionRiskManager()
    
    # Start monitoring
    risk_manager.start_monitoring()
    
    # Example: Simulate account balance update
    risk_manager.update_account_balance(10500.0)
    
    # Example: Assess trade risk
    market_data = {
        'rsi': {'value': 65},
        'atr_ratio': 0.015
    }
    
    assessment = risk_manager.assess_trade_risk('EUR/USD', 'BUY', 87.5, market_data)
    print(f"Trade Assessment: {assessment}")
    
    # Get risk status
    status = risk_manager.get_risk_status()
    print(f"Risk Status: {json.dumps(status, indent=2)}")
    
    # Stop monitoring
    risk_manager.stop_monitoring()