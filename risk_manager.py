import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3

from config import RISK_MANAGEMENT, DATABASE_CONFIG, SIGNAL_CONFIG

class RiskManager:
    def __init__(self):
        self.logger = self._setup_logger()
        self.daily_trades = []
        self.current_exposure = 0.0
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = 0.0
        
        # Initialize database
        self._initialize_database()
        
    def _setup_logger(self):
        logger = logging.getLogger('RiskManager')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('/workspace/logs/risk_manager.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def _initialize_database(self):
        """Initialize risk management database tables"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            # Create risk metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    daily_pnl REAL,
                    daily_trades INTEGER,
                    max_drawdown REAL,
                    current_exposure REAL,
                    risk_score REAL,
                    account_balance REAL,
                    win_rate REAL
                )
            ''')
            
            # Create trade limits table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_limits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    position_size REAL,
                    risk_amount REAL,
                    stop_loss REAL,
                    max_loss REAL,
                    confidence_level REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
    
    def calculate_position_size(self, account_balance: float, signal_data: Dict) -> Dict:
        """Calculate optimal position size based on risk parameters"""
        try:
            signal_strength = signal_data.get('strength', 5)
            accuracy = signal_data.get('accuracy', 85.0)
            volatility_level = signal_data.get('volatility_level', 'Medium')
            
            # Base position size (percentage of account)
            base_risk = RISK_MANAGEMENT['max_risk_per_trade']
            
            # Adjust based on signal strength
            strength_multiplier = signal_strength / 10.0
            
            # Adjust based on accuracy
            accuracy_multiplier = min(1.5, accuracy / 100.0)
            
            # Adjust based on volatility
            volatility_multipliers = {
                'Low': 1.2,
                'Medium': 1.0,
                'High': 0.7
            }
            volatility_multiplier = volatility_multipliers.get(volatility_level, 1.0)
            
            # Calculate final position size
            adjusted_risk = base_risk * strength_multiplier * accuracy_multiplier * volatility_multiplier
            
            # Apply maximum limits
            max_position_risk = min(adjusted_risk, RISK_MANAGEMENT['max_risk_per_trade'] * 1.5)
            
            # Calculate actual amounts
            risk_amount = account_balance * (max_position_risk / 100)
            
            # For binary options, position size is typically the investment amount
            position_size = risk_amount
            
            # Calculate maximum loss (100% for binary options)
            max_loss = position_size
            
            return {
                'position_size': round(position_size, 2),
                'risk_percentage': round(max_position_risk, 2),
                'risk_amount': round(risk_amount, 2),
                'max_loss': round(max_loss, 2),
                'recommended': True if max_position_risk <= RISK_MANAGEMENT['max_risk_per_trade'] else False,
                'risk_level': self._classify_position_risk(max_position_risk),
                'adjustments': {
                    'strength_multiplier': strength_multiplier,
                    'accuracy_multiplier': accuracy_multiplier,
                    'volatility_multiplier': volatility_multiplier
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return {
                'position_size': account_balance * 0.01,  # 1% default
                'risk_percentage': 1.0,
                'risk_amount': account_balance * 0.01,
                'max_loss': account_balance * 0.01,
                'recommended': True,
                'risk_level': 'Low',
                'adjustments': {}
            }
    
    def _classify_position_risk(self, risk_percentage: float) -> str:
        """Classify position risk level"""
        if risk_percentage <= 1.0:
            return 'Very Low'
        elif risk_percentage <= 2.0:
            return 'Low'
        elif risk_percentage <= 3.5:
            return 'Medium'
        elif risk_percentage <= 5.0:
            return 'High'
        else:
            return 'Very High'
    
    def validate_trade(self, signal_data: Dict, account_balance: float) -> Dict:
        """Validate if trade meets risk management criteria"""
        try:
            validation_result = {
                'approved': True,
                'reasons': [],
                'warnings': [],
                'risk_score': 0.0
            }
            
            # Check daily trade limit
            today_trades = self._get_today_trade_count()
            if today_trades >= 20:  # Max 20 trades per day
                validation_result['approved'] = False
                validation_result['reasons'].append('Daily trade limit exceeded')
            
            # Check daily loss limit
            daily_loss_pct = abs(self.daily_pnl) / account_balance * 100
            if daily_loss_pct >= RISK_MANAGEMENT['max_daily_loss']:
                validation_result['approved'] = False
                validation_result['reasons'].append(f'Daily loss limit exceeded: {daily_loss_pct:.1f}%')
            
            # Check signal accuracy threshold
            accuracy = signal_data.get('accuracy', 0)
            if accuracy < SIGNAL_CONFIG['min_accuracy']:
                validation_result['approved'] = False
                validation_result['reasons'].append(f'Signal accuracy too low: {accuracy:.1f}%')
            
            # Check signal strength
            strength = signal_data.get('strength', 0)
            if strength < 6:
                validation_result['warnings'].append(f'Low signal strength: {strength}/10')
            
            # Check volatility conditions
            volatility_level = signal_data.get('volatility_level', 'High')
            if volatility_level == 'High':
                validation_result['warnings'].append('High volatility detected')
            
            # Check concurrent trades
            active_trades = self._get_active_trades_count()
            if active_trades >= RISK_MANAGEMENT['max_concurrent_trades']:
                validation_result['approved'] = False
                validation_result['reasons'].append('Maximum concurrent trades reached')
            
            # Calculate risk score
            validation_result['risk_score'] = self._calculate_risk_score(
                signal_data, account_balance, today_trades, daily_loss_pct
            )
            
            # Final approval based on risk score
            if validation_result['risk_score'] > 8.0:
                validation_result['approved'] = False
                validation_result['reasons'].append('Overall risk score too high')
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating trade: {e}")
            return {
                'approved': False,
                'reasons': ['Validation error'],
                'warnings': [],
                'risk_score': 10.0
            }
    
    def _calculate_risk_score(self, signal_data: Dict, account_balance: float, 
                            today_trades: int, daily_loss_pct: float) -> float:
        """Calculate overall risk score (0-10, lower is better)"""
        try:
            risk_score = 0.0
            
            # Signal quality risk (0-3)
            accuracy = signal_data.get('accuracy', 85)
            strength = signal_data.get('strength', 5)
            signal_risk = 3.0 - (accuracy / 100 * 1.5 + strength / 10 * 1.5)
            risk_score += max(0, signal_risk)
            
            # Volume risk (0-2)
            volume_risk = today_trades / 20 * 2.0  # Max 20 trades per day
            risk_score += volume_risk
            
            # Loss risk (0-3)
            loss_risk = daily_loss_pct / RISK_MANAGEMENT['max_daily_loss'] * 3.0
            risk_score += loss_risk
            
            # Volatility risk (0-2)
            volatility_level = signal_data.get('volatility_level', 'Medium')
            volatility_risks = {'Low': 0.5, 'Medium': 1.0, 'High': 2.0}
            risk_score += volatility_risks.get(volatility_level, 1.0)
            
            return min(10.0, risk_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return 5.0
    
    def _get_today_trade_count(self) -> int:
        """Get number of trades executed today"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) FROM signals 
                WHERE DATE(timestamp) = ? AND status = 'executed'
            ''', (today,))
            
            count = cursor.fetchone()[0]
            conn.close()
            
            return count
            
        except Exception as e:
            self.logger.error(f"Error getting today's trade count: {e}")
            return 0
    
    def _get_active_trades_count(self) -> int:
        """Get number of currently active trades"""
        try:
            now = datetime.now()
            
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) FROM signals 
                WHERE status = 'active' AND expiry_time > ?
            ''', (now.isoformat(),))
            
            count = cursor.fetchone()[0]
            conn.close()
            
            return count
            
        except Exception as e:
            self.logger.error(f"Error getting active trades count: {e}")
            return 0
    
    def calculate_stop_loss(self, signal_data: Dict, position_size: float) -> Dict:
        """Calculate stop-loss parameters for risk management"""
        try:
            # For binary options, stop-loss is typically not applicable
            # as you either win or lose the full amount
            # However, we can calculate risk metrics
            
            entry_price = signal_data.get('entry_price', 0)
            volatility_level = signal_data.get('volatility_level', 'Medium')
            
            # Calculate theoretical stop-loss for analysis
            volatility_multipliers = {
                'Low': 0.5,
                'Medium': 1.0,
                'High': 1.5
            }
            
            volatility_multiplier = volatility_multipliers.get(volatility_level, 1.0)
            
            # Typical stop-loss percentage for forex
            stop_loss_pct = 0.5 * volatility_multiplier  # 0.5% base
            
            if signal_data.get('direction') == 'BUY':
                stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
            else:
                stop_loss_price = entry_price * (1 + stop_loss_pct / 100)
            
            # For binary options, max loss is the position size
            max_loss_amount = position_size
            
            return {
                'stop_loss_price': round(stop_loss_price, 5),
                'stop_loss_percentage': round(stop_loss_pct, 2),
                'max_loss_amount': round(max_loss_amount, 2),
                'risk_reward_ratio': 1.0,  # Binary options typically 1:1 or less
                'applicable': False,  # Not applicable for binary options
                'note': 'Binary options have fixed risk (100% of position)'
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating stop-loss: {e}")
            return {
                'stop_loss_price': 0,
                'stop_loss_percentage': 0,
                'max_loss_amount': position_size,
                'risk_reward_ratio': 1.0,
                'applicable': False,
                'note': 'Error in calculation'
            }
    
    def update_daily_metrics(self, trade_result: Dict):
        """Update daily risk metrics after trade completion"""
        try:
            pnl = trade_result.get('pnl', 0)
            self.daily_pnl += pnl
            
            # Update peak balance and drawdown
            current_balance = trade_result.get('account_balance', 0)
            if current_balance > self.peak_balance:
                self.peak_balance = current_balance
            
            if self.peak_balance > 0:
                current_drawdown = (self.peak_balance - current_balance) / self.peak_balance * 100
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Log metrics to database
            self._save_risk_metrics(current_balance)
            
        except Exception as e:
            self.logger.error(f"Error updating daily metrics: {e}")
    
    def _save_risk_metrics(self, account_balance: float):
        """Save current risk metrics to database"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            today_trades = self._get_today_trade_count()
            win_rate = self._calculate_daily_win_rate()
            risk_score = self._calculate_current_risk_score(account_balance)
            
            cursor.execute('''
                INSERT INTO risk_metrics 
                (timestamp, daily_pnl, daily_trades, max_drawdown, current_exposure, 
                 risk_score, account_balance, win_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                self.daily_pnl,
                today_trades,
                self.max_drawdown,
                self.current_exposure,
                risk_score,
                account_balance,
                win_rate
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving risk metrics: {e}")
    
    def _calculate_daily_win_rate(self) -> float:
        """Calculate today's win rate"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) FROM signals 
                WHERE DATE(timestamp) = ? AND result = 'win'
            ''', (today,))
            wins = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT COUNT(*) FROM signals 
                WHERE DATE(timestamp) = ? AND result IN ('win', 'loss')
            ''', (today,))
            total = cursor.fetchone()[0]
            
            conn.close()
            
            return (wins / total * 100) if total > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating daily win rate: {e}")
            return 0.0
    
    def _calculate_current_risk_score(self, account_balance: float) -> float:
        """Calculate current overall risk score"""
        try:
            risk_score = 0.0
            
            # Daily loss risk
            if account_balance > 0:
                daily_loss_pct = abs(self.daily_pnl) / account_balance * 100
                risk_score += min(5.0, daily_loss_pct / RISK_MANAGEMENT['max_daily_loss'] * 5.0)
            
            # Drawdown risk
            risk_score += min(3.0, self.max_drawdown / 10.0 * 3.0)
            
            # Trade frequency risk
            today_trades = self._get_today_trade_count()
            risk_score += min(2.0, today_trades / 20 * 2.0)
            
            return min(10.0, risk_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating current risk score: {e}")
            return 5.0
    
    def get_risk_report(self, account_balance: float) -> Dict:
        """Generate comprehensive risk report"""
        try:
            today_trades = self._get_today_trade_count()
            daily_win_rate = self._calculate_daily_win_rate()
            risk_score = self._calculate_current_risk_score(account_balance)
            active_trades = self._get_active_trades_count()
            
            # Calculate risk metrics
            daily_loss_pct = abs(self.daily_pnl) / account_balance * 100 if account_balance > 0 else 0
            
            return {
                'overall_risk_score': risk_score,
                'risk_level': self._classify_overall_risk(risk_score),
                'daily_metrics': {
                    'trades_today': today_trades,
                    'daily_pnl': self.daily_pnl,
                    'daily_pnl_pct': (self.daily_pnl / account_balance * 100) if account_balance > 0 else 0,
                    'daily_win_rate': daily_win_rate,
                    'daily_loss_pct': daily_loss_pct
                },
                'position_metrics': {
                    'active_trades': active_trades,
                    'current_exposure': self.current_exposure,
                    'max_drawdown': self.max_drawdown,
                    'peak_balance': self.peak_balance
                },
                'limits': {
                    'max_daily_trades': 20,
                    'max_daily_loss_pct': RISK_MANAGEMENT['max_daily_loss'],
                    'max_concurrent_trades': RISK_MANAGEMENT['max_concurrent_trades'],
                    'max_risk_per_trade': RISK_MANAGEMENT['max_risk_per_trade']
                },
                'recommendations': self._generate_risk_recommendations(risk_score, daily_loss_pct, today_trades)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating risk report: {e}")
            return {
                'overall_risk_score': 10.0,
                'risk_level': 'High',
                'daily_metrics': {},
                'position_metrics': {},
                'limits': {},
                'recommendations': ['Error generating report']
            }
    
    def _classify_overall_risk(self, risk_score: float) -> str:
        """Classify overall risk level"""
        if risk_score <= 2.0:
            return 'Very Low'
        elif risk_score <= 4.0:
            return 'Low'
        elif risk_score <= 6.0:
            return 'Medium'
        elif risk_score <= 8.0:
            return 'High'
        else:
            return 'Very High'
    
    def _generate_risk_recommendations(self, risk_score: float, daily_loss_pct: float, 
                                     today_trades: int) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if risk_score > 7.0:
            recommendations.append("Consider stopping trading for today - high risk score")
        
        if daily_loss_pct > RISK_MANAGEMENT['max_daily_loss'] * 0.8:
            recommendations.append("Approaching daily loss limit - reduce position sizes")
        
        if today_trades > 15:
            recommendations.append("High number of trades today - consider taking a break")
        
        if self.max_drawdown > 5.0:
            recommendations.append("Significant drawdown detected - review strategy")
        
        if risk_score <= 3.0 and daily_loss_pct < 2.0:
            recommendations.append("Good risk management - continue current approach")
        
        return recommendations if recommendations else ["Risk levels are acceptable"]
    
    def reset_daily_metrics(self):
        """Reset daily metrics (call at start of new trading day)"""
        self.daily_pnl = 0.0
        self.daily_trades = []
        self.logger.info("Daily risk metrics reset")
    
    def get_risk_status(self) -> Dict:
        """Get current risk management status"""
        try:
            # Calculate current risk metrics
            daily_risk_used = self._calculate_daily_risk_used()
            current_win_rate = self._calculate_current_win_rate()
            max_position_size = self._calculate_max_position_size()
            current_positions = len(self.daily_trades)
            
            # Determine risk level
            risk_level = self._determine_risk_level(daily_risk_used, current_win_rate)
            
            # Check if safe to trade
            safe_to_trade = self._is_safe_to_trade(daily_risk_used, current_win_rate)
            
            # Get market volatility
            market_volatility = self._get_market_volatility()
            volatility_risk = self._assess_volatility_risk(market_volatility)
            
            # Determine recommended action
            recommended_action = self._get_recommended_action(safe_to_trade, risk_level)
            
            return {
                'risk_level': risk_level,
                'safe_to_trade': safe_to_trade,
                'daily_risk_used': daily_risk_used,
                'current_win_rate': current_win_rate,
                'max_position_size': max_position_size,
                'current_positions': current_positions,
                'stop_loss_active': True,
                'stop_loss_level': RISK_MANAGEMENT['stop_loss_threshold'],
                'take_profit_level': 100.0 - RISK_MANAGEMENT['stop_loss_threshold'],
                'market_volatility': market_volatility,
                'volatility_risk': volatility_risk,
                'recommended_action': recommended_action
            }
            
        except Exception as e:
            self.logger.error(f"Error getting risk status: {e}")
            return {
                'risk_level': 'Medium',
                'safe_to_trade': True,
                'daily_risk_used': 0.0,
                'current_win_rate': 0.0,
                'max_position_size': RISK_MANAGEMENT['max_risk_per_trade'],
                'current_positions': 0,
                'stop_loss_active': True,
                'stop_loss_level': RISK_MANAGEMENT['stop_loss_threshold'],
                'take_profit_level': 100.0 - RISK_MANAGEMENT['stop_loss_threshold'],
                'market_volatility': 'Medium',
                'volatility_risk': 'Low',
                'recommended_action': 'Continue Trading'
            }
    
    def _calculate_daily_risk_used(self) -> float:
        """Calculate daily risk used percentage"""
        try:
            total_risk = sum(trade.get('risk_amount', 0) for trade in self.daily_trades)
            max_daily_risk = RISK_MANAGEMENT['max_daily_loss']
            return min(100.0, (total_risk / max_daily_risk) * 100.0)
        except:
            return 0.0
    
    def _calculate_current_win_rate(self) -> float:
        """Calculate current win rate"""
        try:
            if not self.daily_trades:
                return 0.0
            
            winning_trades = sum(1 for trade in self.daily_trades if trade.get('result', 'loss') == 'win')
            return (winning_trades / len(self.daily_trades)) * 100.0
        except:
            return 0.0
    
    def _calculate_max_position_size(self) -> float:
        """Calculate maximum position size based on current risk"""
        try:
            daily_risk_used = self._calculate_daily_risk_used()
            current_win_rate = self._calculate_current_win_rate()
            
            # Reduce position size if risk is high or win rate is low
            risk_factor = 1.0 - (daily_risk_used / 100.0)
            win_rate_factor = current_win_rate / 100.0
            
            base_position = RISK_MANAGEMENT['max_risk_per_trade']
            adjusted_position = base_position * risk_factor * win_rate_factor
            
            return max(1.0, adjusted_position)  # Minimum 1%
        except:
            return RISK_MANAGEMENT['max_risk_per_trade']
    
    def _determine_risk_level(self, daily_risk_used: float, win_rate: float) -> str:
        """Determine current risk level"""
        if daily_risk_used > 80.0 or win_rate < 60.0:
            return 'High'
        elif daily_risk_used > 50.0 or win_rate < 75.0:
            return 'Medium'
        else:
            return 'Low'
    
    def _is_safe_to_trade(self, daily_risk_used: float, win_rate: float) -> bool:
        """Determine if it's safe to continue trading"""
        return daily_risk_used < 90.0 and win_rate >= 50.0
    
    def _get_market_volatility(self) -> str:
        """Get current market volatility level"""
        # This would normally come from market data
        # For now, return a default value
        return 'Medium'
    
    def _assess_volatility_risk(self, volatility: str) -> str:
        """Assess risk based on volatility"""
        volatility_risk_map = {
            'Low': 'Low',
            'Medium': 'Medium',
            'High': 'High'
        }
        return volatility_risk_map.get(volatility, 'Medium')
    
    def _get_recommended_action(self, safe_to_trade: bool, risk_level: str) -> str:
        """Get recommended trading action"""
        if not safe_to_trade:
            return 'Stop Trading'
        elif risk_level == 'High':
            return 'Reduce Position Sizes'
        elif risk_level == 'Medium':
            return 'Trade Cautiously'
        else:
            return 'Continue Trading'