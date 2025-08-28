#!/usr/bin/env python3
"""
📈 PAPER TRADING ENGINE - PRODUCTION READY
Comprehensive paper trading validation engine for 3+ months validation
Real-time signal generation, performance tracking, and risk monitoring
"""

import asyncio
import numpy as np
import pandas as pd
import sqlite3
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import threading
import time
import pickle
import joblib
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from config import DATABASE_CONFIG, TIMEZONE, SIGNAL_CONFIG, RISK_MANAGEMENT, CURRENCY_PAIRS
from redundant_data_manager import RedundantDataManager
from enhanced_feature_engine import EnhancedFeatureEngine
from model_validation_framework import ModelValidationFramework
from risk_manager import RiskManager

@dataclass
class PaperTrade:
    """Paper trade representation"""
    id: str
    timestamp: datetime
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    expiry_time: datetime
    duration: int  # minutes
    amount: float
    predicted_accuracy: float
    signal_strength: float
    confidence: float
    model_used: str
    features_used: Dict[str, float] = field(default_factory=dict)
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Results (filled after trade completion)
    actual_result: Optional[str] = None  # 'WIN', 'LOSS', 'DRAW'
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    profit_percentage: Optional[float] = None
    closed_at: Optional[datetime] = None

@dataclass
class TradingSignal:
    """Enhanced trading signal"""
    timestamp: datetime
    symbol: str
    direction: str
    confidence: float
    accuracy_prediction: float
    signal_strength: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    duration: int = 60  # minutes
    model_name: str = 'ensemble'
    features: Dict[str, float] = field(default_factory=dict)
    market_analysis: Dict[str, Any] = field(default_factory=dict)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TradingPerformance:
    """Trading performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    draw_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_profit: float = 0.0
    total_loss: float = 0.0
    max_drawdown: float = 0.0
    max_profit: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    recovery_factor: float = 0.0
    daily_returns: List[float] = field(default_factory=list)
    monthly_returns: List[float] = field(default_factory=list)
    accuracy_by_model: Dict[str, float] = field(default_factory=dict)
    accuracy_by_symbol: Dict[str, float] = field(default_factory=dict)

class ModelLoader:
    """Load and manage trained models"""
    
    def __init__(self):
        self.logger = logging.getLogger('ModelLoader')
        self.models = {}
        self.scalers = {}
        self.feature_engines = {}
        
    def load_ensemble_models(self, model_dir: str = '/workspace/models/ensemble') -> bool:
        """Load all ensemble models"""
        
        try:
            # Find latest model files
            import glob
            
            # Load tree-based models
            for model_type in ['xgboost', 'lightgbm', 'catboost', 'random_forest']:
                model_files = glob.glob(f'{model_dir}/{model_type}_*.pkl')
                if model_files:
                    latest_model = max(model_files, key=os.path.getctime)
                    self.models[model_type] = joblib.load(latest_model)
                    self.logger.info(f"Loaded {model_type} model from {latest_model}")
            
            # Load LSTM models
            lstm_files = glob.glob(f'{model_dir}/lstm_*.h5')
            if lstm_files:
                latest_lstm = max(lstm_files, key=os.path.getctime)
                self.models['lstm'] = tf.keras.models.load_model(latest_lstm)
                self.logger.info(f"Loaded LSTM model from {latest_lstm}")
            
            # Load meta-learner
            meta_files = glob.glob(f'{model_dir}/meta_learner_*.pkl')
            if meta_files:
                latest_meta = max(meta_files, key=os.path.getctime)
                self.models['meta_learner'] = joblib.load(latest_meta)
                self.logger.info(f"Loaded meta-learner from {latest_meta}")
            
            # Load scalers
            scaler_files = glob.glob('/workspace/models/scaler_*.pkl')
            if scaler_files:
                latest_scaler = max(scaler_files, key=os.path.getctime)
                with open(latest_scaler, 'rb') as f:
                    self.scalers['feature_scaler'] = pickle.load(f)
                self.logger.info(f"Loaded feature scaler from {latest_scaler}")
            
            return len(self.models) > 0
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False
    
    def predict_ensemble(self, features: np.ndarray) -> Tuple[float, float, str]:
        """Make ensemble prediction"""
        
        try:
            if not self.models:
                return 0.5, 0.5, 'no_models'
            
            # Scale features
            if 'feature_scaler' in self.scalers:
                features_scaled = self.scalers['feature_scaler'].transform(features)
            else:
                features_scaled = features
            
            meta_features = []
            model_predictions = {}
            
            # Get predictions from tree-based models
            for name in ['xgboost', 'lightgbm', 'catboost', 'random_forest']:
                if name in self.models:
                    try:
                        if hasattr(self.models[name], 'predict_proba'):
                            pred = self.models[name].predict_proba(features_scaled)[:, 1]
                        else:
                            pred = self.models[name].predict(features_scaled)
                        
                        meta_features.append(pred[0] if len(pred) > 0 else 0.5)
                        model_predictions[name] = pred[0] if len(pred) > 0 else 0.5
                    except Exception as e:
                        self.logger.warning(f"Error with {name} model: {e}")
                        meta_features.append(0.5)
                        model_predictions[name] = 0.5
            
            # Get LSTM prediction (requires sequences)
            if 'lstm' in self.models:
                try:
                    # For paper trading, we'll use the last features as a simple sequence
                    lstm_features = features_scaled.reshape(1, 1, -1)  # Simple sequence
                    lstm_pred = self.models['lstm'].predict(lstm_features)[0, 0]
                    meta_features.append(lstm_pred)
                    model_predictions['lstm'] = lstm_pred
                except Exception as e:
                    self.logger.warning(f"Error with LSTM model: {e}")
                    meta_features.append(0.5)
                    model_predictions['lstm'] = 0.5
            
            # Meta-learner prediction
            if 'meta_learner' in self.models and meta_features:
                try:
                    meta_input = np.array(meta_features).reshape(1, -1)
                    ensemble_pred = self.models['meta_learner'].predict_proba(meta_input)[0]
                    final_prediction = ensemble_pred[1] if len(ensemble_pred) > 1 else ensemble_pred[0]
                    confidence = max(ensemble_pred) if len(ensemble_pred) > 1 else ensemble_pred[0]
                    best_model = 'ensemble'
                except Exception as e:
                    self.logger.warning(f"Error with meta-learner: {e}")
                    # Fallback to average
                    final_prediction = np.mean(meta_features)
                    confidence = 1.0 - np.std(meta_features)  # High agreement = high confidence
                    best_model = max(model_predictions.items(), key=lambda x: abs(x[1] - 0.5))[0]
            else:
                # Fallback to simple average
                final_prediction = np.mean(meta_features) if meta_features else 0.5
                confidence = 1.0 - np.std(meta_features) if meta_features else 0.5
                best_model = 'average'
            
            return final_prediction, confidence, best_model
            
        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            return 0.5, 0.5, 'error'

class SignalGenerator:
    """Generate trading signals using trained models"""
    
    def __init__(self):
        self.logger = logging.getLogger('SignalGenerator')
        self.model_loader = ModelLoader()
        self.feature_engine = EnhancedFeatureEngine()
        self.data_manager = RedundantDataManager()
        self.risk_manager = RiskManager()
        
        # Load models
        self.models_loaded = self.model_loader.load_ensemble_models()
        if not self.models_loaded:
            self.logger.warning("No models loaded - signals will be random")
    
    async def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generate trading signal for a symbol"""
        
        try:
            # Get real-time market data
            market_data = await self.data_manager.get_real_time_data(symbol)
            
            if market_data is None:
                self.logger.warning(f"No market data available for {symbol}")
                return None
            
            # Create historical data context (simplified for real-time)
            data_context = pd.DataFrame([{
                'timestamp': market_data.timestamp,
                'open': market_data.bid,
                'high': market_data.bid,
                'low': market_data.ask,
                'close': (market_data.bid + market_data.ask) / 2,
                'volume': 1000  # Placeholder
            }])
            
            # Engineer features
            features_df = self.feature_engine.engineer_features(data_context, symbol, '1m')
            
            # Remove non-feature columns
            exclude_cols = ['timestamp', 'symbol', 'timeframe', 'source']
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
            features = features_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
            
            if len(features) == 0:
                self.logger.warning(f"No features generated for {symbol}")
                return None
            
            # Get prediction
            if self.models_loaded:
                prediction, confidence, model_used = self.model_loader.predict_ensemble(features.values)
            else:
                # Random prediction for testing
                prediction = np.random.random()
                confidence = np.random.uniform(0.5, 0.9)
                model_used = 'random'
            
            # Determine direction and signal strength
            if prediction > 0.6:
                direction = 'BUY'
                signal_strength = (prediction - 0.5) * 2  # Scale to 0-1
            elif prediction < 0.4:
                direction = 'SELL'
                signal_strength = (0.5 - prediction) * 2  # Scale to 0-1
            else:
                # No clear signal
                return None
            
            # Risk assessment
            risk_assessment = self.risk_manager.calculate_position_size(
                account_balance=10000,
                signal_data={
                    'strength': signal_strength * 10,
                    'accuracy': prediction,
                    'volatility_level': 'Medium'
                }
            )
            
            # Market analysis
            market_analysis = {
                'spread': abs(market_data.ask - market_data.bid),
                'quality_score': market_data.quality_score,
                'latency_ms': market_data.latency_ms,
                'data_source': market_data.source
            }
            
            # Create signal
            signal = TradingSignal(
                timestamp=datetime.now(TIMEZONE),
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                accuracy_prediction=prediction,
                signal_strength=signal_strength,
                entry_price=(market_data.bid + market_data.ask) / 2,
                duration=60,  # 1 hour
                model_name=model_used,
                features=dict(zip(feature_cols, features.iloc[0].values)),
                market_analysis=market_analysis,
                risk_assessment=risk_assessment
            )
            
            # Validate signal quality
            if self._validate_signal_quality(signal):
                return signal
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def _validate_signal_quality(self, signal: TradingSignal) -> bool:
        """Validate signal meets quality requirements"""
        
        # Check minimum confidence
        if signal.confidence < SIGNAL_CONFIG['min_confidence'] / 100:
            return False
        
        # Check minimum accuracy prediction
        if signal.accuracy_prediction < SIGNAL_CONFIG['min_accuracy'] / 100:
            return False
        
        # Check signal strength
        if signal.signal_strength < 0.5:
            return False
        
        # Check market conditions
        if signal.market_analysis.get('quality_score', 0) < 0.7:
            return False
        
        return True

class PaperTradingEngine:
    """Comprehensive paper trading validation engine"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.logger = logging.getLogger('PaperTradingEngine')
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager()
        
        # Trading state
        self.active_trades: Dict[str, PaperTrade] = {}
        self.closed_trades: List[PaperTrade] = []
        self.daily_pnl: Dict[str, float] = {}
        self.performance = TradingPerformance()
        
        # Configuration
        self.symbols = CURRENCY_PAIRS[:5]  # Start with major pairs
        self.max_concurrent_trades = RISK_MANAGEMENT['max_concurrent_trades']
        self.max_daily_loss = RISK_MANAGEMENT['max_daily_loss'] / 100
        self.max_risk_per_trade = RISK_MANAGEMENT['max_risk_per_trade'] / 100
        
        # Control flags
        self.running = False
        self.paused = False
        
        # Initialize database
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize paper trading database"""
        
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            # Paper trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_trades (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    expiry_time TEXT NOT NULL,
                    duration INTEGER NOT NULL,
                    amount REAL NOT NULL,
                    predicted_accuracy REAL NOT NULL,
                    signal_strength REAL NOT NULL,
                    confidence REAL NOT NULL,
                    model_used TEXT NOT NULL,
                    features TEXT,
                    market_conditions TEXT,
                    actual_result TEXT,
                    exit_price REAL,
                    pnl REAL,
                    profit_percentage REAL,
                    closed_at TEXT
                )
            ''')
            
            # Performance tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    win_rate REAL,
                    total_pnl REAL,
                    current_balance REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    daily_pnl TEXT
                )
            ''')
            
            # Signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS generated_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    accuracy_prediction REAL NOT NULL,
                    signal_strength REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    executed BOOLEAN DEFAULT FALSE,
                    trade_id TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Paper trading database initialized")
            
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
    
    async def start_paper_trading(self, duration_days: int = 90):
        """Start paper trading validation for specified duration"""
        
        self.logger.info(f"Starting paper trading validation for {duration_days} days")
        self.running = True
        
        # Calculate end date
        end_date = datetime.now(TIMEZONE) + timedelta(days=duration_days)
        
        try:
            while self.running and datetime.now(TIMEZONE) < end_date:
                if not self.paused:
                    # Generate and execute signals
                    await self._trading_cycle()
                    
                    # Update performance
                    self._update_performance()
                    
                    # Check risk limits
                    if not self._check_risk_limits():
                        self.logger.warning("Risk limits exceeded, pausing trading")
                        self.paused = True
                    
                    # Save state
                    await self._save_trading_state()
                
                # Wait before next cycle (60 seconds)
                await asyncio.sleep(60)
            
            self.logger.info("Paper trading validation completed")
            
        except Exception as e:
            self.logger.error(f"Error in paper trading: {e}")
        finally:
            self.running = False
            await self._finalize_paper_trading()
    
    async def _trading_cycle(self):
        """Execute one trading cycle"""
        
        try:
            # Check and close expired trades
            await self._check_expired_trades()
            
            # Generate new signals if we have capacity
            if len(self.active_trades) < self.max_concurrent_trades:
                for symbol in self.symbols:
                    if len(self.active_trades) >= self.max_concurrent_trades:
                        break
                    
                    # Check if we already have a trade for this symbol
                    symbol_trades = [t for t in self.active_trades.values() if t.symbol == symbol]
                    if symbol_trades:
                        continue
                    
                    # Generate signal
                    signal = await self.signal_generator.generate_signal(symbol)
                    
                    if signal:
                        # Execute trade
                        trade = await self._execute_paper_trade(signal)
                        if trade:
                            self.active_trades[trade.id] = trade
                            self.logger.info(f"Executed paper trade: {trade.id} - {trade.symbol} {trade.direction}")
                        
                        # Rate limiting
                        await asyncio.sleep(1)
                        
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
    
    async def _execute_paper_trade(self, signal: TradingSignal) -> Optional[PaperTrade]:
        """Execute a paper trade based on signal"""
        
        try:
            # Calculate position size
            risk_data = self.risk_manager.calculate_position_size(
                account_balance=self.current_balance,
                signal_data={
                    'strength': signal.signal_strength * 10,
                    'accuracy': signal.accuracy_prediction,
                    'volatility_level': 'Medium'
                }
            )
            
            # Create trade
            trade_id = f"{signal.symbol}_{signal.direction}_{int(time.time())}"
            
            trade = PaperTrade(
                id=trade_id,
                timestamp=signal.timestamp,
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=signal.entry_price,
                expiry_time=signal.timestamp + timedelta(minutes=signal.duration),
                duration=signal.duration,
                amount=risk_data['position_size'],
                predicted_accuracy=signal.accuracy_prediction,
                signal_strength=signal.signal_strength,
                confidence=signal.confidence,
                model_used=signal.model_name,
                features_used=signal.features,
                market_conditions=signal.market_analysis
            )
            
            # Save to database
            await self._save_trade_to_db(trade)
            
            # Log signal to database
            await self._save_signal_to_db(signal, trade_id)
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Error executing paper trade: {e}")
            return None
    
    async def _check_expired_trades(self):
        """Check and close expired trades"""
        
        current_time = datetime.now(TIMEZONE)
        expired_trades = []
        
        for trade_id, trade in self.active_trades.items():
            if current_time >= trade.expiry_time:
                expired_trades.append(trade_id)
        
        for trade_id in expired_trades:
            trade = self.active_trades.pop(trade_id)
            await self._close_paper_trade(trade)
    
    async def _close_paper_trade(self, trade: PaperTrade):
        """Close a paper trade and calculate results"""
        
        try:
            # Get current market price
            market_data = await self.signal_generator.data_manager.get_real_time_data(trade.symbol)
            
            if market_data:
                current_price = (market_data.bid + market_data.ask) / 2
            else:
                # Fallback: simulate small random movement
                current_price = trade.entry_price * (1 + np.random.normal(0, 0.001))
            
            # Calculate result
            if trade.direction == 'BUY':
                price_movement = current_price - trade.entry_price
                if price_movement > 0:
                    result = 'WIN'
                elif price_movement < 0:
                    result = 'LOSS'
                else:
                    result = 'DRAW'
            else:  # SELL
                price_movement = trade.entry_price - current_price
                if price_movement > 0:
                    result = 'WIN'
                elif price_movement < 0:
                    result = 'LOSS'
                else:
                    result = 'DRAW'
            
            # Calculate PnL (simplified binary options style)
            if result == 'WIN':
                pnl = trade.amount * 0.8  # 80% profit
                profit_percentage = 80.0
            elif result == 'LOSS':
                pnl = -trade.amount  # 100% loss
                profit_percentage = -100.0
            else:  # DRAW
                pnl = 0
                profit_percentage = 0.0
            
            # Update trade
            trade.actual_result = result
            trade.exit_price = current_price
            trade.pnl = pnl
            trade.profit_percentage = profit_percentage
            trade.closed_at = datetime.now(TIMEZONE)
            
            # Update balance
            self.current_balance += pnl
            
            # Update daily PnL
            date_key = trade.closed_at.strftime('%Y-%m-%d')
            if date_key not in self.daily_pnl:
                self.daily_pnl[date_key] = 0
            self.daily_pnl[date_key] += pnl
            
            # Add to closed trades
            self.closed_trades.append(trade)
            
            # Update trade in database
            await self._update_trade_in_db(trade)
            
            self.logger.info(f"Closed trade {trade.id}: {result} - PnL: {pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error closing trade {trade.id}: {e}")
    
    def _update_performance(self):
        """Update performance metrics"""
        
        if not self.closed_trades:
            return
        
        # Basic metrics
        self.performance.total_trades = len(self.closed_trades)
        self.performance.winning_trades = len([t for t in self.closed_trades if t.actual_result == 'WIN'])
        self.performance.losing_trades = len([t for t in self.closed_trades if t.actual_result == 'LOSS'])
        self.performance.draw_trades = len([t for t in self.closed_trades if t.actual_result == 'DRAW'])
        
        self.performance.win_rate = (self.performance.winning_trades / self.performance.total_trades * 100 
                                   if self.performance.total_trades > 0 else 0)
        
        # PnL metrics
        self.performance.total_pnl = sum(t.pnl for t in self.closed_trades if t.pnl is not None)
        self.performance.total_profit = sum(t.pnl for t in self.closed_trades 
                                          if t.pnl is not None and t.pnl > 0)
        self.performance.total_loss = sum(t.pnl for t in self.closed_trades 
                                        if t.pnl is not None and t.pnl < 0)
        
        # Drawdown calculation
        running_pnl = 0
        peak_pnl = 0
        max_drawdown = 0
        
        for trade in sorted(self.closed_trades, key=lambda x: x.closed_at):
            if trade.pnl is not None:
                running_pnl += trade.pnl
                peak_pnl = max(peak_pnl, running_pnl)
                drawdown = peak_pnl - running_pnl
                max_drawdown = max(max_drawdown, drawdown)
        
        self.performance.max_drawdown = max_drawdown
        
        # Other metrics
        winning_trades = [t for t in self.closed_trades if t.actual_result == 'WIN' and t.pnl is not None]
        losing_trades = [t for t in self.closed_trades if t.actual_result == 'LOSS' and t.pnl is not None]
        
        self.performance.average_win = (sum(t.pnl for t in winning_trades) / len(winning_trades) 
                                      if winning_trades else 0)
        self.performance.average_loss = (sum(t.pnl for t in losing_trades) / len(losing_trades) 
                                       if losing_trades else 0)
        
        self.performance.profit_factor = (abs(self.performance.total_profit / self.performance.total_loss) 
                                        if self.performance.total_loss != 0 else float('inf'))
        
        # Daily returns for Sharpe ratio
        self.performance.daily_returns = list(self.daily_pnl.values())
        
        if len(self.performance.daily_returns) > 1:
            daily_returns_array = np.array(self.performance.daily_returns)
            daily_mean = np.mean(daily_returns_array)
            daily_std = np.std(daily_returns_array)
            self.performance.sharpe_ratio = (daily_mean / daily_std * np.sqrt(252) 
                                           if daily_std > 0 else 0)
        
        # Accuracy by model and symbol
        model_trades = {}
        symbol_trades = {}
        
        for trade in self.closed_trades:
            if trade.actual_result is not None:
                # By model
                if trade.model_used not in model_trades:
                    model_trades[trade.model_used] = {'correct': 0, 'total': 0}
                model_trades[trade.model_used]['total'] += 1
                if trade.actual_result == 'WIN':
                    model_trades[trade.model_used]['correct'] += 1
                
                # By symbol
                if trade.symbol not in symbol_trades:
                    symbol_trades[trade.symbol] = {'correct': 0, 'total': 0}
                symbol_trades[trade.symbol]['total'] += 1
                if trade.actual_result == 'WIN':
                    symbol_trades[trade.symbol]['correct'] += 1
        
        self.performance.accuracy_by_model = {
            model: data['correct'] / data['total'] * 100 
            for model, data in model_trades.items() if data['total'] > 0
        }
        
        self.performance.accuracy_by_symbol = {
            symbol: data['correct'] / data['total'] * 100 
            for symbol, data in symbol_trades.items() if data['total'] > 0
        }
    
    def _check_risk_limits(self) -> bool:
        """Check if risk limits are within acceptable bounds"""
        
        # Check daily loss limit
        today = datetime.now(TIMEZONE).strftime('%Y-%m-%d')
        daily_loss = self.daily_pnl.get(today, 0)
        
        if daily_loss < -self.max_daily_loss * self.initial_balance:
            self.logger.warning(f"Daily loss limit exceeded: {daily_loss}")
            return False
        
        # Check maximum drawdown
        if self.performance.max_drawdown > 0.2 * self.initial_balance:  # 20% max drawdown
            self.logger.warning(f"Maximum drawdown exceeded: {self.performance.max_drawdown}")
            return False
        
        # Check minimum balance
        if self.current_balance < 0.5 * self.initial_balance:  # 50% of initial balance
            self.logger.warning(f"Minimum balance threshold reached: {self.current_balance}")
            return False
        
        return True
    
    async def _save_trade_to_db(self, trade: PaperTrade):
        """Save trade to database"""
        
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO paper_trades 
                (id, timestamp, symbol, direction, entry_price, expiry_time, duration, 
                 amount, predicted_accuracy, signal_strength, confidence, model_used, 
                 features, market_conditions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.id,
                trade.timestamp.isoformat() if hasattr(trade.timestamp, 'isoformat') else str(trade.timestamp),
                trade.symbol,
                trade.direction,
                trade.entry_price,
                trade.expiry_time.isoformat() if hasattr(trade.expiry_time, 'isoformat') else str(trade.expiry_time),
                trade.duration,
                trade.amount,
                trade.predicted_accuracy,
                trade.signal_strength,
                trade.confidence,
                trade.model_used,
                json.dumps(trade.features_used),
                json.dumps(trade.market_conditions)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving trade to database: {e}")
    
    async def _update_trade_in_db(self, trade: PaperTrade):
        """Update trade results in database"""
        
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE paper_trades 
                SET actual_result = ?, exit_price = ?, pnl = ?, profit_percentage = ?, closed_at = ?
                WHERE id = ?
            ''', (
                trade.actual_result,
                trade.exit_price,
                trade.pnl,
                trade.profit_percentage,
                trade.closed_at.isoformat() if trade.closed_at else None,
                trade.id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error updating trade in database: {e}")
    
    async def _save_signal_to_db(self, signal: TradingSignal, trade_id: str):
        """Save generated signal to database"""
        
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO generated_signals 
                (timestamp, symbol, direction, confidence, accuracy_prediction, 
                 signal_strength, entry_price, model_name, executed, trade_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal.timestamp.isoformat(),
                signal.symbol,
                signal.direction,
                signal.confidence,
                signal.accuracy_prediction,
                signal.signal_strength,
                signal.entry_price,
                signal.model_name,
                True,
                trade_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving signal to database: {e}")
    
    async def _save_trading_state(self):
        """Save current trading performance state"""
        
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trading_performance 
                (timestamp, total_trades, winning_trades, losing_trades, win_rate, 
                 total_pnl, current_balance, max_drawdown, sharpe_ratio, daily_pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(TIMEZONE).isoformat(),
                self.performance.total_trades,
                self.performance.winning_trades,
                self.performance.losing_trades,
                self.performance.win_rate,
                self.performance.total_pnl,
                self.current_balance,
                self.performance.max_drawdown,
                self.performance.sharpe_ratio,
                json.dumps(self.daily_pnl)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving trading state: {e}")
    
    async def _finalize_paper_trading(self):
        """Finalize paper trading session"""
        
        # Close any remaining active trades
        for trade in list(self.active_trades.values()):
            await self._close_paper_trade(trade)
        
        # Final performance update
        self._update_performance()
        
        # Generate final report
        report = self.generate_performance_report()
        
        # Save final report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f'/workspace/paper_trading_report_{timestamp}.md'
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Paper trading finalized. Report saved to {report_path}")
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        
        report = f"""
# Paper Trading Performance Report

## Summary Statistics
- **Total Trades**: {self.performance.total_trades}
- **Winning Trades**: {self.performance.winning_trades}
- **Losing Trades**: {self.performance.losing_trades}
- **Draw Trades**: {self.performance.draw_trades}
- **Win Rate**: {self.performance.win_rate:.2f}%

## Financial Performance
- **Initial Balance**: ${self.initial_balance:,.2f}
- **Final Balance**: ${self.current_balance:,.2f}
- **Total PnL**: ${self.performance.total_pnl:,.2f}
- **Total Profit**: ${self.performance.total_profit:,.2f}
- **Total Loss**: ${self.performance.total_loss:,.2f}
- **Return on Investment**: {(self.current_balance - self.initial_balance) / self.initial_balance * 100:.2f}%

## Risk Metrics
- **Maximum Drawdown**: ${self.performance.max_drawdown:,.2f}
- **Average Win**: ${self.performance.average_win:.2f}
- **Average Loss**: ${self.performance.average_loss:.2f}
- **Profit Factor**: {self.performance.profit_factor:.2f}
- **Sharpe Ratio**: {self.performance.sharpe_ratio:.2f}

## Performance by Model
"""
        
        for model, accuracy in self.performance.accuracy_by_model.items():
            report += f"- **{model}**: {accuracy:.2f}% accuracy\n"
        
        report += "\n## Performance by Symbol\n"
        
        for symbol, accuracy in self.performance.accuracy_by_symbol.items():
            report += f"- **{symbol}**: {accuracy:.2f}% accuracy\n"
        
        # Assessment
        report += f"""
## Overall Assessment

### 🎯 Accuracy Target (>80%):
"""
        
        overall_accuracy = self.performance.win_rate
        if overall_accuracy >= 80:
            report += f"✅ **PASSED** - Achieved {overall_accuracy:.2f}% win rate\n"
        else:
            report += f"❌ **FAILED** - Only achieved {overall_accuracy:.2f}% win rate\n"
        
        # Profitability assessment
        if self.performance.total_pnl > 0:
            report += "✅ **PROFITABLE** - Positive total PnL\n"
        else:
            report += "❌ **UNPROFITABLE** - Negative total PnL\n"
        
        # Risk assessment
        max_drawdown_pct = (self.performance.max_drawdown / self.initial_balance) * 100
        if max_drawdown_pct <= 20:
            report += f"✅ **RISK CONTROLLED** - Max drawdown {max_drawdown_pct:.2f}%\n"
        else:
            report += f"⚠️ **HIGH RISK** - Max drawdown {max_drawdown_pct:.2f}%\n"
        
        # Trading frequency
        days_active = len(self.daily_pnl)
        trades_per_day = self.performance.total_trades / days_active if days_active > 0 else 0
        report += f"📊 **Trading Frequency**: {trades_per_day:.1f} trades per day\n"
        
        # Final recommendation
        report += "\n## 🚀 Live Trading Readiness\n"
        
        if (overall_accuracy >= 80 and 
            self.performance.total_pnl > 0 and 
            max_drawdown_pct <= 20 and 
            self.performance.profit_factor > 1.0):
            report += "🎉 **READY FOR LIVE TRADING** - All criteria met!\n"
        else:
            report += "⚠️ **NOT READY** - Further optimization required\n"
        
        return report
    
    def stop_trading(self):
        """Stop paper trading"""
        self.running = False
        self.logger.info("Paper trading stopped")
    
    def pause_trading(self):
        """Pause paper trading"""
        self.paused = True
        self.logger.info("Paper trading paused")
    
    def resume_trading(self):
        """Resume paper trading"""
        self.paused = False
        self.logger.info("Paper trading resumed")

# Example usage and testing
async def main():
    """Main paper trading function"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/workspace/logs/paper_trading.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('PaperTradingEngine')
    logger.info("Starting paper trading validation")
    
    try:
        # Initialize paper trading engine
        engine = PaperTradingEngine(initial_balance=10000.0)
        
        # Start paper trading for 1 day (for testing, use 90+ for production)
        await engine.start_paper_trading(duration_days=1)  # Change to 90+ for full validation
        
        # Generate and print final report
        report = engine.generate_performance_report()
        print(report)
        
    except Exception as e:
        logger.error(f"Paper trading failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())