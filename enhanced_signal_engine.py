import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from config import DATABASE_CONFIG, SIGNAL_CONFIG, TIMEZONE
from data_manager import DataManager, MarketData
from ensemble_models import EnsembleSignalGenerator, EnsemblePrediction
from advanced_features import AdvancedFeatureEngine
from alternative_data import AlternativeDataManager
from backtesting_engine import BacktestingEngine
import sqlite3
import talib

@dataclass
class EnhancedSignal:
    """Enhanced signal with comprehensive analysis"""
    symbol: str
    direction: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    accuracy_prediction: float
    entry_time: datetime
    expiry_time: datetime
    expiry_duration: int  # minutes
    entry_price: float
    
    # Ensemble model results
    ensemble_prediction: EnsemblePrediction
    individual_model_scores: Dict[str, float]
    
    # Technical analysis
    technical_strength: float
    trend_alignment: str
    volatility_level: str
    support_resistance_level: float
    
    # Alternative data factors
    news_sentiment_score: float
    social_sentiment_score: float
    economic_impact_score: float
    
    # Risk assessment
    risk_level: str
    position_size_recommendation: float
    stop_loss_level: Optional[float]
    
    # Market context
    market_session: str
    volume_profile: str
    correlation_signals: Dict[str, float]
    
    # Signal quality metrics
    signal_strength: float  # 1-10 scale
    historical_accuracy: float
    backtest_performance: Dict[str, float]
    
    # Execution guidance
    optimal_entry_timing: datetime
    execution_urgency: str  # 'immediate', 'within_5min', 'wait_for_better'
    market_impact_warning: bool

class SignalQualityFilter:
    """Filters signals based on multiple quality criteria"""
    
    def __init__(self):
        self.logger = logging.getLogger('SignalQualityFilter')
        self.min_requirements = {
            'ensemble_confidence': 0.75,
            'signal_strength': 7.0,
            'technical_strength': 0.6,
            'min_accuracy_prediction': 0.8,
            'max_risk_level': 'Medium'
        }
    
    def evaluate_signal_quality(self, signal: EnhancedSignal) -> Dict[str, Any]:
        """Comprehensive signal quality evaluation"""
        try:
            quality_score = 0.0
            max_score = 100.0
            issues = []
            strengths = []
            
            # Ensemble model evaluation (30 points)
            if signal.ensemble_prediction.final_confidence >= 0.85:
                quality_score += 30
                strengths.append("Very high ensemble confidence")
            elif signal.ensemble_prediction.final_confidence >= 0.75:
                quality_score += 25
                strengths.append("High ensemble confidence")
            elif signal.ensemble_prediction.final_confidence >= 0.65:
                quality_score += 15
                issues.append("Moderate ensemble confidence")
            else:
                issues.append("Low ensemble confidence")
            
            # Model consensus evaluation (20 points)
            consensus = signal.ensemble_prediction.consensus_level
            if consensus >= 0.8:
                quality_score += 20
                strengths.append("Strong model consensus")
            elif consensus >= 0.6:
                quality_score += 15
                strengths.append("Good model consensus")
            elif consensus >= 0.4:
                quality_score += 8
                issues.append("Weak model consensus")
            else:
                issues.append("Poor model consensus")
            
            # Technical analysis evaluation (20 points)
            if signal.technical_strength >= 0.8:
                quality_score += 20
                strengths.append("Strong technical signals")
            elif signal.technical_strength >= 0.6:
                quality_score += 15
                strengths.append("Good technical alignment")
            elif signal.technical_strength >= 0.4:
                quality_score += 8
                issues.append("Weak technical signals")
            else:
                issues.append("Poor technical alignment")
            
            # Alternative data evaluation (15 points)
            alt_data_score = (
                abs(signal.news_sentiment_score) * 0.4 +
                abs(signal.social_sentiment_score) * 0.3 +
                signal.economic_impact_score * 0.3
            )
            
            if alt_data_score >= 0.7:
                quality_score += 15
                strengths.append("Strong alternative data support")
            elif alt_data_score >= 0.5:
                quality_score += 10
                strengths.append("Moderate alternative data support")
            elif alt_data_score >= 0.3:
                quality_score += 5
            else:
                issues.append("Weak alternative data support")
            
            # Risk assessment evaluation (10 points)
            if signal.risk_level == 'Low':
                quality_score += 10
                strengths.append("Low risk profile")
            elif signal.risk_level == 'Medium':
                quality_score += 6
            else:
                quality_score += 2
                issues.append("High risk profile")
            
            # Market timing evaluation (5 points)
            if signal.market_session in ['london', 'ny', 'overlap']:
                quality_score += 5
                strengths.append("Optimal market session")
            elif signal.market_session == 'asian':
                quality_score += 3
            else:
                quality_score += 1
                issues.append("Low liquidity session")
            
            # Final quality assessment
            quality_percentage = quality_score / max_score
            
            if quality_percentage >= 0.85:
                quality_grade = 'A+'
                recommendation = 'STRONG_BUY' if signal.direction == 'BUY' else 'STRONG_SELL'
            elif quality_percentage >= 0.75:
                quality_grade = 'A'
                recommendation = 'BUY' if signal.direction == 'BUY' else 'SELL'
            elif quality_percentage >= 0.65:
                quality_grade = 'B+'
                recommendation = 'WEAK_BUY' if signal.direction == 'BUY' else 'WEAK_SELL'
            elif quality_percentage >= 0.55:
                quality_grade = 'B'
                recommendation = 'HOLD'
            else:
                quality_grade = 'C'
                recommendation = 'AVOID'
            
            return {
                'quality_score': quality_score,
                'quality_percentage': quality_percentage,
                'quality_grade': quality_grade,
                'recommendation': recommendation,
                'strengths': strengths,
                'issues': issues,
                'passes_filter': quality_percentage >= 0.75 and signal.risk_level != 'High'
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating signal quality: {e}")
            return {'quality_percentage': 0, 'passes_filter': False}
    
    def apply_market_condition_filters(self, signal: EnhancedSignal) -> bool:
        """Apply market condition-based filters"""
        try:
            # Volatility filter
            if signal.volatility_level == 'Very High':
                self.logger.info("Rejecting signal due to extreme volatility")
                return False
            
            # News impact filter
            if abs(signal.news_sentiment_score) > 0.8 and signal.economic_impact_score > 0.8:
                # Major news event, require higher confidence
                if signal.ensemble_prediction.final_confidence < 0.9:
                    self.logger.info("Rejecting signal due to major news without high confidence")
                    return False
            
            # Time-based filters
            current_hour = datetime.now(TIMEZONE).hour
            if current_hour in [22, 23, 0, 1, 2, 3, 4, 5]:  # Low liquidity hours
                if signal.signal_strength < 8.0:
                    self.logger.info("Rejecting weak signal during low liquidity hours")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying market condition filters: {e}")
            return False

class MarketTimingOptimizer:
    """Optimizes signal timing for best execution"""
    
    def __init__(self):
        self.logger = logging.getLogger('MarketTimingOptimizer')
    
    def optimize_entry_timing(self, signal: EnhancedSignal, market_data: pd.DataFrame) -> datetime:
        """Determine optimal entry timing"""
        try:
            current_time = datetime.now(TIMEZONE)
            
            # Analyze short-term momentum
            recent_data = market_data.tail(20)
            price_momentum = self._calculate_momentum(recent_data)
            
            # Check for immediate execution conditions
            if self._should_execute_immediately(signal, price_momentum):
                return current_time
            
            # Look for better entry in next 5 minutes
            optimal_delay = self._calculate_optimal_delay(signal, price_momentum)
            
            return current_time + timedelta(minutes=optimal_delay)
            
        except Exception as e:
            self.logger.error(f"Error optimizing entry timing: {e}")
            return datetime.now(TIMEZONE)
    
    def _calculate_momentum(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate short-term momentum indicators"""
        try:
            if len(data) < 10:
                return {'trend': 0, 'strength': 0, 'acceleration': 0}
            
            close_prices = data['close']
            
            # Price trend
            trend = (close_prices.iloc[-1] - close_prices.iloc[-10]) / close_prices.iloc[-10]
            
            # Momentum strength
            price_changes = close_prices.pct_change().dropna()
            strength = abs(price_changes.mean()) / price_changes.std() if price_changes.std() > 0 else 0
            
            # Acceleration
            short_ma = close_prices.rolling(3).mean()
            long_ma = close_prices.rolling(10).mean()
            acceleration = (short_ma.iloc[-1] - long_ma.iloc[-1]) / long_ma.iloc[-1]
            
            return {
                'trend': trend,
                'strength': strength,
                'acceleration': acceleration
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum: {e}")
            return {'trend': 0, 'strength': 0, 'acceleration': 0}
    
    def _should_execute_immediately(self, signal: EnhancedSignal, momentum: Dict[str, float]) -> bool:
        """Determine if signal should be executed immediately"""
        try:
            # High confidence signals
            if signal.ensemble_prediction.final_confidence > 0.9:
                return True
            
            # Strong momentum alignment
            if signal.direction == 'BUY' and momentum['trend'] > 0.001 and momentum['acceleration'] > 0:
                return True
            elif signal.direction == 'SELL' and momentum['trend'] < -0.001 and momentum['acceleration'] < 0:
                return True
            
            # High urgency market conditions
            if signal.execution_urgency == 'immediate':
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error determining immediate execution: {e}")
            return True
    
    def _calculate_optimal_delay(self, signal: EnhancedSignal, momentum: Dict[str, float]) -> int:
        """Calculate optimal delay in minutes"""
        try:
            base_delay = 1  # Default 1 minute
            
            # Adjust based on momentum
            if abs(momentum['trend']) < 0.0005:  # Low momentum
                base_delay += 2
            
            # Adjust based on signal strength
            if signal.signal_strength < 7:
                base_delay += 1
            
            # Cap at 5 minutes
            return min(base_delay, 5)
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal delay: {e}")
            return 1

class EnhancedSignalEngine:
    """Main enhanced signal engine combining all advanced components"""
    
    def __init__(self):
        self.logger = logging.getLogger('EnhancedSignalEngine')
        
        # Initialize components
        self.data_manager = DataManager()
        self.feature_engine = AdvancedFeatureEngine()
        self.ensemble_generator = EnsembleSignalGenerator()
        self.alternative_data_manager = AlternativeDataManager()
        self.backtesting_engine = BacktestingEngine()
        
        # Signal processing components
        self.quality_filter = SignalQualityFilter()
        self.timing_optimizer = MarketTimingOptimizer()
        
        # Performance tracking
        self.signal_history = []
        self.model_performance = {}
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize enhanced signals database"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS enhanced_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL,
                    accuracy_prediction REAL,
                    entry_time TEXT,
                    expiry_time TEXT,
                    expiry_duration INTEGER,
                    entry_price REAL,
                    signal_strength REAL,
                    technical_strength REAL,
                    quality_score REAL,
                    quality_grade TEXT,
                    recommendation TEXT,
                    ensemble_data TEXT,
                    alternative_data TEXT,
                    execution_result TEXT,
                    actual_accuracy REAL,
                    pnl REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing enhanced signals database: {e}")
    
    async def generate_enhanced_signal(self, symbol: str, force_signal: bool = False) -> Optional[EnhancedSignal]:
        """Generate a comprehensive enhanced signal"""
        try:
            self.logger.info(f"Generating enhanced signal for {symbol}")
            
            # Step 1: Get market data
            market_data = await self.data_manager.get_historical_data(
                symbol, period="1d", interval="1m"
            )
            
            if market_data is None or len(market_data) < 100:
                self.logger.warning(f"Insufficient market data for {symbol}")
                return None
            
            # Step 2: Get alternative data
            alt_data = await self.alternative_data_manager.fetch_all_alternative_data([symbol])
            sentiment_features = self.alternative_data_manager.get_sentiment_features(symbol)
            
            # Step 3: Generate advanced features
            enhanced_features = await self.feature_engine.generate_realtime_features(market_data, symbol)
            
            # Step 4: Generate ensemble prediction
            ensemble_pred = self.ensemble_generator.predict(market_data)
            
            # Step 5: Perform comprehensive technical analysis
            tech_analysis = await self._perform_comprehensive_technical_analysis(market_data)
            
            # Step 6: Calculate market context
            market_context = await self._analyze_market_context(symbol, market_data)
            
            # Step 7: Create enhanced signal
            enhanced_signal = await self._create_enhanced_signal(
                symbol, market_data, ensemble_pred, tech_analysis, 
                sentiment_features, market_context, enhanced_features
            )
            
            if enhanced_signal is None:
                return None
            
            # Step 8: Apply quality filters
            quality_assessment = self.quality_filter.evaluate_signal_quality(enhanced_signal)
            enhanced_signal.backtest_performance = quality_assessment
            
            # Step 9: Check market condition filters
            if not self.quality_filter.apply_market_condition_filters(enhanced_signal):
                self.logger.info(f"Signal for {symbol} rejected by market condition filters")
                return None
            
            # Step 10: Optimize timing
            enhanced_signal.optimal_entry_timing = self.timing_optimizer.optimize_entry_timing(
                enhanced_signal, market_data
            )
            
            # Step 11: Final quality check
            if not force_signal and not quality_assessment.get('passes_filter', False):
                self.logger.info(f"Signal for {symbol} rejected by quality filter")
                return None
            
            # Step 12: Store signal
            await self._store_enhanced_signal(enhanced_signal, quality_assessment)
            
            self.logger.info(f"Enhanced signal generated for {symbol} with quality grade {quality_assessment.get('quality_grade', 'N/A')}")
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced signal for {symbol}: {e}")
            return None
    
    async def _perform_comprehensive_technical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive technical analysis"""
        try:
            close = data['close']
            high = data['high']
            low = data['low']
            volume = data['volume']
            
            analysis = {}
            
            # Trend analysis
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            
            trend_score = 0
            if close.iloc[-1] > sma_20.iloc[-1]:
                trend_score += 1
            if sma_20.iloc[-1] > sma_50.iloc[-1]:
                trend_score += 1
            if ema_12.iloc[-1] > ema_26.iloc[-1]:
                trend_score += 1
            
            analysis['trend_strength'] = trend_score / 3
            analysis['trend_direction'] = 'bullish' if trend_score >= 2 else 'bearish' if trend_score <= 1 else 'neutral'
            
            # Momentum analysis
            rsi = self._calculate_rsi(close, 14)
            stoch_k = self._calculate_stochastic(high, low, close, 14)
            
            momentum_score = 0
            if 30 <= rsi.iloc[-1] <= 70:  # RSI in normal range
                momentum_score += 1
            if 20 <= stoch_k.iloc[-1] <= 80:  # Stochastic in normal range
                momentum_score += 1
            
            analysis['momentum_strength'] = momentum_score / 2
            analysis['rsi'] = rsi.iloc[-1]
            analysis['stochastic'] = stoch_k.iloc[-1]
            
            # Volatility analysis
            atr = self._calculate_atr(high, low, close, 14)
            bb_upper, bb_lower = self._calculate_bollinger_bands(close, 20, 2)
            
            volatility_level = 'low'
            if atr.iloc[-1] > atr.rolling(50).mean().iloc[-1] * 1.5:
                volatility_level = 'high'
            elif atr.iloc[-1] > atr.rolling(50).mean().iloc[-1] * 1.2:
                volatility_level = 'medium'
            
            analysis['volatility_level'] = volatility_level
            analysis['atr'] = atr.iloc[-1]
            
            # Support/Resistance analysis
            support_level = low.rolling(20).min().iloc[-1]
            resistance_level = high.rolling(20).max().iloc[-1]
            current_price = close.iloc[-1]
            
            sr_score = min(
                (current_price - support_level) / (resistance_level - support_level),
                1.0
            )
            
            analysis['support_resistance_score'] = sr_score
            analysis['support_level'] = support_level
            analysis['resistance_level'] = resistance_level
            
            # Volume analysis
            volume_sma = volume.rolling(20).mean()
            volume_ratio = volume.iloc[-1] / volume_sma.iloc[-1]
            
            analysis['volume_strength'] = min(volume_ratio / 2, 1.0)  # Normalize
            
            # Overall technical strength
            technical_strength = np.mean([
                analysis['trend_strength'],
                analysis['momentum_strength'],
                1 - abs(0.5 - sr_score),  # Closer to middle is better
                min(analysis['volume_strength'], 1.0)
            ])
            
            analysis['overall_strength'] = technical_strength
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive technical analysis: {e}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Stochastic %K"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        return 100 * (close - lowest_low) / (highest_high - lowest_low)
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    async def _analyze_market_context(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current market context"""
        try:
            context = {}
            
            # Determine market session
            current_hour = datetime.now(TIMEZONE).hour
            if 0 <= current_hour < 8:
                context['session'] = 'asian'
            elif 8 <= current_hour < 13:
                context['session'] = 'london'
            elif 13 <= current_hour < 16:
                context['session'] = 'overlap'
            elif 16 <= current_hour < 21:
                context['session'] = 'ny'
            else:
                context['session'] = 'after_hours'
            
            # Volume profile
            recent_volume = data['volume'].tail(20).mean()
            avg_volume = data['volume'].rolling(100).mean().iloc[-1]
            
            if recent_volume > avg_volume * 1.5:
                context['volume_profile'] = 'high'
            elif recent_volume > avg_volume * 1.2:
                context['volume_profile'] = 'above_average'
            elif recent_volume < avg_volume * 0.8:
                context['volume_profile'] = 'low'
            else:
                context['volume_profile'] = 'normal'
            
            # Get correlation signals
            correlation_symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'XAU/USD']
            context['correlation_signals'] = {}
            
            for corr_symbol in correlation_symbols:
                if corr_symbol != symbol:
                    try:
                        corr_data = await self.data_manager.get_historical_data(
                            corr_symbol, period="1d", interval="1m"
                        )
                        if corr_data is not None and len(corr_data) > 50:
                            correlation = data['close'].tail(50).corr(corr_data['close'].tail(50))
                            context['correlation_signals'][corr_symbol] = correlation
                    except:
                        continue
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error analyzing market context: {e}")
            return {}
    
    async def _create_enhanced_signal(self, symbol: str, market_data: pd.DataFrame,
                                    ensemble_pred: EnsemblePrediction, tech_analysis: Dict[str, Any],
                                    sentiment_features: Dict[str, float], market_context: Dict[str, Any],
                                    enhanced_features: Dict[str, float]) -> Optional[EnhancedSignal]:
        """Create enhanced signal from all analysis components"""
        try:
            current_time = datetime.now(TIMEZONE)
            current_price = market_data['close'].iloc[-1]
            
            # Determine signal direction and confidence
            if ensemble_pred.final_prediction == 2:  # HOLD
                return None
            
            direction = 'BUY' if ensemble_pred.final_prediction == 0 else 'SELL'
            base_confidence = ensemble_pred.final_confidence
            
            # Adjust confidence based on technical analysis
            tech_adjustment = tech_analysis.get('overall_strength', 0.5) * 0.1
            adjusted_confidence = min(base_confidence + tech_adjustment, 1.0)
            
            # Calculate signal strength (1-10 scale)
            signal_strength = self._calculate_signal_strength(
                ensemble_pred, tech_analysis, sentiment_features, market_context
            )
            
            # Determine expiry duration based on volatility and market conditions
            volatility_level = tech_analysis.get('volatility_level', 'medium')
            if volatility_level == 'low':
                expiry_duration = 5  # 5 minutes for low volatility
            elif volatility_level == 'high':
                expiry_duration = 2  # 2 minutes for high volatility
            else:
                expiry_duration = 3  # 3 minutes for medium volatility
            
            expiry_time = current_time + timedelta(minutes=expiry_duration)
            
            # Calculate risk level
            risk_level = self._calculate_risk_level(
                adjusted_confidence, volatility_level, sentiment_features
            )
            
            # Position sizing recommendation (as percentage of balance)
            position_size_pct = self._calculate_position_size(
                adjusted_confidence, risk_level, signal_strength
            )
            
            # Calculate accuracy prediction
            accuracy_prediction = self._predict_signal_accuracy(
                ensemble_pred, tech_analysis, sentiment_features, market_context
            )
            
            # Get individual model scores
            individual_scores = {}
            for pred in ensemble_pred.individual_predictions:
                individual_scores[pred.model_name] = pred.confidence
            
            # Determine execution urgency
            if signal_strength >= 9 and adjusted_confidence >= 0.9:
                execution_urgency = 'immediate'
            elif signal_strength >= 7 and adjusted_confidence >= 0.8:
                execution_urgency = 'within_5min'
            else:
                execution_urgency = 'wait_for_better'
            
            enhanced_signal = EnhancedSignal(
                symbol=symbol,
                direction=direction,
                confidence=adjusted_confidence,
                accuracy_prediction=accuracy_prediction,
                entry_time=current_time,
                expiry_time=expiry_time,
                expiry_duration=expiry_duration,
                entry_price=current_price,
                
                ensemble_prediction=ensemble_pred,
                individual_model_scores=individual_scores,
                
                technical_strength=tech_analysis.get('overall_strength', 0),
                trend_alignment=tech_analysis.get('trend_direction', 'neutral'),
                volatility_level=volatility_level,
                support_resistance_level=tech_analysis.get('support_resistance_score', 0.5),
                
                news_sentiment_score=sentiment_features.get('news_sentiment_avg', 0),
                social_sentiment_score=sentiment_features.get('social_sentiment_avg', 0),
                economic_impact_score=sentiment_features.get('economic_impact_avg', 0),
                
                risk_level=risk_level,
                position_size_recommendation=position_size_pct,
                stop_loss_level=None,  # Binary options don't use traditional stop loss
                
                market_session=market_context.get('session', 'unknown'),
                volume_profile=market_context.get('volume_profile', 'normal'),
                correlation_signals=market_context.get('correlation_signals', {}),
                
                signal_strength=signal_strength,
                historical_accuracy=0.0,  # Will be calculated based on historical performance
                backtest_performance={},
                
                optimal_entry_timing=current_time,
                execution_urgency=execution_urgency,
                market_impact_warning=False
            )
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"Error creating enhanced signal: {e}")
            return None
    
    def _calculate_signal_strength(self, ensemble_pred: EnsemblePrediction,
                                 tech_analysis: Dict[str, Any], sentiment_features: Dict[str, float],
                                 market_context: Dict[str, Any]) -> float:
        """Calculate overall signal strength (1-10 scale)"""
        try:
            # Base score from ensemble confidence
            base_score = ensemble_pred.final_confidence * 5  # Max 5 points
            
            # Technical analysis contribution (max 2 points)
            tech_score = tech_analysis.get('overall_strength', 0) * 2
            
            # Model consensus contribution (max 1.5 points)
            consensus_score = ensemble_pred.consensus_level * 1.5
            
            # Alternative data contribution (max 1 point)
            alt_data_score = min(
                abs(sentiment_features.get('news_sentiment_avg', 0)) * 0.5 +
                sentiment_features.get('economic_impact_avg', 0) * 0.3 +
                abs(sentiment_features.get('social_sentiment_avg', 0)) * 0.2,
                1.0
            )
            
            # Market context contribution (max 0.5 points)
            context_score = 0
            if market_context.get('session') in ['london', 'ny', 'overlap']:
                context_score += 0.3
            if market_context.get('volume_profile') in ['above_average', 'high']:
                context_score += 0.2
            
            total_score = base_score + tech_score + consensus_score + alt_data_score + context_score
            
            return min(total_score, 10.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating signal strength: {e}")
            return 5.0
    
    def _calculate_risk_level(self, confidence: float, volatility: str, sentiment: Dict[str, float]) -> str:
        """Calculate risk level for the signal"""
        try:
            risk_score = 0
            
            # Confidence factor
            if confidence < 0.7:
                risk_score += 2
            elif confidence < 0.8:
                risk_score += 1
            
            # Volatility factor
            if volatility == 'high':
                risk_score += 2
            elif volatility == 'medium':
                risk_score += 1
            
            # Sentiment conflict factor
            news_sentiment = sentiment.get('news_sentiment_avg', 0)
            social_sentiment = sentiment.get('social_sentiment_avg', 0)
            
            if abs(news_sentiment - social_sentiment) > 0.5:
                risk_score += 1
            
            # Economic impact factor
            if sentiment.get('economic_impact_avg', 0) > 0.7:
                risk_score += 1
            
            if risk_score <= 1:
                return 'Low'
            elif risk_score <= 3:
                return 'Medium'
            else:
                return 'High'
                
        except Exception as e:
            self.logger.error(f"Error calculating risk level: {e}")
            return 'Medium'
    
    def _calculate_position_size(self, confidence: float, risk_level: str, signal_strength: float) -> float:
        """Calculate recommended position size as percentage of balance"""
        try:
            base_size = 0.02  # 2% base
            
            # Adjust for confidence
            confidence_multiplier = confidence * 1.5
            
            # Adjust for signal strength
            strength_multiplier = signal_strength / 10
            
            # Adjust for risk level
            risk_multipliers = {'Low': 1.2, 'Medium': 1.0, 'High': 0.6}
            risk_multiplier = risk_multipliers.get(risk_level, 1.0)
            
            recommended_size = base_size * confidence_multiplier * strength_multiplier * risk_multiplier
            
            # Cap at maximum position size
            return min(recommended_size, 0.05)  # Max 5% of balance
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.02
    
    def _predict_signal_accuracy(self, ensemble_pred: EnsemblePrediction,
                               tech_analysis: Dict[str, Any], sentiment_features: Dict[str, float],
                               market_context: Dict[str, Any]) -> float:
        """Predict the likely accuracy of this signal"""
        try:
            # Base accuracy from ensemble confidence
            base_accuracy = 0.5 + (ensemble_pred.final_confidence * 0.4)
            
            # Adjustments based on various factors
            adjustments = 0
            
            # Technical alignment
            if tech_analysis.get('overall_strength', 0) > 0.7:
                adjustments += 0.05
            
            # Model consensus
            if ensemble_pred.consensus_level > 0.8:
                adjustments += 0.05
            
            # Market session (higher accuracy during active sessions)
            if market_context.get('session') in ['london', 'ny', 'overlap']:
                adjustments += 0.03
            
            # Volume confirmation
            if market_context.get('volume_profile') in ['above_average', 'high']:
                adjustments += 0.02
            
            # Alternative data alignment
            sentiment_alignment = self._check_sentiment_alignment(sentiment_features)
            if sentiment_alignment:
                adjustments += 0.03
            
            predicted_accuracy = base_accuracy + adjustments
            
            return min(predicted_accuracy, 0.98)  # Cap at 98%
            
        except Exception as e:
            self.logger.error(f"Error predicting signal accuracy: {e}")
            return 0.75
    
    def _check_sentiment_alignment(self, sentiment_features: Dict[str, float]) -> bool:
        """Check if sentiment factors are aligned"""
        try:
            news_sentiment = sentiment_features.get('news_sentiment_avg', 0)
            social_sentiment = sentiment_features.get('social_sentiment_avg', 0)
            
            # Check if sentiments are aligned (same direction and not conflicting)
            if abs(news_sentiment) > 0.1 and abs(social_sentiment) > 0.1:
                return (news_sentiment > 0) == (social_sentiment > 0)
            
            return True  # Neutral is considered aligned
            
        except Exception as e:
            self.logger.error(f"Error checking sentiment alignment: {e}")
            return False
    
    async def _store_enhanced_signal(self, signal: EnhancedSignal, quality_assessment: Dict[str, Any]):
        """Store enhanced signal in database"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO enhanced_signals 
                (timestamp, symbol, direction, confidence, accuracy_prediction,
                 entry_time, expiry_time, expiry_duration, entry_price,
                 signal_strength, technical_strength, quality_score,
                 quality_grade, recommendation, ensemble_data, alternative_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(TIMEZONE).isoformat(),
                signal.symbol,
                signal.direction,
                signal.confidence,
                signal.accuracy_prediction,
                signal.entry_time.isoformat(),
                signal.expiry_time.isoformat(),
                signal.expiry_duration,
                signal.entry_price,
                signal.signal_strength,
                signal.technical_strength,
                quality_assessment.get('quality_score', 0),
                quality_assessment.get('quality_grade', 'N/A'),
                quality_assessment.get('recommendation', 'HOLD'),
                json.dumps({
                    'final_confidence': signal.ensemble_prediction.final_confidence,
                    'consensus_level': signal.ensemble_prediction.consensus_level,
                    'individual_scores': signal.individual_model_scores
                }),
                json.dumps({
                    'news_sentiment': signal.news_sentiment_score,
                    'social_sentiment': signal.social_sentiment_score,
                    'economic_impact': signal.economic_impact_score
                })
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing enhanced signal: {e}")
    
    async def scan_for_signals(self, symbols: List[str] = None) -> List[EnhancedSignal]:
        """Scan multiple symbols for high-quality signals"""
        try:
            if symbols is None:
                symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'XAU/USD']
            
            self.logger.info(f"Scanning {len(symbols)} symbols for signals")
            
            signals = []
            
            # Process symbols concurrently
            tasks = []
            for symbol in symbols:
                task = self.generate_enhanced_signal(symbol, force_signal=False)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, EnhancedSignal):
                    signals.append(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"Error generating signal: {result}")
            
            # Sort by signal strength
            signals.sort(key=lambda x: x.signal_strength, reverse=True)
            
            self.logger.info(f"Found {len(signals)} high-quality signals")
            return signals
            
        except Exception as e:
            self.logger.error(f"Error scanning for signals: {e}")
            return []
    
    def format_signal_for_telegram(self, signal: EnhancedSignal) -> str:
        """Format enhanced signal for Telegram display"""
        try:
            # Determine emojis based on direction and quality
            direction_emoji = "🟢" if signal.direction == "BUY" else "🔴"
            quality_emoji = "⭐⭐⭐" if signal.signal_strength >= 8 else "⭐⭐" if signal.signal_strength >= 6 else "⭐"
            
            # Format expiry time
            expiry_str = signal.expiry_time.strftime("%H:%M")
            entry_str = signal.entry_time.strftime("%H:%M")
            
            message = f"""
{direction_emoji} **ENHANCED SIGNAL** {quality_emoji}

**Currency Pair:** {signal.symbol}
**Direction:** {signal.direction}
**Entry Time:** {entry_str}
**Expiry:** {expiry_str} ({signal.expiry_duration}min)

**📊 ANALYSIS:**
• AI Confidence: {signal.confidence:.1%}
• Signal Strength: {signal.signal_strength:.1f}/10
• Predicted Accuracy: {signal.accuracy_prediction:.1%}
• Risk Level: {signal.risk_level}

**🔍 TECHNICAL:**
• Trend: {signal.trend_alignment.title()}
• Volatility: {signal.volatility_level.title()}
• Technical Strength: {signal.technical_strength:.1%}

**📰 MARKET CONTEXT:**
• Session: {signal.market_session.title()}
• Volume: {signal.volume_profile.title()}
• News Sentiment: {signal.news_sentiment_score:+.2f}

**💰 RECOMMENDATION:**
• Position Size: {signal.position_size_recommendation:.1%} of balance
• Execution: {signal.execution_urgency.replace('_', ' ').title()}

**🤖 ENSEMBLE ANALYSIS:**
• Model Consensus: {signal.ensemble_prediction.consensus_level:.1%}
• Processing Time: {signal.ensemble_prediction.processing_time:.2f}s

*Signal generated by Enhanced AI Trading System*
            """.strip()
            
            return message
            
        except Exception as e:
            self.logger.error(f"Error formatting signal for Telegram: {e}")
            return f"Signal: {signal.symbol} {signal.direction} at {signal.confidence:.1%} confidence"