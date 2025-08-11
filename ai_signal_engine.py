#!/usr/bin/env python3
"""
AI Signal Engine - Enhanced Version
Integrates the trained Binary Options LSTM AI model with the existing Telegram bot system
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys
from typing import Dict, List, Optional, Tuple
import time

# Add workspace to path
sys.path.append('/workspace')

# Import our AI components
from binary_options_ai_model import BinaryOptionsAIModel, create_realistic_market_data
from pocket_option_enhanced_api import EnhancedPocketOptionAPI, TradingSignal

class AISignalEngine:
    def __init__(self):
        self.logger = logging.getLogger('AISignalEngine')
        self.setup_logging()
        
        # Initialize AI model
        self.ai_model = BinaryOptionsAIModel()
        self.api = EnhancedPocketOptionAPI(demo_mode=True)
        
        # Signal tracking
        self.last_signals = {}
        self.signal_cache = {}
        self.market_conditions = {}
        
        # Configuration
        self.config = {
            'min_confidence': 60.0,           # Minimum confidence for signal
            'signal_cooldown': 300,           # 5 minutes between signals per pair
            'supported_pairs': [
                'EURUSD_OTC', 'GBPUSD_OTC', 'USDJPY_OTC', 'AUDUSD_OTC',
                'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD'
            ],
            'fallback_data_length': 120       # Fallback data points for demo
        }
        
        # Initialize model
        self._initialize_model()
    
    def setup_logging(self):
        """Setup logging configuration"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _initialize_model(self):
        """Initialize the AI model"""
        try:
            if self.ai_model.is_model_trained():
                success = self.ai_model.load_model()
                if success:
                    self.logger.info("✅ AI model loaded successfully")
                else:
                    self.logger.error("❌ Failed to load AI model")
            else:
                self.logger.warning("⚠️ No pre-trained model found")
        except Exception as e:
            self.logger.error(f"Error initializing AI model: {e}")
    
    async def generate_signal(self, pair: str = None) -> Optional[Dict]:
        """Generate AI-powered trading signal"""
        try:
            # Select best pair if none specified
            if not pair:
                pair = await self._select_best_pair()
            
            # Check cooldown
            if self._is_in_cooldown(pair):
                self.logger.debug(f"Pair {pair} is in cooldown")
                return None
            
            # Get market data
            market_data = await self._get_market_data(pair)
            if market_data is None or len(market_data) < 60:
                self.logger.warning(f"Insufficient market data for {pair}")
                return None
            
            # Generate AI signal
            ai_signal = self.ai_model.predict_signal(market_data)
            
            if ai_signal['signal'] in ['CALL', 'PUT'] and ai_signal['confidence'] >= self.config['min_confidence']:
                # Create enhanced signal with additional analysis
                signal = await self._create_enhanced_signal(pair, ai_signal, market_data)
                
                # Update cooldown
                self.last_signals[pair] = time.time()
                
                # Cache signal
                self.signal_cache[pair] = signal
                
                self.logger.info(f"✅ Generated signal for {pair}: {signal['direction']} ({signal['accuracy']:.1f}%)")
                return signal
            else:
                self.logger.debug(f"Low confidence signal for {pair}: {ai_signal['confidence']:.1f}%")
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return None
    
    async def _select_best_pair(self) -> str:
        """Select the best currency pair for signal generation"""
        # Check each supported pair for signal opportunities
        best_pair = None
        best_confidence = 0
        
        for pair in self.config['supported_pairs']:
            if self._is_in_cooldown(pair):
                continue
            
            try:
                market_data = await self._get_market_data(pair)
                if market_data is None or len(market_data) < 60:
                    continue
                
                # Quick confidence check
                ai_signal = self.ai_model.predict_signal(market_data)
                if ai_signal['confidence'] > best_confidence and ai_signal['signal'] in ['CALL', 'PUT']:
                    best_confidence = ai_signal['confidence']
                    best_pair = pair
            except Exception as e:
                self.logger.debug(f"Error checking pair {pair}: {e}")
                continue
        
        return best_pair or self.config['supported_pairs'][0]  # Fallback to first pair
    
    async def _get_market_data(self, pair: str) -> Optional[pd.DataFrame]:
        """Get market data for the specified pair"""
        try:
            # Try to get real data from API first
            if hasattr(self.api, 'get_candle_data'):
                data = await self.api.get_candle_data(pair, timeframe='1m', count=120)
                if not data.empty:
                    return data
            
            # Fallback to simulated data for demo purposes
            self.logger.debug(f"Using simulated data for {pair}")
            return create_realistic_market_data(self.config['fallback_data_length'])
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {pair}: {e}")
            return None
    
    async def _create_enhanced_signal(self, pair: str, ai_signal: Dict, market_data: pd.DataFrame) -> Dict:
        """Create enhanced signal with additional technical analysis"""
        try:
            # Calculate additional technical indicators
            technical_analysis = self._perform_technical_analysis(market_data)
            
            # Determine signal strength
            strength = self._calculate_signal_strength(ai_signal, technical_analysis)
            
            # Calculate volatility-based expiry
            volatility = self._calculate_volatility(market_data)
            expiry = self._determine_expiry(ai_signal['confidence'], volatility)
            
            # Create comprehensive signal
            signal = {
                'pair': pair,
                'direction': ai_signal['direction'],
                'accuracy': ai_signal['confidence'],
                'ai_confidence': ai_signal['confidence'],
                'time_expiry': f"{expiry}m",
                'expiry_minutes': expiry,
                'signal_time': datetime.now().strftime('%H:%M:%S'),
                'entry_price': self._get_current_price(market_data),
                'strength': min(10, max(1, int(strength))),
                'volatility_level': self._get_volatility_level(volatility),
                'trend': technical_analysis.get('trend', 'Neutral'),
                'risk_level': self._calculate_risk_level(ai_signal['confidence'], volatility),
                'probabilities': ai_signal.get('probabilities', {}),
                'technical_indicators': technical_analysis,
                'signal_id': f"{pair}_{int(time.time())}",
                'generated_at': datetime.now().isoformat()
            }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error creating enhanced signal: {e}")
            # Return basic signal as fallback
            return {
                'pair': pair,
                'direction': ai_signal['direction'],
                'accuracy': ai_signal['confidence'],
                'ai_confidence': ai_signal['confidence'],
                'time_expiry': '3m',
                'expiry_minutes': 3,
                'signal_time': datetime.now().strftime('%H:%M:%S'),
                'entry_price': 'N/A',
                'strength': 7,
                'volatility_level': 'Medium',
                'trend': 'Neutral',
                'risk_level': 'Medium'
            }
    
    def _perform_technical_analysis(self, data: pd.DataFrame) -> Dict:
        """Perform additional technical analysis"""
        try:
            # Use the AI model's feature creation for consistency
            features = self.ai_model.create_features(data)
            
            # Extract key indicators from the last values
            latest = features.iloc[-1]
            
            # Determine trend
            if latest['price_vs_sma_20'] > 0.002:
                trend = 'Strong Bullish'
            elif latest['price_vs_sma_20'] > 0:
                trend = 'Bullish'
            elif latest['price_vs_sma_20'] < -0.002:
                trend = 'Strong Bearish'
            elif latest['price_vs_sma_20'] < 0:
                trend = 'Bearish'
            else:
                trend = 'Neutral'
            
            # RSI signal
            rsi = latest['rsi']
            if rsi > 70:
                rsi_signal = 'Overbought'
            elif rsi < 30:
                rsi_signal = 'Oversold'
            else:
                rsi_signal = 'Neutral'
            
            # MACD signal
            macd = latest['macd']
            macd_signal = 'Bullish' if macd > 0 else 'Bearish'
            
            # Bollinger Bands position
            bb_pos = latest['bb_position']
            if bb_pos > 0.8:
                bb_signal = 'Near Upper Band'
            elif bb_pos < 0.2:
                bb_signal = 'Near Lower Band'
            else:
                bb_signal = 'Middle Range'
            
            return {
                'trend': trend,
                'rsi': f"{rsi:.1f}",
                'rsi_signal': rsi_signal,
                'macd_signal': macd_signal,
                'bb_position': bb_signal,
                'volatility': f"{latest['volatility']:.4f}",
                'momentum': f"{latest['price_momentum']:.4f}"
            }
            
        except Exception as e:
            self.logger.error(f"Error in technical analysis: {e}")
            return {
                'trend': 'Neutral',
                'rsi': 'N/A',
                'rsi_signal': 'Neutral',
                'macd_signal': 'Neutral',
                'bb_position': 'Middle Range'
            }
    
    def _calculate_signal_strength(self, ai_signal: Dict, technical_analysis: Dict) -> float:
        """Calculate overall signal strength"""
        try:
            base_strength = ai_signal['confidence'] / 10  # Scale confidence to 1-10
            
            # Add technical analysis bonus
            trend_bonus = 0
            if technical_analysis.get('trend', '').startswith('Strong'):
                trend_bonus = 1.5
            elif technical_analysis.get('trend', '') in ['Bullish', 'Bearish']:
                trend_bonus = 1.0
            
            # RSI bonus
            rsi_bonus = 0
            if technical_analysis.get('rsi_signal') in ['Overbought', 'Oversold']:
                rsi_bonus = 0.5
            
            total_strength = base_strength + trend_bonus + rsi_bonus
            return min(10, max(1, total_strength))
            
        except Exception as e:
            return ai_signal['confidence'] / 10
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate current market volatility"""
        try:
            returns = data['close'].pct_change().dropna()
            return returns.tail(20).std()
        except Exception:
            return 0.01  # Default volatility
    
    def _determine_expiry(self, confidence: float, volatility: float) -> int:
        """Determine optimal expiry time based on confidence and volatility"""
        if confidence > 80 and volatility > 0.02:
            return 2  # High confidence, high volatility - quick trade
        elif confidence > 70:
            return 3  # Medium-high confidence
        elif volatility < 0.005:
            return 5  # Low volatility - longer expiry
        else:
            return 3  # Default expiry
    
    def _get_current_price(self, data: pd.DataFrame) -> str:
        """Get current price from market data"""
        try:
            return f"{data['close'].iloc[-1]:.5f}"
        except Exception:
            return 'N/A'
    
    def _get_volatility_level(self, volatility: float) -> str:
        """Get volatility level description"""
        if volatility > 0.02:
            return 'High'
        elif volatility > 0.01:
            return 'Medium'
        else:
            return 'Low'
    
    def _calculate_risk_level(self, confidence: float, volatility: float) -> str:
        """Calculate risk level for the signal"""
        if confidence > 75 and volatility < 0.015:
            return 'Low'
        elif confidence > 65 and volatility < 0.02:
            return 'Medium'
        else:
            return 'High'
    
    def _is_in_cooldown(self, pair: str) -> bool:
        """Check if pair is in cooldown period"""
        if pair not in self.last_signals:
            return False
        
        time_since_last = time.time() - self.last_signals[pair]
        return time_since_last < self.config['signal_cooldown']
    
    # Methods for compatibility with existing Telegram bot
    
    def get_available_pairs(self) -> List[str]:
        """Get list of available trading pairs"""
        return self.config['supported_pairs']
    
    def get_market_status(self) -> Dict:
        """Get current market status"""
        current_time = datetime.now()
        
        # Determine market session
        hour = current_time.hour
        if 0 <= hour < 6:
            session = 'Asian Session'
        elif 6 <= hour < 14:
            session = 'European Session'
        elif 14 <= hour < 22:
            session = 'American Session'
        else:
            session = 'Overlap Session'
        
        return {
            'session': session,
            'is_open': True,  # Binary options trade 24/7
            'volatility': 'Medium',
            'signal_quality': 'High',
            'active_pairs': len(self.config['supported_pairs']),
            'risk_level': 'Medium',
            'position_size': 'Standard',
            'next_event': 'Market continues 24/7'
        }
    
    async def analyze_pair(self, pair: str) -> Optional[Dict]:
        """Analyze specific currency pair"""
        try:
            # Get market data
            market_data = await self._get_market_data(pair)
            if market_data is None or len(market_data) < 60:
                return None
            
            # Generate AI signal
            ai_signal = self.ai_model.predict_signal(market_data)
            
            # Perform technical analysis
            technical_analysis = self._perform_technical_analysis(market_data)
            
            # Calculate additional metrics
            current_price = float(market_data['close'].iloc[-1])
            price_change = ((current_price - float(market_data['close'].iloc[-2])) / float(market_data['close'].iloc[-2])) * 100
            
            analysis = {
                'current_price': f"{current_price:.5f}",
                'price_change': f"{price_change:+.3f}",
                'volatility': technical_analysis.get('volatility', 'N/A'),
                'rsi': technical_analysis.get('rsi', 'N/A'),
                'rsi_signal': technical_analysis.get('rsi_signal', 'Neutral'),
                'macd_signal': technical_analysis.get('macd_signal', 'Neutral'),
                'bb_position': technical_analysis.get('bb_position', 'N/A'),
                'support': f"{market_data['low'].tail(20).min():.5f}",
                'resistance': f"{market_data['high'].tail(20).max():.5f}",
                'price_position': f"{((current_price - market_data['low'].tail(20).min()) / (market_data['high'].tail(20).max() - market_data['low'].tail(20).min()) * 100):.1f}",
                'recommendation': ai_signal['direction'] if ai_signal['confidence'] > 60 else 'HOLD',
                'signal_strength': min(10, max(1, int(ai_signal['confidence'] / 10))),
                'risk_level': self._calculate_risk_level(ai_signal['confidence'], float(technical_analysis.get('volatility', '0.01')))
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing pair {pair}: {e}")
            return None
    
    def is_model_loaded(self) -> bool:
        """Check if AI model is loaded"""
        return self.ai_model.is_model_trained()
    
    def is_data_connected(self) -> bool:
        """Check if data connection is available"""
        return True  # Always true for demo mode

# Test function
async def test_ai_signal_engine():
    """Test the AI signal engine"""
    engine = AISignalEngine()
    
    print("🧪 Testing AI Signal Engine")
    print("=" * 40)
    
    # Test signal generation
    signal = await engine.generate_signal()
    if signal:
        print(f"✅ Signal Generated:")
        print(f"   Pair: {signal['pair']}")
        print(f"   Direction: {signal['direction']}")
        print(f"   Confidence: {signal['accuracy']:.1f}%")
        print(f"   Expiry: {signal['time_expiry']}")
        print(f"   Strength: {signal['strength']}/10")
    else:
        print("❌ No signal generated")
    
    # Test market status
    status = engine.get_market_status()
    print(f"\n📊 Market Status: {status['session']}")
    
    # Test pair analysis
    analysis = await engine.analyze_pair('EUR/USD')
    if analysis:
        print(f"\n📈 EUR/USD Analysis:")
        print(f"   Recommendation: {analysis['recommendation']}")
        print(f"   Strength: {analysis['signal_strength']}/10")
    
    print("\n✅ AI Signal Engine test completed!")

if __name__ == "__main__":
    asyncio.run(test_ai_signal_engine())