import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import talib
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from config import TIMEZONE
from data_manager import DataManager

class MarketRegimeDetector:
    """Detects market regimes using multiple methods"""
    
    def __init__(self):
        self.logger = logging.getLogger('MarketRegimeDetector')
        self.regime_model = None
        self.scaler = StandardScaler()
        
    def detect_volatility_regimes(self, data: pd.DataFrame, lookback: int = 50) -> pd.Series:
        """Detect volatility regimes (high/low volatility periods)"""
        try:
            # Calculate rolling volatility
            returns = data['close'].pct_change()
            volatility = returns.rolling(window=lookback).std()
            
            # Use quantiles to classify regimes
            low_vol_threshold = volatility.quantile(0.33)
            high_vol_threshold = volatility.quantile(0.67)
            
            regimes = pd.Series(index=data.index, dtype=int)
            regimes[volatility <= low_vol_threshold] = 0  # Low volatility
            regimes[(volatility > low_vol_threshold) & (volatility <= high_vol_threshold)] = 1  # Medium
            regimes[volatility > high_vol_threshold] = 2  # High volatility
            
            return regimes.fillna(1)
            
        except Exception as e:
            self.logger.error(f"Error detecting volatility regimes: {e}")
            return pd.Series(index=data.index, data=1)
    
    def detect_trend_regimes(self, data: pd.DataFrame, lookback: int = 50) -> pd.Series:
        """Detect trend regimes (bull/bear/sideways)"""
        try:
            close = data['close']
            
            # Calculate trend indicators
            sma_short = close.rolling(window=20).mean()
            sma_long = close.rolling(window=50).mean()
            
            # Trend strength
            trend_strength = (sma_short - sma_long) / sma_long
            
            # Classify regimes
            bull_threshold = trend_strength.quantile(0.67)
            bear_threshold = trend_strength.quantile(0.33)
            
            regimes = pd.Series(index=data.index, dtype=int)
            regimes[trend_strength >= bull_threshold] = 2  # Bull market
            regimes[(trend_strength < bull_threshold) & (trend_strength > bear_threshold)] = 1  # Sideways
            regimes[trend_strength <= bear_threshold] = 0  # Bear market
            
            return regimes.fillna(1)
            
        except Exception as e:
            self.logger.error(f"Error detecting trend regimes: {e}")
            return pd.Series(index=data.index, data=1)
    
    def detect_momentum_regimes(self, data: pd.DataFrame, lookback: int = 14) -> pd.Series:
        """Detect momentum regimes"""
        try:
            close = data['close']
            
            # Calculate RSI and rate of change
            rsi = talib.RSI(close.values, timeperiod=lookback)
            roc = talib.ROC(close.values, timeperiod=lookback)
            
            # Combine momentum indicators
            momentum_score = (rsi - 50) / 50 + np.tanh(roc * 100)
            
            # Classify regimes
            high_momentum_threshold = np.nanquantile(momentum_score, 0.67)
            low_momentum_threshold = np.nanquantile(momentum_score, 0.33)
            
            regimes = pd.Series(index=data.index, dtype=int)
            regimes[momentum_score >= high_momentum_threshold] = 2  # High momentum
            regimes[(momentum_score < high_momentum_threshold) & (momentum_score > low_momentum_threshold)] = 1  # Medium
            regimes[momentum_score <= low_momentum_threshold] = 0  # Low momentum
            
            return pd.Series(regimes).fillna(1)
            
        except Exception as e:
            self.logger.error(f"Error detecting momentum regimes: {e}")
            return pd.Series(index=data.index, data=1)

class VolumeProfileAnalyzer:
    """Analyzes volume profile and market microstructure"""
    
    def __init__(self):
        self.logger = logging.getLogger('VolumeProfileAnalyzer')
    
    def calculate_vwap_bands(self, data: pd.DataFrame, periods: int = 20) -> Dict[str, pd.Series]:
        """Calculate VWAP and bands"""
        try:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            cumulative_tpv = (typical_price * data['volume']).cumsum()
            cumulative_volume = data['volume'].cumsum()
            vwap = cumulative_tpv / cumulative_volume
            
            # Rolling VWAP for bands
            rolling_vwap = (typical_price * data['volume']).rolling(periods).sum() / data['volume'].rolling(periods).sum()
            
            # Calculate standard deviation bands
            price_diff = typical_price - rolling_vwap
            variance = (price_diff ** 2 * data['volume']).rolling(periods).sum() / data['volume'].rolling(periods).sum()
            std_dev = np.sqrt(variance)
            
            return {
                'vwap': vwap,
                'rolling_vwap': rolling_vwap,
                'upper_band_1': rolling_vwap + std_dev,
                'lower_band_1': rolling_vwap - std_dev,
                'upper_band_2': rolling_vwap + 2 * std_dev,
                'lower_band_2': rolling_vwap - 2 * std_dev
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating VWAP bands: {e}")
            return {}
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate advanced volume indicators"""
        try:
            volume = data['volume']
            close = data['close']
            high = data['high']
            low = data['low']
            
            # On-Balance Volume
            obv = talib.OBV(close.values, volume.values)
            
            # Accumulation/Distribution Line
            ad_line = talib.AD(high.values, low.values, close.values, volume.values)
            
            # Money Flow Index
            mfi = talib.MFI(high.values, low.values, close.values, volume.values, timeperiod=14)
            
            # Volume Rate of Change
            volume_roc = talib.ROC(volume.values, timeperiod=10)
            
            # Volume moving averages
            volume_sma_10 = volume.rolling(10).mean()
            volume_sma_20 = volume.rolling(20).mean()
            
            # Volume relative to average
            volume_ratio = volume / volume_sma_20
            
            return {
                'obv': pd.Series(obv, index=data.index),
                'ad_line': pd.Series(ad_line, index=data.index),
                'mfi': pd.Series(mfi, index=data.index),
                'volume_roc': pd.Series(volume_roc, index=data.index),
                'volume_ratio': volume_ratio,
                'volume_sma_10': volume_sma_10,
                'volume_sma_20': volume_sma_20
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volume indicators: {e}")
            return {}
    
    def detect_volume_anomalies(self, data: pd.DataFrame, threshold: float = 2.0) -> pd.Series:
        """Detect volume anomalies"""
        try:
            volume = data['volume']
            volume_zscore = stats.zscore(volume.dropna())
            
            anomalies = pd.Series(index=data.index, data=False)
            anomalies[volume.dropna().index] = np.abs(volume_zscore) > threshold
            
            return anomalies.fillna(False)
            
        except Exception as e:
            self.logger.error(f"Error detecting volume anomalies: {e}")
            return pd.Series(index=data.index, data=False)

class CrossAssetCorrelationAnalyzer:
    """Analyzes correlations across different assets"""
    
    def __init__(self):
        self.logger = logging.getLogger('CrossAssetCorrelationAnalyzer')
        self.data_manager = DataManager()
        self.correlation_cache = {}
        
    async def get_correlation_features(self, symbol: str, data: pd.DataFrame, 
                                     lookback: int = 50) -> Dict[str, float]:
        """Calculate correlation features with other assets"""
        try:
            # Define correlation assets based on the main symbol
            correlation_assets = await self._get_correlation_assets(symbol)
            
            features = {}
            
            for asset in correlation_assets:
                try:
                    # Get correlation asset data
                    asset_data = await self.data_manager.get_historical_data(
                        asset, period="1mo", interval="1m"
                    )
                    
                    if asset_data is not None and len(asset_data) > lookback:
                        # Calculate rolling correlation
                        correlation = self._calculate_rolling_correlation(
                            data['close'], asset_data['close'], lookback
                        )
                        
                        if correlation is not None:
                            features[f'corr_{asset.replace("/", "_")}'] = correlation
                            
                            # Correlation change
                            prev_correlation = self._calculate_rolling_correlation(
                                data['close'], asset_data['close'], lookback, lag=10
                            )
                            
                            if prev_correlation is not None:
                                features[f'corr_change_{asset.replace("/", "_")}'] = correlation - prev_correlation
                                
                except Exception as e:
                    self.logger.warning(f"Error calculating correlation with {asset}: {e}")
                    continue
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error getting correlation features: {e}")
            return {}
    
    async def _get_correlation_assets(self, symbol: str) -> List[str]:
        """Get relevant assets for correlation analysis"""
        
        # Base correlation assets for all symbols
        base_assets = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'XAU/USD']
        
        # Add specific assets based on symbol type
        if 'USD' in symbol:
            base_assets.extend(['DXY', 'XAU/USD', 'BTC/USD'])
        elif 'EUR' in symbol:
            base_assets.extend(['EUR/GBP', 'EUR/JPY'])
        elif 'GBP' in symbol:
            base_assets.extend(['GBP/JPY', 'GBP/CHF'])
        
        # Remove the symbol itself if present
        return [asset for asset in base_assets if asset != symbol]
    
    def _calculate_rolling_correlation(self, series1: pd.Series, series2: pd.Series, 
                                     window: int, lag: int = 0) -> Optional[float]:
        """Calculate rolling correlation between two series"""
        try:
            # Align series by timestamp
            combined = pd.concat([series1, series2], axis=1, join='inner')
            combined.columns = ['series1', 'series2']
            
            if len(combined) < window + lag:
                return None
            
            # Apply lag if specified
            if lag > 0:
                combined = combined.iloc[:-lag]
            
            # Calculate correlation
            correlation = combined['series1'].tail(window).corr(combined['series2'].tail(window))
            
            return float(correlation) if not np.isnan(correlation) else None
            
        except Exception as e:
            self.logger.warning(f"Error calculating correlation: {e}")
            return None

class SeasonalPatternAnalyzer:
    """Analyzes seasonal and time-based patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger('SeasonalPatternAnalyzer')
    
    def extract_time_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Extract time-based features"""
        try:
            # Ensure index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            features = {}
            
            # Basic time features
            features['hour'] = data.index.hour
            features['day_of_week'] = data.index.dayofweek
            features['day_of_month'] = data.index.day
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
            
            # Cyclical encoding
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
            
            features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
            
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
            
            # Market session indicators
            features['asian_session'] = ((features['hour'] >= 0) & (features['hour'] < 8)).astype(int)
            features['london_session'] = ((features['hour'] >= 8) & (features['hour'] < 16)).astype(int)
            features['ny_session'] = ((features['hour'] >= 13) & (features['hour'] < 21)).astype(int)
            features['overlap_london_ny'] = ((features['hour'] >= 13) & (features['hour'] < 16)).astype(int)
            
            # Weekend/weekday
            features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting time features: {e}")
            return {}
    
    def calculate_seasonal_strength(self, data: pd.DataFrame, periods: List[int] = None) -> Dict[str, float]:
        """Calculate seasonal strength for different periods"""
        try:
            if periods is None:
                periods = [24, 168, 720]  # Hourly, weekly, monthly patterns
            
            seasonal_strengths = {}
            returns = data['close'].pct_change().dropna()
            
            for period in periods:
                if len(returns) >= period * 3:  # Need at least 3 cycles
                    # Group by period and calculate variance
                    period_groups = returns.groupby(returns.index.hour if period == 24 else 
                                                   returns.index.dayofweek if period == 168 else
                                                   returns.index.day).var()
                    
                    # Seasonal strength is the variance of group means relative to overall variance
                    seasonal_strength = period_groups.var() / returns.var()
                    seasonal_strengths[f'seasonal_strength_{period}'] = float(seasonal_strength)
            
            return seasonal_strengths
            
        except Exception as e:
            self.logger.error(f"Error calculating seasonal strength: {e}")
            return {}

class OrderBookFeatureExtractor:
    """Extracts features from order book data (when available)"""
    
    def __init__(self):
        self.logger = logging.getLogger('OrderBookFeatureExtractor')
    
    def calculate_spread_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate bid-ask spread features"""
        try:
            features = {}
            
            if 'bid' in data.columns and 'ask' in data.columns:
                # Basic spread
                spread = data['ask'] - data['bid']
                features['spread'] = spread
                features['spread_pct'] = spread / data['close'] * 100
                
                # Rolling spread statistics
                features['spread_ma'] = spread.rolling(20).mean()
                features['spread_std'] = spread.rolling(20).std()
                features['spread_zscore'] = (spread - features['spread_ma']) / features['spread_std']
                
                # Mid price
                mid_price = (data['bid'] + data['ask']) / 2
                features['mid_price'] = mid_price
                features['mid_close_diff'] = data['close'] - mid_price
                
            else:
                # Estimate spread from high-low if bid-ask not available
                estimated_spread = data['high'] - data['low']
                features['estimated_spread'] = estimated_spread
                features['estimated_spread_pct'] = estimated_spread / data['close'] * 100
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating spread features: {e}")
            return {}
    
    def calculate_imbalance_features(self, order_book_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Calculate order book imbalance features"""
        try:
            if order_book_data is None:
                return {}
            
            features = {}
            
            # Order imbalance
            if 'bid_depth' in order_book_data.columns and 'ask_depth' in order_book_data.columns:
                total_depth = order_book_data['bid_depth'] + order_book_data['ask_depth']
                imbalance = (order_book_data['bid_depth'] - order_book_data['ask_depth']) / total_depth
                
                features['order_imbalance'] = float(imbalance.iloc[-1]) if len(imbalance) > 0 else 0.0
                features['order_imbalance_ma'] = float(imbalance.rolling(10).mean().iloc[-1]) if len(imbalance) >= 10 else 0.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating imbalance features: {e}")
            return {}

class VolatilityClusteringAnalyzer:
    """Analyzes volatility clustering and GARCH effects"""
    
    def __init__(self):
        self.logger = logging.getLogger('VolatilityClusteringAnalyzer')
    
    def calculate_volatility_features(self, data: pd.DataFrame, windows: List[int] = None) -> Dict[str, pd.Series]:
        """Calculate various volatility measures"""
        try:
            if windows is None:
                windows = [5, 10, 20, 50]
            
            features = {}
            returns = data['close'].pct_change()
            
            for window in windows:
                # Historical volatility
                vol = returns.rolling(window).std() * np.sqrt(252 * 24 * 60)  # Annualized
                features[f'volatility_{window}'] = vol
                
                # Parkinson volatility (high-low)
                parkinson_vol = np.sqrt(
                    (1 / (4 * np.log(2))) * 
                    (np.log(data['high'] / data['low']) ** 2).rolling(window).mean()
                ) * np.sqrt(252 * 24 * 60)
                features[f'parkinson_vol_{window}'] = parkinson_vol
                
                # Garman-Klass volatility
                gk_vol = np.sqrt(
                    (np.log(data['high'] / data['low']) ** 2).rolling(window).mean() / 2 -
                    (2 * np.log(2) - 1) * (np.log(data['close'] / data['open']) ** 2).rolling(window).mean()
                ) * np.sqrt(252 * 24 * 60)
                features[f'gk_vol_{window}'] = gk_vol
            
            # Volatility of volatility
            base_vol = returns.rolling(20).std()
            features['vol_of_vol'] = base_vol.rolling(20).std()
            
            # Volatility skew
            features['vol_skew'] = returns.rolling(50).skew()
            features['vol_kurtosis'] = returns.rolling(50).kurtosis()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility features: {e}")
            return {}
    
    def detect_volatility_clusters(self, data: pd.DataFrame, threshold: float = 1.5) -> pd.Series:
        """Detect volatility clustering periods"""
        try:
            returns = data['close'].pct_change()
            vol = returns.rolling(20).std()
            
            # Standardize volatility
            vol_zscore = stats.zscore(vol.dropna())
            
            # Detect clusters (periods of high volatility persistence)
            high_vol_periods = pd.Series(index=data.index, data=False)
            high_vol_periods[vol.dropna().index] = vol_zscore > threshold
            
            return high_vol_periods.fillna(False)
            
        except Exception as e:
            self.logger.error(f"Error detecting volatility clusters: {e}")
            return pd.Series(index=data.index, data=False)

class AdvancedFeatureEngine:
    """Main feature engineering engine that combines all analyzers"""
    
    def __init__(self):
        self.logger = logging.getLogger('AdvancedFeatureEngine')
        
        # Initialize analyzers
        self.regime_detector = MarketRegimeDetector()
        self.volume_analyzer = VolumeProfileAnalyzer()
        self.correlation_analyzer = CrossAssetCorrelationAnalyzer()
        self.seasonal_analyzer = SeasonalPatternAnalyzer()
        self.orderbook_extractor = OrderBookFeatureExtractor()
        self.volatility_analyzer = VolatilityClusteringAnalyzer()
        
        self.scaler = StandardScaler()
        
    def generate_all_features(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Generate all advanced features"""
        try:
            self.logger.info("Starting advanced feature generation...")
            
            # Start with original data
            features_df = data.copy()
            
            # Basic technical indicators (from original LSTM model)
            features_df.update(self._calculate_basic_indicators(data))
            
            # Market regime features
            features_df['volatility_regime'] = self.regime_detector.detect_volatility_regimes(data)
            features_df['trend_regime'] = self.regime_detector.detect_trend_regimes(data)
            features_df['momentum_regime'] = self.regime_detector.detect_momentum_regimes(data)
            
            # Volume features
            volume_features = self.volume_analyzer.calculate_volume_indicators(data)
            for name, series in volume_features.items():
                features_df[name] = series
            
            # VWAP features
            vwap_features = self.volume_analyzer.calculate_vwap_bands(data)
            for name, series in vwap_features.items():
                features_df[name] = series
            
            # Volume anomalies
            features_df['volume_anomaly'] = self.volume_analyzer.detect_volume_anomalies(data)
            
            # Time-based features
            time_features = self.seasonal_analyzer.extract_time_features(data)
            for name, series in time_features.items():
                features_df[name] = series
            
            # Seasonal strength
            seasonal_strengths = self.seasonal_analyzer.calculate_seasonal_strength(data)
            for name, value in seasonal_strengths.items():
                features_df[name] = value
            
            # Spread features
            spread_features = self.orderbook_extractor.calculate_spread_features(data)
            for name, series in spread_features.items():
                features_df[name] = series
            
            # Volatility features
            vol_features = self.volatility_analyzer.calculate_volatility_features(data)
            for name, series in vol_features.items():
                features_df[name] = series
            
            # Volatility clusters
            features_df['volatility_cluster'] = self.volatility_analyzer.detect_volatility_clusters(data)
            
            # Support and resistance levels
            features_df.update(self._calculate_support_resistance(data))
            
            # Price action patterns
            features_df.update(self._calculate_price_patterns(data))
            
            # Statistical features
            features_df.update(self._calculate_statistical_features(data))
            
            # Generate target variable
            features_df['target'] = self._generate_target(data)
            
            # Clean up features
            features_df = self._clean_features(features_df)
            
            self.logger.info(f"Generated {len(features_df.columns)} features")
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error generating advanced features: {e}")
            raise
    
    async def generate_realtime_features(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Generate features for real-time prediction"""
        try:
            # Generate full feature set
            full_features = self.generate_all_features(data, symbol)
            
            # Get correlation features (async)
            if symbol:
                correlation_features = await self.correlation_analyzer.get_correlation_features(
                    symbol, data
                )
                
                # Add correlation features to the latest row
                for name, value in correlation_features.items():
                    full_features.loc[full_features.index[-1], name] = value
            
            # Return the latest row as a dictionary
            latest_features = full_features.iloc[-1].to_dict()
            
            # Remove non-numeric values
            numeric_features = {k: v for k, v in latest_features.items() 
                              if isinstance(v, (int, float, np.number)) and not np.isnan(v)}
            
            return numeric_features
            
        except Exception as e:
            self.logger.error(f"Error generating real-time features: {e}")
            return {}
    
    def _calculate_basic_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate basic technical indicators"""
        try:
            features = {}
            
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            volume = data['volume'].values
            
            # Moving averages
            features['sma_5'] = pd.Series(talib.SMA(close, timeperiod=5), index=data.index)
            features['sma_10'] = pd.Series(talib.SMA(close, timeperiod=10), index=data.index)
            features['sma_20'] = pd.Series(talib.SMA(close, timeperiod=20), index=data.index)
            features['ema_5'] = pd.Series(talib.EMA(close, timeperiod=5), index=data.index)
            features['ema_10'] = pd.Series(talib.EMA(close, timeperiod=10), index=data.index)
            features['ema_20'] = pd.Series(talib.EMA(close, timeperiod=20), index=data.index)
            
            # Oscillators
            features['rsi'] = pd.Series(talib.RSI(close, timeperiod=14), index=data.index)
            features['stoch_k'], features['stoch_d'] = [
                pd.Series(x, index=data.index) for x in talib.STOCH(high, low, close)
            ]
            features['williams_r'] = pd.Series(talib.WILLR(high, low, close), index=data.index)
            features['cci'] = pd.Series(talib.CCI(high, low, close), index=data.index)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            features['macd'] = pd.Series(macd, index=data.index)
            features['macd_signal'] = pd.Series(macd_signal, index=data.index)
            features['macd_histogram'] = pd.Series(macd_hist, index=data.index)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            features['bb_upper'] = pd.Series(bb_upper, index=data.index)
            features['bb_middle'] = pd.Series(bb_middle, index=data.index)
            features['bb_lower'] = pd.Series(bb_lower, index=data.index)
            features['bb_width'] = features['bb_upper'] - features['bb_lower']
            features['bb_position'] = (data['close'] - features['bb_lower']) / features['bb_width']
            
            # ADX
            features['adx'] = pd.Series(talib.ADX(high, low, close), index=data.index)
            features['plus_di'] = pd.Series(talib.PLUS_DI(high, low, close), index=data.index)
            features['minus_di'] = pd.Series(talib.MINUS_DI(high, low, close), index=data.index)
            
            # ATR
            features['atr'] = pd.Series(talib.ATR(high, low, close), index=data.index)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating basic indicators: {e}")
            return {}
    
    def _calculate_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Dict[str, pd.Series]:
        """Calculate dynamic support and resistance levels"""
        try:
            features = {}
            
            # Rolling support and resistance
            features['support'] = data['low'].rolling(window).min()
            features['resistance'] = data['high'].rolling(window).max()
            
            # Distance to support/resistance
            features['dist_to_support'] = (data['close'] - features['support']) / data['close']
            features['dist_to_resistance'] = (features['resistance'] - data['close']) / data['close']
            
            # Pivot points
            pivot = (data['high'] + data['low'] + data['close']) / 3
            features['pivot'] = pivot
            features['r1'] = 2 * pivot - data['low']
            features['s1'] = 2 * pivot - data['high']
            features['r2'] = pivot + (data['high'] - data['low'])
            features['s2'] = pivot - (data['high'] - data['low'])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {e}")
            return {}
    
    def _calculate_price_patterns(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate price action patterns"""
        try:
            features = {}
            
            # Candlestick patterns
            close = data['close'].values
            open_prices = data['open'].values if 'open' in data.columns else close
            high = data['high'].values
            low = data['low'].values
            
            # Doji
            features['doji'] = pd.Series(talib.CDLDOJI(open_prices, high, low, close), index=data.index)
            
            # Hammer
            features['hammer'] = pd.Series(talib.CDLHAMMER(open_prices, high, low, close), index=data.index)
            
            # Engulfing patterns
            features['engulfing'] = pd.Series(talib.CDLENGULFING(open_prices, high, low, close), index=data.index)
            
            # Price momentum patterns
            returns = data['close'].pct_change()
            
            # Consecutive up/down days
            features['consecutive_up'] = (returns > 0).rolling(5).sum()
            features['consecutive_down'] = (returns < 0).rolling(5).sum()
            
            # Gap analysis
            if 'open' in data.columns:
                gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
                features['gap'] = gap
                features['gap_up'] = (gap > 0.001).astype(int)
                features['gap_down'] = (gap < -0.001).astype(int)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating price patterns: {e}")
            return {}
    
    def _calculate_statistical_features(self, data: pd.DataFrame, windows: List[int] = None) -> Dict[str, pd.Series]:
        """Calculate statistical features"""
        try:
            if windows is None:
                windows = [5, 10, 20, 50]
            
            features = {}
            returns = data['close'].pct_change()
            
            for window in windows:
                # Rolling statistics
                features[f'mean_{window}'] = returns.rolling(window).mean()
                features[f'std_{window}'] = returns.rolling(window).std()
                features[f'skew_{window}'] = returns.rolling(window).skew()
                features[f'kurt_{window}'] = returns.rolling(window).kurtosis()
                
                # Quantiles
                features[f'q25_{window}'] = returns.rolling(window).quantile(0.25)
                features[f'q75_{window}'] = returns.rolling(window).quantile(0.75)
                
                # Z-score
                mean_ret = features[f'mean_{window}']
                std_ret = features[f'std_{window}']
                features[f'zscore_{window}'] = (returns - mean_ret) / std_ret
            
            # Autocorrelation
            for lag in [1, 2, 3, 5, 10]:
                features[f'autocorr_{lag}'] = returns.rolling(50).apply(
                    lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
                )
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating statistical features: {e}")
            return {}
    
    def _generate_target(self, data: pd.DataFrame, lookahead: int = 2) -> pd.Series:
        """Generate target variable for training"""
        try:
            close_prices = data['close']
            
            # Calculate future returns
            future_returns = close_prices.shift(-lookahead) / close_prices - 1
            
            # Classify into BUY (0), SELL (1), HOLD (2)
            target = pd.Series(index=data.index, dtype=int)
            
            # Use dynamic thresholds based on volatility
            rolling_vol = close_prices.pct_change().rolling(20).std()
            buy_threshold = rolling_vol * 0.5
            sell_threshold = -rolling_vol * 0.5
            
            target[future_returns > buy_threshold] = 0  # BUY
            target[future_returns < sell_threshold] = 1  # SELL
            target[(future_returns >= sell_threshold) & (future_returns <= buy_threshold)] = 2  # HOLD
            
            return target.fillna(2)
            
        except Exception as e:
            self.logger.error(f"Error generating target: {e}")
            return pd.Series(index=data.index, data=2)
    
    def _clean_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features"""
        try:
            # Remove infinite values
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill missing values (limited)
            features_df = features_df.fillna(method='ffill', limit=5)
            
            # Drop columns with too many missing values
            threshold = 0.5 * len(features_df)
            features_df = features_df.dropna(axis=1, thresh=threshold)
            
            # Drop rows with any remaining missing values
            features_df = features_df.dropna()
            
            # Remove constant columns
            nunique = features_df.nunique()
            constant_columns = nunique[nunique == 1].index
            features_df = features_df.drop(columns=constant_columns)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error cleaning features: {e}")
            return features_df