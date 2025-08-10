import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sqlite3
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from dataclasses import dataclass
import pytz
import redis
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

from institutional_config import (
    DATA_PROVIDERS, DATA_QUALITY, DATABASE_CONFIG_INSTITUTIONAL, 
    GLOBAL_TIMEZONES, MARKET_SESSIONS
)

@dataclass
class ProfessionalMarketData:
    """Enhanced market data structure for institutional use"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    vwap: Optional[float] = None
    tick_count: Optional[int] = None
    
    # Level II data
    bid_depth: Optional[float] = None
    ask_depth: Optional[float] = None
    order_book: Optional[Dict] = None
    
    # Quality metrics
    source: str = ""
    latency_ms: Optional[float] = None
    quality_score: Optional[float] = None
    
    # Derived metrics
    returns: Optional[float] = None
    volatility: Optional[float] = None
    market_impact: Optional[float] = None

@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics"""
    source: str
    symbol: str
    timestamp: datetime
    latency_ms: float
    completeness: float  # 0-1
    accuracy: float      # 0-1
    consistency: float   # 0-1
    timeliness: float    # 0-1
    overall_score: float # 0-1

class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(f"DataProvider.{self.__class__.__name__}")
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config.get('timeout', 10)))
        
    @abstractmethod
    async def fetch_market_data(self, symbol: str, timeframe: str) -> Optional[ProfessionalMarketData]:
        pass
        
    @abstractmethod
    async def fetch_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        pass
        
    async def close(self):
        await self.session.close()

class PolygonDataProvider(DataProvider):
    """Polygon.io professional data provider"""
    
    async def fetch_market_data(self, symbol: str, timeframe: str = '1min') -> Optional[ProfessionalMarketData]:
        try:
            start_time = time.time()
            
            # Convert symbol format for Polygon
            polygon_symbol = self._convert_symbol_format(symbol)
            
            url = f"{self.config['endpoint']}/aggs/ticker/{polygon_symbol}/range/1/{timeframe}/2024-01-01/2024-12-31"
            params = {
                'apikey': self.config['api_key'],
                'limit': 1,
                'sort': 'desc'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('results'):
                        result = data['results'][0]
                        latency = (time.time() - start_time) * 1000
                        
                        return ProfessionalMarketData(
                            symbol=symbol,
                            timestamp=datetime.fromtimestamp(result['t'] / 1000, tz=pytz.UTC),
                            open=result['o'],
                            high=result['h'],
                            low=result['l'],
                            close=result['c'],
                            volume=result['v'],
                            vwap=result.get('vw'),
                            tick_count=result.get('n'),
                            source='polygon',
                            latency_ms=latency,
                            quality_score=self._calculate_quality_score(result, latency)
                        )
        except Exception as e:
            self.logger.error(f"Error fetching Polygon data for {symbol}: {e}")
            return None
    
    async def fetch_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        try:
            polygon_symbol = self._convert_symbol_format(symbol)
            
            url = f"{self.config['endpoint']}/aggs/ticker/{polygon_symbol}/range/1/minute/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {
                'apikey': self.config['api_key'],
                'limit': 50000
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('results'):
                        df = pd.DataFrame(data['results'])
                        df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
                        df.set_index('timestamp', inplace=True)
                        df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
                        df['source'] = 'polygon'
                        return df
                        
        except Exception as e:
            self.logger.error(f"Error fetching Polygon historical data: {e}")
            return None
    
    def _convert_symbol_format(self, symbol: str) -> str:
        """Convert symbol to Polygon format"""
        if '/' in symbol:
            base, quote = symbol.split('/')
            return f"C:{base}{quote}"
        return symbol
    
    def _calculate_quality_score(self, data: Dict, latency: float) -> float:
        """Calculate data quality score"""
        score = 1.0
        
        # Latency penalty
        if latency > 1000:  # > 1 second
            score -= 0.3
        elif latency > 500:  # > 500ms
            score -= 0.1
            
        # Data completeness
        required_fields = ['o', 'h', 'l', 'c', 'v']
        missing_fields = sum(1 for field in required_fields if field not in data or data[field] is None)
        score -= missing_fields * 0.1
        
        return max(0.0, score)

class AlphaVantageDataProvider(DataProvider):
    """Alpha Vantage professional data provider"""
    
    async def fetch_market_data(self, symbol: str, timeframe: str = '1min') -> Optional[ProfessionalMarketData]:
        try:
            start_time = time.time()
            
            url = self.config['endpoint']
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': timeframe,
                'apikey': self.config['api_key'],
                'outputsize': 'compact'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    time_series_key = f'Time Series ({timeframe})'
                    if time_series_key in data:
                        latest_timestamp = max(data[time_series_key].keys())
                        latest_data = data[time_series_key][latest_timestamp]
                        
                        latency = (time.time() - start_time) * 1000
                        
                        return ProfessionalMarketData(
                            symbol=symbol,
                            timestamp=datetime.strptime(latest_timestamp, '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.UTC),
                            open=float(latest_data['1. open']),
                            high=float(latest_data['2. high']),
                            low=float(latest_data['3. low']),
                            close=float(latest_data['4. close']),
                            volume=float(latest_data['5. volume']),
                            source='alpha_vantage',
                            latency_ms=latency,
                            quality_score=self._calculate_quality_score(latest_data, latency)
                        )
        except Exception as e:
            self.logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
            return None
    
    async def fetch_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        try:
            url = self.config['endpoint']
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': '1min',
                'apikey': self.config['api_key'],
                'outputsize': 'full'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    time_series_key = 'Time Series (1min)'
                    if time_series_key in data:
                        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
                        df.index = pd.to_datetime(df.index, utc=True)
                        df = df.rename(columns={
                            '1. open': 'open',
                            '2. high': 'high',
                            '3. low': 'low',
                            '4. close': 'close',
                            '5. volume': 'volume'
                        })
                        df = df.astype(float)
                        df['source'] = 'alpha_vantage'
                        df = df.sort_index()
                        
                        # Filter by date range
                        mask = (df.index >= start_date) & (df.index <= end_date)
                        return df.loc[mask]
                        
        except Exception as e:
            self.logger.error(f"Error fetching Alpha Vantage historical data: {e}")
            return None
    
    def _calculate_quality_score(self, data: Dict, latency: float) -> float:
        """Calculate data quality score"""
        score = 1.0
        
        # Latency penalty
        if latency > 2000:  # > 2 seconds
            score -= 0.3
        elif latency > 1000:  # > 1 second
            score -= 0.1
            
        # Data completeness
        required_fields = ['1. open', '2. high', '3. low', '4. close', '5. volume']
        missing_fields = sum(1 for field in required_fields if field not in data)
        score -= missing_fields * 0.1
        
        return max(0.0, score)

class IEXCloudDataProvider(DataProvider):
    """IEX Cloud professional data provider"""
    
    async def fetch_market_data(self, symbol: str, timeframe: str = '1m') -> Optional[ProfessionalMarketData]:
        try:
            start_time = time.time()
            
            url = f"{self.config['endpoint']}/stock/{symbol}/quote"
            params = {'token': self.config['api_key']}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    latency = (time.time() - start_time) * 1000
                    
                    return ProfessionalMarketData(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(data['latestUpdate'] / 1000, tz=pytz.UTC),
                        open=data.get('open', data['latestPrice']),
                        high=data.get('high', data['latestPrice']),
                        low=data.get('low', data['latestPrice']),
                        close=data['latestPrice'],
                        volume=data.get('volume', 0),
                        bid=data.get('iexBidPrice'),
                        ask=data.get('iexAskPrice'),
                        source='iex_cloud',
                        latency_ms=latency,
                        quality_score=self._calculate_quality_score(data, latency)
                    )
        except Exception as e:
            self.logger.error(f"Error fetching IEX Cloud data for {symbol}: {e}")
            return None
    
    async def fetch_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        try:
            # IEX Cloud uses different endpoint for historical data
            period = self._calculate_period(start_date, end_date)
            
            url = f"{self.config['endpoint']}/stock/{symbol}/chart/{period}"
            params = {'token': self.config['api_key']}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data:
                        df = pd.DataFrame(data)
                        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df.get('minute', '09:30'), utc=True)
                        df.set_index('timestamp', inplace=True)
                        df = df[['open', 'high', 'low', 'close', 'volume']]
                        df['source'] = 'iex_cloud'
                        return df
                        
        except Exception as e:
            self.logger.error(f"Error fetching IEX Cloud historical data: {e}")
            return None
    
    def _calculate_period(self, start_date: datetime, end_date: datetime) -> str:
        """Calculate IEX period string"""
        days = (end_date - start_date).days
        if days <= 1:
            return '1d'
        elif days <= 5:
            return '5d'
        elif days <= 30:
            return '1m'
        elif days <= 90:
            return '3m'
        elif days <= 180:
            return '6m'
        elif days <= 365:
            return '1y'
        else:
            return '2y'
    
    def _calculate_quality_score(self, data: Dict, latency: float) -> float:
        """Calculate data quality score"""
        score = 1.0
        
        # Latency penalty
        if latency > 1000:
            score -= 0.2
        elif latency > 500:
            score -= 0.1
            
        # Data freshness (IEX provides real-time data)
        if 'latestUpdate' in data:
            age_seconds = (time.time() * 1000 - data['latestUpdate']) / 1000
            if age_seconds > 60:  # > 1 minute old
                score -= 0.2
                
        return max(0.0, score)

class DataQualityValidator:
    """Advanced data quality validation and monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger('DataQualityValidator')
        self.quality_thresholds = DATA_QUALITY['validation_rules']
        
    def validate_data_point(self, data: ProfessionalMarketData, previous_data: Optional[ProfessionalMarketData] = None) -> DataQualityMetrics:
        """Comprehensive data quality validation"""
        
        quality_scores = {
            'completeness': self._check_completeness(data),
            'accuracy': self._check_accuracy(data, previous_data),
            'consistency': self._check_consistency(data, previous_data),
            'timeliness': self._check_timeliness(data)
        }
        
        overall_score = np.mean(list(quality_scores.values()))
        
        return DataQualityMetrics(
            source=data.source,
            symbol=data.symbol,
            timestamp=data.timestamp,
            latency_ms=data.latency_ms or 0,
            completeness=quality_scores['completeness'],
            accuracy=quality_scores['accuracy'],
            consistency=quality_scores['consistency'],
            timeliness=quality_scores['timeliness'],
            overall_score=overall_score
        )
    
    def _check_completeness(self, data: ProfessionalMarketData) -> float:
        """Check data completeness"""
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        present_fields = sum(1 for field in required_fields if getattr(data, field) is not None)
        return present_fields / len(required_fields)
    
    def _check_accuracy(self, data: ProfessionalMarketData, previous_data: Optional[ProfessionalMarketData]) -> float:
        """Check data accuracy"""
        score = 1.0
        
        # Basic sanity checks
        if data.high < data.low:
            score -= 0.5
        if data.close < 0:
            score -= 0.3
        if data.volume < 0:
            score -= 0.2
            
        # Price movement validation
        if previous_data:
            price_change = abs((data.close - previous_data.close) / previous_data.close)
            if price_change > self.quality_thresholds['price_movement_threshold']:
                score -= 0.3
                
        return max(0.0, score)
    
    def _check_consistency(self, data: ProfessionalMarketData, previous_data: Optional[ProfessionalMarketData]) -> float:
        """Check data consistency"""
        score = 1.0
        
        # Volume spike detection
        if previous_data and previous_data.volume > 0:
            volume_ratio = data.volume / previous_data.volume
            if volume_ratio > self.quality_thresholds['volume_spike_threshold']:
                score -= 0.2
                
        # Bid-ask spread validation
        if data.bid and data.ask:
            spread = (data.ask - data.bid) / data.close
            if spread > self.quality_thresholds['bid_ask_spread_threshold']:
                score -= 0.1
                
        return max(0.0, score)
    
    def _check_timeliness(self, data: ProfessionalMarketData) -> float:
        """Check data timeliness"""
        if not data.latency_ms:
            return 0.8  # Default score if latency not available
            
        latency_target = DATA_QUALITY['latency_requirements']['market_data']
        
        if data.latency_ms <= latency_target:
            return 1.0
        elif data.latency_ms <= latency_target * 2:
            return 0.8
        elif data.latency_ms <= latency_target * 5:
            return 0.5
        else:
            return 0.2

class ProfessionalDataManager:
    """Institutional-grade data management with redundancy and quality control"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.providers: Dict[str, DataProvider] = {}
        self.cache = self._setup_cache()
        self.quality_validator = DataQualityValidator()
        self.failover_active = False
        
        # Initialize providers
        self._initialize_providers()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('ProfessionalDataManager')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('/workspace/logs/professional_data_manager.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def _setup_cache(self) -> Optional[redis.Redis]:
        """Setup Redis cache for high-frequency data access"""
        try:
            cache_config = DATABASE_CONFIG_INSTITUTIONAL['cache']['redis']
            return redis.Redis(
                host=cache_config['host'],
                port=cache_config['port'],
                db=cache_config['db'],
                password=cache_config.get('password'),
                decode_responses=True
            )
        except Exception as e:
            self.logger.warning(f"Redis cache not available: {e}")
            return None
    
    def _initialize_providers(self):
        """Initialize all enabled data providers"""
        for provider_name, config in DATA_PROVIDERS.items():
            if config.get('enabled', False):
                try:
                    if provider_name == 'polygon':
                        self.providers[provider_name] = PolygonDataProvider(config)
                    elif provider_name == 'alpha_vantage':
                        self.providers[provider_name] = AlphaVantageDataProvider(config)
                    elif provider_name == 'iex_cloud':
                        self.providers[provider_name] = IEXCloudDataProvider(config)
                    # Add more providers as needed
                    
                    self.logger.info(f"Initialized {provider_name} data provider")
                except Exception as e:
                    self.logger.error(f"Failed to initialize {provider_name}: {e}")
    
    async def get_real_time_data(self, symbol: str, max_sources: int = 3) -> Optional[ProfessionalMarketData]:
        """Get real-time data with automatic failover and quality validation"""
        
        # Check cache first
        cached_data = self._get_from_cache(symbol)
        if cached_data:
            return cached_data
        
        # Try providers in priority order
        providers_tried = 0
        best_data = None
        best_quality = 0.0
        
        sorted_providers = sorted(
            [(name, provider) for name, provider in self.providers.items()], 
            key=lambda x: DATA_PROVIDERS[x[0]]['priority']
        )
        
        tasks = []
        for provider_name, provider in sorted_providers[:max_sources]:
            task = asyncio.create_task(
                self._fetch_with_timeout(provider, symbol, provider_name)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete or timeout
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Evaluate results and select best quality data
        for result in results:
            if isinstance(result, ProfessionalMarketData):
                quality_metrics = self.quality_validator.validate_data_point(result)
                
                if quality_metrics.overall_score > best_quality:
                    best_quality = quality_metrics.overall_score
                    best_data = result
                    
                # Log quality metrics
                self.logger.info(f"Data quality for {symbol} from {result.source}: {quality_metrics.overall_score:.3f}")
        
        if best_data and best_quality > 0.5:  # Minimum quality threshold
            # Cache the result
            self._cache_data(symbol, best_data)
            return best_data
        else:
            self.logger.warning(f"No quality data available for {symbol}")
            return None
    
    async def _fetch_with_timeout(self, provider: DataProvider, symbol: str, provider_name: str) -> Optional[ProfessionalMarketData]:
        """Fetch data with timeout handling"""
        try:
            timeout = DATA_PROVIDERS[provider_name]['timeout']
            return await asyncio.wait_for(
                provider.fetch_market_data(symbol), 
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout fetching data from {provider_name} for {symbol}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching from {provider_name}: {e}")
            return None
    
    def _get_from_cache(self, symbol: str) -> Optional[ProfessionalMarketData]:
        """Get data from cache if available and fresh"""
        if not self.cache:
            return None
            
        try:
            cached_json = self.cache.get(f"market_data:{symbol}")
            if cached_json:
                data_dict = json.loads(cached_json)
                # Convert back to ProfessionalMarketData object
                data_dict['timestamp'] = datetime.fromisoformat(data_dict['timestamp'])
                return ProfessionalMarketData(**data_dict)
        except Exception as e:
            self.logger.warning(f"Error reading from cache: {e}")
        
        return None
    
    def _cache_data(self, symbol: str, data: ProfessionalMarketData):
        """Cache data with expiration"""
        if not self.cache:
            return
            
        try:
            # Convert to JSON-serializable format
            data_dict = {
                'symbol': data.symbol,
                'timestamp': data.timestamp.isoformat(),
                'open': data.open,
                'high': data.high,
                'low': data.low,
                'close': data.close,
                'volume': data.volume,
                'bid': data.bid,
                'ask': data.ask,
                'spread': data.spread,
                'vwap': data.vwap,
                'source': data.source,
                'latency_ms': data.latency_ms,
                'quality_score': data.quality_score
            }
            
            cache_key = f"market_data:{symbol}"
            cache_ttl = DATABASE_CONFIG_INSTITUTIONAL['cache']['redis']['ttl']
            
            self.cache.setex(
                cache_key, 
                cache_ttl, 
                json.dumps(data_dict, default=str)
            )
        except Exception as e:
            self.logger.warning(f"Error caching data: {e}")
    
    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Get historical data with provider failover"""
        
        sorted_providers = sorted(
            [(name, provider) for name, provider in self.providers.items()], 
            key=lambda x: DATA_PROVIDERS[x[0]]['priority']
        )
        
        for provider_name, provider in sorted_providers:
            try:
                data = await provider.fetch_historical_data(symbol, start_date, end_date)
                if data is not None and len(data) > 0:
                    self.logger.info(f"Successfully fetched historical data from {provider_name}")
                    return data
            except Exception as e:
                self.logger.warning(f"Failed to fetch historical data from {provider_name}: {e}")
                continue
        
        self.logger.error(f"Failed to fetch historical data for {symbol} from all providers")
        return None
    
    async def close(self):
        """Cleanup resources"""
        for provider in self.providers.values():
            await provider.close()
        
        if self.cache:
            self.cache.close()

# Example usage and testing
async def main():
    """Test the professional data manager"""
    data_manager = ProfessionalDataManager()
    
    try:
        # Test real-time data
        symbols = ['AAPL', 'EURUSD', 'BTCUSD']
        
        for symbol in symbols:
            print(f"\nFetching real-time data for {symbol}...")
            data = await data_manager.get_real_time_data(symbol)
            
            if data:
                print(f"  Source: {data.source}")
                print(f"  Price: {data.close}")
                print(f"  Volume: {data.volume}")
                print(f"  Latency: {data.latency_ms}ms")
                print(f"  Quality: {data.quality_score}")
            else:
                print(f"  No data available")
        
        # Test historical data
        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=1)
        
        print(f"\nFetching historical data for AAPL...")
        historical = await data_manager.get_historical_data('AAPL', start_date, end_date)
        
        if historical is not None:
            print(f"  Records: {len(historical)}")
            print(f"  Date range: {historical.index[0]} to {historical.index[-1]}")
        else:
            print("  No historical data available")
            
    finally:
        await data_manager.close()

if __name__ == "__main__":
    asyncio.run(main())