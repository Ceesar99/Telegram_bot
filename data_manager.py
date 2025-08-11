import asyncio
import aiohttp
import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
from datetime import datetime, timedelta
import logging
import sqlite3
import json
import time
from typing import Dict, List, Optional, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor
import requests
from dataclasses import dataclass
import pytz

from config import DATABASE_CONFIG, TIMEZONE

@dataclass
class MarketData:
    """Structured market data container"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float = None
    ask: float = None
    spread: float = None
    vwap: float = None
    tick_count: int = None

@dataclass
class OrderBookData:
    """Order book data structure"""
    symbol: str
    timestamp: datetime
    bids: List[Tuple[float, float]]  # [(price, volume), ...]
    asks: List[Tuple[float, float]]
    bid_depth: float
    ask_depth: float
    imbalance: float
    
class DataQualityValidator:
    """Validates data quality and detects anomalies"""
    
    def __init__(self):
        self.logger = logging.getLogger('DataQualityValidator')
        
    def validate_candle(self, candle: MarketData) -> bool:
        """Validate individual candle data"""
        try:
            # Basic sanity checks
            if candle.high < candle.low:
                self.logger.warning(f"Invalid OHLC: high < low for {candle.symbol}")
                return False
                
            if candle.open < 0 or candle.close < 0:
                self.logger.warning(f"Negative prices for {candle.symbol}")
                return False
                
            if candle.volume < 0:
                self.logger.warning(f"Negative volume for {candle.symbol}")
                return False
                
            # Check for extreme price movements (>20% in 1 minute)
            if abs((candle.close - candle.open) / candle.open) > 0.20:
                self.logger.warning(f"Extreme price movement for {candle.symbol}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating candle: {e}")
            return False
    
    def detect_gaps(self, data: pd.DataFrame, symbol: str) -> List[datetime]:
        """Detect missing data gaps"""
        gaps = []
        data_sorted = data.sort_index()
        
        for i in range(1, len(data_sorted)):
            time_diff = (data_sorted.index[i] - data_sorted.index[i-1]).total_seconds()
            if time_diff > 120:  # More than 2 minutes gap
                gaps.append(data_sorted.index[i-1])
                
        if gaps:
            self.logger.warning(f"Found {len(gaps)} data gaps for {symbol}")
            
        return gaps
    
    def detect_outliers(self, data: pd.DataFrame, column: str = 'close') -> pd.Series:
        """Detect price outliers using IQR method"""
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
        
        if outliers.sum() > 0:
            self.logger.warning(f"Found {outliers.sum()} outliers in {column}")
            
        return outliers

class MultiSourceDataProvider:
    """Aggregates data from multiple sources for redundancy"""
    
    def __init__(self):
        self.logger = logging.getLogger('MultiSourceDataProvider')
        self.sources = {}
        self.quality_validator = DataQualityValidator()
        
        # Initialize data sources
        self._init_data_sources()
        
    def _init_data_sources(self):
        """Initialize various data sources"""
        try:
            # Yahoo Finance for backup historical data
            self.sources['yahoo'] = {
                'active': True,
                'priority': 3,
                'type': 'historical'
            }
            
            # Alpha Vantage (if API key available)
            self.sources['alpha_vantage'] = {
                'active': False,  # Requires API key
                'priority': 2,
                'type': 'real_time'
            }
            
            # Twelve Data (if API key available)
            self.sources['twelve_data'] = {
                'active': False,  # Requires API key
                'priority': 1,
                'type': 'real_time'
            }
            
            # CCXT for crypto data
            try:
                import ccxt
                self.sources['binance'] = {
                    'active': True,
                    'priority': 2,
                    'type': 'crypto',
                    'exchange': ccxt.binance()
                }
            except:
                pass
                
        except Exception as e:
            self.logger.error(f"Error initializing data sources: {e}")
    
    async def get_historical_data(self, symbol: str, period: str = "1y", 
                                interval: str = "1m") -> Optional[pd.DataFrame]:
        """Get historical data with fallback sources"""
        
        # Try primary sources first
        for source_name in sorted(self.sources.keys(), 
                                key=lambda x: self.sources[x]['priority']):
            
            if not self.sources[source_name]['active']:
                continue
                
            try:
                data = await self._fetch_from_source(source_name, symbol, period, interval)
                
                if data is not None and len(data) > 0:
                    # Validate data quality
                    if self._validate_dataset(data, symbol):
                        self.logger.info(f"Successfully fetched data from {source_name}")
                        return data
                        
            except Exception as e:
                self.logger.warning(f"Failed to fetch from {source_name}: {e}")
                continue
        
        self.logger.error(f"Failed to fetch data for {symbol} from all sources")
        return None
    
    async def _fetch_from_source(self, source: str, symbol: str, 
                                period: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch data from specific source"""
        
        if source == 'yahoo':
            return await self._fetch_yahoo_data(symbol, period, interval)
        elif source == 'binance' and symbol.endswith('USDT'):
            return await self._fetch_binance_data(symbol, period, interval)
        elif source == 'alpha_vantage':
            return await self._fetch_alpha_vantage_data(symbol, interval)
        
        return None
    
    async def _fetch_yahoo_data(self, symbol: str, period: str, 
                               interval: str) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance"""
        try:
            # Convert symbol format for Yahoo Finance
            yahoo_symbol = self._convert_symbol_to_yahoo(symbol)
            
            # Use asyncio to run in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                ticker = yf.Ticker(yahoo_symbol)
                data = await loop.run_in_executor(
                    executor, 
                    ticker.history, 
                    period, 
                    interval
                )
            
            if data.empty:
                return None
                
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            
            # Add derived fields
            data['vwap'] = (data['volume'] * data['close']).cumsum() / data['volume'].cumsum()
            data['spread'] = np.nan  # Not available from Yahoo
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching Yahoo data: {e}")
            return None
    
    async def _fetch_binance_data(self, symbol: str, period: str, 
                                 interval: str) -> Optional[pd.DataFrame]:
        """Fetch data from Binance"""
        try:
            exchange = self.sources['binance']['exchange']
            
            # Convert timeframe
            timeframe = self._convert_interval_to_ccxt(interval)
            
            # Calculate start time
            since = self._get_since_timestamp(period)
            
            # Fetch data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
            
            if not ohlcv:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add derived fields
            df['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching Binance data: {e}")
            return None
    
    def _convert_symbol_to_yahoo(self, symbol: str) -> str:
        """Convert symbol to Yahoo Finance format"""
        symbol_map = {
            'EUR/USD': 'EURUSD=X',
            'GBP/USD': 'GBPUSD=X',
            'USD/JPY': 'USDJPY=X',
            'AUD/USD': 'AUDUSD=X',
            'USD/CAD': 'USDCAD=X',
            'USD/CHF': 'USDCHF=X',
            'NZD/USD': 'NZDUSD=X',
            'BTC/USD': 'BTC-USD',
            'ETH/USD': 'ETH-USD',
            'XAU/USD': 'GC=F',
            'XAG/USD': 'SI=F',
            'OIL/USD': 'CL=F'
        }
        
        return symbol_map.get(symbol, symbol)
    
    def _convert_interval_to_ccxt(self, interval: str) -> str:
        """Convert interval to CCXT format"""
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '1h': '1h',
            '1d': '1d'
        }
        return interval_map.get(interval, '1m')
    
    def _get_since_timestamp(self, period: str) -> int:
        """Get timestamp for period start"""
        now = datetime.now()
        
        if period == '1d':
            start = now - timedelta(days=1)
        elif period == '7d':
            start = now - timedelta(days=7)
        elif period == '1mo':
            start = now - timedelta(days=30)
        elif period == '1y':
            start = now - timedelta(days=365)
        else:
            start = now - timedelta(days=365)
            
        return int(start.timestamp() * 1000)
    
    def _validate_dataset(self, data: pd.DataFrame, symbol: str) -> bool:
        """Validate entire dataset"""
        try:
            if data.empty:
                return False
            
            # Check for minimum data points
            if len(data) < 100:
                self.logger.warning(f"Insufficient data points for {symbol}")
                return False
            
            # Check for gaps
            gaps = self.quality_validator.detect_gaps(data, symbol)
            
            # Check for outliers
            outliers = self.quality_validator.detect_outliers(data)
            
            # Accept data if gaps and outliers are within acceptable limits
            gap_ratio = len(gaps) / len(data)
            outlier_ratio = outliers.sum() / len(data)
            
            if gap_ratio > 0.05:  # More than 5% gaps
                self.logger.warning(f"Too many gaps in data for {symbol}")
                return False
            
            if outlier_ratio > 0.02:  # More than 2% outliers
                self.logger.warning(f"Too many outliers in data for {symbol}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating dataset: {e}")
            return False

class RealTimeDataStream:
    """Manages real-time data streaming"""
    
    def __init__(self):
        self.logger = logging.getLogger('RealTimeDataStream')
        self.subscribers = {}
        self.is_streaming = False
        self.stream_tasks = []
        
    async def subscribe(self, symbol: str, callback):
        """Subscribe to real-time data for a symbol"""
        if symbol not in self.subscribers:
            self.subscribers[symbol] = []
        
        self.subscribers[symbol].append(callback)
        
        # Start streaming if not already started
        if not self.is_streaming:
            await self.start_streaming()
    
    async def start_streaming(self):
        """Start real-time data streaming"""
        self.is_streaming = True
        
        # Create streaming tasks for each subscribed symbol
        for symbol in self.subscribers.keys():
            task = asyncio.create_task(self._stream_symbol_data(symbol))
            self.stream_tasks.append(task)
    
    async def _stream_symbol_data(self, symbol: str):
        """Stream data for a specific symbol"""
        while self.is_streaming:
            try:
                # Simulate real-time data (replace with actual WebSocket/API calls)
                data = await self._fetch_current_data(symbol)
                
                if data and symbol in self.subscribers:
                    for callback in self.subscribers[symbol]:
                        try:
                            await callback(symbol, data)
                        except Exception as e:
                            self.logger.error(f"Error in callback for {symbol}: {e}")
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                self.logger.error(f"Error streaming data for {symbol}: {e}")
                await asyncio.sleep(5)  # Wait before retry
    
    async def _fetch_current_data(self, symbol: str) -> Optional[MarketData]:
        """Fetch current market data"""
        try:
            # This would be replaced with actual real-time data source
            # For now, simulate with random walk
            
            # Get last known price
            last_price = getattr(self, f'_last_price_{symbol}', 1.0)
            
            # Random walk simulation
            change = np.random.normal(0, 0.0001)
            new_price = last_price * (1 + change)
            
            # Update last price
            setattr(self, f'_last_price_{symbol}', new_price)
            
            # Create market data
            now = datetime.now(TIMEZONE)
            
            return MarketData(
                symbol=symbol,
                timestamp=now,
                open=new_price,
                high=new_price * 1.0001,
                low=new_price * 0.9999,
                close=new_price,
                volume=np.random.randint(1000, 10000),
                bid=new_price * 0.99999,
                ask=new_price * 1.00001,
                spread=(new_price * 1.00001) - (new_price * 0.99999),
                vwap=new_price
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching current data for {symbol}: {e}")
            return None

class DataManager:
    """Main data management class"""
    
    def __init__(self):
        self.logger = logging.getLogger('DataManager')
        self.provider = MultiSourceDataProvider()
        self.stream = RealTimeDataStream()
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize data storage database"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            # Market data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    bid REAL,
                    ask REAL,
                    spread REAL,
                    vwap REAL,
                    UNIQUE(symbol, timestamp)
                )
            ''')
            
            # Order book data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS order_book_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    bids TEXT,
                    asks TEXT,
                    bid_depth REAL,
                    ask_depth REAL,
                    imbalance REAL,
                    UNIQUE(symbol, timestamp)
                )
            ''')
            
            # Data quality metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    total_points INTEGER,
                    missing_points INTEGER,
                    outliers INTEGER,
                    gaps INTEGER,
                    quality_score REAL,
                    UNIQUE(symbol, date)
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
    
    async def get_historical_data(self, symbol: str, period: str = "1y", 
                                interval: str = "1m", force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """Get historical data with caching"""
        
        cache_key = f"{symbol}_{period}_{interval}"
        
        # Check cache first
        if not force_refresh and cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if (datetime.now() - cache_time).total_seconds() < self.cache_ttl:
                return data
        
        # Fetch fresh data
        data = await self.provider.get_historical_data(symbol, period, interval)
        
        if data is not None:
            # Cache the data
            self.cache[cache_key] = (datetime.now(), data)
            
            # Store in database
            await self._store_market_data(symbol, data)
        
        return data
    
    async def _store_market_data(self, symbol: str, data: pd.DataFrame):
        """Store market data in database"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            
            for timestamp, row in data.iterrows():
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO market_data 
                    (symbol, timestamp, open, high, low, close, volume, vwap)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    timestamp.isoformat(),
                    row.get('open', 0),
                    row.get('high', 0),
                    row.get('low', 0),
                    row.get('close', 0),
                    row.get('volume', 0),
                    row.get('vwap', 0)
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing market data: {e}")
    
    async def subscribe_real_time(self, symbol: str, callback):
        """Subscribe to real-time data updates"""
        await self.stream.subscribe(symbol, callback)
    
    def get_data_quality_report(self, symbol: str, days: int = 30) -> Dict:
        """Get data quality report for a symbol"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            # Get data quality metrics
            cursor.execute('''
                SELECT * FROM data_quality 
                WHERE symbol = ? AND date >= date('now', '-{} days')
                ORDER BY date DESC
            '''.format(days), (symbol,))
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return {"error": "No quality data available"}
            
            # Calculate summary statistics
            total_points = sum(row[3] for row in results)
            total_missing = sum(row[4] for row in results)
            total_outliers = sum(row[5] for row in results)
            total_gaps = sum(row[6] for row in results)
            avg_quality = sum(row[7] for row in results) / len(results)
            
            return {
                "symbol": symbol,
                "period_days": days,
                "total_data_points": total_points,
                "missing_points": total_missing,
                "outliers": total_outliers,
                "gaps": total_gaps,
                "average_quality_score": avg_quality,
                "data_completeness": (total_points - total_missing) / total_points if total_points > 0 else 0,
                "outlier_ratio": total_outliers / total_points if total_points > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting quality report: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old market data"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
            
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM market_data WHERE timestamp < ?', (cutoff_date,))
            cursor.execute('DELETE FROM order_book_data WHERE timestamp < ?', (cutoff_date,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleaned up {deleted_count} old data records")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up data: {e}")
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            cursor.execute('SELECT DISTINCT symbol FROM market_data ORDER BY symbol')
            symbols = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return symbols
            
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []
    
    def generate_sample_training_data(self, symbol: str = "EUR/USD", days: int = 30):
        """Generate sample training data when no historical data is available"""
        try:
            import numpy as np
            import pandas as pd
            from datetime import datetime, timedelta
            
            # Generate sample OHLCV data
            np.random.seed(42)  # For reproducible data
            
            # Start from 30 days ago
            start_date = datetime.now() - timedelta(days=days)
            dates = pd.date_range(start=start_date, end=datetime.now(), freq='1H')
            
            # Generate realistic price movements
            base_price = 1.1000  # Starting price for EUR/USD
            
            prices = []
            for i in range(len(dates)):
                # Add some randomness and trend
                change = np.random.normal(0, 0.0005)  # Small random change
                trend = 0.0001 * np.sin(i / 24)  # Daily trend
                base_price += change + trend
                
                # Generate OHLC from base price
                high = base_price + abs(np.random.normal(0, 0.0003))
                low = base_price - abs(np.random.normal(0, 0.0003))
                open_price = base_price + np.random.normal(0, 0.0002)
                close_price = base_price + np.random.normal(0, 0.0002)
                volume = np.random.uniform(1000, 10000)
                
                prices.append({
                    'timestamp': dates[i],
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': volume
                })
            
            df = pd.DataFrame(prices)
            df.set_index('timestamp', inplace=True)
            
            # Save sample data
            sample_file = f"/workspace/data/sample_{symbol.replace('/', '_')}_data.csv"
            df.to_csv(sample_file)
            
            self.logger.info(f"Generated sample training data for {symbol}: {len(df)} records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating sample training data: {e}")
            return None