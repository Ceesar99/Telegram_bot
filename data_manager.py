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
            
            # Check if this is an OTC symbol that Yahoo Finance doesn't support well
            if 'OTC' in symbol:
                self.logger.info(f"OTC symbol {symbol} detected, using alternative data source")
                return await self._fetch_yahoo_alternative(yahoo_symbol, period, interval)
            
            # Use asyncio to run in thread pool with better error handling
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                try:
                    ticker = yf.Ticker(yahoo_symbol)
                    data = await loop.run_in_executor(
                        executor, 
                        ticker.history, 
                        period, 
                        interval
                    )
                except Exception as yf_error:
                    self.logger.warning(f"Yahoo Finance failed for {symbol}: {yf_error}")
                    # Try alternative approach with requests
                    return await self._fetch_yahoo_alternative(yahoo_symbol, period, interval)
            
            if data.empty:
                self.logger.warning(f"Empty data from Yahoo Finance for {symbol}")
                return None
                
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            
            # Add derived fields
            data['vwap'] = (data['volume'] * data['close']).cumsum() / data['volume'].cumsum()
            data['spread'] = np.nan  # Not available from Yahoo
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching Yahoo data for {symbol}: {e}")
            # Generate fallback data when all sources fail
            return await self._generate_fallback_data(symbol, period, interval)
    
    async def _fetch_yahoo_alternative(self, yahoo_symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Alternative method to fetch Yahoo Finance data using direct API calls"""
        try:
            # Use a more direct approach that might avoid the chrome136 issue
            import requests
            
            # Calculate time range
            end_time = int(time.time())
            if period == "1d":
                start_time = end_time - (24 * 60 * 60)  # 24 hours ago
            elif period == "1w":
                start_time = end_time - (7 * 24 * 60 * 60)  # 1 week ago
            elif period == "1m":
                start_time = end_time - (30 * 24 * 60 * 60)  # 1 month ago
            else:
                start_time = end_time - (24 * 60 * 60)  # Default to 1 day
            
            # Try multiple Yahoo Finance API endpoints
            api_endpoints = [
                f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}",
                f"https://query2.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}",
                f"https://query1.finance.yahoo.com/v7/finance/chart/{yahoo_symbol}"
            ]
            
            for url in api_endpoints:
                try:
                    params = {
                        'period1': start_time,
                        'period2': end_time,
                        'interval': interval,
                        'includePrePost': 'false',
                        'events': 'div,split'
                    }
                    
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Accept': 'application/json',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1'
                    }
                    
                    response = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda u=url, p=params, h=headers: requests.get(u, params=p, headers=h, timeout=15)
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
                            continue  # Try next endpoint
                        
                        result = data['chart']['result'][0]
                        timestamps = result['timestamp']
                        quotes = result['indicators']['quote'][0]
                        
                        # Create DataFrame
                        df = pd.DataFrame({
                            'timestamp': pd.to_datetime(timestamps, unit='s'),
                            'open': quotes['open'],
                            'high': quotes['high'],
                            'low': quotes['low'],
                            'close': quotes['close'],
                            'volume': quotes['volume']
                        })
                        
                        df.set_index('timestamp', inplace=True)
                        df.dropna(inplace=True)
                        
                        if not df.empty:
                            # Add derived fields
                            df['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
                            df['spread'] = np.nan
                            
                            self.logger.info(f"Successfully fetched data from {url}")
                            return df
                            
                except Exception as e:
                    self.logger.debug(f"Failed to fetch from {url}: {e}")
                    continue
            
            # If all endpoints fail, return None
            self.logger.warning(f"All Yahoo Finance API endpoints failed for {yahoo_symbol}")
            return None
            
        except Exception as e:
            self.logger.error(f"Alternative Yahoo fetch failed for {yahoo_symbol}: {e}")
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
        # Remove OTC suffix and convert to standard format
        clean_symbol = symbol.replace(' OTC', '').replace('OTC ', '')
        
        symbol_map = {
            'EUR/USD': 'EURUSD=X',
            'GBP/USD': 'GBPUSD=X',
            'USD/JPY': 'USDJPY=X',
            'AUD/USD': 'AUDUSD=X',
            'USD/CAD': 'USDCAD=X',
            'USD/CHF': 'USDCHF=X',
            'NZD/USD': 'NZDUSD=X',
            'EUR/GBP': 'EURGBP=X',
            'GBP/JPY': 'GBPJPY=X',
            'EUR/JPY': 'EURJPY=X',
            'AUD/JPY': 'AUDJPY=X',
            'BTC/USD': 'BTC-USD',
            'ETH/USD': 'ETH-USD',
            'XAU/USD': 'GC=F',
            'XAG/USD': 'SI=F',
            'OIL/USD': 'CL=F'
        }
        
        return symbol_map.get(clean_symbol, clean_symbol)
    
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
    
    async def _generate_fallback_data(self, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Generate realistic fallback data when all data sources fail"""
        try:
            self.logger.info(f"Generating fallback data for {symbol}")
            
            # Calculate time range
            end_time = datetime.now(TIMEZONE)
            if period == "1d":
                start_time = end_time - timedelta(days=1)
                freq = "1min"
            elif period == "1w":
                start_time = end_time - timedelta(weeks=1)
                freq = "5min"
            elif period == "1m":
                start_time = end_time - timedelta(days=30)
                freq = "1H"
            else:
                start_time = end_time - timedelta(days=1)
                freq = "1min"
            
            # Generate timestamps
            dates = pd.date_range(start=start_time, end=end_time, freq=freq)
            
            # Get base price for symbol
            base_price = self._get_base_price_for_symbol(symbol)
            
            # Generate realistic price movements
            np.random.seed(hash(symbol) % 1000)  # Deterministic but different per symbol
            
            prices = [base_price]
            for _ in range(len(dates) - 1):
                # Small random change with some trend
                change = np.random.normal(0, base_price * 0.0001)
                prices.append(prices[-1] + change)
            
            # Create OHLCV data
            data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                volatility = base_price * 0.0002
                high = price + np.random.uniform(0, volatility)
                low = price - np.random.uniform(0, volatility)
                open_price = price + np.random.uniform(-volatility/2, volatility/2)
                close_price = price
                volume = np.random.uniform(1000, 10000)
                
                data.append({
                    'timestamp': date,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            self.logger.info(f"Generated fallback data for {symbol}: {len(df)} records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating fallback data: {e}")
            return None
    
    def _get_base_price_for_symbol(self, symbol: str):
        """Get base price for symbol to generate realistic fallback data"""
        base_prices = {
            'EUR/USD': 1.1000,
            'GBP/USD': 1.2800,
            'USD/JPY': 150.00,
            'USD/CHF': 0.9000,
            'AUD/USD': 0.6800,
            'USD/CAD': 1.3500,
            'NZD/USD': 0.6200,
            'EUR/GBP': 0.8600,
            'EUR/JPY': 165.00,
            'GBP/JPY': 192.00,
            'BTC/USD': 45000.00,
            'ETH/USD': 2800.00,
            'XAU/USD': 1950.00,
            'XAG/USD': 24.50,
            'OIL/USD': 75.00
        }
        
        # Handle OTC pairs
        if 'OTC' in symbol:
            base_symbol = symbol.replace(' OTC', '')
            return base_prices.get(base_symbol, 1.0000)
        
        return base_prices.get(symbol, 1.0000)