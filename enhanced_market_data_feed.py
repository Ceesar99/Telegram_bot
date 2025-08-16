#!/usr/bin/env python3
"""
🚀 Enhanced Market Data Feed System
Real-time data integration for maximum trading accuracy
"""

import asyncio
import aiohttp
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import websockets
from dataclasses import dataclass

@dataclass
class MarketTick:
    """Real-time market tick data"""
    symbol: str
    price: float
    bid: float
    ask: float
    volume: float
    timestamp: datetime
    source: str
    spread: float = 0.0
    
    def __post_init__(self):
        self.spread = self.ask - self.bid if self.ask and self.bid else 0.0

class EnhancedMarketDataFeed:
    """
    🔥 Professional Market Data Feed System
    
    Features:
    - Multiple data sources for redundancy
    - Real-time WebSocket streaming
    - Automatic failover
    - Data quality validation
    - Latency optimization
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.logger = self._setup_logger()
        
        # Data sources
        self.data_sources = {}
        self.active_connections = {}
        self.data_buffer = {}
        
        # Quality metrics
        self.data_quality = {}
        self.latency_metrics = {}
        
        # Callbacks
        self.data_callbacks = []
        
        # Initialize sources
        self._initialize_data_sources()
    
    def _get_default_config(self) -> Dict:
        """Default configuration with API keys"""
        return {
            # API Keys (Set these with your actual keys)
            'alpha_vantage_key': 'YOUR_ALPHA_VANTAGE_KEY',
            'iex_token': 'YOUR_IEX_TOKEN', 
            'finnhub_key': 'YOUR_FINNHUB_KEY',
            'fixer_key': 'YOUR_FIXER_KEY',
            
            # Data Sources Priority
            'primary_sources': ['alpha_vantage', 'iex_cloud'],
            'backup_sources': ['finnhub', 'yahoo_finance'],
            'crypto_sources': ['coingecko', 'binance'],
            
            # Symbols to track
            'forex_pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'],
            'crypto_pairs': ['BTCUSD', 'ETHUSD', 'ADAUSD'],
            
            # Quality controls
            'max_latency_ms': 1000,  # 1 second max latency
            'min_update_frequency': 1,  # At least 1 update per second
            'price_change_threshold': 0.05,  # 5% max price change validation
            
            # Failover settings
            'connection_timeout': 10,
            'retry_attempts': 3,
            'failover_delay': 5,
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for market data feed"""
        logger = logging.getLogger('MarketDataFeed')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('/workspace/logs/market_data_feed.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _initialize_data_sources(self):
        """Initialize all data source configurations"""
        
        # Alpha Vantage Configuration
        self.data_sources['alpha_vantage'] = {
            'name': 'Alpha Vantage',
            'base_url': 'https://www.alphavantage.co/query',
            'api_key': self.config['alpha_vantage_key'],
            'rate_limit': 5,  # calls per minute
            'supports_realtime': True,
            'supports_websocket': False,
            'data_quality': 'HIGH',
            'cost': 'PAID'
        }
        
        # IEX Cloud Configuration  
        self.data_sources['iex_cloud'] = {
            'name': 'IEX Cloud',
            'base_url': 'https://cloud.iexapis.com/stable',
            'api_key': self.config['iex_token'],
            'rate_limit': 100,  # calls per second
            'supports_realtime': True,
            'supports_websocket': True,
            'data_quality': 'HIGH',
            'cost': 'FREEMIUM'
        }
        
        # Finnhub Configuration
        self.data_sources['finnhub'] = {
            'name': 'Finnhub',
            'base_url': 'https://finnhub.io/api/v1',
            'websocket_url': 'wss://ws.finnhub.io',
            'api_key': self.config['finnhub_key'],
            'rate_limit': 60,  # calls per minute
            'supports_realtime': True,
            'supports_websocket': True,
            'data_quality': 'HIGH',
            'cost': 'FREEMIUM'
        }
        
        # Yahoo Finance (Free backup)
        self.data_sources['yahoo_finance'] = {
            'name': 'Yahoo Finance',
            'base_url': 'https://query1.finance.yahoo.com/v8/finance/chart',
            'rate_limit': 2000,  # daily limit
            'supports_realtime': True,
            'supports_websocket': False,
            'data_quality': 'MEDIUM',
            'cost': 'FREE'
        }
        
        # CoinGecko for Crypto
        self.data_sources['coingecko'] = {
            'name': 'CoinGecko',
            'base_url': 'https://api.coingecko.com/api/v3',
            'rate_limit': 50,  # calls per minute
            'supports_realtime': True,
            'supports_websocket': False,
            'data_quality': 'HIGH',
            'cost': 'FREE'
        }
        
        # Binance for Crypto (Free)
        self.data_sources['binance'] = {
            'name': 'Binance',
            'base_url': 'https://api.binance.com/api/v3',
            'websocket_url': 'wss://stream.binance.com:9443/ws',
            'rate_limit': 1200,  # per minute
            'supports_realtime': True,
            'supports_websocket': True,
            'data_quality': 'HIGH',
            'cost': 'FREE'
        }
        
        self.logger.info(f"📊 Initialized {len(self.data_sources)} data sources")
    
    async def start_real_time_feed(self):
        """Start real-time data feed from multiple sources"""
        self.logger.info("🚀 Starting real-time market data feed...")
        
        tasks = []
        
        # Start primary sources
        for source_name in self.config['primary_sources']:
            if source_name in self.data_sources:
                task = asyncio.create_task(self._start_source_feed(source_name))
                tasks.append(task)
        
        # Start crypto sources if crypto pairs configured
        if self.config['crypto_pairs']:
            for source_name in self.config['crypto_sources']:
                if source_name in self.data_sources:
                    task = asyncio.create_task(self._start_crypto_feed(source_name))
                    tasks.append(task)
        
        # Start data quality monitoring
        monitor_task = asyncio.create_task(self._monitor_data_quality())
        tasks.append(monitor_task)
        
        # Wait for all tasks
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _start_source_feed(self, source_name: str):
        """Start feed from specific data source"""
        source = self.data_sources[source_name]
        
        try:
            if source.get('supports_websocket', False):
                await self._start_websocket_feed(source_name)
            else:
                await self._start_polling_feed(source_name)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to start {source_name}: {e}")
            await self._handle_source_failure(source_name)
    
    async def _start_websocket_feed(self, source_name: str):
        """Start WebSocket real-time feed"""
        source = self.data_sources[source_name]
        
        if source_name == 'finnhub':
            await self._start_finnhub_websocket()
        elif source_name == 'binance':
            await self._start_binance_websocket()
        elif source_name == 'iex_cloud':
            await self._start_iex_websocket()
    
    async def _start_finnhub_websocket(self):
        """Start Finnhub WebSocket feed"""
        uri = f"wss://ws.finnhub.io?token={self.config['finnhub_key']}"
        
        try:
            async with websockets.connect(uri) as websocket:
                self.logger.info("✅ Connected to Finnhub WebSocket")
                
                # Subscribe to forex pairs
                for pair in self.config['forex_pairs']:
                    subscribe_msg = {
                        "type": "subscribe",
                        "symbol": f"FX:{pair}"
                    }
                    await websocket.send(json.dumps(subscribe_msg))
                
                # Listen for messages
                async for message in websocket:
                    await self._process_finnhub_message(message)
                    
        except Exception as e:
            self.logger.error(f"❌ Finnhub WebSocket error: {e}")
    
    async def _start_binance_websocket(self):
        """Start Binance WebSocket feed for crypto"""
        # Create stream names for crypto pairs
        streams = []
        for pair in self.config['crypto_pairs']:
            # Convert BTCUSD to btcusdt format
            binance_pair = pair.lower().replace('usd', 'usdt')
            streams.append(f"{binance_pair}@ticker")
        
        stream_string = '/'.join(streams)
        uri = f"wss://stream.binance.com:9443/ws/{stream_string}"
        
        try:
            async with websockets.connect(uri) as websocket:
                self.logger.info("✅ Connected to Binance WebSocket")
                
                async for message in websocket:
                    await self._process_binance_message(message)
                    
        except Exception as e:
            self.logger.error(f"❌ Binance WebSocket error: {e}")
    
    async def _start_polling_feed(self, source_name: str):
        """Start polling-based feed for sources without WebSocket"""
        source = self.data_sources[source_name]
        
        while True:
            try:
                if source_name == 'alpha_vantage':
                    await self._poll_alpha_vantage()
                elif source_name == 'yahoo_finance':
                    await self._poll_yahoo_finance()
                elif source_name == 'coingecko':
                    await self._poll_coingecko()
                
                # Wait based on rate limits
                rate_limit = source.get('rate_limit', 60)
                await asyncio.sleep(60 / rate_limit)  # Convert to seconds between calls
                
            except Exception as e:
                self.logger.error(f"❌ Polling error for {source_name}: {e}")
                await asyncio.sleep(10)  # Wait before retry
    
    async def _poll_alpha_vantage(self):
        """Poll Alpha Vantage for forex data"""
        async with aiohttp.ClientSession() as session:
            for pair in self.config['forex_pairs']:
                try:
                    # Convert EURUSD to EUR,USD format
                    from_currency = pair[:3]
                    to_currency = pair[3:]
                    
                    url = f"{self.data_sources['alpha_vantage']['base_url']}"
                    params = {
                        'function': 'CURRENCY_EXCHANGE_RATE',
                        'from_currency': from_currency,
                        'to_currency': to_currency,
                        'apikey': self.config['alpha_vantage_key']
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            await self._process_alpha_vantage_data(pair, data)
                        
                except Exception as e:
                    self.logger.error(f"❌ Alpha Vantage error for {pair}: {e}")
    
    async def _poll_yahoo_finance(self):
        """Poll Yahoo Finance for backup data"""
        async with aiohttp.ClientSession() as session:
            for pair in self.config['forex_pairs']:
                try:
                    # Convert EURUSD to EURUSD=X format
                    yahoo_symbol = f"{pair}=X"
                    url = f"{self.data_sources['yahoo_finance']['base_url']}/{yahoo_symbol}"
                    
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            await self._process_yahoo_data(pair, data)
                            
                except Exception as e:
                    self.logger.error(f"❌ Yahoo Finance error for {pair}: {e}")
    
    async def _process_finnhub_message(self, message: str):
        """Process Finnhub WebSocket message"""
        try:
            data = json.loads(message)
            if data.get('type') == 'trade':
                for trade in data.get('data', []):
                    symbol = trade.get('s', '').replace('FX:', '')
                    price = trade.get('p')
                    timestamp = datetime.fromtimestamp(trade.get('t', 0) / 1000)
                    
                    tick = MarketTick(
                        symbol=symbol,
                        price=price,
                        bid=price - 0.0001,  # Estimate spread
                        ask=price + 0.0001,
                        volume=trade.get('v', 0),
                        timestamp=timestamp,
                        source='finnhub'
                    )
                    
                    await self._process_market_tick(tick)
                    
        except Exception as e:
            self.logger.error(f"❌ Error processing Finnhub message: {e}")
    
    async def _process_market_tick(self, tick: MarketTick):
        """Process and validate market tick"""
        try:
            # Data quality validation
            if self._validate_tick_quality(tick):
                # Update buffer
                self.data_buffer[tick.symbol] = tick
                
                # Calculate latency
                latency_ms = (datetime.now() - tick.timestamp).total_seconds() * 1000
                self.latency_metrics[tick.symbol] = latency_ms
                
                # Call registered callbacks
                for callback in self.data_callbacks:
                    try:
                        await callback(tick)
                    except Exception as e:
                        self.logger.error(f"❌ Callback error: {e}")
                
                self.logger.debug(f"📊 {tick.symbol}: {tick.price} (latency: {latency_ms:.1f}ms)")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing tick for {tick.symbol}: {e}")
    
    def _validate_tick_quality(self, tick: MarketTick) -> bool:
        """Validate tick data quality"""
        try:
            # Check for reasonable price
            if tick.price <= 0:
                return False
            
            # Check for reasonable spread
            if tick.spread > tick.price * 0.1:  # 10% spread is unreasonable
                return False
            
            # Check for timestamp freshness
            age_seconds = (datetime.now() - tick.timestamp).total_seconds()
            if age_seconds > 60:  # Data older than 1 minute
                return False
            
            # Check for price change validation
            if tick.symbol in self.data_buffer:
                last_tick = self.data_buffer[tick.symbol]
                price_change = abs(tick.price - last_tick.price) / last_tick.price
                if price_change > self.config['price_change_threshold']:
                    self.logger.warning(f"⚠️ Large price change for {tick.symbol}: {price_change:.2%}")
                    return False
            
            return True
            
        except Exception:
            return False
    
    def add_data_callback(self, callback):
        """Add callback for real-time data"""
        self.data_callbacks.append(callback)
    
    def get_latest_price(self, symbol: str) -> Optional[MarketTick]:
        """Get latest price for symbol"""
        return self.data_buffer.get(symbol)
    
    def get_data_quality_report(self) -> Dict:
        """Get data quality and performance report"""
        report = {
            'active_symbols': len(self.data_buffer),
            'avg_latency_ms': sum(self.latency_metrics.values()) / len(self.latency_metrics) if self.latency_metrics else 0,
            'data_sources_active': len(self.active_connections),
            'last_update': datetime.now().isoformat()
        }
        
        return report

# Usage example
async def example_usage():
    """Example of how to use the enhanced market data feed"""
    
    # Initialize feed
    feed = EnhancedMarketDataFeed({
        'alpha_vantage_key': 'YOUR_ALPHA_VANTAGE_KEY',
        'iex_token': 'YOUR_IEX_TOKEN',
        'finnhub_key': 'YOUR_FINNHUB_KEY',
        'forex_pairs': ['EURUSD', 'GBPUSD', 'USDJPY'],
        'crypto_pairs': ['BTCUSD', 'ETHUSD']
    })
    
    # Add callback for trading system
    async def price_callback(tick: MarketTick):
        print(f"📊 {tick.symbol}: ${tick.price:.4f} from {tick.source}")
        # Here you would feed data to your trading models
    
    feed.add_data_callback(price_callback)
    
    # Start real-time feed
    await feed.start_real_time_feed()

if __name__ == "__main__":
    asyncio.run(example_usage())