import asyncio
import redis
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor
import websockets
import numpy as np
from collections import deque
import pickle
import zlib

@dataclass
class StreamMessage:
    """High-performance message structure for streaming"""
    timestamp: float
    symbol: str
    message_type: str
    data: Dict[str, Any]
    sequence_id: int
    source: str

class HighPerformanceRedisStream:
    """Ultra-fast Redis streaming with compression and batching"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.async_redis = None
        self.logger = logging.getLogger('RedisStream')
        
        # Performance optimization settings
        self.batch_size = 100
        self.compression_enabled = True
        self.max_memory_usage = 1024 * 1024 * 100  # 100MB
        
        # Stream statistics
        self.messages_processed = 0
        self.bytes_processed = 0
        self.start_time = time.time()
        
    async def initialize(self):
        """Initialize Redis connections"""
        try:
            # High-performance Redis connection pool
            self.redis_client = redis.Redis.from_url(
                self.redis_url,
                max_connections=50,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            
            # Async Redis for streaming (using standard redis with async support)
            self.async_redis = redis.Redis.from_url(
                self.redis_url,
                max_connections=20,
                retry_on_timeout=True
            )
            
            # Configure Redis for optimal performance
            await self._optimize_redis_config()
            
            self.logger.info("✅ Redis streaming engine initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis: {e}")
            return False
    
    async def _optimize_redis_config(self):
        """Optimize Redis configuration for high-throughput streaming"""
        try:
            # Set memory policy
            await self.async_redis.config_set('maxmemory-policy', 'allkeys-lru')
            
            # Enable compression
            await self.async_redis.config_set('rdbcompression', 'yes')
            
            # Optimize for performance
            await self.async_redis.config_set('tcp-keepalive', '60')
            await self.async_redis.config_set('timeout', '0')
            
        except Exception as e:
            self.logger.warning(f"Could not optimize Redis config: {e}")
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data for efficient storage and transmission"""
        if self.compression_enabled and len(data) > 100:
            return zlib.compress(data)
        return data
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data"""
        if self.compression_enabled and data.startswith(b'\x78'):  # zlib magic number
            return zlib.decompress(data)
        return data
    
    async def publish_message(self, stream_name: str, message: StreamMessage) -> bool:
        """Publish message to Redis stream with high performance"""
        try:
            # Serialize message
            serialized_data = pickle.dumps(asdict(message))
            compressed_data = self._compress_data(serialized_data)
            
            # Add to Redis stream
            message_id = await self.async_redis.xadd(
                stream_name,
                {'data': compressed_data},
                maxlen=10000,  # Keep only recent messages
                approximate=True
            )
            
            # Update statistics
            self.messages_processed += 1
            self.bytes_processed += len(compressed_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish message: {e}")
            return False
    
    async def consume_stream(self, stream_name: str, callback: Callable[[StreamMessage], None],
                           group_name: str = "trading_group"):
        """Consume messages from Redis stream with high performance"""
        try:
            # Create consumer group if it doesn't exist
            try:
                await self.async_redis.xgroup_create(stream_name, group_name, id='0', mkstream=True)
            except Exception:
                pass  # Group already exists
            
            consumer_name = f"consumer_{int(time.time())}"
            
            while True:
                try:
                    # Read from stream
                    messages = await self.async_redis.xreadgroup(
                        group_name,
                        consumer_name,
                        {stream_name: '>'},
                        count=self.batch_size,
                        block=100  # 100ms timeout
                    )
                    
                    for stream, msgs in messages:
                        for msg_id, fields in msgs:
                            try:
                                # Deserialize message
                                compressed_data = fields[b'data']
                                decompressed_data = self._decompress_data(compressed_data)
                                message_dict = pickle.loads(decompressed_data)
                                message = StreamMessage(**message_dict)
                                
                                # Process message
                                await self._process_message_async(callback, message)
                                
                                # Acknowledge message
                                await self.async_redis.xack(stream_name, group_name, msg_id)
                                
                            except Exception as e:
                                self.logger.error(f"Error processing message {msg_id}: {e}")
                
                except Exception as e:
                    self.logger.error(f"Error in stream consumption: {e}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            self.logger.error(f"Fatal error in stream consumer: {e}")
    
    async def _process_message_async(self, callback: Callable, message: StreamMessage):
        """Process message asynchronously"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(message)
            else:
                callback(message)
        except Exception as e:
            self.logger.error(f"Error in message callback: {e}")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get streaming performance statistics"""
        runtime = time.time() - self.start_time
        return {
            'messages_per_second': self.messages_processed / max(runtime, 1),
            'bytes_per_second': self.bytes_processed / max(runtime, 1),
            'total_messages': self.messages_processed,
            'total_bytes': self.bytes_processed,
            'runtime_seconds': runtime,
            'compression_ratio': (self.bytes_processed / max(self.messages_processed * 1000, 1)) if self.compression_enabled else 1.0
        }

class EventDrivenMarketDataProcessor:
    """Event-driven market data processing engine"""
    
    def __init__(self, redis_stream: HighPerformanceRedisStream):
        self.redis_stream = redis_stream
        self.logger = logging.getLogger('EventDrivenProcessor')
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.processing_stats = {
            'events_processed': 0,
            'avg_processing_time': 0.0,
            'max_processing_time': 0.0
        }
        
        # High-frequency data buffers
        self.price_buffers: Dict[str, deque] = {}
        self.volume_buffers: Dict[str, deque] = {}
        self.buffer_size = 1000
        
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register event handler for specific event types"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        self.logger.info(f"Registered handler for event type: {event_type}")
    
    async def process_market_data_event(self, message: StreamMessage):
        """Process incoming market data events"""
        start_time = time.perf_counter()
        
        try:
            # Update buffers for ultra-fast access
            symbol = message.symbol
            if symbol not in self.price_buffers:
                self.price_buffers[symbol] = deque(maxlen=self.buffer_size)
                self.volume_buffers[symbol] = deque(maxlen=self.buffer_size)
            
            # Extract data
            data = message.data
            if 'price' in data:
                self.price_buffers[symbol].append(data['price'])
            if 'volume' in data:
                self.volume_buffers[symbol].append(data['volume'])
            
            # Trigger event handlers
            if message.message_type in self.event_handlers:
                tasks = []
                for handler in self.event_handlers[message.message_type]:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(message))
                    else:
                        # Run sync handlers in thread pool
                        tasks.append(asyncio.get_event_loop().run_in_executor(
                            None, handler, message
                        ))
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update statistics
            processing_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            self.processing_stats['events_processed'] += 1
            self.processing_stats['avg_processing_time'] = (
                (self.processing_stats['avg_processing_time'] * (self.processing_stats['events_processed'] - 1) + processing_time) /
                self.processing_stats['events_processed']
            )
            self.processing_stats['max_processing_time'] = max(
                self.processing_stats['max_processing_time'], processing_time
            )
            
            # Log performance for very fast processing
            if processing_time < 1.0:  # Less than 1ms
                self.logger.debug(f"Ultra-fast processing: {processing_time:.3f}ms for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error processing market data event: {e}")
    
    def get_latest_prices(self, symbol: str, count: int = 100) -> np.ndarray:
        """Get latest prices for a symbol with zero-copy access"""
        if symbol in self.price_buffers:
            buffer = self.price_buffers[symbol]
            return np.array(list(buffer)[-count:]) if buffer else np.array([])
        return np.array([])
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing performance statistics"""
        return self.processing_stats.copy()

class WebSocketMarketDataFeed:
    """High-performance WebSocket market data feed"""
    
    def __init__(self, redis_stream: HighPerformanceRedisStream):
        self.redis_stream = redis_stream
        self.logger = logging.getLogger('WebSocketFeed')
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.is_running = False
        self.sequence_counter = 0
        
    async def start_feed(self, symbols: List[str], sources: List[str]):
        """Start high-frequency market data feed"""
        self.is_running = True
        
        # Start multiple feed connections for redundancy
        tasks = []
        for source in sources:
            tasks.append(self._connect_to_source(source, symbols))
        
        # Start all connections concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _connect_to_source(self, source: str, symbols: List[str]):
        """Connect to a market data source"""
        try:
            if source == "pocket_option":
                await self._connect_pocket_option(symbols)
            elif source == "binance":
                await self._connect_binance(symbols)
            elif source == "polygon":
                await self._connect_polygon(symbols)
            else:
                self.logger.warning(f"Unknown data source: {source}")
                
        except Exception as e:
            self.logger.error(f"Error connecting to {source}: {e}")
    
    async def _connect_pocket_option(self, symbols: List[str]):
        """Connect to Pocket Option WebSocket feed"""
        # This would implement the actual Pocket Option WebSocket connection
        # For now, we'll simulate with dummy data
        while self.is_running:
            for symbol in symbols:
                # Simulate market data
                price = 1.2345 + np.random.normal(0, 0.001)
                volume = np.random.randint(100, 1000)
                
                message = StreamMessage(
                    timestamp=time.time(),
                    symbol=symbol,
                    message_type="market_data",
                    data={
                        'price': price,
                        'volume': volume,
                        'bid': price - 0.0001,
                        'ask': price + 0.0001,
                        'source': 'pocket_option'
                    },
                    sequence_id=self.sequence_counter,
                    source="pocket_option"
                )
                
                await self.redis_stream.publish_message(f"market_data_{symbol}", message)
                self.sequence_counter += 1
            
            await asyncio.sleep(0.1)  # 100ms delay for simulation
    
    async def _connect_binance(self, symbols: List[str]):
        """Connect to Binance WebSocket feed"""
        # Simulate Binance connection
        while self.is_running:
            for symbol in symbols:
                # Simulate crypto data
                price = 50000 + np.random.normal(0, 100)
                volume = np.random.randint(1, 10)
                
                message = StreamMessage(
                    timestamp=time.time(),
                    symbol=f"{symbol}_CRYPTO",
                    message_type="crypto_data",
                    data={
                        'price': price,
                        'volume': volume,
                        'source': 'binance'
                    },
                    sequence_id=self.sequence_counter,
                    source="binance"
                )
                
                await self.redis_stream.publish_message(f"crypto_data_{symbol}", message)
                self.sequence_counter += 1
            
            await asyncio.sleep(0.05)  # 50ms delay
    
    async def _connect_polygon(self, symbols: List[str]):
        """Connect to Polygon WebSocket feed"""
        # Simulate Polygon connection for stocks
        while self.is_running:
            for symbol in symbols:
                # Simulate stock data
                price = 100 + np.random.normal(0, 1)
                volume = np.random.randint(100, 5000)
                
                message = StreamMessage(
                    timestamp=time.time(),
                    symbol=f"{symbol}_STOCK",
                    message_type="stock_data",
                    data={
                        'price': price,
                        'volume': volume,
                        'source': 'polygon'
                    },
                    sequence_id=self.sequence_counter,
                    source="polygon"
                )
                
                await self.redis_stream.publish_message(f"stock_data_{symbol}", message)
                self.sequence_counter += 1
            
            await asyncio.sleep(0.2)  # 200ms delay
    
    def stop_feed(self):
        """Stop all market data feeds"""
        self.is_running = False
        self.logger.info("Market data feeds stopped")

class StreamingTradingEngine:
    """Main streaming trading engine coordinator"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.logger = logging.getLogger('StreamingTradingEngine')
        
        # Initialize components
        self.redis_stream = HighPerformanceRedisStream(redis_url)
        self.processor = EventDrivenMarketDataProcessor(self.redis_stream)
        self.market_feed = WebSocketMarketDataFeed(self.redis_stream)
        
        self.is_running = False
        
        # Register event handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default event handlers"""
        self.processor.register_event_handler("market_data", self._handle_market_data)
        self.processor.register_event_handler("crypto_data", self._handle_crypto_data)
        self.processor.register_event_handler("stock_data", self._handle_stock_data)
    
    async def _handle_market_data(self, message: StreamMessage):
        """Handle market data events"""
        # This would trigger signal generation and trading decisions
        symbol = message.symbol
        price = message.data.get('price', 0)
        
        # Log high-frequency updates
        self.logger.debug(f"Market data: {symbol} @ {price}")
        
        # Here you would integrate with your enhanced signal engine
        # For now, we'll just update the price buffers (already done in processor)
    
    async def _handle_crypto_data(self, message: StreamMessage):
        """Handle cryptocurrency data events"""
        symbol = message.symbol
        price = message.data.get('price', 0)
        self.logger.debug(f"Crypto data: {symbol} @ {price}")
    
    async def _handle_stock_data(self, message: StreamMessage):
        """Handle stock data events"""
        symbol = message.symbol
        price = message.data.get('price', 0)
        self.logger.debug(f"Stock data: {symbol} @ {price}")
    
    async def start(self, symbols: List[str] = None, sources: List[str] = None):
        """Start the streaming trading engine"""
        if symbols is None:
            symbols = ["EURUSD", "GBPUSD", "USDJPY", "BTCUSD"]
        
        if sources is None:
            sources = ["pocket_option", "binance", "polygon"]
        
        try:
            # Initialize Redis
            await self.redis_stream.initialize()
            
            # Start market data feed
            feed_task = asyncio.create_task(
                self.market_feed.start_feed(symbols, sources)
            )
            
            # Start stream consumers
            consumer_tasks = []
            for symbol in symbols:
                for source in sources:
                    stream_name = f"market_data_{symbol}" if source == "pocket_option" else f"{source.split('_')[0]}_data_{symbol}"
                    consumer_task = asyncio.create_task(
                        self.redis_stream.consume_stream(
                            stream_name,
                            self.processor.process_market_data_event
                        )
                    )
                    consumer_tasks.append(consumer_task)
            
            self.is_running = True
            self.logger.info("🚀 Streaming trading engine started")
            
            # Wait for all tasks
            all_tasks = [feed_task] + consumer_tasks
            await asyncio.gather(*all_tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Error starting streaming engine: {e}")
            raise
    
    def stop(self):
        """Stop the streaming trading engine"""
        self.is_running = False
        self.market_feed.stop_feed()
        self.logger.info("⏹️ Streaming trading engine stopped")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            'redis_stats': self.redis_stream.get_performance_stats(),
            'processing_stats': self.processor.get_processing_stats(),
            'engine_status': 'running' if self.is_running else 'stopped'
        }

# Example usage and testing
async def main():
    """Example usage of the streaming trading engine"""
    engine = StreamingTradingEngine()
    
    try:
        # Start the engine
        await engine.start(
            symbols=["EURUSD", "GBPUSD"],
            sources=["pocket_option", "binance"]
        )
        
    except KeyboardInterrupt:
        print("Stopping engine...")
        engine.stop()

if __name__ == "__main__":
    asyncio.run(main())