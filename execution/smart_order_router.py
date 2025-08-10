import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import json
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

from institutional_config import SOR_CONFIG, DATA_QUALITY, INSTITUTIONAL_PERFORMANCE_TARGETS

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TWAP = "twap"
    VWAP = "vwap"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"

class OrderStatus(Enum):
    PENDING = "pending"
    WORKING = "working"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class Venue(Enum):
    PRIMARY_EXCHANGE = "primary_exchange"
    DARK_POOL = "dark_pool"
    ECN = "ecn"
    INTERNAL = "internal"

@dataclass
class Order:
    """Institutional-grade order representation"""
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # Execution parameters
    venue_preference: Optional[Venue] = None
    time_in_force: str = "DAY"  # DAY, GTC, IOC, FOK
    min_quantity: Optional[float] = None
    display_quantity: Optional[float] = None
    
    # Algo parameters
    algo_params: Dict[str, Any] = field(default_factory=dict)
    
    # Status and tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    
    # Timestamps
    created_time: datetime = field(default_factory=datetime.now)
    submitted_time: Optional[datetime] = None
    fill_time: Optional[datetime] = None
    
    # Risk and compliance
    risk_approved: bool = False
    compliance_approved: bool = False
    
    # Execution quality metrics
    implementation_shortfall: Optional[float] = None
    market_impact: Optional[float] = None
    timing_risk: Optional[float] = None

@dataclass
class Fill:
    """Trade fill representation"""
    fill_id: str
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    venue: Venue
    timestamp: datetime
    liquidity_flag: str  # 'maker', 'taker', 'aggressive'

@dataclass
class MarketData:
    """Real-time market data for execution"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last_price: float
    last_size: float
    volume: float
    vwap: Optional[float] = None

class ExecutionAlgorithm(ABC):
    """Base class for execution algorithms"""
    
    def __init__(self, order: Order, market_data_feed):
        self.order = order
        self.market_data_feed = market_data_feed
        self.logger = logging.getLogger(f"ExecutionAlgorithm.{self.__class__.__name__}")
        self.child_orders: List[Order] = []
        self.is_active = False
        
    @abstractmethod
    async def execute(self) -> List[Fill]:
        """Execute the order using the algorithm"""
        pass
        
    @abstractmethod
    def cancel(self):
        """Cancel the algorithm execution"""
        pass
        
    def calculate_market_impact(self, quantity: float, market_data: MarketData) -> float:
        """Calculate estimated market impact"""
        # Simplified market impact model
        spread = market_data.ask - market_data.bid
        mid_price = (market_data.bid + market_data.ask) / 2
        
        # Impact as percentage of spread based on order size relative to market depth
        depth = market_data.bid_size + market_data.ask_size
        size_ratio = quantity / max(depth, 1)
        
        impact_factor = min(0.5, size_ratio * 0.1)  # Cap at 50% of spread
        return impact_factor * spread / mid_price

class TWAPAlgorithm(ExecutionAlgorithm):
    """Time-Weighted Average Price execution algorithm"""
    
    def __init__(self, order: Order, market_data_feed):
        super().__init__(order, market_data_feed)
        self.duration = order.algo_params.get('duration_seconds', 600)  # 10 minutes default
        self.slice_size = order.algo_params.get('slice_size_percent', 0.1)  # 10% slices
        self.interval = order.algo_params.get('interval_seconds', 30)  # 30 second intervals
        
    async def execute(self) -> List[Fill]:
        """Execute TWAP algorithm"""
        fills = []
        self.is_active = True
        
        try:
            total_slices = int(self.duration / self.interval)
            slice_quantity = self.order.quantity / total_slices
            
            self.logger.info(f"Starting TWAP execution for {self.order.order_id}: "
                           f"{total_slices} slices of {slice_quantity:.2f} over {self.duration}s")
            
            for slice_num in range(total_slices):
                if not self.is_active:
                    break
                    
                # Get current market data
                market_data = await self.market_data_feed.get_market_data(self.order.symbol)
                if not market_data:
                    await asyncio.sleep(self.interval)
                    continue
                
                # Calculate limit price based on market conditions
                if self.order.side == 'BUY':
                    limit_price = market_data.bid + (market_data.ask - market_data.bid) * 0.3
                else:
                    limit_price = market_data.ask - (market_data.ask - market_data.bid) * 0.3
                
                # Create child order
                child_order = Order(
                    order_id=f"{self.order.order_id}_slice_{slice_num}",
                    symbol=self.order.symbol,
                    side=self.order.side,
                    quantity=slice_quantity,
                    order_type=OrderType.LIMIT,
                    price=limit_price,
                    time_in_force="IOC"  # Immediate or Cancel
                )
                
                # Submit child order
                child_fills = await self._submit_child_order(child_order, market_data)
                fills.extend(child_fills)
                
                # Wait for next interval
                if slice_num < total_slices - 1:
                    await asyncio.sleep(self.interval)
                    
        except Exception as e:
            self.logger.error(f"Error in TWAP execution: {e}")
        finally:
            self.is_active = False
            
        return fills
    
    async def _submit_child_order(self, child_order: Order, market_data: MarketData) -> List[Fill]:
        """Submit and manage child order"""
        fills = []
        
        try:
            # Simulate order submission and execution
            # In production, this would interface with actual trading venues
            
            # Calculate execution probability based on limit price aggressiveness
            if child_order.side == 'BUY':
                aggressiveness = (child_order.price - market_data.bid) / (market_data.ask - market_data.bid)
            else:
                aggressiveness = (market_data.ask - child_order.price) / (market_data.ask - market_data.bid)
            
            fill_probability = min(0.95, max(0.1, aggressiveness))
            
            if np.random.random() < fill_probability:
                # Simulate partial or full fill
                fill_ratio = np.random.uniform(0.7, 1.0)
                filled_quantity = child_order.quantity * fill_ratio
                
                # Add some price improvement/slippage
                price_improvement = np.random.uniform(-0.0001, 0.0001)
                fill_price = child_order.price + price_improvement
                
                fill = Fill(
                    fill_id=f"fill_{child_order.order_id}_{int(time.time()*1000)}",
                    order_id=child_order.order_id,
                    symbol=child_order.symbol,
                    side=child_order.side,
                    quantity=filled_quantity,
                    price=fill_price,
                    commission=filled_quantity * 0.0001,  # 1 basis point
                    venue=Venue.PRIMARY_EXCHANGE,
                    timestamp=datetime.now(),
                    liquidity_flag='maker' if aggressiveness < 0.5 else 'taker'
                )
                
                fills.append(fill)
                self.logger.info(f"Child order filled: {filled_quantity:.2f} @ {fill_price:.5f}")
                
        except Exception as e:
            self.logger.error(f"Error submitting child order: {e}")
            
        return fills
    
    def cancel(self):
        """Cancel TWAP execution"""
        self.is_active = False
        self.logger.info(f"TWAP algorithm cancelled for order {self.order.order_id}")

class VWAPAlgorithm(ExecutionAlgorithm):
    """Volume-Weighted Average Price execution algorithm"""
    
    def __init__(self, order: Order, market_data_feed):
        super().__init__(order, market_data_feed)
        self.lookback_minutes = order.algo_params.get('lookback_minutes', 20)
        self.participation_rate = order.algo_params.get('participation_rate', 0.1)  # 10%
        self.max_volume_percent = order.algo_params.get('max_volume_percent', 0.25)  # 25%
        
    async def execute(self) -> List[Fill]:
        """Execute VWAP algorithm"""
        fills = []
        self.is_active = True
        
        try:
            # Get historical volume profile
            volume_profile = await self._get_volume_profile()
            
            total_intervals = len(volume_profile)
            executed_quantity = 0.0
            
            self.logger.info(f"Starting VWAP execution for {self.order.order_id}: "
                           f"participation_rate={self.participation_rate:.2%}")
            
            for interval_idx, expected_volume_ratio in enumerate(volume_profile):
                if not self.is_active or executed_quantity >= self.order.quantity:
                    break
                
                # Get current market data
                market_data = await self.market_data_feed.get_market_data(self.order.symbol)
                if not market_data:
                    await asyncio.sleep(30)  # Wait 30 seconds
                    continue
                
                # Calculate target quantity for this interval
                current_volume = market_data.volume
                target_volume = current_volume * expected_volume_ratio * self.participation_rate
                
                # Limit by max volume percentage
                max_quantity = current_volume * self.max_volume_percent
                target_quantity = min(target_volume, max_quantity)
                
                # Don't exceed remaining order quantity
                remaining_quantity = self.order.quantity - executed_quantity
                slice_quantity = min(target_quantity, remaining_quantity)
                
                if slice_quantity > 0:
                    # Execute slice
                    child_fills = await self._execute_vwap_slice(slice_quantity, market_data)
                    fills.extend(child_fills)
                    executed_quantity += sum(fill.quantity for fill in child_fills)
                
                await asyncio.sleep(60)  # 1 minute intervals
                
        except Exception as e:
            self.logger.error(f"Error in VWAP execution: {e}")
        finally:
            self.is_active = False
            
        return fills
    
    async def _get_volume_profile(self) -> List[float]:
        """Get historical volume profile for the day"""
        # Simplified volume profile - in production, use real historical data
        # Typical intraday volume pattern (higher at open/close, lower midday)
        hours = 6.5  # Trading hours
        intervals = int(hours * 60 / 1)  # 1-minute intervals
        
        profile = []
        for i in range(intervals):
            hour_of_day = 9.5 + (i / 60)  # Starting at 9:30 AM
            
            # U-shaped volume pattern
            if hour_of_day < 10.5:  # Opening hour
                volume_factor = 1.5
            elif hour_of_day > 15.0:  # Closing hour
                volume_factor = 1.3
            elif 11.5 <= hour_of_day <= 13.5:  # Lunch time
                volume_factor = 0.6
            else:
                volume_factor = 1.0
                
            profile.append(volume_factor)
        
        # Normalize to sum to 1
        total = sum(profile)
        return [x / total for x in profile]
    
    async def _execute_vwap_slice(self, quantity: float, market_data: MarketData) -> List[Fill]:
        """Execute a single VWAP slice"""
        # Similar to TWAP child order execution but with volume considerations
        fills = []
        
        # Use mid-price for VWAP execution
        mid_price = (market_data.bid + market_data.ask) / 2
        
        child_order = Order(
            order_id=f"{self.order.order_id}_vwap_{int(time.time()*1000)}",
            symbol=self.order.symbol,
            side=self.order.side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            price=mid_price,
            time_in_force="IOC"
        )
        
        # Simulate execution
        fill_ratio = np.random.uniform(0.8, 1.0)  # Higher fill rate for VWAP
        filled_quantity = quantity * fill_ratio
        
        if filled_quantity > 0:
            fill = Fill(
                fill_id=f"fill_{child_order.order_id}",
                order_id=child_order.order_id,
                symbol=child_order.symbol,
                side=child_order.side,
                quantity=filled_quantity,
                price=mid_price + np.random.uniform(-0.00005, 0.00005),  # Small price variation
                commission=filled_quantity * 0.0001,
                venue=Venue.PRIMARY_EXCHANGE,
                timestamp=datetime.now(),
                liquidity_flag='maker'
            )
            
            fills.append(fill)
            
        return fills
    
    def cancel(self):
        """Cancel VWAP execution"""
        self.is_active = False
        self.logger.info(f"VWAP algorithm cancelled for order {self.order.order_id}")

class ImplementationShortfallAlgorithm(ExecutionAlgorithm):
    """Implementation Shortfall execution algorithm"""
    
    def __init__(self, order: Order, market_data_feed):
        super().__init__(order, market_data_feed)
        self.risk_aversion = order.algo_params.get('risk_aversion', 0.5)  # 0-1 scale
        self.max_participation = order.algo_params.get('max_participation', 0.3)
        
    async def execute(self) -> List[Fill]:
        """Execute Implementation Shortfall algorithm"""
        fills = []
        self.is_active = True
        
        try:
            # Get initial market data
            initial_market_data = await self.market_data_feed.get_market_data(self.order.symbol)
            if not initial_market_data:
                return fills
                
            initial_price = (initial_market_data.bid + initial_market_data.ask) / 2
            executed_quantity = 0.0
            
            self.logger.info(f"Starting Implementation Shortfall execution for {self.order.order_id}: "
                           f"risk_aversion={self.risk_aversion}")
            
            # Adaptive execution based on market conditions
            while self.is_active and executed_quantity < self.order.quantity:
                current_market_data = await self.market_data_feed.get_market_data(self.order.symbol)
                if not current_market_data:
                    await asyncio.sleep(10)
                    continue
                
                # Calculate optimal execution rate based on market impact and timing risk
                remaining_quantity = self.order.quantity - executed_quantity
                optimal_rate = self._calculate_optimal_execution_rate(
                    remaining_quantity, current_market_data, initial_price
                )
                
                # Execute slice
                slice_quantity = min(remaining_quantity, optimal_rate)
                if slice_quantity > 0:
                    child_fills = await self._execute_is_slice(slice_quantity, current_market_data)
                    fills.extend(child_fills)
                    executed_quantity += sum(fill.quantity for fill in child_fills)
                
                await asyncio.sleep(30)  # 30-second intervals
                
        except Exception as e:
            self.logger.error(f"Error in Implementation Shortfall execution: {e}")
        finally:
            self.is_active = False
            
        return fills
    
    def _calculate_optimal_execution_rate(self, remaining_quantity: float, 
                                        market_data: MarketData, initial_price: float) -> float:
        """Calculate optimal execution rate balancing market impact and timing risk"""
        
        # Market impact cost (increases with execution rate)
        market_impact_factor = self.calculate_market_impact(remaining_quantity, market_data)
        
        # Timing risk (price movement risk from waiting)
        current_price = (market_data.bid + market_data.ask) / 2
        price_momentum = (current_price - initial_price) / initial_price
        
        # Adjust for price momentum
        if self.order.side == 'BUY' and price_momentum > 0:
            urgency_factor = 1.5  # Accelerate buying if price rising
        elif self.order.side == 'SELL' and price_momentum < 0:
            urgency_factor = 1.5  # Accelerate selling if price falling
        else:
            urgency_factor = 1.0
        
        # Base execution rate
        base_rate = remaining_quantity * 0.1  # 10% per interval
        
        # Adjust based on risk aversion
        # High risk aversion = prefer lower market impact = slower execution
        risk_adjustment = 1.0 - (self.risk_aversion * 0.5)
        
        optimal_rate = base_rate * risk_adjustment * urgency_factor
        
        # Cap by maximum participation rate
        max_rate = market_data.volume * self.max_participation
        return min(optimal_rate, max_rate)
    
    async def _execute_is_slice(self, quantity: float, market_data: MarketData) -> List[Fill]:
        """Execute Implementation Shortfall slice"""
        fills = []
        
        # Adaptive pricing based on market conditions
        spread = market_data.ask - market_data.bid
        mid_price = (market_data.bid + market_data.ask) / 2
        
        # Aggressive pricing when risk aversion is low
        if self.order.side == 'BUY':
            aggressiveness = 0.5 + (1 - self.risk_aversion) * 0.3
            limit_price = market_data.bid + spread * aggressiveness
        else:
            aggressiveness = 0.5 + (1 - self.risk_aversion) * 0.3
            limit_price = market_data.ask - spread * aggressiveness
        
        child_order = Order(
            order_id=f"{self.order.order_id}_is_{int(time.time()*1000)}",
            symbol=self.order.symbol,
            side=self.order.side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            price=limit_price,
            time_in_force="IOC"
        )
        
        # Simulate execution with higher fill rate for aggressive orders
        fill_probability = 0.7 + aggressiveness * 0.25
        if np.random.random() < fill_probability:
            filled_quantity = quantity * np.random.uniform(0.8, 1.0)
            
            fill = Fill(
                fill_id=f"fill_{child_order.order_id}",
                order_id=child_order.order_id,
                symbol=child_order.symbol,
                side=child_order.side,
                quantity=filled_quantity,
                price=limit_price + np.random.uniform(-0.00002, 0.00002),
                commission=filled_quantity * 0.0001,
                venue=Venue.PRIMARY_EXCHANGE,
                timestamp=datetime.now(),
                liquidity_flag='taker'
            )
            
            fills.append(fill)
            
        return fills
    
    def cancel(self):
        """Cancel Implementation Shortfall execution"""
        self.is_active = False
        self.logger.info(f"Implementation Shortfall algorithm cancelled for order {self.order.order_id}")

class SmartOrderRouter:
    """Institutional-grade smart order routing engine"""
    
    def __init__(self, market_data_feed):
        self.logger = self._setup_logger()
        self.market_data_feed = market_data_feed
        self.active_orders: Dict[str, Order] = {}
        self.active_algorithms: Dict[str, ExecutionAlgorithm] = {}
        self.fills: List[Fill] = []
        self.execution_metrics = {}
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('SmartOrderRouter')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('/workspace/logs/smart_order_router.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    async def submit_order(self, order: Order) -> str:
        """Submit order for smart routing"""
        try:
            # Pre-trade risk checks
            if not await self._pre_trade_risk_check(order):
                order.status = OrderStatus.REJECTED
                self.logger.warning(f"Order {order.order_id} rejected by risk check")
                return order.order_id
            
            # Store order
            self.active_orders[order.order_id] = order
            order.submitted_time = datetime.now()
            order.status = OrderStatus.WORKING
            
            # Route order based on type
            if order.order_type in [OrderType.TWAP, OrderType.VWAP, OrderType.IMPLEMENTATION_SHORTFALL]:
                await self._route_algorithmic_order(order)
            else:
                await self._route_simple_order(order)
            
            self.logger.info(f"Order {order.order_id} submitted successfully")
            return order.order_id
            
        except Exception as e:
            self.logger.error(f"Error submitting order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
            return order.order_id
    
    async def _pre_trade_risk_check(self, order: Order) -> bool:
        """Comprehensive pre-trade risk validation"""
        
        # Check order size limits
        if order.quantity <= 0:
            self.logger.warning(f"Invalid order quantity: {order.quantity}")
            return False
        
        # Check price reasonableness
        if order.price and order.price <= 0:
            self.logger.warning(f"Invalid order price: {order.price}")
            return False
        
        # Get current market data for validation
        market_data = await self.market_data_feed.get_market_data(order.symbol)
        if market_data:
            mid_price = (market_data.bid + market_data.ask) / 2
            
            # Check for extreme prices (more than 10% away from mid)
            if order.price:
                price_deviation = abs(order.price - mid_price) / mid_price
                if price_deviation > 0.1:
                    self.logger.warning(f"Order price {order.price} deviates {price_deviation:.2%} from market {mid_price}")
                    return False
        
        # Additional risk checks can be added here
        order.risk_approved = True
        return True
    
    async def _route_algorithmic_order(self, order: Order):
        """Route algorithmic order to appropriate execution algorithm"""
        
        algorithm = None
        
        if order.order_type == OrderType.TWAP:
            algorithm = TWAPAlgorithm(order, self.market_data_feed)
        elif order.order_type == OrderType.VWAP:
            algorithm = VWAPAlgorithm(order, self.market_data_feed)
        elif order.order_type == OrderType.IMPLEMENTATION_SHORTFALL:
            algorithm = ImplementationShortfallAlgorithm(order, self.market_data_feed)
        
        if algorithm:
            self.active_algorithms[order.order_id] = algorithm
            
            # Execute algorithm in background task
            asyncio.create_task(self._execute_algorithm(order.order_id, algorithm))
    
    async def _execute_algorithm(self, order_id: str, algorithm: ExecutionAlgorithm):
        """Execute algorithm and handle results"""
        try:
            fills = await algorithm.execute()
            
            # Process fills
            for fill in fills:
                await self._process_fill(fill)
            
            # Update order status
            order = self.active_orders.get(order_id)
            if order:
                if order.filled_quantity >= order.quantity:
                    order.status = OrderStatus.FILLED
                    order.fill_time = datetime.now()
                elif order.filled_quantity > 0:
                    order.status = OrderStatus.PARTIALLY_FILLED
                
                # Calculate execution metrics
                self._calculate_execution_metrics(order, fills)
                
        except Exception as e:
            self.logger.error(f"Error executing algorithm for order {order_id}: {e}")
        finally:
            # Cleanup
            if order_id in self.active_algorithms:
                del self.active_algorithms[order_id]
    
    async def _route_simple_order(self, order: Order):
        """Route simple market/limit orders"""
        
        # Get current market data
        market_data = await self.market_data_feed.get_market_data(order.symbol)
        if not market_data:
            order.status = OrderStatus.REJECTED
            return
        
        # Determine execution price
        if order.order_type == OrderType.MARKET:
            execution_price = market_data.ask if order.side == 'BUY' else market_data.bid
        else:
            execution_price = order.price
        
        # Simulate execution
        if order.order_type == OrderType.MARKET or self._should_fill_limit_order(order, market_data):
            fill = Fill(
                fill_id=f"fill_{order.order_id}_{int(time.time()*1000)}",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=execution_price,
                commission=order.quantity * 0.0001,
                venue=Venue.PRIMARY_EXCHANGE,
                timestamp=datetime.now(),
                liquidity_flag='taker' if order.order_type == OrderType.MARKET else 'maker'
            )
            
            await self._process_fill(fill)
    
    def _should_fill_limit_order(self, order: Order, market_data: MarketData) -> bool:
        """Determine if limit order should fill based on market conditions"""
        if order.side == 'BUY':
            return order.price >= market_data.ask
        else:
            return order.price <= market_data.bid
    
    async def _process_fill(self, fill: Fill):
        """Process trade fill"""
        self.fills.append(fill)
        
        # Update order
        order = self.active_orders.get(fill.order_id)
        if order:
            order.filled_quantity += fill.quantity
            
            # Update average fill price
            total_value = order.avg_fill_price * (order.filled_quantity - fill.quantity) + fill.price * fill.quantity
            order.avg_fill_price = total_value / order.filled_quantity
            
            order.total_commission += fill.commission
        
        self.logger.info(f"Fill processed: {fill.quantity:.2f} {fill.symbol} @ {fill.price:.5f}")
    
    def _calculate_execution_metrics(self, order: Order, fills: List[Fill]):
        """Calculate execution quality metrics"""
        if not fills:
            return
        
        # Calculate VWAP of fills
        total_value = sum(fill.price * fill.quantity for fill in fills)
        total_quantity = sum(fill.quantity for fill in fills)
        execution_vwap = total_value / total_quantity if total_quantity > 0 else 0
        
        # Calculate implementation shortfall (simplified)
        # This would normally use arrival price and benchmark
        arrival_price = fills[0].price  # Simplified
        implementation_shortfall = (execution_vwap - arrival_price) / arrival_price
        
        # Store metrics
        metrics = {
            'order_id': order.order_id,
            'execution_vwap': execution_vwap,
            'implementation_shortfall': implementation_shortfall,
            'total_commission': order.total_commission,
            'fill_rate': order.filled_quantity / order.quantity,
            'execution_time': (fills[-1].timestamp - fills[0].timestamp).total_seconds(),
            'venue_breakdown': self._calculate_venue_breakdown(fills)
        }
        
        self.execution_metrics[order.order_id] = metrics
    
    def _calculate_venue_breakdown(self, fills: List[Fill]) -> Dict[str, float]:
        """Calculate execution breakdown by venue"""
        venue_quantities = {}
        total_quantity = sum(fill.quantity for fill in fills)
        
        for fill in fills:
            venue = fill.venue.value
            venue_quantities[venue] = venue_quantities.get(venue, 0) + fill.quantity
        
        return {venue: qty / total_quantity for venue, qty in venue_quantities.items()}
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        try:
            order = self.active_orders.get(order_id)
            if not order:
                return False
            
            # Cancel algorithm if running
            if order_id in self.active_algorithms:
                self.active_algorithms[order_id].cancel()
                del self.active_algorithms[order_id]
            
            order.status = OrderStatus.CANCELLED
            self.logger.info(f"Order {order_id} cancelled")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current order status"""
        return self.active_orders.get(order_id)
    
    def get_execution_metrics(self, order_id: str) -> Optional[Dict]:
        """Get execution metrics for an order"""
        return self.execution_metrics.get(order_id)
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        return [order for order in self.active_orders.values() 
                if order.status in [OrderStatus.WORKING, OrderStatus.PARTIALLY_FILLED]]

# Mock market data feed for testing
class MockMarketDataFeed:
    """Mock market data feed for testing"""
    
    def __init__(self):
        self.data = {
            'EURUSD': MarketData('EURUSD', datetime.now(), 1.0950, 1.0952, 1000000, 1000000, 1.0951, 10000, 50000000),
            'GBPUSD': MarketData('GBPUSD', datetime.now(), 1.2650, 1.2652, 800000, 800000, 1.2651, 8000, 40000000),
            'USDJPY': MarketData('USDJPY', datetime.now(), 149.45, 149.47, 1200000, 1200000, 149.46, 12000, 60000000),
        }
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get market data for symbol"""
        base_data = self.data.get(symbol)
        if not base_data:
            return None
        
        # Add some realistic price movement
        price_change = np.random.uniform(-0.0001, 0.0001)
        
        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            bid=base_data.bid + price_change,
            ask=base_data.ask + price_change,
            bid_size=base_data.bid_size * np.random.uniform(0.8, 1.2),
            ask_size=base_data.ask_size * np.random.uniform(0.8, 1.2),
            last_price=base_data.last_price + price_change,
            last_size=base_data.last_size,
            volume=base_data.volume * np.random.uniform(0.9, 1.1)
        )

# Example usage
async def main():
    """Test the smart order router"""
    
    # Create mock market data feed
    market_data_feed = MockMarketDataFeed()
    
    # Create smart order router
    sor = SmartOrderRouter(market_data_feed)
    
    # Test TWAP order
    twap_order = Order(
        order_id="TWAP_001",
        symbol="EURUSD",
        side="BUY",
        quantity=1000000,
        order_type=OrderType.TWAP,
        algo_params={
            'duration_seconds': 300,  # 5 minutes
            'slice_size_percent': 0.1,
            'interval_seconds': 30
        }
    )
    
    print("Submitting TWAP order...")
    await sor.submit_order(twap_order)
    
    # Wait for execution
    await asyncio.sleep(10)
    
    # Check status
    status = sor.get_order_status("TWAP_001")
    if status:
        print(f"Order Status: {status.status.value}")
        print(f"Filled Quantity: {status.filled_quantity}")
        print(f"Average Fill Price: {status.avg_fill_price:.5f}")
    
    # Get execution metrics
    metrics = sor.get_execution_metrics("TWAP_001")
    if metrics:
        print(f"Execution Metrics: {metrics}")

if __name__ == "__main__":
    asyncio.run(main())