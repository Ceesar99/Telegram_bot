#!/usr/bin/env python3
"""
Unified Trading System - Original + Institutional Grade

This system combines both the original binary options trading bot
and the institutional-grade trading system into one comprehensive
platform that can run both systems simultaneously or individually.

Features:
- Original Bot: LSTM AI-powered signals, Telegram interface, Pocket Option
- Institutional System: Professional data feeds, advanced risk management, 
  portfolio optimization, compliance monitoring
- Unified Interface: Single entry point for both systems
- Mode Selection: Choose between original, institutional, or hybrid mode
- Shared Components: Common data management, logging, and monitoring

Author: Unified Trading System
Version: 2.0.0
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json
import signal
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/workspace')

# Import core components
from telegram_bot import TradingBot
from signal_engine import SignalEngine
from pocket_option_api import PocketOptionAPI
from performance_tracker import PerformanceTracker
from risk_manager import RiskManager
from config import (
    LOGGING_CONFIG, DATABASE_CONFIG, TELEGRAM_BOT_TOKEN,
    TELEGRAM_USER_ID, POCKET_OPTION_SSID
)

# Import institutional components
try:
    from professional_data_manager import ProfessionalDataManager
    from enhanced_signal_engine import EnhancedSignalEngine
    from execution.smart_order_router import SmartOrderRouter
    from portfolio.institutional_risk_manager import InstitutionalRiskManager
    from monitoring.institutional_monitoring import InstitutionalMonitoringSystem
    INSTITUTIONAL_AVAILABLE = True
except ImportError:
    INSTITUTIONAL_AVAILABLE = False
    print("⚠️  Institutional components not available - running in original mode only")

@dataclass
class SystemMode:
    """Trading system mode configuration"""
    name: str
    description: str
    original_bot: bool
    institutional_system: bool
    hybrid_mode: bool

@dataclass
class TradingSession:
    """Trading session information"""
    session_id: str
    start_time: datetime
    mode: str
    end_time: Optional[datetime] = None
    total_signals: int = 0
    successful_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    accuracy: float = 0.0
    sharpe_ratio: float = 0.0

@dataclass
class SystemHealth:
    """Overall system health status"""
    timestamp: datetime
    overall_status: str  # 'healthy', 'degraded', 'critical'
    original_bot_status: str
    institutional_status: str
    data_feeds_status: str
    execution_status: str
    risk_status: str
    monitoring_status: str
    active_alerts: int
    uptime_hours: float

class UnifiedTradingSystem:
    """
    Comprehensive unified trading system that orchestrates
    both original and institutional-grade components
    """
    
    def __init__(self, mode: str = "hybrid"):
        # Setup logger first
        self.logger = self._setup_logger()
        
        # Now validate mode (after logger is available)
        self.mode = self._validate_mode(mode)
        
        self.system_start_time = datetime.now()
        self.is_running = False
        self.shutdown_requested = False
        
        # Core components (always available)
        self.telegram_bot: Optional[TradingBot] = None
        self.signal_engine: Optional[SignalEngine] = None
        self.pocket_api: Optional[PocketOptionAPI] = None
        self.performance_tracker: Optional[PerformanceTracker] = None
        self.risk_manager: Optional[RiskManager] = None
        
        # Institutional components (conditional)
        self.data_manager: Optional[Any] = None
        self.enhanced_signal_engine: Optional[Any] = None
        self.order_router: Optional[Any] = None
        self.institutional_risk_manager: Optional[Any] = None
        self.monitoring_system: Optional[Any] = None
        
        # Session tracking
        self.current_session: Optional[TradingSession] = None
        self.session_history: List[TradingSession] = []
        
        # Performance metrics
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'daily_accuracy': 0.0,
            'system_uptime': 0.0,
            'mode': self.mode.name
        }
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
    def _validate_mode(self, mode: str) -> SystemMode:
        """Validate and set system mode"""
        modes = {
            "original": SystemMode("Original", "Original binary options bot only", True, False, False),
            "institutional": SystemMode("Institutional", "Institutional-grade system only", False, True, False),
            "hybrid": SystemMode("Hybrid", "Both systems running simultaneously", True, True, True)
        }
        
        if mode not in modes:
            self.logger.warning(f"Invalid mode '{mode}', defaulting to hybrid")
            mode = "hybrid"
            
        if mode == "institutional" and not INSTITUTIONAL_AVAILABLE:
            self.logger.warning("Institutional mode requested but components not available, defaulting to original")
            mode = "original"
            
        return modes[mode]
    
    def _setup_logger(self) -> logging.Logger:
        """Setup main system logger"""
        # Create logs directory if it doesn't exist
        os.makedirs('/workspace/logs', exist_ok=True)
        
        logger = logging.getLogger('UnifiedTradingSystem')
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create file handler
        file_handler = logging.FileHandler('/workspace/logs/unified_system.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_requested = True
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize(self) -> bool:
        """Initialize all system components"""
        self.logger.info(f"Initializing Unified Trading System in {self.mode.name} mode")
        
        try:
            # Initialize core components
            await self._initialize_core_components()
            
            # Initialize institutional components if needed
            if self.mode.institutional_system and INSTITUTIONAL_AVAILABLE:
                await self._initialize_institutional_components()
            
            # Start new session
            self._start_new_session()
            
            self.logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            return False
    
    async def _initialize_core_components(self):
        """Initialize core trading bot components"""
        self.logger.info("Initializing core components...")
        
        # Initialize Telegram bot
        self.telegram_bot = TradingBot()
        
        # Initialize signal engine
        self.signal_engine = SignalEngine()
        
        # Initialize Pocket Option API
        self.pocket_api = PocketOptionAPI()
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker()
        
        # Initialize risk manager
        self.risk_manager = RiskManager()
        
        self.logger.info("Core components initialized")
    
    async def _initialize_institutional_components(self):
        """Initialize institutional-grade components"""
        if not INSTITUTIONAL_AVAILABLE:
            return
            
        self.logger.info("Initializing institutional components...")
        
        try:
            # Initialize professional data manager
            self.data_manager = ProfessionalDataManager()
            
            # Initialize enhanced signal engine
            self.enhanced_signal_engine = EnhancedSignalEngine()
            
            # Initialize smart order router with mock market data feed
            from execution.smart_order_router import MockMarketDataFeed
            self.order_router = SmartOrderRouter(MockMarketDataFeed())
            
            # Initialize institutional risk manager
            self.institutional_risk_manager = InstitutionalRiskManager()
            
            # Initialize monitoring system
            self.monitoring_system = InstitutionalMonitoringSystem()
            
            self.logger.info("Institutional components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize institutional components: {e}")
            self.mode = SystemMode("Original", "Fallback to original mode", True, False, False)
    
    def _start_new_session(self):
        """Start a new trading session"""
        session_id = f"session_{int(time.time())}"
        self.current_session = TradingSession(
            session_id=session_id,
            start_time=datetime.now(),
            mode=self.mode.name
        )
        self.logger.info(f"Started new trading session: {session_id}")
    
    async def start(self):
        """Start the unified trading system"""
        if not await self.initialize():
            self.logger.error("Failed to initialize system")
            return
        
        self.is_running = True
        self.logger.info("Starting Unified Trading System...")
        
        try:
            # Start main trading loop
            await self._main_trading_loop()
        except Exception as e:
            self.logger.error(f"Main trading loop error: {e}")
        finally:
            await self.shutdown()
    
    async def _main_trading_loop(self):
        """Main trading loop that orchestrates both systems"""
        self.logger.info("Entering main trading loop...")
        
        while self.is_running and not self.shutdown_requested:
            try:
                # Perform system health check
                health = await self._perform_system_health_check()
                
                # Generate and process signals based on mode
                if self.mode.original_bot:
                    await self._process_original_signals()
                
                if self.mode.institutional_system and INSTITUTIONAL_AVAILABLE:
                    await self._process_institutional_signals()
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Monitor portfolio risk
                await self._monitor_portfolio_risk()
                
                # Wait before next iteration
                await asyncio.sleep(30)  # 30-second intervals
                
            except Exception as e:
                self.logger.error(f"Error in main trading loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _process_original_signals(self):
        """Process signals using the original trading bot"""
        try:
            # Generate signal using original signal engine
            signal = await self.signal_engine.generate_signal()
            
            if signal:
                self.logger.info(f"Generated original signal: {signal.get('pair', 'Unknown')} {signal.get('direction', 'Unknown')}")
                
                # Apply risk management
                if await self.risk_manager.validate_trade(signal, 1000.0):  # Default balance
                    # Execute trade via Pocket Option
                    if self.pocket_api:
                        # Execute trade via Pocket Option (synchronous stub)
                        try:
                            self.pocket_api.execute_trade(signal)
                        except Exception as e:
                            self.logger.error(f"Trade execution failed: {e}")
                    
                    # Update performance tracking
                    if self.performance_tracker:
                        self.performance_tracker.save_signal(signal)
                    
                    self.current_session.total_signals += 1
                        
        except Exception as e:
            self.logger.error(f"Error processing original signals: {e}")
    
    async def _process_institutional_signals(self):
        """Process signals using the institutional system"""
        if not INSTITUTIONAL_AVAILABLE:
            return
            
        try:
            # Generate enhanced signals
            if self.enhanced_signal_engine:
                # Generate enhanced signals for major pairs
                symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"]
                signals = []
                
                for symbol in symbols:
                    try:
                        signal = await self.enhanced_signal_engine.generate_enhanced_signal(symbol)
                        if signal:
                            signals.append(signal)
                    except Exception as e:
                        self.logger.warning(f"Failed to generate signal for {symbol}: {e}")
                
                if signals:
                    self.logger.info(f"Generated {len(signals)} institutional signals")
                    
                    # Assess and filter signals
                    filtered_signals = await self._assess_and_filter_signals(signals)
                    
                    # Execute institutional trades
                    if filtered_signals:
                        await self._execute_institutional_trades(filtered_signals)
                        
        except Exception as e:
            self.logger.error(f"Error processing institutional signals: {e}")
    
    async def _assess_and_filter_signals(self, signals: List[Dict]) -> List[Dict]:
        """Assess and filter institutional signals"""
        if not INSTITUTIONAL_AVAILABLE:
            return signals
            
        filtered_signals = []
        
        for signal in signals:
            try:
                # Assess signal risk
                risk_assessment = await self._assess_signal_risk(signal)
                
                # Analyze portfolio impact
                portfolio_impact = await self._analyze_portfolio_impact(signal)
                
                # Validate market conditions
                market_validation = await self._validate_market_conditions(signal)
                
                # Apply institutional filters
                if (risk_assessment['risk_score'] < 0.7 and
                    portfolio_impact['impact_score'] < 0.8 and
                    market_validation['is_valid']):
                    filtered_signals.append(signal)
                    
            except Exception as e:
                self.logger.error(f"Error assessing signal {signal.get('id', 'unknown')}: {e}")
        
        return filtered_signals
    
    async def _assess_signal_risk(self, signal: Dict) -> Dict[str, Any]:
        """Assess risk for institutional signals"""
        # Placeholder for risk assessment logic
        return {
            'risk_score': 0.5,
            'risk_factors': ['market_volatility', 'liquidity'],
            'recommendation': 'proceed'
        }
    
    async def _analyze_portfolio_impact(self, signal: Dict) -> Dict[str, Any]:
        """Analyze portfolio impact of institutional signals"""
        # Placeholder for portfolio impact analysis
        return {
            'impact_score': 0.6,
            'exposure_change': 0.02,
            'correlation_risk': 0.3
        }
    
    async def _validate_market_conditions(self, signal: Dict) -> Dict[str, Any]:
        """Validate market conditions for institutional signals"""
        # Placeholder for market validation
        return {
            'is_valid': True,
            'market_conditions': 'favorable',
            'volatility': 'normal'
        }
    
    async def _execute_institutional_trades(self, signals: List[Dict]):
        """Execute institutional trades"""
        if not INSTITUTIONAL_AVAILABLE:
            return
            
        try:
            for signal in signals:
                # Execute via smart order router
                if self.order_router:
                    order = await self.order_router.execute_order(signal)
                    self.logger.info(f"Executed institutional order: {order}")
                
                # Update institutional risk management
                if self.institutional_risk_manager:
                    await self.institutional_risk_manager.record_trade(signal)
                    
        except Exception as e:
            self.logger.error(f"Error executing institutional trades: {e}")
    
    async def _perform_system_health_check(self) -> SystemHealth:
        """Perform comprehensive system health check"""
        try:
            # Check original bot status
            original_status = "healthy" if self.telegram_bot else "degraded"
            
            # Check institutional status
            institutional_status = "healthy" if (self.mode.institutional_system and 
                                              INSTITUTIONAL_AVAILABLE and 
                                              self.data_manager) else "not_available"
            
            # Determine overall status
            if original_status == "healthy" and institutional_status in ["healthy", "not_available"]:
                overall_status = "healthy"
            elif original_status == "degraded" or institutional_status == "degraded":
                overall_status = "degraded"
            else:
                overall_status = "critical"
            
            # Calculate uptime
            uptime = (datetime.now() - self.system_start_time).total_seconds() / 3600
            
            health = SystemHealth(
                timestamp=datetime.now(),
                overall_status=overall_status,
                original_bot_status=original_status,
                institutional_status=institutional_status,
                data_feeds_status="healthy" if self.pocket_api else "degraded",
                execution_status="healthy" if self.pocket_api else "degraded",
                risk_status="healthy" if self.risk_manager else "degraded",
                monitoring_status="healthy" if self.performance_tracker else "degraded",
                active_alerts=0,
                uptime_hours=uptime
            )
            
            # Log health status
            if health.overall_status != "healthy":
                self.logger.warning(f"System health: {health.overall_status}")
            
            return health
            
        except Exception as e:
            self.logger.error(f"Error during health check: {e}")
            return SystemHealth(
                timestamp=datetime.now(),
                overall_status="critical",
                original_bot_status="unknown",
                institutional_status="unknown",
                data_feeds_status="unknown",
                execution_status="unknown",
                risk_status="unknown",
                monitoring_status="unknown",
                active_alerts=0,
                uptime_hours=0.0
            )
    
    async def _update_performance_metrics(self):
        """Update system performance metrics"""
        try:
            if self.performance_tracker:
                metrics = self.performance_tracker.get_statistics()
                
                self.performance_metrics.update({
                    'total_trades': metrics.get('total_trades', 0),
                    'winning_trades': metrics.get('winning_trades', 0),
                    'total_pnl': metrics.get('total_pnl', 0.0),
                    'daily_accuracy': metrics.get('daily_accuracy', 0.0),
                    'system_uptime': (datetime.now() - self.system_start_time).total_seconds() / 3600
                })
                
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    async def _monitor_portfolio_risk(self):
        """Monitor portfolio risk across both systems"""
        try:
            # Monitor original bot risk
            if self.risk_manager:
                risk_report = self.risk_manager.get_risk_report(1000.0)  # Default balance
            
            # Monitor institutional risk
            if self.mode.institutional_system and INSTITUTIONAL_AVAILABLE:
                if self.institutional_risk_manager:
                    risk_report = self.institutional_risk_manager.generate_risk_report()
                    self.logger.info(f"Institutional risk report: {risk_report.get('overall_risk_level', 'N/A')}")
                    
        except Exception as e:
            self.logger.error(f"Error monitoring portfolio risk: {e}")
    
    async def shutdown(self):
        """Gracefully shutdown the system"""
        self.logger.info("Initiating system shutdown...")
        self.is_running = False
        
        try:
            # End current session
            if self.current_session:
                self.current_session.end_time = datetime.now()
                self.session_history.append(self.current_session)
                
                # Generate session report
                await self._generate_session_report(self.current_session)
            
            # Shutdown components
            if self.telegram_bot:
                # Check if shutdown method exists
                if hasattr(self.telegram_bot, 'shutdown'):
                    await self.telegram_bot.shutdown()
                else:
                    self.logger.info("Telegram bot shutdown method not available")
            
            if self.pocket_api:
                # Check if close method exists
                if hasattr(self.pocket_api, 'close'):
                    await self.pocket_api.close()
                else:
                    self.logger.info("PocketOptionAPI close method not available")
            
            # Generate shutdown report
            await self._generate_shutdown_report()
            
            self.logger.info("System shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def _generate_session_report(self, session: TradingSession):
        """Generate session completion report"""
        try:
            duration = session.end_time - session.start_time
            report = {
                'session_id': session.session_id,
                'mode': session.mode,
                'duration_hours': duration.total_seconds() / 3600,
                'total_signals': session.total_signals,
                'successful_trades': session.successful_trades,
                'total_pnl': session.total_pnl,
                'max_drawdown': session.max_drawdown,
                'accuracy': session.accuracy,
                'sharpe_ratio': session.sharpe_ratio
            }
            
            # Save report
            report_path = f"/workspace/logs/session_{session.session_id}_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Session report saved: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating session report: {e}")
    
    async def _generate_shutdown_report(self):
        """Generate system shutdown report"""
        try:
            uptime = (datetime.now() - self.system_start_time).total_seconds() / 3600
            
            report = {
                'shutdown_time': datetime.now().isoformat(),
                'total_uptime_hours': uptime,
                'mode': self.mode.name,
                'total_sessions': len(self.session_history),
                'final_performance_metrics': self.performance_metrics,
                'system_health': await self._perform_system_health_check()
            }
            
            # Save report
            report_path = f"/workspace/logs/shutdown_report_{int(time.time())}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Shutdown report saved: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating shutdown report: {e}")
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'mode': self.mode.name,
            'is_running': self.is_running,
            'uptime_hours': (datetime.now() - self.system_start_time).total_seconds() / 3600,
            'performance_metrics': self.performance_metrics,
            'current_session': {
                'id': self.current_session.session_id if self.current_session else None,
                'start_time': self.current_session.start_time.isoformat() if self.current_session else None,
                'total_signals': self.current_session.total_signals if self.current_session else 0
            } if self.current_session else None,
            'institutional_available': INSTITUTIONAL_AVAILABLE
        }

async def main():
    """Main entry point for the unified trading system"""
    print("🚀 Starting Unified Trading System...")
    print("=" * 50)
    
    # Parse command line arguments for mode
    mode = "hybrid"  # Default mode
    test_mode = False
    
    for i, arg in enumerate(sys.argv):
        if arg == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
        elif arg == "--test":
            test_mode = True
    
    print(f"Mode: {mode}")
    if test_mode:
        print("Test mode: enabled")
    print("=" * 50)
    
    # Create and start the system
    system = UnifiedTradingSystem(mode=mode)
    
    try:
        if test_mode:
            # Run for a limited time in test mode
            import signal
            def timeout_handler(signum, frame):
                print("\n⏰ Test mode timeout reached")
                raise KeyboardInterrupt
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(15)  # 15 seconds timeout
            
        await system.start()
    except KeyboardInterrupt:
        print("\n⚠️  Received interrupt signal")
    except Exception as e:
        print(f"❌ System error: {e}")
    finally:
        await system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())