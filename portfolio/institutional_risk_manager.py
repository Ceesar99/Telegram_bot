import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import sqlite3
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from institutional_config import INSTITUTIONAL_RISK, INSTITUTIONAL_PERFORMANCE_TARGETS

class RiskModel(Enum):
    HISTORICAL_SIMULATION = "historical_simulation"
    MONTE_CARLO = "monte_carlo"
    PARAMETRIC = "parametric"
    COPULA = "copula"

class StressScenario(Enum):
    CRISIS_2008 = "2008_crisis"
    COVID_2020 = "2020_covid"
    INTEREST_RATE_SHOCK = "interest_rate_shock"
    CURRENCY_CRISIS = "currency_crisis"
    CUSTOM = "custom"

@dataclass
class Position:
    """Portfolio position representation"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    sector: Optional[str] = None
    country: Optional[str] = None
    currency: str = "USD"
    
    # Risk metrics
    beta: Optional[float] = None
    volatility: Optional[float] = None
    var_contribution: Optional[float] = None
    
    # Timestamps
    entry_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    timestamp: datetime
    
    # VaR metrics
    var_1d_95: float
    var_1d_99: float
    cvar_1d_95: float  # Conditional VaR (Expected Shortfall)
    cvar_1d_99: float
    
    # Portfolio metrics
    total_exposure: float
    net_exposure: float
    gross_exposure: float
    leverage: float
    
    # Concentration metrics
    largest_position_pct: float
    top_5_positions_pct: float
    sector_concentration: Dict[str, float]
    
    # Correlation metrics
    avg_correlation: float
    max_correlation: float
    correlation_risk: float
    
    # Drawdown metrics
    current_drawdown: float
    max_drawdown: float
    drawdown_duration_days: int
    
    # Stress test results
    stress_test_results: Dict[str, float]
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float

@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_portfolio_var: float = 0.02  # 2% daily VaR
    max_position_size: float = 0.05  # 5% of portfolio
    max_sector_concentration: float = 0.25  # 25% in any sector
    max_correlation: float = 0.7  # Max correlation between positions
    max_leverage: float = 3.0  # Max portfolio leverage
    max_drawdown: float = 0.05  # 5% max drawdown
    
    # Stop loss levels
    position_stop_loss: float = 0.02  # 2% position stop loss
    portfolio_stop_loss: float = 0.03  # 3% portfolio stop loss

class VaRCalculator:
    """Value at Risk calculation engine"""
    
    def __init__(self):
        self.logger = logging.getLogger('VaRCalculator')
        
    def calculate_historical_var(self, returns: pd.Series, confidence_level: float = 0.95,
                                lookback_days: int = 252) -> Tuple[float, float]:
        """Calculate VaR using historical simulation"""
        try:
            # Use recent returns for calculation
            recent_returns = returns.tail(lookback_days)
            
            if len(recent_returns) < 30:
                self.logger.warning("Insufficient data for VaR calculation")
                return 0.0, 0.0
            
            # Calculate VaR as percentile
            var_percentile = (1 - confidence_level) * 100
            var = np.percentile(recent_returns, var_percentile)
            
            # Calculate Conditional VaR (Expected Shortfall)
            cvar_returns = recent_returns[recent_returns <= var]
            cvar = cvar_returns.mean() if len(cvar_returns) > 0 else var
            
            return abs(var), abs(cvar)
            
        except Exception as e:
            self.logger.error(f"Error calculating historical VaR: {e}")
            return 0.0, 0.0
    
    def calculate_parametric_var(self, returns: pd.Series, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate VaR using parametric method (normal distribution)"""
        try:
            if len(returns) < 30:
                return 0.0, 0.0
            
            # Calculate mean and standard deviation
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Z-score for confidence level
            z_score = stats.norm.ppf(1 - confidence_level)
            
            # VaR calculation
            var = -(mean_return + z_score * std_return)
            
            # Conditional VaR for normal distribution
            cvar = -(mean_return - std_return * stats.norm.pdf(z_score) / (1 - confidence_level))
            
            return max(0, var), max(0, cvar)
            
        except Exception as e:
            self.logger.error(f"Error calculating parametric VaR: {e}")
            return 0.0, 0.0
    
    def calculate_monte_carlo_var(self, returns: pd.Series, confidence_level: float = 0.95,
                                 simulations: int = 10000) -> Tuple[float, float]:
        """Calculate VaR using Monte Carlo simulation"""
        try:
            if len(returns) < 30:
                return 0.0, 0.0
            
            # Fit distribution parameters
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Generate random scenarios
            np.random.seed(42)  # For reproducibility
            simulated_returns = np.random.normal(mean_return, std_return, simulations)
            
            # Calculate VaR and CVaR
            var_percentile = (1 - confidence_level) * 100
            var = np.percentile(simulated_returns, var_percentile)
            
            cvar_returns = simulated_returns[simulated_returns <= var]
            cvar = cvar_returns.mean() if len(cvar_returns) > 0 else var
            
            return abs(var), abs(cvar)
            
        except Exception as e:
            self.logger.error(f"Error calculating Monte Carlo VaR: {e}")
            return 0.0, 0.0

class StressTesting:
    """Stress testing engine for portfolio analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger('StressTesting')
        self.scenarios = INSTITUTIONAL_RISK['stress_testing']['scenarios']
    
    def run_stress_test(self, portfolio: Dict[str, Position], 
                       scenario: StressScenario) -> Dict[str, float]:
        """Run stress test on portfolio"""
        try:
            scenario_params = self._get_scenario_parameters(scenario)
            results = {}
            
            total_pnl = 0.0
            total_value = sum(pos.market_value for pos in portfolio.values())
            
            for symbol, position in portfolio.items():
                # Apply scenario stress to position
                stressed_pnl = self._apply_stress_to_position(position, scenario_params)
                total_pnl += stressed_pnl
                results[symbol] = stressed_pnl
            
            # Calculate portfolio-level metrics
            portfolio_pnl_pct = total_pnl / total_value if total_value > 0 else 0
            
            results.update({
                'total_pnl': total_pnl,
                'portfolio_pnl_pct': portfolio_pnl_pct,
                'scenario': scenario.value,
                'timestamp': datetime.now()
            })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running stress test: {e}")
            return {}
    
    def _get_scenario_parameters(self, scenario: StressScenario) -> Dict:
        """Get parameters for stress scenario"""
        if scenario.value in self.scenarios:
            return self.scenarios[scenario.value]
        
        # Default mild stress scenario
        return {'equity_shock': -0.1, 'volatility_shock': 1.5}
    
    def _apply_stress_to_position(self, position: Position, scenario_params: Dict) -> float:
        """Apply stress scenario to individual position"""
        
        # Base price shock
        price_shock = scenario_params.get('equity_shock', 0)
        
        # Adjust shock based on asset type/sector
        if position.sector:
            sector_adjustments = {
                'Technology': 1.2,  # Tech more volatile
                'Finance': 1.5,     # Banks sensitive to rates
                'Energy': 1.3,      # Commodity exposure
                'Healthcare': 0.8,  # Defensive
                'Utilities': 0.6    # Most defensive
            }
            shock_multiplier = sector_adjustments.get(position.sector, 1.0)
            price_shock *= shock_multiplier
        
        # Currency shock for foreign positions
        if position.currency != 'USD':
            fx_shock = scenario_params.get('fx_shock', 0)
            price_shock += fx_shock
        
        # Calculate stressed P&L
        stressed_price = position.current_price * (1 + price_shock)
        stressed_value = position.quantity * stressed_price
        stressed_pnl = stressed_value - position.market_value
        
        return stressed_pnl
    
    def run_multiple_scenarios(self, portfolio: Dict[str, Position]) -> Dict[str, Dict]:
        """Run multiple stress scenarios"""
        results = {}
        
        for scenario in StressScenario:
            if scenario != StressScenario.CUSTOM:
                results[scenario.value] = self.run_stress_test(portfolio, scenario)
        
        return results

class CorrelationAnalyzer:
    """Portfolio correlation analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger('CorrelationAnalyzer')
    
    def calculate_correlation_matrix(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix for portfolio assets"""
        try:
            return returns_data.corr()
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    
    def calculate_portfolio_correlation_risk(self, positions: Dict[str, Position],
                                           correlation_matrix: pd.DataFrame) -> float:
        """Calculate portfolio-level correlation risk"""
        try:
            if correlation_matrix.empty:
                return 0.0
            
            # Get position weights
            total_value = sum(pos.market_value for pos in positions.values())
            weights = {}
            
            for symbol, position in positions.items():
                if symbol in correlation_matrix.index:
                    weights[symbol] = position.market_value / total_value
            
            # Calculate portfolio variance considering correlations
            portfolio_variance = 0.0
            
            for symbol1 in weights:
                for symbol2 in weights:
                    if symbol1 in correlation_matrix.index and symbol2 in correlation_matrix.columns:
                        vol1 = positions[symbol1].volatility or 0.02  # Default 2% daily vol
                        vol2 = positions[symbol2].volatility or 0.02
                        corr = correlation_matrix.loc[symbol1, symbol2]
                        
                        portfolio_variance += (weights[symbol1] * weights[symbol2] * 
                                             vol1 * vol2 * corr)
            
            return np.sqrt(portfolio_variance)
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation risk: {e}")
            return 0.0
    
    def find_highly_correlated_pairs(self, correlation_matrix: pd.DataFrame,
                                    threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Find highly correlated asset pairs"""
        high_corr_pairs = []
        
        try:
            for i in range(len(correlation_matrix.index)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    symbol1 = correlation_matrix.index[i]
                    symbol2 = correlation_matrix.columns[j]
                    corr = correlation_matrix.iloc[i, j]
                    
                    if abs(corr) >= threshold:
                        high_corr_pairs.append((symbol1, symbol2, corr))
            
            # Sort by correlation strength
            high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error finding correlated pairs: {e}")
        
        return high_corr_pairs

class InstitutionalRiskManager:
    """Comprehensive institutional-grade risk management system"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.var_calculator = VaRCalculator()
        self.stress_tester = StressTesting()
        self.correlation_analyzer = CorrelationAnalyzer()
        
        # Risk limits from config
        self.limits = RiskLimits(
            max_portfolio_var=INSTITUTIONAL_RISK['portfolio_level']['max_portfolio_var'],
            max_position_size=INSTITUTIONAL_RISK['portfolio_level']['max_single_position'],
            max_sector_concentration=INSTITUTIONAL_RISK['portfolio_level']['max_sector_concentration'],
            max_correlation=INSTITUTIONAL_RISK['portfolio_level']['correlation_limit'],
            max_leverage=INSTITUTIONAL_RISK['portfolio_level']['leverage_limit']
        )
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.portfolio_returns = pd.Series(dtype=float)
        self.risk_metrics_history: List[RiskMetrics] = []
        
        # Initialize database
        self._initialize_database()
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('InstitutionalRiskManager')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('/workspace/logs/institutional_risk_manager.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def _initialize_database(self):
        """Initialize risk management database"""
        try:
            conn = sqlite3.connect('/workspace/data/risk_management.db')
            cursor = conn.cursor()
            
            # Risk metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    var_1d_95 REAL,
                    var_1d_99 REAL,
                    cvar_1d_95 REAL,
                    cvar_1d_99 REAL,
                    total_exposure REAL,
                    leverage REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    stress_test_results TEXT
                )
            ''')
            
            # Positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity REAL,
                    entry_price REAL,
                    current_price REAL,
                    market_value REAL,
                    unrealized_pnl REAL,
                    sector TEXT,
                    var_contribution REAL
                )
            ''')
            
            # Risk violations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    description TEXT,
                    severity TEXT,
                    current_value REAL,
                    limit_value REAL,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
    
    def update_position(self, symbol: str, quantity: float, price: float, 
                       sector: str = None, country: str = None):
        """Update or create portfolio position"""
        try:
            if symbol in self.positions:
                # Update existing position
                position = self.positions[symbol]
                
                # Calculate new average entry price if adding to position
                if (quantity > 0 and position.quantity > 0) or (quantity < 0 and position.quantity < 0):
                    total_value = position.quantity * position.entry_price + quantity * price
                    total_quantity = position.quantity + quantity
                    position.entry_price = total_value / total_quantity if total_quantity != 0 else price
                
                position.quantity += quantity
                position.current_price = price
                position.market_value = position.quantity * price
                position.unrealized_pnl = (price - position.entry_price) * position.quantity
                position.last_update = datetime.now()
                
                # Remove position if quantity is zero
                if abs(position.quantity) < 1e-8:
                    del self.positions[symbol]
            else:
                # Create new position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    current_price=price,
                    market_value=quantity * price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    sector=sector,
                    country=country
                )
            
            self.logger.info(f"Position updated: {symbol} qty={quantity} price={price}")
            
        except Exception as e:
            self.logger.error(f"Error updating position {symbol}: {e}")
    
    def calculate_portfolio_risk_metrics(self, returns_data: pd.DataFrame = None) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            timestamp = datetime.now()
            
            # Portfolio exposure metrics
            total_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
            net_exposure = sum(pos.market_value for pos in self.positions.values())
            gross_exposure = total_exposure
            leverage = gross_exposure / max(abs(net_exposure), 1)  # Avoid division by zero
            
            # VaR calculations
            var_1d_95, cvar_1d_95 = 0.0, 0.0
            var_1d_99, cvar_1d_99 = 0.0, 0.0
            
            if len(self.portfolio_returns) > 30:
                var_1d_95, cvar_1d_95 = self.var_calculator.calculate_historical_var(
                    self.portfolio_returns, confidence_level=0.95
                )
                var_1d_99, cvar_1d_99 = self.var_calculator.calculate_historical_var(
                    self.portfolio_returns, confidence_level=0.99
                )
            
            # Concentration metrics
            position_values = [abs(pos.market_value) for pos in self.positions.values()]
            largest_position_pct = max(position_values) / max(total_exposure, 1) if position_values else 0
            
            sorted_positions = sorted(position_values, reverse=True)[:5]
            top_5_positions_pct = sum(sorted_positions) / max(total_exposure, 1)
            
            # Sector concentration
            sector_exposure = {}
            for pos in self.positions.values():
                sector = pos.sector or 'Unknown'
                sector_exposure[sector] = sector_exposure.get(sector, 0) + abs(pos.market_value)
            
            sector_concentration = {
                sector: exposure / max(total_exposure, 1) 
                for sector, exposure in sector_exposure.items()
            }
            
            # Correlation metrics
            correlation_matrix = pd.DataFrame()
            if returns_data is not None and not returns_data.empty:
                correlation_matrix = self.correlation_analyzer.calculate_correlation_matrix(returns_data)
                
            avg_correlation = 0.0
            max_correlation = 0.0
            correlation_risk = 0.0
            
            if not correlation_matrix.empty:
                # Calculate average correlation (excluding diagonal)
                corr_values = correlation_matrix.values
                np.fill_diagonal(corr_values, np.nan)
                avg_correlation = np.nanmean(np.abs(corr_values))
                max_correlation = np.nanmax(np.abs(corr_values))
                
                correlation_risk = self.correlation_analyzer.calculate_portfolio_correlation_risk(
                    self.positions, correlation_matrix
                )
            
            # Drawdown metrics
            current_drawdown, max_drawdown, drawdown_duration = self._calculate_drawdown_metrics()
            
            # Stress test results
            stress_results = self.stress_tester.run_multiple_scenarios(self.positions)
            stress_test_summary = {
                scenario: results.get('portfolio_pnl_pct', 0) 
                for scenario, results in stress_results.items()
            }
            
            # Risk-adjusted return metrics
            sharpe_ratio, sortino_ratio, calmar_ratio, information_ratio = self._calculate_risk_adjusted_returns()
            
            # Create risk metrics object
            risk_metrics = RiskMetrics(
                timestamp=timestamp,
                var_1d_95=var_1d_95,
                var_1d_99=var_1d_99,
                cvar_1d_95=cvar_1d_95,
                cvar_1d_99=cvar_1d_99,
                total_exposure=total_exposure,
                net_exposure=net_exposure,
                gross_exposure=gross_exposure,
                leverage=leverage,
                largest_position_pct=largest_position_pct,
                top_5_positions_pct=top_5_positions_pct,
                sector_concentration=sector_concentration,
                avg_correlation=avg_correlation,
                max_correlation=max_correlation,
                correlation_risk=correlation_risk,
                current_drawdown=current_drawdown,
                max_drawdown=max_drawdown,
                drawdown_duration_days=drawdown_duration,
                stress_test_results=stress_test_summary,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                information_ratio=information_ratio
            )
            
            # Store in history
            self.risk_metrics_history.append(risk_metrics)
            
            # Save to database
            self._save_risk_metrics(risk_metrics)
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(
                timestamp=datetime.now(),
                var_1d_95=0, var_1d_99=0, cvar_1d_95=0, cvar_1d_99=0,
                total_exposure=0, net_exposure=0, gross_exposure=0, leverage=0,
                largest_position_pct=0, top_5_positions_pct=0, sector_concentration={},
                avg_correlation=0, max_correlation=0, correlation_risk=0,
                current_drawdown=0, max_drawdown=0, drawdown_duration_days=0,
                stress_test_results={}, sharpe_ratio=0, sortino_ratio=0,
                calmar_ratio=0, information_ratio=0
            )
    
    def check_risk_limits(self, risk_metrics: RiskMetrics) -> List[Dict]:
        """Check for risk limit violations"""
        violations = []
        
        try:
            # VaR limit check
            if risk_metrics.var_1d_95 > self.limits.max_portfolio_var:
                violations.append({
                    'type': 'VAR_VIOLATION',
                    'description': f'Portfolio VaR {risk_metrics.var_1d_95:.3f} exceeds limit {self.limits.max_portfolio_var:.3f}',
                    'severity': 'HIGH',
                    'current_value': risk_metrics.var_1d_95,
                    'limit_value': self.limits.max_portfolio_var
                })
            
            # Position size limit check
            if risk_metrics.largest_position_pct > self.limits.max_position_size:
                violations.append({
                    'type': 'POSITION_SIZE_VIOLATION',
                    'description': f'Largest position {risk_metrics.largest_position_pct:.3f} exceeds limit {self.limits.max_position_size:.3f}',
                    'severity': 'MEDIUM',
                    'current_value': risk_metrics.largest_position_pct,
                    'limit_value': self.limits.max_position_size
                })
            
            # Sector concentration check
            for sector, concentration in risk_metrics.sector_concentration.items():
                if concentration > self.limits.max_sector_concentration:
                    violations.append({
                        'type': 'SECTOR_CONCENTRATION_VIOLATION',
                        'description': f'Sector {sector} concentration {concentration:.3f} exceeds limit {self.limits.max_sector_concentration:.3f}',
                        'severity': 'MEDIUM',
                        'current_value': concentration,
                        'limit_value': self.limits.max_sector_concentration
                    })
            
            # Leverage limit check
            if risk_metrics.leverage > self.limits.max_leverage:
                violations.append({
                    'type': 'LEVERAGE_VIOLATION',
                    'description': f'Portfolio leverage {risk_metrics.leverage:.2f} exceeds limit {self.limits.max_leverage:.2f}',
                    'severity': 'HIGH',
                    'current_value': risk_metrics.leverage,
                    'limit_value': self.limits.max_leverage
                })
            
            # Correlation limit check
            if risk_metrics.max_correlation > self.limits.max_correlation:
                violations.append({
                    'type': 'CORRELATION_VIOLATION',
                    'description': f'Max correlation {risk_metrics.max_correlation:.3f} exceeds limit {self.limits.max_correlation:.3f}',
                    'severity': 'MEDIUM',
                    'current_value': risk_metrics.max_correlation,
                    'limit_value': self.limits.max_correlation
                })
            
            # Drawdown limit check
            if risk_metrics.current_drawdown > self.limits.max_drawdown:
                violations.append({
                    'type': 'DRAWDOWN_VIOLATION',
                    'description': f'Current drawdown {risk_metrics.current_drawdown:.3f} exceeds limit {self.limits.max_drawdown:.3f}',
                    'severity': 'HIGH',
                    'current_value': risk_metrics.current_drawdown,
                    'limit_value': self.limits.max_drawdown
                })
            
            # Save violations to database
            for violation in violations:
                self._save_risk_violation(violation)
            
            if violations:
                self.logger.warning(f"Found {len(violations)} risk limit violations")
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
        
        return violations
    
    def _calculate_drawdown_metrics(self) -> Tuple[float, float, int]:
        """Calculate drawdown metrics"""
        try:
            if len(self.portfolio_returns) < 10:
                return 0.0, 0.0, 0
            
            # Calculate cumulative returns
            cumulative_returns = (1 + self.portfolio_returns).cumprod()
            
            # Calculate running maximum (peak)
            peak = cumulative_returns.expanding(min_periods=1).max()
            
            # Calculate drawdown
            drawdown = (cumulative_returns - peak) / peak
            
            # Current drawdown
            current_drawdown = abs(drawdown.iloc[-1])
            
            # Maximum drawdown
            max_drawdown = abs(drawdown.min())
            
            # Drawdown duration (simplified)
            drawdown_duration = 0
            if current_drawdown > 0.001:  # If currently in drawdown
                # Count days since peak
                last_peak_idx = peak.index[peak == peak.iloc[-1]][-1]
                current_idx = cumulative_returns.index[-1]
                drawdown_duration = (current_idx - last_peak_idx).days
            
            return current_drawdown, max_drawdown, drawdown_duration
            
        except Exception as e:
            self.logger.error(f"Error calculating drawdown metrics: {e}")
            return 0.0, 0.0, 0
    
    def _calculate_risk_adjusted_returns(self) -> Tuple[float, float, float, float]:
        """Calculate risk-adjusted return metrics"""
        try:
            if len(self.portfolio_returns) < 30:
                return 0.0, 0.0, 0.0, 0.0
            
            # Annualize metrics
            annual_factor = 252  # Trading days
            
            # Mean return and volatility
            mean_return = self.portfolio_returns.mean() * annual_factor
            volatility = self.portfolio_returns.std() * np.sqrt(annual_factor)
            
            # Risk-free rate (simplified - use 2% annually)
            risk_free_rate = 0.02
            
            # Sharpe ratio
            sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Sortino ratio (downside deviation)
            negative_returns = self.portfolio_returns[self.portfolio_returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(annual_factor) if len(negative_returns) > 0 else volatility
            sortino_ratio = (mean_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Calmar ratio
            max_drawdown = abs(self.portfolio_returns.min())  # Simplified
            calmar_ratio = mean_return / max_drawdown if max_drawdown > 0 else 0
            
            # Information ratio (simplified - assume benchmark return is risk-free rate)
            excess_return = mean_return - risk_free_rate
            tracking_error = volatility  # Simplified
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
            
            return sharpe_ratio, sortino_ratio, calmar_ratio, information_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-adjusted returns: {e}")
            return 0.0, 0.0, 0.0, 0.0
    
    def _save_risk_metrics(self, metrics: RiskMetrics):
        """Save risk metrics to database"""
        try:
            conn = sqlite3.connect('/workspace/data/risk_management.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO risk_metrics 
                (timestamp, var_1d_95, var_1d_99, cvar_1d_95, cvar_1d_99, 
                 total_exposure, leverage, max_drawdown, sharpe_ratio, stress_test_results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp.isoformat(),
                metrics.var_1d_95,
                metrics.var_1d_99,
                metrics.cvar_1d_95,
                metrics.cvar_1d_99,
                metrics.total_exposure,
                metrics.leverage,
                metrics.max_drawdown,
                metrics.sharpe_ratio,
                str(metrics.stress_test_results)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving risk metrics: {e}")
    
    def _save_risk_violation(self, violation: Dict):
        """Save risk violation to database"""
        try:
            conn = sqlite3.connect('/workspace/data/risk_management.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO risk_violations 
                (timestamp, violation_type, description, severity, current_value, limit_value)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                violation['type'],
                violation['description'],
                violation['severity'],
                violation['current_value'],
                violation['limit_value']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving risk violation: {e}")
    
    def generate_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        try:
            latest_metrics = self.risk_metrics_history[-1] if self.risk_metrics_history else None
            
            if not latest_metrics:
                return {'error': 'No risk metrics available'}
            
            # Get recent violations
            violations = self.check_risk_limits(latest_metrics)
            
            report = {
                'timestamp': latest_metrics.timestamp,
                'portfolio_summary': {
                    'total_positions': len(self.positions),
                    'total_exposure': latest_metrics.total_exposure,
                    'net_exposure': latest_metrics.net_exposure,
                    'leverage': latest_metrics.leverage
                },
                'risk_metrics': {
                    'var_95': latest_metrics.var_1d_95,
                    'var_99': latest_metrics.var_1d_99,
                    'cvar_95': latest_metrics.cvar_1d_95,
                    'max_drawdown': latest_metrics.max_drawdown,
                    'correlation_risk': latest_metrics.correlation_risk
                },
                'concentration_analysis': {
                    'largest_position': latest_metrics.largest_position_pct,
                    'top_5_positions': latest_metrics.top_5_positions_pct,
                    'sector_breakdown': latest_metrics.sector_concentration
                },
                'stress_test_results': latest_metrics.stress_test_results,
                'risk_adjusted_returns': {
                    'sharpe_ratio': latest_metrics.sharpe_ratio,
                    'sortino_ratio': latest_metrics.sortino_ratio,
                    'calmar_ratio': latest_metrics.calmar_ratio,
                    'information_ratio': latest_metrics.information_ratio
                },
                'violations': violations,
                'recommendations': self._generate_risk_recommendations(latest_metrics, violations)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating risk report: {e}")
            return {'error': str(e)}
    
    def _generate_risk_recommendations(self, metrics: RiskMetrics, violations: List[Dict]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        try:
            # VaR recommendations
            if metrics.var_1d_95 > self.limits.max_portfolio_var * 0.8:
                recommendations.append("Consider reducing position sizes to lower portfolio VaR")
            
            # Concentration recommendations
            if metrics.largest_position_pct > self.limits.max_position_size * 0.8:
                recommendations.append("Reduce largest position size to improve diversification")
            
            # Correlation recommendations
            if metrics.avg_correlation > 0.5:
                recommendations.append("High average correlation detected - consider adding uncorrelated assets")
            
            # Leverage recommendations
            if metrics.leverage > self.limits.max_leverage * 0.8:
                recommendations.append("Consider reducing leverage to lower portfolio risk")
            
            # Drawdown recommendations
            if metrics.current_drawdown > self.limits.max_drawdown * 0.5:
                recommendations.append("Significant drawdown detected - review risk management strategy")
            
            # Stress test recommendations
            worst_scenario = min(metrics.stress_test_results.values()) if metrics.stress_test_results else 0
            if worst_scenario < -0.15:  # -15% in worst case
                recommendations.append("Portfolio shows high sensitivity to stress scenarios - consider hedging")
            
            # Sector concentration recommendations
            max_sector_concentration = max(metrics.sector_concentration.values()) if metrics.sector_concentration else 0
            if max_sector_concentration > self.limits.max_sector_concentration * 0.8:
                recommendations.append("High sector concentration - diversify across sectors")
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
        
        return recommendations

# Example usage and testing
async def main():
    """Test the institutional risk manager"""
    
    # Create risk manager
    risk_manager = InstitutionalRiskManager()
    
    # Add some test positions
    risk_manager.update_position('AAPL', 1000, 150.0, 'Technology')
    risk_manager.update_position('MSFT', 800, 250.0, 'Technology')
    risk_manager.update_position('JPM', 500, 120.0, 'Finance')
    risk_manager.update_position('JNJ', 300, 160.0, 'Healthcare')
    
    # Generate some mock portfolio returns
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    portfolio_returns = pd.Series(
        np.random.normal(0.001, 0.02, 100),  # 0.1% daily return, 2% volatility
        index=dates
    )
    risk_manager.portfolio_returns = portfolio_returns
    
    # Calculate risk metrics
    print("Calculating portfolio risk metrics...")
    risk_metrics = risk_manager.calculate_portfolio_risk_metrics()
    
    print(f"Portfolio VaR (95%): {risk_metrics.var_1d_95:.4f}")
    print(f"Portfolio Leverage: {risk_metrics.leverage:.2f}")
    print(f"Largest Position: {risk_metrics.largest_position_pct:.2%}")
    print(f"Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
    
    # Check risk limits
    violations = risk_manager.check_risk_limits(risk_metrics)
    if violations:
        print(f"\nRisk violations found: {len(violations)}")
        for violation in violations:
            print(f"  - {violation['description']}")
    else:
        print("\nNo risk limit violations")
    
    # Generate risk report
    print("\nGenerating risk report...")
    report = risk_manager.generate_risk_report()
    
    if 'error' not in report:
        print("Risk report generated successfully")
        print(f"Recommendations: {len(report['recommendations'])}")
        for rec in report['recommendations']:
            print(f"  - {rec}")
    else:
        print(f"Error generating report: {report['error']}")

if __name__ == "__main__":
    asyncio.run(main())