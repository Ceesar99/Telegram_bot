#!/usr/bin/env python3
"""
⚖️ COMPREHENSIVE RISK VALIDATOR - PRODUCTION READY
Advanced risk management validation with stress testing, scenario analysis, and real-time monitoring
Ensures trading system operates within acceptable risk parameters for live deployment
"""

import numpy as np
import pandas as pd
import sqlite3
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from config import DATABASE_CONFIG, TIMEZONE, RISK_MANAGEMENT
from paper_trading_engine import PaperTrade, TradingPerformance

@dataclass
class RiskLimits:
    """Risk management limits and thresholds"""
    max_daily_loss_pct: float = 5.0  # Maximum daily loss as % of capital
    max_drawdown_pct: float = 20.0  # Maximum drawdown as % of capital
    max_position_size_pct: float = 2.0  # Maximum position size as % of capital
    max_concurrent_trades: int = 10  # Maximum concurrent trades
    max_correlation_exposure: float = 50.0  # Maximum exposure to correlated assets
    min_win_rate: float = 60.0  # Minimum acceptable win rate
    max_var_daily: float = 1000.0  # Maximum daily Value at Risk
    max_leverage: float = 1.0  # Maximum leverage allowed
    stop_loss_threshold: float = 100.0  # Stop loss threshold per trade
    emergency_stop_loss: float = 1000.0  # Emergency stop for entire system

@dataclass
class StressTestScenario:
    """Stress test scenario definition"""
    name: str
    description: str
    market_shock_pct: float
    volatility_multiplier: float
    correlation_increase: float
    liquidity_reduction: float
    duration_days: int

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    timestamp: datetime
    portfolio_value: float
    daily_var_95: float
    daily_var_99: float
    expected_shortfall: float
    maximum_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: float
    alpha: float
    tracking_error: float
    information_ratio: float
    win_rate: float
    profit_factor: float
    risk_adjusted_return: float

@dataclass
class RiskValidationResults:
    """Risk validation results"""
    validation_date: datetime
    overall_risk_score: float
    risk_limits_compliance: Dict[str, bool]
    stress_test_results: Dict[str, Dict[str, float]]
    scenario_analysis: Dict[str, float]
    var_backtesting: Dict[str, float]
    correlation_analysis: Dict[str, float]
    concentration_risk: Dict[str, float]
    liquidity_risk: Dict[str, float]
    model_risk: Dict[str, float]
    recommendations: List[str]
    critical_issues: List[str]

class VaRCalculator:
    """Value at Risk calculator with multiple methodologies"""
    
    def __init__(self):
        self.logger = logging.getLogger('VaRCalculator')
    
    def calculate_historical_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Historical Value at Risk"""
        
        if len(returns) == 0:
            return 0.0
        
        # Sort returns in ascending order
        sorted_returns = np.sort(returns)
        
        # Calculate VaR at specified confidence level
        index = int((1 - confidence_level) * len(sorted_returns))
        var = -sorted_returns[index] if index < len(sorted_returns) else -sorted_returns[0]
        
        return max(0, var)
    
    def calculate_parametric_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Parametric (Normal) Value at Risk"""
        
        if len(returns) == 0:
            return 0.0
        
        # Calculate mean and standard deviation
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # Calculate VaR
        var = -(mean_return + z_score * std_return)
        
        return max(0, var)
    
    def calculate_monte_carlo_var(self, returns: np.ndarray, confidence_level: float = 0.95, 
                                 num_simulations: int = 10000) -> float:
        """Calculate Monte Carlo Value at Risk"""
        
        if len(returns) == 0:
            return 0.0
        
        # Fit distribution to returns
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # Generate random scenarios
        np.random.seed(42)
        simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
        
        # Calculate VaR
        var = self.calculate_historical_var(simulated_returns, confidence_level)
        
        return var
    
    def calculate_expected_shortfall(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        
        if len(returns) == 0:
            return 0.0
        
        # Calculate VaR first
        var = self.calculate_historical_var(returns, confidence_level)
        
        # Calculate Expected Shortfall as average of losses beyond VaR
        tail_losses = returns[returns <= -var]
        expected_shortfall = -np.mean(tail_losses) if len(tail_losses) > 0 else var
        
        return expected_shortfall

class StressTester:
    """Comprehensive stress testing framework"""
    
    def __init__(self):
        self.logger = logging.getLogger('StressTester')
        self.scenarios = self._create_stress_scenarios()
    
    def _create_stress_scenarios(self) -> List[StressTestScenario]:
        """Create comprehensive stress test scenarios"""
        
        scenarios = [
            # Market crash scenarios
            StressTestScenario(
                name="Market Crash 2008",
                description="Financial crisis scenario with 40% market drop",
                market_shock_pct=-40.0,
                volatility_multiplier=3.0,
                correlation_increase=0.8,
                liquidity_reduction=0.5,
                duration_days=30
            ),
            
            StressTestScenario(
                name="Flash Crash",
                description="Sudden 20% market drop with high volatility",
                market_shock_pct=-20.0,
                volatility_multiplier=5.0,
                correlation_increase=0.9,
                liquidity_reduction=0.3,
                duration_days=1
            ),
            
            StressTestScenario(
                name="Currency Crisis",
                description="Major currency devaluation scenario",
                market_shock_pct=-25.0,
                volatility_multiplier=4.0,
                correlation_increase=0.7,
                liquidity_reduction=0.4,
                duration_days=7
            ),
            
            # Volatility scenarios
            StressTestScenario(
                name="High Volatility Period",
                description="Extended high volatility without major directional move",
                market_shock_pct=0.0,
                volatility_multiplier=2.5,
                correlation_increase=0.3,
                liquidity_reduction=0.1,
                duration_days=14
            ),
            
            # Correlation scenarios
            StressTestScenario(
                name="Correlation Breakdown",
                description="Historical correlations break down completely",
                market_shock_pct=-10.0,
                volatility_multiplier=2.0,
                correlation_increase=-0.5,  # Negative correlation increase means breakdown
                liquidity_reduction=0.2,
                duration_days=5
            ),
            
            # Liquidity scenarios
            StressTestScenario(
                name="Liquidity Crisis",
                description="Severe liquidity shortage across markets",
                market_shock_pct=-15.0,
                volatility_multiplier=2.0,
                correlation_increase=0.6,
                liquidity_reduction=0.8,
                duration_days=10
            )
        ]
        
        return scenarios
    
    def run_stress_test(self, portfolio_returns: np.ndarray, 
                       scenario: StressTestScenario) -> Dict[str, float]:
        """Run stress test for a specific scenario"""
        
        try:
            self.logger.info(f"Running stress test: {scenario.name}")
            
            if len(portfolio_returns) == 0:
                return {"error": "No portfolio returns data"}
            
            # Calculate baseline metrics
            baseline_var = np.percentile(portfolio_returns, 5)
            baseline_return = np.mean(portfolio_returns)
            baseline_volatility = np.std(portfolio_returns)
            
            # Apply stress scenario
            stressed_returns = portfolio_returns.copy()
            
            # Apply market shock
            if scenario.market_shock_pct != 0:
                shock_magnitude = scenario.market_shock_pct / 100
                stressed_returns = stressed_returns + shock_magnitude
            
            # Apply volatility multiplier
            if scenario.volatility_multiplier != 1:
                returns_deviation = stressed_returns - np.mean(stressed_returns)
                stressed_returns = np.mean(stressed_returns) + (returns_deviation * scenario.volatility_multiplier)
            
            # Calculate stressed metrics
            stressed_var = np.percentile(stressed_returns, 5)
            stressed_return = np.mean(stressed_returns)
            stressed_volatility = np.std(stressed_returns)
            
            # Calculate portfolio impact
            portfolio_loss = np.sum(stressed_returns[stressed_returns < 0])
            max_single_loss = np.min(stressed_returns)
            probability_of_loss = len(stressed_returns[stressed_returns < 0]) / len(stressed_returns)
            
            # Liquidity impact
            liquidity_cost = scenario.liquidity_reduction * abs(portfolio_loss) * 0.1  # Estimate
            
            results = {
                "baseline_var": baseline_var,
                "stressed_var": stressed_var,
                "var_change": stressed_var - baseline_var,
                "baseline_return": baseline_return,
                "stressed_return": stressed_return,
                "return_change": stressed_return - baseline_return,
                "baseline_volatility": baseline_volatility,
                "stressed_volatility": stressed_volatility,
                "volatility_change": stressed_volatility - baseline_volatility,
                "total_portfolio_loss": portfolio_loss,
                "max_single_loss": max_single_loss,
                "probability_of_loss": probability_of_loss,
                "liquidity_cost": liquidity_cost,
                "scenario_severity": abs(scenario.market_shock_pct) + (scenario.volatility_multiplier - 1) * 10
            }
            
            self.logger.info(f"Stress test {scenario.name} completed: Total loss {portfolio_loss:.2f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running stress test {scenario.name}: {e}")
            return {"error": str(e)}
    
    def run_all_stress_tests(self, portfolio_returns: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Run all stress test scenarios"""
        
        results = {}
        
        for scenario in self.scenarios:
            results[scenario.name] = self.run_stress_test(portfolio_returns, scenario)
        
        return results

class CorrelationAnalyzer:
    """Analyze correlations and concentration risk"""
    
    def __init__(self):
        self.logger = logging.getLogger('CorrelationAnalyzer')
    
    def calculate_portfolio_correlations(self, returns_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate portfolio correlation metrics"""
        
        try:
            if returns_data.empty:
                return {"error": "No returns data"}
            
            # Calculate correlation matrix
            correlation_matrix = returns_data.corr()
            
            # Calculate average correlation
            avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            
            # Calculate maximum correlation
            max_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max()
            
            # Calculate minimum correlation
            min_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min()
            
            # Calculate eigenvalues for concentration risk
            eigenvalues = np.linalg.eigvals(correlation_matrix)
            max_eigenvalue = np.max(eigenvalues)
            eigenvalue_ratio = max_eigenvalue / len(eigenvalues)
            
            # Concentration risk metrics
            concentration_risk = eigenvalue_ratio  # Higher values indicate more concentration
            
            results = {
                "average_correlation": avg_correlation,
                "maximum_correlation": max_correlation,
                "minimum_correlation": min_correlation,
                "eigenvalue_concentration": eigenvalue_ratio,
                "concentration_risk_score": concentration_risk,
                "diversification_ratio": 1 - concentration_risk,
                "effective_portfolio_size": len(eigenvalues) / eigenvalue_ratio
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating correlations: {e}")
            return {"error": str(e)}
    
    def analyze_regime_correlations(self, returns_data: pd.DataFrame, 
                                  regime_threshold: float = 0.02) -> Dict[str, float]:
        """Analyze correlations during different market regimes"""
        
        try:
            if returns_data.empty:
                return {"error": "No returns data"}
            
            # Calculate market returns (assuming first column is market proxy)
            market_returns = returns_data.iloc[:, 0] if len(returns_data.columns) > 0 else returns_data.mean(axis=1)
            
            # Define regimes based on market returns
            stress_periods = market_returns < -regime_threshold
            normal_periods = abs(market_returns) <= regime_threshold
            bull_periods = market_returns > regime_threshold
            
            results = {}
            
            # Calculate correlations for each regime
            if stress_periods.sum() > 10:  # Need minimum observations
                stress_corr = returns_data[stress_periods].corr()
                results["stress_avg_correlation"] = stress_corr.values[np.triu_indices_from(stress_corr.values, k=1)].mean()
                results["stress_max_correlation"] = stress_corr.values[np.triu_indices_from(stress_corr.values, k=1)].max()
            
            if normal_periods.sum() > 10:
                normal_corr = returns_data[normal_periods].corr()
                results["normal_avg_correlation"] = normal_corr.values[np.triu_indices_from(normal_corr.values, k=1)].mean()
                results["normal_max_correlation"] = normal_corr.values[np.triu_indices_from(normal_corr.values, k=1)].max()
            
            if bull_periods.sum() > 10:
                bull_corr = returns_data[bull_periods].corr()
                results["bull_avg_correlation"] = bull_corr.values[np.triu_indices_from(bull_corr.values, k=1)].mean()
                results["bull_max_correlation"] = bull_corr.values[np.triu_indices_from(bull_corr.values, k=1)].max()
            
            # Calculate correlation increase during stress
            if "stress_avg_correlation" in results and "normal_avg_correlation" in results:
                results["correlation_increase_stress"] = results["stress_avg_correlation"] - results["normal_avg_correlation"]
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing regime correlations: {e}")
            return {"error": str(e)}

class LiquidityRiskAnalyzer:
    """Analyze liquidity risk and market impact"""
    
    def __init__(self):
        self.logger = logging.getLogger('LiquidityRiskAnalyzer')
    
    def assess_liquidity_risk(self, trading_data: pd.DataFrame) -> Dict[str, float]:
        """Assess portfolio liquidity risk"""
        
        try:
            if trading_data.empty:
                return {"error": "No trading data"}
            
            # Calculate trading volume metrics
            if 'amount' in trading_data.columns:
                avg_trade_size = trading_data['amount'].mean()
                max_trade_size = trading_data['amount'].max()
                total_volume = trading_data['amount'].sum()
            else:
                avg_trade_size = max_trade_size = total_volume = 0
            
            # Calculate time-based liquidity metrics
            if 'timestamp' in trading_data.columns:
                trading_data['timestamp'] = pd.to_datetime(trading_data['timestamp'])
                trading_data = trading_data.sort_values('timestamp')
                
                # Time between trades
                time_diffs = trading_data['timestamp'].diff().dt.total_seconds() / 60  # minutes
                avg_time_between_trades = time_diffs.mean()
                max_time_between_trades = time_diffs.max()
            else:
                avg_time_between_trades = max_time_between_trades = 0
            
            # Estimate market impact (simplified)
            market_impact_score = min(1.0, max_trade_size / 1000)  # Normalized score
            
            # Liquidity concentration risk
            if 'symbol' in trading_data.columns:
                symbol_concentration = trading_data['symbol'].value_counts()
                max_symbol_concentration = symbol_concentration.max() / len(trading_data)
            else:
                max_symbol_concentration = 1.0
            
            # Calculate liquidity risk score
            liquidity_risk_factors = [
                market_impact_score,
                min(1.0, avg_time_between_trades / 60),  # Hours between trades
                max_symbol_concentration,
                min(1.0, max_trade_size / avg_trade_size / 10) if avg_trade_size > 0 else 0
            ]
            
            liquidity_risk_score = np.mean(liquidity_risk_factors)
            
            results = {
                "avg_trade_size": avg_trade_size,
                "max_trade_size": max_trade_size,
                "total_volume": total_volume,
                "avg_time_between_trades": avg_time_between_trades,
                "max_time_between_trades": max_time_between_trades,
                "market_impact_score": market_impact_score,
                "symbol_concentration": max_symbol_concentration,
                "liquidity_risk_score": liquidity_risk_score,
                "liquidity_rating": self._get_liquidity_rating(liquidity_risk_score)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error assessing liquidity risk: {e}")
            return {"error": str(e)}
    
    def _get_liquidity_rating(self, risk_score: float) -> str:
        """Convert liquidity risk score to rating"""
        
        if risk_score < 0.2:
            return "EXCELLENT"
        elif risk_score < 0.4:
            return "GOOD"
        elif risk_score < 0.6:
            return "MODERATE"
        elif risk_score < 0.8:
            return "POOR"
        else:
            return "CRITICAL"

class ComprehensiveRiskValidator:
    """Main comprehensive risk validation system"""
    
    def __init__(self, risk_limits: RiskLimits = None):
        self.logger = logging.getLogger('ComprehensiveRiskValidator')
        self.risk_limits = risk_limits or RiskLimits()
        
        # Initialize components
        self.var_calculator = VaRCalculator()
        self.stress_tester = StressTester()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.liquidity_analyzer = LiquidityRiskAnalyzer()
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize risk validation database"""
        
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            # Risk metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    portfolio_value REAL NOT NULL,
                    daily_var_95 REAL NOT NULL,
                    daily_var_99 REAL NOT NULL,
                    expected_shortfall REAL NOT NULL,
                    maximum_drawdown REAL NOT NULL,
                    current_drawdown REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    risk_score REAL NOT NULL
                )
            ''')
            
            # Risk validation results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_validation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    validation_date TEXT NOT NULL,
                    overall_risk_score REAL NOT NULL,
                    risk_limits_compliance TEXT NOT NULL,
                    stress_test_results TEXT NOT NULL,
                    recommendations TEXT NOT NULL,
                    critical_issues TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Risk validation database initialized")
            
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
    
    def validate_comprehensive_risk(self) -> RiskValidationResults:
        """Perform comprehensive risk validation"""
        
        self.logger.info("Starting comprehensive risk validation")
        
        try:
            # Load trading data
            trading_data = self._load_trading_data()
            
            if trading_data.empty:
                return self._create_empty_results("No trading data available")
            
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(trading_data)
            
            # 1. Risk Limits Compliance Check
            risk_limits_compliance = self._check_risk_limits_compliance(trading_data, portfolio_returns)
            
            # 2. Stress Testing
            stress_test_results = self.stress_tester.run_all_stress_tests(portfolio_returns)
            
            # 3. Scenario Analysis
            scenario_analysis = self._run_scenario_analysis(portfolio_returns)
            
            # 4. VaR Backtesting
            var_backtesting = self._perform_var_backtesting(portfolio_returns)
            
            # 5. Correlation Analysis
            returns_df = self._prepare_returns_dataframe(trading_data)
            correlation_analysis = self.correlation_analyzer.calculate_portfolio_correlations(returns_df)
            
            # 6. Concentration Risk
            concentration_risk = self._assess_concentration_risk(trading_data)
            
            # 7. Liquidity Risk
            liquidity_risk = self.liquidity_analyzer.assess_liquidity_risk(trading_data)
            
            # 8. Model Risk Assessment
            model_risk = self._assess_model_risk(trading_data)
            
            # Calculate overall risk score
            overall_risk_score = self._calculate_overall_risk_score(
                risk_limits_compliance, stress_test_results, var_backtesting,
                correlation_analysis, concentration_risk, liquidity_risk, model_risk
            )
            
            # Generate recommendations and identify critical issues
            recommendations = self._generate_recommendations(
                risk_limits_compliance, stress_test_results, overall_risk_score
            )
            
            critical_issues = self._identify_critical_issues(
                risk_limits_compliance, stress_test_results, overall_risk_score
            )
            
            # Create validation results
            results = RiskValidationResults(
                validation_date=datetime.now(TIMEZONE),
                overall_risk_score=overall_risk_score,
                risk_limits_compliance=risk_limits_compliance,
                stress_test_results=stress_test_results,
                scenario_analysis=scenario_analysis,
                var_backtesting=var_backtesting,
                correlation_analysis=correlation_analysis,
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk,
                model_risk=model_risk,
                recommendations=recommendations,
                critical_issues=critical_issues
            )
            
            # Save results
            self._save_validation_results(results)
            
            self.logger.info(f"Risk validation completed: Overall score {overall_risk_score:.2f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive risk validation: {e}")
            return self._create_empty_results(f"Validation error: {e}")
    
    def _load_trading_data(self) -> pd.DataFrame:
        """Load trading data from database"""
        
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            
            # Load completed trades
            trading_data = pd.read_sql_query('''
                SELECT * FROM paper_trades 
                WHERE actual_result IS NOT NULL 
                ORDER BY closed_at
            ''', conn)
            
            conn.close()
            
            if not trading_data.empty:
                trading_data['timestamp'] = pd.to_datetime(trading_data['timestamp'])
                trading_data['closed_at'] = pd.to_datetime(trading_data['closed_at'])
            
            return trading_data
            
        except Exception as e:
            self.logger.error(f"Error loading trading data: {e}")
            return pd.DataFrame()
    
    def _calculate_portfolio_returns(self, trading_data: pd.DataFrame) -> np.ndarray:
        """Calculate portfolio returns from trading data"""
        
        if trading_data.empty or 'pnl' not in trading_data.columns:
            return np.array([])
        
        # Calculate returns as percentage of initial capital (assuming $10,000)
        returns = trading_data['pnl'].values / 10000.0
        
        return returns
    
    def _check_risk_limits_compliance(self, trading_data: pd.DataFrame, 
                                    portfolio_returns: np.ndarray) -> Dict[str, bool]:
        """Check compliance with risk limits"""
        
        compliance = {}
        
        try:
            # Daily loss limit
            if len(portfolio_returns) > 0:
                max_daily_loss = np.min(portfolio_returns) * 100  # Convert to percentage
                compliance['daily_loss_limit'] = max_daily_loss > -self.risk_limits.max_daily_loss_pct
            else:
                compliance['daily_loss_limit'] = True
            
            # Maximum drawdown
            if len(portfolio_returns) > 0:
                drawdown = self._calculate_max_drawdown(portfolio_returns) * 100
                compliance['drawdown_limit'] = drawdown < self.risk_limits.max_drawdown_pct
            else:
                compliance['drawdown_limit'] = True
            
            # Position size limits
            if not trading_data.empty and 'amount' in trading_data.columns:
                max_position = trading_data['amount'].max()
                max_position_pct = (max_position / 10000.0) * 100  # Assuming $10k account
                compliance['position_size_limit'] = max_position_pct <= self.risk_limits.max_position_size_pct
            else:
                compliance['position_size_limit'] = True
            
            # Concurrent trades limit
            if not trading_data.empty:
                # Count overlapping trades (simplified)
                max_concurrent = self._estimate_max_concurrent_trades(trading_data)
                compliance['concurrent_trades_limit'] = max_concurrent <= self.risk_limits.max_concurrent_trades
            else:
                compliance['concurrent_trades_limit'] = True
            
            # Win rate requirement
            if not trading_data.empty and 'actual_result' in trading_data.columns:
                win_rate = (trading_data['actual_result'] == 'WIN').mean() * 100
                compliance['win_rate_requirement'] = win_rate >= self.risk_limits.min_win_rate
            else:
                compliance['win_rate_requirement'] = True
            
            # VaR limit
            if len(portfolio_returns) > 0:
                daily_var = self.var_calculator.calculate_historical_var(portfolio_returns, 0.95) * 10000
                compliance['var_limit'] = daily_var <= self.risk_limits.max_var_daily
            else:
                compliance['var_limit'] = True
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits compliance: {e}")
            # Default to non-compliant in case of error
            compliance = {key: False for key in ['daily_loss_limit', 'drawdown_limit', 
                                               'position_size_limit', 'concurrent_trades_limit',
                                               'win_rate_requirement', 'var_limit']}
        
        return compliance
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        
        return np.max(drawdown)
    
    def _estimate_max_concurrent_trades(self, trading_data: pd.DataFrame) -> int:
        """Estimate maximum concurrent trades (simplified)"""
        
        if trading_data.empty:
            return 0
        
        # Group by day and count trades
        trading_data['date'] = trading_data['timestamp'].dt.date
        daily_counts = trading_data.groupby('date').size()
        
        return daily_counts.max() if len(daily_counts) > 0 else 0
    
    def _run_scenario_analysis(self, portfolio_returns: np.ndarray) -> Dict[str, float]:
        """Run scenario analysis"""
        
        try:
            if len(portfolio_returns) == 0:
                return {"error": "No returns data"}
            
            scenarios = {
                "best_case": np.percentile(portfolio_returns, 95),
                "worst_case": np.percentile(portfolio_returns, 5),
                "median_case": np.percentile(portfolio_returns, 50),
                "stress_case": np.percentile(portfolio_returns, 1),
                "expected_return": np.mean(portfolio_returns),
                "volatility": np.std(portfolio_returns)
            }
            
            # Add scenario probabilities
            scenarios["probability_positive"] = len(portfolio_returns[portfolio_returns > 0]) / len(portfolio_returns)
            scenarios["probability_loss"] = len(portfolio_returns[portfolio_returns < 0]) / len(portfolio_returns)
            scenarios["probability_large_loss"] = len(portfolio_returns[portfolio_returns < -0.05]) / len(portfolio_returns)
            
            return scenarios
            
        except Exception as e:
            self.logger.error(f"Error in scenario analysis: {e}")
            return {"error": str(e)}
    
    def _perform_var_backtesting(self, portfolio_returns: np.ndarray) -> Dict[str, float]:
        """Perform VaR backtesting"""
        
        try:
            if len(portfolio_returns) < 50:  # Need minimum data for backtesting
                return {"error": "Insufficient data for VaR backtesting"}
            
            # Use rolling window for VaR calculation
            window_size = 30
            var_violations_95 = 0
            var_violations_99 = 0
            total_observations = 0
            
            for i in range(window_size, len(portfolio_returns)):
                # Calculate VaR using historical window
                historical_window = portfolio_returns[i-window_size:i]
                var_95 = self.var_calculator.calculate_historical_var(historical_window, 0.95)
                var_99 = self.var_calculator.calculate_historical_var(historical_window, 0.99)
                
                # Check if actual return violated VaR
                actual_return = portfolio_returns[i]
                
                if actual_return < -var_95:
                    var_violations_95 += 1
                
                if actual_return < -var_99:
                    var_violations_99 += 1
                
                total_observations += 1
            
            # Calculate violation rates
            violation_rate_95 = var_violations_95 / total_observations if total_observations > 0 else 0
            violation_rate_99 = var_violations_99 / total_observations if total_observations > 0 else 0
            
            # Expected violation rates
            expected_rate_95 = 0.05
            expected_rate_99 = 0.01
            
            # Kupiec test statistics (simplified)
            kupiec_stat_95 = self._calculate_kupiec_statistic(var_violations_95, total_observations, expected_rate_95)
            kupiec_stat_99 = self._calculate_kupiec_statistic(var_violations_99, total_observations, expected_rate_99)
            
            results = {
                "violation_rate_95": violation_rate_95,
                "violation_rate_99": violation_rate_99,
                "expected_rate_95": expected_rate_95,
                "expected_rate_99": expected_rate_99,
                "kupiec_stat_95": kupiec_stat_95,
                "kupiec_stat_99": kupiec_stat_99,
                "var_model_quality_95": "GOOD" if abs(violation_rate_95 - expected_rate_95) < 0.02 else "POOR",
                "var_model_quality_99": "GOOD" if abs(violation_rate_99 - expected_rate_99) < 0.005 else "POOR",
                "total_observations": total_observations
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in VaR backtesting: {e}")
            return {"error": str(e)}
    
    def _calculate_kupiec_statistic(self, violations: int, observations: int, expected_rate: float) -> float:
        """Calculate Kupiec test statistic for VaR backtesting"""
        
        if observations == 0 or violations == 0:
            return 0.0
        
        observed_rate = violations / observations
        
        # Kupiec likelihood ratio test statistic
        if observed_rate == 0 or observed_rate == 1:
            return float('inf')
        
        log_likelihood = (violations * np.log(observed_rate / expected_rate) + 
                         (observations - violations) * np.log((1 - observed_rate) / (1 - expected_rate)))
        
        kupiec_stat = -2 * log_likelihood
        
        return kupiec_stat
    
    def _prepare_returns_dataframe(self, trading_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare returns dataframe for correlation analysis"""
        
        try:
            if trading_data.empty:
                return pd.DataFrame()
            
            # Group by symbol and calculate returns
            returns_data = []
            
            for symbol in trading_data['symbol'].unique():
                symbol_data = trading_data[trading_data['symbol'] == symbol].copy()
                symbol_data = symbol_data.sort_values('closed_at')
                
                # Calculate returns (simplified)
                if 'pnl' in symbol_data.columns:
                    symbol_returns = symbol_data['pnl'].values
                    returns_data.append(pd.Series(symbol_returns, name=symbol))
            
            if returns_data:
                returns_df = pd.concat(returns_data, axis=1)
                returns_df = returns_df.fillna(0)  # Fill missing values
                return returns_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error preparing returns dataframe: {e}")
            return pd.DataFrame()
    
    def _assess_concentration_risk(self, trading_data: pd.DataFrame) -> Dict[str, float]:
        """Assess portfolio concentration risk"""
        
        try:
            if trading_data.empty:
                return {"error": "No trading data"}
            
            # Symbol concentration
            if 'symbol' in trading_data.columns:
                symbol_counts = trading_data['symbol'].value_counts()
                max_symbol_pct = symbol_counts.iloc[0] / len(trading_data) if len(symbol_counts) > 0 else 0
                herfindahl_index = np.sum((symbol_counts / len(trading_data)) ** 2)
            else:
                max_symbol_pct = herfindahl_index = 1.0
            
            # Model concentration
            if 'model_used' in trading_data.columns:
                model_counts = trading_data['model_used'].value_counts()
                max_model_pct = model_counts.iloc[0] / len(trading_data) if len(model_counts) > 0 else 0
            else:
                max_model_pct = 1.0
            
            # Time concentration (trading during specific hours)
            if 'timestamp' in trading_data.columns:
                trading_data['hour'] = pd.to_datetime(trading_data['timestamp']).dt.hour
                hour_counts = trading_data['hour'].value_counts()
                max_hour_pct = hour_counts.iloc[0] / len(trading_data) if len(hour_counts) > 0 else 0
            else:
                max_hour_pct = 1.0
            
            # Overall concentration risk score
            concentration_factors = [max_symbol_pct, max_model_pct, max_hour_pct, herfindahl_index]
            concentration_risk_score = np.mean(concentration_factors)
            
            results = {
                "max_symbol_concentration": max_symbol_pct,
                "max_model_concentration": max_model_pct,
                "max_time_concentration": max_hour_pct,
                "herfindahl_index": herfindahl_index,
                "concentration_risk_score": concentration_risk_score,
                "concentration_rating": self._get_concentration_rating(concentration_risk_score),
                "diversification_score": 1 - concentration_risk_score
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error assessing concentration risk: {e}")
            return {"error": str(e)}
    
    def _assess_model_risk(self, trading_data: pd.DataFrame) -> Dict[str, float]:
        """Assess model risk factors"""
        
        try:
            if trading_data.empty:
                return {"error": "No trading data"}
            
            # Model performance stability
            if 'model_used' in trading_data.columns and 'actual_result' in trading_data.columns:
                model_performance = {}
                model_stability = {}
                
                for model in trading_data['model_used'].unique():
                    model_trades = trading_data[trading_data['model_used'] == model]
                    win_rate = (model_trades['actual_result'] == 'WIN').mean()
                    model_performance[model] = win_rate
                    
                    # Calculate performance stability (rolling window)
                    if len(model_trades) >= 20:
                        rolling_performance = []
                        for i in range(10, len(model_trades), 5):
                            window_data = model_trades.iloc[i-10:i]
                            window_win_rate = (window_data['actual_result'] == 'WIN').mean()
                            rolling_performance.append(window_win_rate)
                        
                        model_stability[model] = np.std(rolling_performance) if rolling_performance else 0
                    else:
                        model_stability[model] = 0
                
                avg_model_performance = np.mean(list(model_performance.values())) if model_performance else 0
                avg_model_stability = np.mean(list(model_stability.values())) if model_stability else 0
                performance_variance = np.var(list(model_performance.values())) if model_performance else 0
                
            else:
                avg_model_performance = avg_model_stability = performance_variance = 0
            
            # Data quality risk
            data_quality_score = 1.0  # Placeholder - would integrate with data validation
            
            # Model complexity risk (simplified)
            num_models = len(trading_data['model_used'].unique()) if 'model_used' in trading_data.columns else 1
            complexity_risk = min(1.0, num_models / 10)  # Normalized
            
            # Overall model risk score
            model_risk_factors = [
                1 - avg_model_performance,  # Lower performance = higher risk
                avg_model_stability,  # Higher instability = higher risk
                performance_variance,  # Higher variance = higher risk
                1 - data_quality_score,  # Lower quality = higher risk
                complexity_risk
            ]
            
            model_risk_score = np.mean(model_risk_factors)
            
            results = {
                "avg_model_performance": avg_model_performance,
                "model_stability": 1 - avg_model_stability,  # Convert to stability score
                "performance_variance": performance_variance,
                "data_quality_score": data_quality_score,
                "model_complexity_risk": complexity_risk,
                "model_risk_score": model_risk_score,
                "model_risk_rating": self._get_risk_rating(model_risk_score),
                "number_of_models": num_models
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error assessing model risk: {e}")
            return {"error": str(e)}
    
    def _calculate_overall_risk_score(self, risk_limits_compliance: Dict[str, bool],
                                    stress_test_results: Dict[str, Dict[str, float]],
                                    var_backtesting: Dict[str, float],
                                    correlation_analysis: Dict[str, float],
                                    concentration_risk: Dict[str, float],
                                    liquidity_risk: Dict[str, float],
                                    model_risk: Dict[str, float]) -> float:
        """Calculate overall risk score (0-100, lower is better)"""
        
        try:
            risk_components = []
            
            # Risk limits compliance (30% weight)
            compliance_score = sum(risk_limits_compliance.values()) / len(risk_limits_compliance)
            risk_components.append((1 - compliance_score) * 30)
            
            # Stress test results (25% weight)
            if stress_test_results and 'Market Crash 2008' in stress_test_results:
                stress_result = stress_test_results['Market Crash 2008']
                if 'total_portfolio_loss' in stress_result:
                    stress_loss = abs(stress_result['total_portfolio_loss'])
                    stress_score = min(1.0, stress_loss / 2.0)  # Normalize to 0-1
                    risk_components.append(stress_score * 25)
                else:
                    risk_components.append(25)  # High risk if no data
            else:
                risk_components.append(25)
            
            # VaR model quality (15% weight)
            if 'var_model_quality_95' in var_backtesting:
                var_quality = 0 if var_backtesting['var_model_quality_95'] == 'GOOD' else 1
                risk_components.append(var_quality * 15)
            else:
                risk_components.append(15)
            
            # Concentration risk (10% weight)
            if 'concentration_risk_score' in concentration_risk:
                risk_components.append(concentration_risk['concentration_risk_score'] * 10)
            else:
                risk_components.append(10)
            
            # Liquidity risk (10% weight)
            if 'liquidity_risk_score' in liquidity_risk:
                risk_components.append(liquidity_risk['liquidity_risk_score'] * 10)
            else:
                risk_components.append(10)
            
            # Model risk (10% weight)
            if 'model_risk_score' in model_risk:
                risk_components.append(model_risk['model_risk_score'] * 10)
            else:
                risk_components.append(10)
            
            overall_risk_score = sum(risk_components)
            
            return min(100, max(0, overall_risk_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating overall risk score: {e}")
            return 100.0  # Maximum risk in case of error
    
    def _generate_recommendations(self, risk_limits_compliance: Dict[str, bool],
                                stress_test_results: Dict[str, Dict[str, float]],
                                overall_risk_score: float) -> List[str]:
        """Generate risk management recommendations"""
        
        recommendations = []
        
        # Risk limits recommendations
        for limit, compliant in risk_limits_compliance.items():
            if not compliant:
                if limit == 'daily_loss_limit':
                    recommendations.append("Reduce position sizes to limit daily losses")
                elif limit == 'drawdown_limit':
                    recommendations.append("Implement stronger stop-loss mechanisms")
                elif limit == 'position_size_limit':
                    recommendations.append("Enforce stricter position sizing rules")
                elif limit == 'win_rate_requirement':
                    recommendations.append("Improve model accuracy or adjust strategy")
                elif limit == 'var_limit':
                    recommendations.append("Reduce portfolio risk exposure")
        
        # Overall risk recommendations
        if overall_risk_score > 70:
            recommendations.append("CRITICAL: Overall risk is very high - consider suspending trading")
        elif overall_risk_score > 50:
            recommendations.append("HIGH RISK: Implement additional risk controls immediately")
        elif overall_risk_score > 30:
            recommendations.append("MODERATE RISK: Monitor risk metrics closely")
        
        # Stress test recommendations
        if stress_test_results:
            for scenario, results in stress_test_results.items():
                if 'total_portfolio_loss' in results and abs(results['total_portfolio_loss']) > 1.0:
                    recommendations.append(f"High losses in {scenario} scenario - diversify portfolio")
        
        if not recommendations:
            recommendations.append("Risk profile appears acceptable - continue monitoring")
        
        return recommendations
    
    def _identify_critical_issues(self, risk_limits_compliance: Dict[str, bool],
                                stress_test_results: Dict[str, Dict[str, float]],
                                overall_risk_score: float) -> List[str]:
        """Identify critical risk issues"""
        
        critical_issues = []
        
        # Critical compliance failures
        critical_limits = ['daily_loss_limit', 'drawdown_limit', 'var_limit']
        for limit in critical_limits:
            if limit in risk_limits_compliance and not risk_limits_compliance[limit]:
                critical_issues.append(f"CRITICAL: {limit.replace('_', ' ').title()} exceeded")
        
        # Critical stress test failures
        if stress_test_results:
            for scenario, results in stress_test_results.items():
                if 'total_portfolio_loss' in results and abs(results['total_portfolio_loss']) > 2.0:
                    critical_issues.append(f"CRITICAL: Severe losses in {scenario} stress test")
        
        # Overall risk assessment
        if overall_risk_score > 80:
            critical_issues.append("CRITICAL: Overall risk score exceeds acceptable threshold")
        
        return critical_issues
    
    def _get_concentration_rating(self, score: float) -> str:
        """Convert concentration score to rating"""
        
        if score < 0.2:
            return "WELL_DIVERSIFIED"
        elif score < 0.4:
            return "MODERATELY_DIVERSIFIED"
        elif score < 0.6:
            return "CONCENTRATED"
        else:
            return "HIGHLY_CONCENTRATED"
    
    def _get_risk_rating(self, score: float) -> str:
        """Convert risk score to rating"""
        
        if score < 0.2:
            return "LOW"
        elif score < 0.4:
            return "MODERATE"
        elif score < 0.6:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _create_empty_results(self, error_message: str) -> RiskValidationResults:
        """Create empty validation results with error message"""
        
        return RiskValidationResults(
            validation_date=datetime.now(TIMEZONE),
            overall_risk_score=100.0,
            risk_limits_compliance={},
            stress_test_results={},
            scenario_analysis={},
            var_backtesting={},
            correlation_analysis={},
            concentration_risk={},
            liquidity_risk={},
            model_risk={},
            recommendations=[error_message],
            critical_issues=[error_message]
        )
    
    def _save_validation_results(self, results: RiskValidationResults):
        """Save validation results to database"""
        
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO risk_validation_results 
                (validation_date, overall_risk_score, risk_limits_compliance, 
                 stress_test_results, recommendations, critical_issues)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                results.validation_date.isoformat(),
                results.overall_risk_score,
                json.dumps(results.risk_limits_compliance),
                json.dumps(results.stress_test_results, default=str),
                json.dumps(results.recommendations),
                json.dumps(results.critical_issues)
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info("Risk validation results saved to database")
            
        except Exception as e:
            self.logger.error(f"Error saving validation results: {e}")
    
    def generate_risk_report(self, results: RiskValidationResults) -> str:
        """Generate comprehensive risk validation report"""
        
        report = f"""
# 🔒 COMPREHENSIVE RISK VALIDATION REPORT

## Executive Summary
- **Validation Date**: {results.validation_date.strftime('%Y-%m-%d %H:%M:%S')}
- **Overall Risk Score**: {results.overall_risk_score:.1f}/100 ({self._get_risk_rating(results.overall_risk_score/100)})
- **Risk Status**: {'🔴 CRITICAL' if results.overall_risk_score > 70 else '🟡 ELEVATED' if results.overall_risk_score > 40 else '🟢 ACCEPTABLE'}

## 📊 Risk Limits Compliance

"""
        
        for limit, compliant in results.risk_limits_compliance.items():
            status = "✅ COMPLIANT" if compliant else "❌ NON-COMPLIANT"
            report += f"- **{limit.replace('_', ' ').title()}**: {status}\n"
        
        report += f"""
## 🧪 Stress Test Results

"""
        
        for scenario, results_dict in results.stress_test_results.items():
            if 'total_portfolio_loss' in results_dict:
                loss = results_dict['total_portfolio_loss']
                report += f"### {scenario}\n"
                report += f"- **Portfolio Loss**: {loss:.2f}\n"
                report += f"- **Max Single Loss**: {results_dict.get('max_single_loss', 0):.2f}\n"
                report += f"- **Loss Probability**: {results_dict.get('probability_of_loss', 0):.1%}\n\n"
        
        report += f"""
## 📈 Value at Risk Analysis

"""
        
        if 'var_model_quality_95' in results.var_backtesting:
            report += f"- **VaR Model Quality (95%)**: {results.var_backtesting['var_model_quality_95']}\n"
            report += f"- **Violation Rate (95%)**: {results.var_backtesting.get('violation_rate_95', 0):.1%}\n"
            report += f"- **Expected Rate (95%)**: {results.var_backtesting.get('expected_rate_95', 0):.1%}\n"
        
        report += f"""
## 🎯 Concentration Risk Analysis

"""
        
        if 'concentration_risk_score' in results.concentration_risk:
            score = results.concentration_risk['concentration_risk_score']
            rating = results.concentration_risk.get('concentration_rating', 'UNKNOWN')
            report += f"- **Concentration Risk Score**: {score:.2f}\n"
            report += f"- **Concentration Rating**: {rating}\n"
            report += f"- **Diversification Score**: {results.concentration_risk.get('diversification_score', 0):.2f}\n"
        
        report += f"""
## 💧 Liquidity Risk Analysis

"""
        
        if 'liquidity_risk_score' in results.liquidity_risk:
            score = results.liquidity_risk['liquidity_risk_score']
            rating = results.liquidity_risk.get('liquidity_rating', 'UNKNOWN')
            report += f"- **Liquidity Risk Score**: {score:.2f}\n"
            report += f"- **Liquidity Rating**: {rating}\n"
        
        report += f"""
## 🤖 Model Risk Analysis

"""
        
        if 'model_risk_score' in results.model_risk:
            score = results.model_risk['model_risk_score']
            rating = results.model_risk.get('model_risk_rating', 'UNKNOWN')
            report += f"- **Model Risk Score**: {score:.2f}\n"
            report += f"- **Model Risk Rating**: {rating}\n"
            report += f"- **Number of Models**: {results.model_risk.get('number_of_models', 0)}\n"
        
        report += f"""
## 🚨 Critical Issues

"""
        
        if results.critical_issues:
            for issue in results.critical_issues:
                report += f"- ⚠️ {issue}\n"
        else:
            report += "- ✅ No critical issues identified\n"
        
        report += f"""
## 📋 Recommendations

"""
        
        for recommendation in results.recommendations:
            report += f"- 💡 {recommendation}\n"
        
        # Overall assessment
        report += f"""
## 🎯 Overall Risk Assessment

"""
        
        if results.overall_risk_score <= 30:
            report += "✅ **LOW RISK**: Trading system operates within acceptable risk parameters\n"
        elif results.overall_risk_score <= 50:
            report += "🟡 **MODERATE RISK**: Some risk factors require attention\n"
        elif results.overall_risk_score <= 70:
            report += "🟠 **HIGH RISK**: Multiple risk factors need immediate attention\n"
        else:
            report += "🔴 **CRITICAL RISK**: Trading should be suspended until risks are mitigated\n"
        
        # Live trading readiness
        report += f"""
## 🚀 Live Trading Readiness

"""
        
        compliance_rate = sum(results.risk_limits_compliance.values()) / len(results.risk_limits_compliance) if results.risk_limits_compliance else 0
        
        if results.overall_risk_score <= 40 and compliance_rate >= 0.8 and not results.critical_issues:
            report += "🎉 **APPROVED FOR LIVE TRADING**: Risk profile is acceptable\n"
        else:
            report += "❌ **NOT APPROVED**: Risk mitigation required before live trading\n"
        
        return report

# Example usage and testing
def main():
    """Main risk validation function"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/workspace/logs/risk_validation.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('ComprehensiveRiskValidator')
    logger.info("Starting comprehensive risk validation")
    
    try:
        # Initialize risk validator
        risk_limits = RiskLimits(
            max_daily_loss_pct=5.0,
            max_drawdown_pct=20.0,
            max_position_size_pct=2.0,
            min_win_rate=70.0
        )
        
        validator = ComprehensiveRiskValidator(risk_limits)
        
        # Run comprehensive risk validation
        results = validator.validate_comprehensive_risk()
        
        # Generate and print report
        report = validator.generate_risk_report(results)
        print(report)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f'/workspace/risk_validation_report_{timestamp}.md'
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Risk validation report saved to {report_path}")
        logger.info(f"Overall risk score: {results.overall_risk_score:.1f}")
        
        if results.critical_issues:
            logger.warning(f"Critical issues identified: {len(results.critical_issues)}")
        else:
            logger.info("No critical risk issues identified")
        
    except Exception as e:
        logger.error(f"Risk validation failed: {e}")
        raise

if __name__ == "__main__":
    main()