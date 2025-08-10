import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from config import DATABASE_CONFIG
from ensemble_models import EnsembleSignalGenerator
from advanced_features import AdvancedFeatureEngine
from data_manager import DataManager

@dataclass
class BacktestTrade:
    """Container for individual backtest trades"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    exit_price: float
    position_size: float
    predicted_accuracy: float
    actual_result: str  # 'WIN' or 'LOSS'
    pnl: float
    transaction_costs: float
    slippage: float
    confidence: float
    features_used: Dict[str, float] = field(default_factory=dict)

@dataclass
class BacktestResults:
    """Container for backtest results"""
    trades: List[BacktestTrade]
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    gross_pnl: float
    total_costs: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_duration: float
    best_trade: float
    worst_trade: float
    consecutive_wins: int
    consecutive_losses: int
    start_date: datetime
    end_date: datetime
    total_days: int
    avg_trades_per_day: float
    return_on_investment: float
    calmar_ratio: float
    sortino_ratio: float
    var_95: float
    cvar_95: float

class TransactionCostModel:
    """Models realistic transaction costs and slippage"""
    
    def __init__(self):
        self.logger = logging.getLogger('TransactionCostModel')
        
        # Default cost parameters (can be customized per broker/symbol)
        self.cost_params = {
            'spread_cost_pct': 0.0001,  # 1 pip spread cost
            'commission_per_trade': 0.0,  # No commission for binary options
            'slippage_base_pct': 0.00005,  # Base slippage
            'slippage_volatility_factor': 0.1,  # Slippage increases with volatility
            'funding_cost_annual': 0.02,  # Annual funding cost
        }
    
    def calculate_entry_costs(self, symbol: str, position_size: float, 
                             market_data: pd.DataFrame, volatility: float) -> Dict[str, float]:
        """Calculate costs for entering a position"""
        try:
            # Spread cost
            spread_cost = position_size * self.cost_params['spread_cost_pct']
            
            # Commission
            commission = self.cost_params['commission_per_trade']
            
            # Slippage (higher during volatile periods)
            base_slippage = position_size * self.cost_params['slippage_base_pct']
            volatility_slippage = base_slippage * volatility * self.cost_params['slippage_volatility_factor']
            total_slippage = base_slippage + volatility_slippage
            
            return {
                'spread_cost': spread_cost,
                'commission': commission,
                'slippage': total_slippage,
                'total_entry_cost': spread_cost + commission + total_slippage
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating entry costs: {e}")
            return {'total_entry_cost': 0.0}
    
    def calculate_exit_costs(self, symbol: str, position_size: float, 
                            hold_duration_hours: float, volatility: float) -> Dict[str, float]:
        """Calculate costs for exiting a position"""
        try:
            # For binary options, exit is automatic at expiry with no additional costs
            # But we can model opportunity costs or early exit penalties
            
            # Spread cost (if early exit is allowed)
            spread_cost = position_size * self.cost_params['spread_cost_pct'] * 0.5  # Half spread
            
            # Funding cost (for positions held longer than expected)
            funding_cost = 0.0
            if hold_duration_hours > 24:  # Only for positions held > 1 day
                daily_funding_rate = self.cost_params['funding_cost_annual'] / 365
                funding_cost = position_size * daily_funding_rate * (hold_duration_hours / 24)
            
            return {
                'spread_cost': spread_cost,
                'funding_cost': funding_cost,
                'total_exit_cost': spread_cost + funding_cost
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating exit costs: {e}")
            return {'total_exit_cost': 0.0}

class WalkForwardAnalyzer:
    """Implements walk-forward analysis for robust backtesting"""
    
    def __init__(self, train_period_days: int = 60, test_period_days: int = 10, 
                 step_size_days: int = 5):
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.step_size_days = step_size_days
        self.logger = logging.getLogger('WalkForwardAnalyzer')
    
    def generate_walk_forward_periods(self, start_date: datetime, 
                                    end_date: datetime) -> List[Dict[str, datetime]]:
        """Generate walk-forward train/test periods"""
        periods = []
        current_date = start_date
        
        while current_date + timedelta(days=self.train_period_days + self.test_period_days) <= end_date:
            train_start = current_date
            train_end = current_date + timedelta(days=self.train_period_days)
            test_start = train_end
            test_end = train_end + timedelta(days=self.test_period_days)
            
            periods.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })
            
            current_date += timedelta(days=self.step_size_days)
        
        self.logger.info(f"Generated {len(periods)} walk-forward periods")
        return periods
    
    def analyze_walk_forward_stability(self, period_results: List[BacktestResults]) -> Dict[str, float]:
        """Analyze stability across walk-forward periods"""
        try:
            if not period_results:
                return {}
            
            # Extract metrics from each period
            win_rates = [r.win_rate for r in period_results]
            sharpe_ratios = [r.sharpe_ratio for r in period_results if not np.isnan(r.sharpe_ratio)]
            profit_factors = [r.profit_factor for r in period_results if not np.isnan(r.profit_factor)]
            
            # Calculate stability metrics
            stability = {
                'periods_count': len(period_results),
                'win_rate_mean': np.mean(win_rates),
                'win_rate_std': np.std(win_rates),
                'win_rate_stability': 1 - (np.std(win_rates) / np.mean(win_rates)) if np.mean(win_rates) > 0 else 0,
                'positive_periods': sum(1 for r in period_results if r.total_pnl > 0),
                'negative_periods': sum(1 for r in period_results if r.total_pnl < 0),
                'consistency_ratio': sum(1 for r in period_results if r.total_pnl > 0) / len(period_results)
            }
            
            if sharpe_ratios:
                stability.update({
                    'sharpe_mean': np.mean(sharpe_ratios),
                    'sharpe_std': np.std(sharpe_ratios),
                    'sharpe_stability': 1 - (np.std(sharpe_ratios) / abs(np.mean(sharpe_ratios))) if np.mean(sharpe_ratios) != 0 else 0
                })
            
            if profit_factors:
                stability.update({
                    'profit_factor_mean': np.mean(profit_factors),
                    'profit_factor_std': np.std(profit_factors)
                })
            
            return stability
            
        except Exception as e:
            self.logger.error(f"Error analyzing walk-forward stability: {e}")
            return {}

class MonteCarloSimulator:
    """Performs Monte Carlo simulations for robustness testing"""
    
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
        self.logger = logging.getLogger('MonteCarloSimulator')
    
    def simulate_trade_outcomes(self, historical_trades: List[BacktestTrade]) -> Dict[str, Any]:
        """Simulate different trade outcome sequences"""
        try:
            if not historical_trades:
                return {}
            
            # Extract trade PnLs
            trade_pnls = [trade.pnl for trade in historical_trades]
            trade_win_rates = [1 if trade.actual_result == 'WIN' else 0 for trade in historical_trades]
            
            simulation_results = []
            
            for _ in range(self.n_simulations):
                # Bootstrap sample trades
                simulated_trades = np.random.choice(trade_pnls, size=len(trade_pnls), replace=True)
                
                # Calculate metrics for this simulation
                total_pnl = np.sum(simulated_trades)
                cumulative_pnl = np.cumsum(simulated_trades)
                max_drawdown = self._calculate_max_drawdown(cumulative_pnl)
                
                simulation_results.append({
                    'total_pnl': total_pnl,
                    'max_drawdown': max_drawdown,
                    'final_balance': 10000 + total_pnl  # Assume $10k starting balance
                })
            
            # Analyze simulation results
            total_pnls = [r['total_pnl'] for r in simulation_results]
            max_drawdowns = [r['max_drawdown'] for r in simulation_results]
            final_balances = [r['final_balance'] for r in simulation_results]
            
            return {
                'simulations_count': self.n_simulations,
                'profit_probability': sum(1 for pnl in total_pnls if pnl > 0) / len(total_pnls),
                'total_pnl_mean': np.mean(total_pnls),
                'total_pnl_std': np.std(total_pnls),
                'total_pnl_percentiles': {
                    '5th': np.percentile(total_pnls, 5),
                    '25th': np.percentile(total_pnls, 25),
                    '50th': np.percentile(total_pnls, 50),
                    '75th': np.percentile(total_pnls, 75),
                    '95th': np.percentile(total_pnls, 95)
                },
                'max_drawdown_mean': np.mean(max_drawdowns),
                'max_drawdown_95th_percentile': np.percentile(max_drawdowns, 95),
                'bankruptcy_probability': sum(1 for balance in final_balances if balance <= 0) / len(final_balances)
            }
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo simulation: {e}")
            return {}
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative returns"""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return abs(np.min(drawdown))

class StatisticalSignificanceTester:
    """Tests statistical significance of backtest results"""
    
    def __init__(self):
        self.logger = logging.getLogger('StatisticalSignificanceTester')
    
    def test_win_rate_significance(self, trades: List[BacktestTrade], 
                                 expected_win_rate: float = 0.5) -> Dict[str, float]:
        """Test if win rate is significantly different from expected"""
        try:
            n_trades = len(trades)
            n_wins = sum(1 for trade in trades if trade.actual_result == 'WIN')
            observed_win_rate = n_wins / n_trades if n_trades > 0 else 0
            
            # Binomial test
            p_value = stats.binom_test(n_wins, n_trades, expected_win_rate, alternative='two-sided')
            
            # Confidence interval for win rate
            confidence_interval = stats.binom.interval(0.95, n_trades, observed_win_rate)
            ci_lower = confidence_interval[0] / n_trades
            ci_upper = confidence_interval[1] / n_trades
            
            return {
                'observed_win_rate': observed_win_rate,
                'expected_win_rate': expected_win_rate,
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'confidence_interval_lower': ci_lower,
                'confidence_interval_upper': ci_upper,
                'sample_size': n_trades
            }
            
        except Exception as e:
            self.logger.error(f"Error testing win rate significance: {e}")
            return {}
    
    def test_sharpe_ratio_significance(self, trades: List[BacktestTrade]) -> Dict[str, float]:
        """Test if Sharpe ratio is significantly different from zero"""
        try:
            if not trades:
                return {}
            
            # Calculate daily returns
            daily_returns = self._calculate_daily_returns(trades)
            
            if len(daily_returns) < 2:
                return {}
            
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns, ddof=1)
            
            if std_return == 0:
                return {}
            
            sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized
            
            # T-test for mean return significantly different from zero
            t_stat, p_value = stats.ttest_1samp(daily_returns, 0)
            
            # Confidence interval for Sharpe ratio
            n = len(daily_returns)
            se_sharpe = np.sqrt((1 + 0.5 * sharpe_ratio**2) / n)
            ci_lower = sharpe_ratio - 1.96 * se_sharpe
            ci_upper = sharpe_ratio + 1.96 * se_sharpe
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'confidence_interval_lower': ci_lower,
                'confidence_interval_upper': ci_upper,
                'sample_size': n
            }
            
        except Exception as e:
            self.logger.error(f"Error testing Sharpe ratio significance: {e}")
            return {}
    
    def _calculate_daily_returns(self, trades: List[BacktestTrade]) -> np.ndarray:
        """Calculate daily returns from trades"""
        try:
            # Group trades by date
            daily_pnl = {}
            
            for trade in trades:
                date = trade.exit_time.date()
                if date not in daily_pnl:
                    daily_pnl[date] = 0
                daily_pnl[date] += trade.pnl
            
            # Convert to returns (assuming starting balance)
            starting_balance = 10000
            returns = []
            
            for pnl in daily_pnl.values():
                daily_return = pnl / starting_balance
                returns.append(daily_return)
            
            return np.array(returns)
            
        except Exception as e:
            self.logger.error(f"Error calculating daily returns: {e}")
            return np.array([])

class BacktestingEngine:
    """Main backtesting engine that orchestrates all components"""
    
    def __init__(self):
        self.logger = logging.getLogger('BacktestingEngine')
        
        # Initialize components
        self.data_manager = DataManager()
        self.feature_engine = AdvancedFeatureEngine()
        self.ensemble_generator = EnsembleSignalGenerator()
        self.cost_model = TransactionCostModel()
        self.walk_forward_analyzer = WalkForwardAnalyzer()
        self.monte_carlo_simulator = MonteCarloSimulator()
        self.significance_tester = StatisticalSignificanceTester()
        
        self.trades = []
        self.results = None
    
    async def run_comprehensive_backtest(self, symbols: List[str], start_date: datetime, 
                                       end_date: datetime, initial_balance: float = 10000,
                                       use_walk_forward: bool = True) -> Dict[str, Any]:
        """Run comprehensive backtesting with all features"""
        try:
            self.logger.info(f"Starting comprehensive backtest for {len(symbols)} symbols")
            
            comprehensive_results = {}
            
            if use_walk_forward:
                # Walk-forward analysis
                comprehensive_results['walk_forward'] = await self._run_walk_forward_backtest(
                    symbols, start_date, end_date, initial_balance
                )
            
            # Single period backtest for comparison
            comprehensive_results['single_period'] = await self._run_single_period_backtest(
                symbols, start_date, end_date, initial_balance
            )
            
            # Monte Carlo simulation
            if self.trades:
                comprehensive_results['monte_carlo'] = self.monte_carlo_simulator.simulate_trade_outcomes(
                    self.trades
                )
            
            # Statistical significance tests
            if self.trades:
                comprehensive_results['significance_tests'] = {
                    'win_rate_test': self.significance_tester.test_win_rate_significance(self.trades),
                    'sharpe_test': self.significance_tester.test_sharpe_ratio_significance(self.trades)
                }
            
            # Generate comprehensive report
            comprehensive_results['summary'] = self._generate_comprehensive_summary(comprehensive_results)
            
            return comprehensive_results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive backtest: {e}")
            raise
    
    async def _run_walk_forward_backtest(self, symbols: List[str], start_date: datetime,
                                       end_date: datetime, initial_balance: float) -> Dict[str, Any]:
        """Run walk-forward backtesting"""
        try:
            # Generate walk-forward periods
            periods = self.walk_forward_analyzer.generate_walk_forward_periods(start_date, end_date)
            
            period_results = []
            all_trades = []
            
            for i, period in enumerate(periods):
                self.logger.info(f"Processing walk-forward period {i+1}/{len(periods)}")
                
                # Train model on training period
                training_data = await self._get_combined_data(
                    symbols, period['train_start'], period['train_end']
                )
                
                if training_data is not None and len(training_data) > 100:
                    # Train ensemble model
                    self.ensemble_generator.train_ensemble(training_data)
                    
                    # Test on out-of-sample period
                    test_data = await self._get_combined_data(
                        symbols, period['test_start'], period['test_end']
                    )
                    
                    if test_data is not None:
                        period_trades = await self._simulate_trading_period(
                            symbols, test_data, initial_balance
                        )
                        
                        all_trades.extend(period_trades)
                        
                        # Calculate period results
                        period_result = self._calculate_backtest_results(
                            period_trades, period['test_start'], period['test_end']
                        )
                        period_results.append(period_result)
            
            # Analyze walk-forward stability
            stability_analysis = self.walk_forward_analyzer.analyze_walk_forward_stability(period_results)
            
            return {
                'periods': period_results,
                'stability_analysis': stability_analysis,
                'combined_results': self._calculate_backtest_results(
                    all_trades, start_date, end_date
                ) if all_trades else None
            }
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward backtest: {e}")
            return {}
    
    async def _run_single_period_backtest(self, symbols: List[str], start_date: datetime,
                                        end_date: datetime, initial_balance: float) -> BacktestResults:
        """Run single-period backtesting"""
        try:
            # Get data for entire period
            combined_data = await self._get_combined_data(symbols, start_date, end_date)
            
            if combined_data is None or len(combined_data) < 200:
                raise ValueError("Insufficient data for backtesting")
            
            # Split into train/test (80/20)
            split_idx = int(len(combined_data) * 0.8)
            train_data = combined_data.iloc[:split_idx]
            test_data = combined_data.iloc[split_idx:]
            
            # Train ensemble model
            self.ensemble_generator.train_ensemble(train_data)
            
            # Simulate trading on test data
            trades = await self._simulate_trading_period(symbols, test_data, initial_balance)
            self.trades = trades
            
            # Calculate results
            results = self._calculate_backtest_results(trades, start_date, end_date)
            self.results = results
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in single period backtest: {e}")
            raise
    
    async def _get_combined_data(self, symbols: List[str], start_date: datetime, 
                               end_date: datetime) -> Optional[pd.DataFrame]:
        """Get and combine data from multiple symbols"""
        try:
            combined_data = None
            
            for symbol in symbols:
                # Calculate period in days
                days = (end_date - start_date).days
                period = f"{days}d" if days <= 365 else "1y"
                
                data = await self.data_manager.get_historical_data(
                    symbol, period=period, interval="1m"
                )
                
                if data is not None and len(data) > 0:
                    # Filter by date range
                    data = data[
                        (data.index >= start_date) & (data.index <= end_date)
                    ]
                    
                    if len(data) > 0:
                        # Add symbol column
                        data['symbol'] = symbol
                        
                        if combined_data is None:
                            combined_data = data
                        else:
                            combined_data = pd.concat([combined_data, data])
            
            if combined_data is not None:
                combined_data = combined_data.sort_index()
                
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error getting combined data: {e}")
            return None
    
    async def _simulate_trading_period(self, symbols: List[str], data: pd.DataFrame, 
                                     initial_balance: float) -> List[BacktestTrade]:
        """Simulate trading for a specific period"""
        try:
            trades = []
            current_balance = initial_balance
            
            # Group data by symbol for processing
            symbol_data = {}
            for symbol in symbols:
                symbol_data[symbol] = data[data['symbol'] == symbol].drop('symbol', axis=1)
            
            # Process each symbol's data
            for symbol, sym_data in symbol_data.items():
                if len(sym_data) < 100:  # Need sufficient data
                    continue
                
                # Generate features and predictions
                features_data = self.feature_engine.generate_all_features(sym_data, symbol)
                
                if len(features_data) < 60:  # Need enough for sequence models
                    continue
                
                # Generate signals at regular intervals (e.g., every hour)
                signal_intervals = range(60, len(features_data), 60)  # Every 60 minutes
                
                for i in signal_intervals:
                    try:
                        # Get data slice for prediction
                        prediction_data = features_data.iloc[:i+1]
                        
                        # Generate ensemble prediction
                        ensemble_pred = self.ensemble_generator.predict(prediction_data)
                        
                        # Check if signal meets criteria
                        if self._should_trade(ensemble_pred, current_balance):
                            trade = await self._execute_backtest_trade(
                                symbol, prediction_data, ensemble_pred, current_balance
                            )
                            
                            if trade:
                                trades.append(trade)
                                current_balance += trade.pnl
                                
                                # Stop if balance gets too low
                                if current_balance < initial_balance * 0.2:
                                    self.logger.warning("Balance too low, stopping trading")
                                    break
                    
                    except Exception as e:
                        self.logger.warning(f"Error processing signal for {symbol}: {e}")
                        continue
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Error simulating trading period: {e}")
            return []
    
    def _should_trade(self, ensemble_pred, current_balance: float) -> bool:
        """Determine if a signal should result in a trade"""
        try:
            # Basic criteria for trading
            min_confidence = 0.7
            min_consensus = 0.6
            
            # Check confidence and consensus
            if (ensemble_pred.final_confidence >= min_confidence and 
                ensemble_pred.consensus_level >= min_consensus and
                ensemble_pred.final_prediction != 2):  # Not HOLD
                
                # Additional risk checks
                if current_balance > 1000:  # Minimum balance required
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error determining trade signal: {e}")
            return False
    
    async def _execute_backtest_trade(self, symbol: str, data: pd.DataFrame, 
                                    ensemble_pred, current_balance: float) -> Optional[BacktestTrade]:
        """Execute a backtest trade"""
        try:
            # Get current market data
            current_time = data.index[-1]
            current_price = data['close'].iloc[-1]
            
            # Calculate position size (2% of balance)
            position_size = current_balance * 0.02
            
            # Calculate volatility for cost model
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1] if len(returns) >= 20 else 0.01
            
            # Calculate transaction costs
            entry_costs = self.cost_model.calculate_entry_costs(
                symbol, position_size, data, volatility
            )
            
            # Determine trade direction
            direction = 'BUY' if ensemble_pred.final_prediction == 0 else 'SELL'
            
            # Set expiry time (binary options typically 2-5 minutes)
            expiry_minutes = 2
            exit_time = current_time + timedelta(minutes=expiry_minutes)
            
            # Simulate trade outcome
            # For backtesting, we need to look ahead to see the actual result
            # In real trading, this would be unknown
            
            # Find the price at expiry
            future_data = data[data.index > current_time]
            if len(future_data) >= expiry_minutes:
                exit_price = future_data['close'].iloc[expiry_minutes-1]
            else:
                # Use last available price if not enough future data
                exit_price = future_data['close'].iloc[-1] if len(future_data) > 0 else current_price
                exit_time = future_data.index[-1] if len(future_data) > 0 else current_time
            
            # Determine if trade was successful
            price_change = exit_price - current_price
            
            if direction == 'BUY':
                won = price_change > 0
            else:  # SELL
                won = price_change < 0
            
            # Calculate PnL (binary options: win = ~80% return, loss = -100%)
            if won:
                pnl = position_size * 0.8  # 80% return
                actual_result = 'WIN'
            else:
                pnl = -position_size  # Lose entire stake
                actual_result = 'LOSS'
            
            # Subtract transaction costs
            total_costs = entry_costs['total_entry_cost']
            net_pnl = pnl - total_costs
            
            # Create trade record
            trade = BacktestTrade(
                entry_time=current_time,
                exit_time=exit_time,
                symbol=symbol,
                direction=direction,
                entry_price=current_price,
                exit_price=exit_price,
                position_size=position_size,
                predicted_accuracy=ensemble_pred.final_confidence,
                actual_result=actual_result,
                pnl=net_pnl,
                transaction_costs=total_costs,
                slippage=entry_costs.get('slippage', 0),
                confidence=ensemble_pred.final_confidence,
                features_used=ensemble_pred.meta_features
            )
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Error executing backtest trade: {e}")
            return None
    
    def _calculate_backtest_results(self, trades: List[BacktestTrade], 
                                  start_date: datetime, end_date: datetime) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        try:
            if not trades:
                return BacktestResults(
                    trades=[], total_trades=0, winning_trades=0, losing_trades=0,
                    win_rate=0, total_pnl=0, gross_pnl=0, total_costs=0,
                    max_drawdown=0, sharpe_ratio=0, profit_factor=0,
                    avg_trade_duration=0, best_trade=0, worst_trade=0,
                    consecutive_wins=0, consecutive_losses=0,
                    start_date=start_date, end_date=end_date, total_days=0,
                    avg_trades_per_day=0, return_on_investment=0,
                    calmar_ratio=0, sortino_ratio=0, var_95=0, cvar_95=0
                )
            
            # Basic statistics
            total_trades = len(trades)
            winning_trades = sum(1 for trade in trades if trade.actual_result == 'WIN')
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # PnL statistics
            total_pnl = sum(trade.pnl for trade in trades)
            gross_pnl = sum(trade.pnl + trade.transaction_costs for trade in trades)
            total_costs = sum(trade.transaction_costs for trade in trades)
            
            # Trade statistics
            trade_pnls = [trade.pnl for trade in trades]
            best_trade = max(trade_pnls)
            worst_trade = min(trade_pnls)
            
            # Duration statistics
            durations = [(trade.exit_time - trade.entry_time).total_seconds() / 60 
                        for trade in trades]  # in minutes
            avg_trade_duration = np.mean(durations)
            
            # Consecutive wins/losses
            consecutive_wins = self._calculate_max_consecutive_wins(trades)
            consecutive_losses = self._calculate_max_consecutive_losses(trades)
            
            # Risk metrics
            cumulative_pnl = np.cumsum(trade_pnls)
            max_drawdown = self._calculate_max_drawdown_from_trades(cumulative_pnl)
            
            # Performance ratios
            daily_returns = self._calculate_daily_returns_from_trades(trades)
            sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
            
            winning_pnl = sum(trade.pnl for trade in trades if trade.actual_result == 'WIN')
            losing_pnl = abs(sum(trade.pnl for trade in trades if trade.actual_result == 'LOSS'))
            profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')
            
            # Time-based metrics
            total_days = (end_date - start_date).days
            avg_trades_per_day = total_trades / total_days if total_days > 0 else 0
            
            # Assuming initial balance of $10,000
            initial_balance = 10000
            return_on_investment = total_pnl / initial_balance
            
            # Additional risk metrics
            calmar_ratio = return_on_investment / max_drawdown if max_drawdown > 0 else 0
            sortino_ratio = self._calculate_sortino_ratio(daily_returns)
            var_95 = np.percentile(trade_pnls, 5) if trade_pnls else 0
            cvar_95 = np.mean([pnl for pnl in trade_pnls if pnl <= var_95]) if trade_pnls else 0
            
            return BacktestResults(
                trades=trades,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                gross_pnl=gross_pnl,
                total_costs=total_costs,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                profit_factor=profit_factor,
                avg_trade_duration=avg_trade_duration,
                best_trade=best_trade,
                worst_trade=worst_trade,
                consecutive_wins=consecutive_wins,
                consecutive_losses=consecutive_losses,
                start_date=start_date,
                end_date=end_date,
                total_days=total_days,
                avg_trades_per_day=avg_trades_per_day,
                return_on_investment=return_on_investment,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                var_95=var_95,
                cvar_95=cvar_95
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating backtest results: {e}")
            raise
    
    def _calculate_max_consecutive_wins(self, trades: List[BacktestTrade]) -> int:
        """Calculate maximum consecutive wins"""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade.actual_result == 'WIN':
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_max_consecutive_losses(self, trades: List[BacktestTrade]) -> int:
        """Calculate maximum consecutive losses"""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade.actual_result == 'LOSS':
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_max_drawdown_from_trades(self, cumulative_pnl: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative PnL"""
        if len(cumulative_pnl) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - peak
        return abs(np.min(drawdown))
    
    def _calculate_daily_returns_from_trades(self, trades: List[BacktestTrade]) -> np.ndarray:
        """Calculate daily returns from trades"""
        daily_pnl = {}
        
        for trade in trades:
            date = trade.exit_time.date()
            if date not in daily_pnl:
                daily_pnl[date] = 0
            daily_pnl[date] += trade.pnl
        
        # Convert to returns (assuming starting balance)
        starting_balance = 10000
        returns = [pnl / starting_balance for pnl in daily_pnl.values()]
        
        return np.array(returns)
    
    def _calculate_sharpe_ratio(self, daily_returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(daily_returns) < 2:
            return 0.0
        
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return * np.sqrt(252)  # Annualized
    
    def _calculate_sortino_ratio(self, daily_returns: np.ndarray) -> float:
        """Calculate Sortino ratio"""
        if len(daily_returns) < 2:
            return 0.0
        
        mean_return = np.mean(daily_returns)
        downside_returns = daily_returns[daily_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(downside_returns, ddof=1)
        
        if downside_deviation == 0:
            return 0.0
        
        return mean_return / downside_deviation * np.sqrt(252)  # Annualized
    
    def _generate_comprehensive_summary(self, comprehensive_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary of all results"""
        try:
            summary = {}
            
            # Single period results
            if 'single_period' in comprehensive_results:
                single_results = comprehensive_results['single_period']
                summary['single_period_summary'] = {
                    'total_trades': single_results.total_trades,
                    'win_rate': single_results.win_rate,
                    'total_pnl': single_results.total_pnl,
                    'sharpe_ratio': single_results.sharpe_ratio,
                    'max_drawdown': single_results.max_drawdown,
                    'profit_factor': single_results.profit_factor
                }
            
            # Walk-forward results
            if 'walk_forward' in comprehensive_results:
                wf_results = comprehensive_results['walk_forward']
                if 'stability_analysis' in wf_results:
                    stability = wf_results['stability_analysis']
                    summary['walk_forward_summary'] = {
                        'periods_tested': stability.get('periods_count', 0),
                        'consistency_ratio': stability.get('consistency_ratio', 0),
                        'win_rate_stability': stability.get('win_rate_stability', 0),
                        'average_win_rate': stability.get('win_rate_mean', 0)
                    }
            
            # Monte Carlo results
            if 'monte_carlo' in comprehensive_results:
                mc_results = comprehensive_results['monte_carlo']
                summary['monte_carlo_summary'] = {
                    'profit_probability': mc_results.get('profit_probability', 0),
                    'bankruptcy_probability': mc_results.get('bankruptcy_probability', 0),
                    'expected_pnl': mc_results.get('total_pnl_mean', 0),
                    'worst_case_5th_percentile': mc_results.get('total_pnl_percentiles', {}).get('5th', 0)
                }
            
            # Statistical significance
            if 'significance_tests' in comprehensive_results:
                sig_tests = comprehensive_results['significance_tests']
                summary['significance_summary'] = {
                    'win_rate_significant': sig_tests.get('win_rate_test', {}).get('is_significant', False),
                    'sharpe_significant': sig_tests.get('sharpe_test', {}).get('is_significant', False)
                }
            
            # Overall assessment
            summary['overall_assessment'] = self._assess_strategy_viability(comprehensive_results)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive summary: {e}")
            return {}
    
    def _assess_strategy_viability(self, comprehensive_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall strategy viability"""
        try:
            assessment = {
                'is_viable': False,
                'confidence_level': 'Low',
                'key_strengths': [],
                'key_weaknesses': [],
                'recommendations': []
            }
            
            score = 0
            max_score = 10
            
            # Check single period performance
            if 'single_period' in comprehensive_results:
                single = comprehensive_results['single_period']
                
                if single.win_rate >= 0.6:
                    score += 2
                    assessment['key_strengths'].append('High win rate')
                elif single.win_rate < 0.5:
                    assessment['key_weaknesses'].append('Low win rate')
                
                if single.sharpe_ratio >= 1.0:
                    score += 2
                    assessment['key_strengths'].append('Good risk-adjusted returns')
                elif single.sharpe_ratio < 0:
                    assessment['key_weaknesses'].append('Negative risk-adjusted returns')
                
                if single.profit_factor >= 1.5:
                    score += 1
                    assessment['key_strengths'].append('Strong profit factor')
            
            # Check walk-forward stability
            if 'walk_forward' in comprehensive_results:
                wf = comprehensive_results['walk_forward']
                stability = wf.get('stability_analysis', {})
                
                if stability.get('consistency_ratio', 0) >= 0.7:
                    score += 2
                    assessment['key_strengths'].append('Consistent across time periods')
                elif stability.get('consistency_ratio', 0) < 0.5:
                    assessment['key_weaknesses'].append('Inconsistent performance')
                
                if stability.get('win_rate_stability', 0) >= 0.8:
                    score += 1
                    assessment['key_strengths'].append('Stable win rate')
            
            # Check Monte Carlo results
            if 'monte_carlo' in comprehensive_results:
                mc = comprehensive_results['monte_carlo']
                
                if mc.get('profit_probability', 0) >= 0.7:
                    score += 1
                    assessment['key_strengths'].append('High probability of profit')
                
                if mc.get('bankruptcy_probability', 1) <= 0.05:
                    score += 1
                    assessment['key_strengths'].append('Low bankruptcy risk')
                elif mc.get('bankruptcy_probability', 1) > 0.2:
                    assessment['key_weaknesses'].append('High bankruptcy risk')
            
            # Check statistical significance
            if 'significance_tests' in comprehensive_results:
                sig = comprehensive_results['significance_tests']
                
                if sig.get('win_rate_test', {}).get('is_significant', False):
                    score += 1
                    assessment['key_strengths'].append('Statistically significant results')
            
            # Determine viability and confidence
            assessment['is_viable'] = score >= 6
            
            if score >= 8:
                assessment['confidence_level'] = 'High'
            elif score >= 6:
                assessment['confidence_level'] = 'Medium'
            else:
                assessment['confidence_level'] = 'Low'
            
            # Generate recommendations
            if score < 6:
                assessment['recommendations'].extend([
                    'Improve feature engineering',
                    'Optimize model parameters',
                    'Consider different timeframes',
                    'Enhance risk management'
                ])
            elif score < 8:
                assessment['recommendations'].extend([
                    'Fine-tune position sizing',
                    'Optimize entry/exit timing',
                    'Monitor performance closely'
                ])
            else:
                assessment['recommendations'].extend([
                    'Consider live trading with small position sizes',
                    'Continue monitoring and optimization',
                    'Scale up gradually'
                ])
            
            assessment['overall_score'] = f"{score}/{max_score}"
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing strategy viability: {e}")
            return {'is_viable': False, 'confidence_level': 'Unknown'}
    
    def generate_backtest_report(self, filepath: str = None) -> str:
        """Generate detailed backtest report"""
        try:
            if not self.results:
                return "No backtest results available"
            
            report = []
            report.append("=" * 60)
            report.append("COMPREHENSIVE BACKTEST REPORT")
            report.append("=" * 60)
            report.append("")
            
            # Summary statistics
            report.append(f"Total Trades: {self.results.total_trades}")
            report.append(f"Winning Trades: {self.results.winning_trades}")
            report.append(f"Losing Trades: {self.results.losing_trades}")
            report.append(f"Win Rate: {self.results.win_rate:.2%}")
            report.append("")
            
            # Performance metrics
            report.append(f"Total PnL: ${self.results.total_pnl:.2f}")
            report.append(f"ROI: {self.results.return_on_investment:.2%}")
            report.append(f"Sharpe Ratio: {self.results.sharpe_ratio:.2f}")
            report.append(f"Profit Factor: {self.results.profit_factor:.2f}")
            report.append(f"Max Drawdown: ${self.results.max_drawdown:.2f}")
            report.append("")
            
            # Risk metrics
            report.append(f"Best Trade: ${self.results.best_trade:.2f}")
            report.append(f"Worst Trade: ${self.results.worst_trade:.2f}")
            report.append(f"VaR (95%): ${self.results.var_95:.2f}")
            report.append(f"CVaR (95%): ${self.results.cvar_95:.2f}")
            report.append("")
            
            # Time-based metrics
            report.append(f"Average Trade Duration: {self.results.avg_trade_duration:.1f} minutes")
            report.append(f"Average Trades per Day: {self.results.avg_trades_per_day:.1f}")
            report.append(f"Consecutive Wins: {self.results.consecutive_wins}")
            report.append(f"Consecutive Losses: {self.results.consecutive_losses}")
            
            report_text = "\n".join(report)
            
            if filepath:
                with open(filepath, 'w') as f:
                    f.write(report_text)
                self.logger.info(f"Backtest report saved to {filepath}")
            
            return report_text
            
        except Exception as e:
            self.logger.error(f"Error generating backtest report: {e}")
            return "Error generating report"