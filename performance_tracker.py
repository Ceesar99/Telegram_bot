import logging
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

from config import DATABASE_CONFIG, PERFORMANCE_TARGETS, SIGNAL_CONFIG

class PerformanceTracker:
    def __init__(self):
        self.logger = self._setup_logger()
        self._initialize_database()
        
    def _setup_logger(self):
        logger = logging.getLogger('PerformanceTracker')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('/workspace/logs/performance_tracker.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def _initialize_database(self):
        """Initialize performance tracking database tables"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            # Create signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    accuracy REAL,
                    ai_confidence REAL,
                    strength INTEGER,
                    entry_price REAL,
                    expiry_time TEXT,
                    duration INTEGER,
                    volatility_level TEXT,
                    risk_level TEXT,
                    status TEXT DEFAULT 'pending',
                    result TEXT,
                    actual_price REAL,
                    pnl REAL,
                    technical_indicators TEXT
                )
            ''')
            
            # Create performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    total_signals INTEGER,
                    winning_signals INTEGER,
                    losing_signals INTEGER,
                    win_rate REAL,
                    avg_accuracy REAL,
                    avg_confidence REAL,
                    total_pnl REAL,
                    best_pair TEXT,
                    worst_pair TEXT,
                    model_performance REAL
                )
            ''')
            
            # Create accuracy tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS accuracy_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    predicted_accuracy REAL,
                    actual_result TEXT,
                    accuracy_error REAL,
                    confidence_level REAL,
                    timeframe INTEGER
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
    
    def save_signal(self, signal_data: Dict):
        """Save signal to database for tracking"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            # Calculate expiry timestamp
            now = datetime.now()
            duration = signal_data.get('recommended_duration', 2)
            expiry_time = now + timedelta(minutes=duration + SIGNAL_CONFIG['signal_advance_time'])
            
            cursor.execute('''
                INSERT INTO signals (
                    timestamp, pair, direction, accuracy, ai_confidence, strength,
                    entry_price, expiry_time, duration, volatility_level, risk_level,
                    status, technical_indicators
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                now.isoformat(),
                signal_data['pair'],
                signal_data['direction'],
                signal_data['accuracy'],
                signal_data['ai_confidence'],
                signal_data['strength'],
                signal_data['entry_price'],
                expiry_time.isoformat(),
                duration,
                signal_data['volatility_level'],
                signal_data['risk_level'],
                'active',
                json.dumps(signal_data.get('technical_indicators', {}))
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Signal saved: {signal_data['pair']} {signal_data['direction']}")
            
        except Exception as e:
            self.logger.error(f"Error saving signal: {e}")
    
    def update_signal_result(self, signal_id: int, result: str, actual_price: float, pnl: float):
        """Update signal with actual result"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE signals 
                SET status = 'completed', result = ?, actual_price = ?, pnl = ?
                WHERE id = ?
            ''', (result, actual_price, pnl, signal_id))
            
            conn.commit()
            conn.close()
            
            # Track accuracy
            self._track_accuracy(signal_id, result)
            
            self.logger.info(f"Signal {signal_id} updated with result: {result}")
            
        except Exception as e:
            self.logger.error(f"Error updating signal result: {e}")
    
    def _track_accuracy(self, signal_id: int, actual_result: str):
        """Track accuracy of predictions"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            # Get signal details
            cursor.execute('''
                SELECT pair, accuracy, ai_confidence, duration, timestamp
                FROM signals WHERE id = ?
            ''', (signal_id,))
            
            signal_data = cursor.fetchone()
            if not signal_data:
                return
            
            pair, predicted_accuracy, confidence, timeframe, timestamp = signal_data
            
            # Calculate accuracy error
            actual_accuracy = 100.0 if actual_result == 'win' else 0.0
            accuracy_error = abs(predicted_accuracy - actual_accuracy)
            
            # Save accuracy tracking
            cursor.execute('''
                INSERT INTO accuracy_tracking (
                    timestamp, pair, predicted_accuracy, actual_result,
                    accuracy_error, confidence_level, timeframe
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, pair, predicted_accuracy, actual_result,
                accuracy_error, confidence, timeframe
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error tracking accuracy: {e}")
    
    def get_statistics(self) -> Dict:
        """Get comprehensive trading statistics"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            # Overall statistics
            cursor.execute('''
                SELECT COUNT(*) as total_signals,
                       SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as winning_trades,
                       SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) as losing_trades,
                       AVG(CASE WHEN result = 'win' THEN 100.0 ELSE 0.0 END) as win_rate,
                       AVG(accuracy) as avg_accuracy,
                       AVG(ai_confidence) as avg_confidence
                FROM signals WHERE status = 'completed'
            ''')
            
            overall_stats = cursor.fetchone()
            
            # Today's statistics
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute('''
                SELECT COUNT(*) as today_signals,
                       AVG(CASE WHEN result = 'win' THEN 100.0 ELSE 0.0 END) as today_win_rate
                FROM signals 
                WHERE DATE(timestamp) = ? AND status = 'completed'
            ''', (today,))
            
            today_stats = cursor.fetchone()
            
            # This week's statistics
            week_start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            cursor.execute('''
                SELECT COUNT(*) as week_signals,
                       AVG(CASE WHEN result = 'win' THEN 100.0 ELSE 0.0 END) as week_win_rate
                FROM signals 
                WHERE timestamp >= ? AND status = 'completed'
            ''', (week_start,))
            
            week_stats = cursor.fetchone()
            
            # This month's statistics
            month_start = datetime.now().replace(day=1).strftime('%Y-%m-%d')
            cursor.execute('''
                SELECT COUNT(*) as month_signals,
                       AVG(CASE WHEN result = 'win' THEN 100.0 ELSE 0.0 END) as month_win_rate
                FROM signals 
                WHERE timestamp >= ? AND status = 'completed'
            ''', (month_start,))
            
            month_stats = cursor.fetchone()
            
            # Accuracy by timeframe
            cursor.execute('''
                SELECT duration,
                       AVG(CASE WHEN result = 'win' THEN 100.0 ELSE 0.0 END) as accuracy
                FROM signals 
                WHERE status = 'completed'
                GROUP BY duration
            ''')
            
            timeframe_accuracy = cursor.fetchall()
            
            # Best performing pairs
            cursor.execute('''
                SELECT pair,
                       AVG(CASE WHEN result = 'win' THEN 100.0 ELSE 0.0 END) as win_rate,
                       COUNT(*) as trade_count
                FROM signals 
                WHERE status = 'completed'
                GROUP BY pair
                HAVING trade_count >= 5
                ORDER BY win_rate DESC
                LIMIT 3
            ''')
            
            best_pairs = cursor.fetchall()
            
            conn.close()
            
            # Build statistics dictionary
            stats = {
                'total_signals': overall_stats[0] or 0,
                'winning_trades': overall_stats[1] or 0,
                'losing_trades': overall_stats[2] or 0,
                'win_rate': overall_stats[3] or 0.0,
                'avg_accuracy': overall_stats[4] or 0.0,
                'avg_confidence': overall_stats[5] or 0.0,
                'today_signals': today_stats[0] or 0,
                'today_win_rate': today_stats[1] or 0.0,
                'week_signals': week_stats[0] or 0,
                'week_win_rate': week_stats[1] or 0.0,
                'month_signals': month_stats[0] or 0,
                'month_win_rate': month_stats[1] or 0.0,
                'model_accuracy': overall_stats[4] or 0.0,
                'target_achievement': min(100, (overall_stats[3] or 0) / PERFORMANCE_TARGETS['daily_win_rate'] * 100)
            }
            
            # Add timeframe accuracy
            for timeframe_data in timeframe_accuracy:
                duration, accuracy = timeframe_data
                stats[f'accuracy_{duration}min'] = accuracy or 0.0
            
            # Add best pairs
            for i, pair_data in enumerate(best_pairs, 1):
                pair, win_rate, count = pair_data
                stats[f'best_pair_{i}'] = pair
                stats[f'best_pair_{i}_rate'] = win_rate
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}
    
    def get_detailed_performance(self) -> Dict:
        """Get detailed performance analysis"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            
            # Load data into DataFrame
            df = pd.read_sql_query('''
                SELECT * FROM signals WHERE status = 'completed'
            ''', conn)
            
            if df.empty:
                return self._empty_performance_report()
            
            # Calculate performance metrics
            df['win'] = (df['result'] == 'win').astype(int)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            
            # Overall metrics
            overall_win_rate = df['win'].mean() * 100
            signal_accuracy = df['accuracy'].mean()
            total_trades = len(df)
            
            # Recent performance (last 30 days)
            recent_df = df[df['timestamp'] >= (datetime.now() - timedelta(days=30))]
            recent_wins = recent_df['win'].sum()
            recent_losses = len(recent_df) - recent_wins
            recent_win_rate = recent_df['win'].mean() * 100 if len(recent_df) > 0 else 0
            
            # Calculate streaks
            df_sorted = df.sort_values('timestamp')
            streaks = []
            current_streak = 0
            current_type = None
            
            for result in df_sorted['result']:
                if result == current_type:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        streaks.append((current_type, current_streak))
                    current_type = result
                    current_streak = 1
            
            if current_streak > 0:
                streaks.append((current_type, current_streak))
            
            win_streaks = [s[1] for s in streaks if s[0] == 'win']
            best_streak = max(win_streaks) if win_streaks else 0
            
            # Performance by timeframe
            timeframe_performance = df.groupby('duration')['win'].agg(['mean', 'count']).reset_index()
            
            # Calculate Sharpe ratio (simplified)
            if 'pnl' in df.columns and df['pnl'].std() > 0:
                sharpe_ratio = df['pnl'].mean() / df['pnl'].std()
            else:
                sharpe_ratio = 0
            
            # Calculate profit factor
            total_wins_pnl = df[df['result'] == 'win']['pnl'].sum() if 'pnl' in df.columns else 0
            total_losses_pnl = abs(df[df['result'] == 'loss']['pnl'].sum()) if 'pnl' in df.columns else 1
            profit_factor = total_wins_pnl / total_losses_pnl if total_losses_pnl > 0 else 0
            
            # Calculate max drawdown
            if 'pnl' in df.columns:
                cumulative_pnl = df['pnl'].cumsum()
                running_max = cumulative_pnl.expanding().max()
                drawdown = (cumulative_pnl - running_max) / running_max * 100
                max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0
            else:
                max_drawdown = 0
            
            # Risk-adjusted return
            risk_adjusted_return = overall_win_rate - max_drawdown if max_drawdown > 0 else overall_win_rate
            
            # VaR calculation (simplified)
            if 'pnl' in df.columns and len(df) > 0:
                var_95 = np.percentile(df['pnl'], 5)
            else:
                var_95 = 0
            
            conn.close()
            
            return {
                'overall_win_rate': overall_win_rate,
                'signal_accuracy': signal_accuracy,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'recent_wins': recent_wins,
                'recent_losses': recent_losses,
                'recent_win_rate': recent_win_rate,
                'best_streak': best_streak,
                'win_rate_2min': timeframe_performance[timeframe_performance['duration'] == 2]['mean'].iloc[0] * 100 if not timeframe_performance[timeframe_performance['duration'] == 2].empty else 0,
                'win_rate_3min': timeframe_performance[timeframe_performance['duration'] == 3]['mean'].iloc[0] * 100 if not timeframe_performance[timeframe_performance['duration'] == 3].empty else 0,
                'win_rate_5min': timeframe_performance[timeframe_performance['duration'] == 5]['mean'].iloc[0] * 100 if not timeframe_performance[timeframe_performance['duration'] == 5].empty else 0,
                'count_2min': timeframe_performance[timeframe_performance['duration'] == 2]['count'].iloc[0] if not timeframe_performance[timeframe_performance['duration'] == 2].empty else 0,
                'count_3min': timeframe_performance[timeframe_performance['duration'] == 3]['count'].iloc[0] if not timeframe_performance[timeframe_performance['duration'] == 3].empty else 0,
                'count_5min': timeframe_performance[timeframe_performance['duration'] == 5]['count'].iloc[0] if not timeframe_performance[timeframe_performance['duration'] == 5].empty else 0,
                'model_accuracy': signal_accuracy,
                'avg_confidence': df['ai_confidence'].mean(),
                'last_retrain': 'Never',  # Would track model retraining
                'max_drawdown': max_drawdown,
                'risk_adjusted_return': risk_adjusted_return,
                'var_95': var_95
            }
            
        except Exception as e:
            self.logger.error(f"Error getting detailed performance: {e}")
            return self._empty_performance_report()
    
    def _empty_performance_report(self) -> Dict:
        """Return empty performance report when no data available"""
        return {
            'overall_win_rate': 0.0,
            'signal_accuracy': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'recent_wins': 0,
            'recent_losses': 0,
            'recent_win_rate': 0.0,
            'best_streak': 0,
            'win_rate_2min': 0.0,
            'win_rate_3min': 0.0,
            'win_rate_5min': 0.0,
            'count_2min': 0,
            'count_3min': 0,
            'count_5min': 0,
            'model_accuracy': 0.0,
            'avg_confidence': 0.0,
            'last_retrain': 'Never',
            'max_drawdown': 0.0,
            'risk_adjusted_return': 0.0,
            'var_95': 0.0
        }
    
    def generate_performance_chart(self) -> Optional[str]:
        """Generate performance chart and return file path"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            
            # Load data
            df = pd.read_sql_query('''
                SELECT timestamp, result, accuracy, ai_confidence, pair
                FROM signals WHERE status = 'completed'
                ORDER BY timestamp
            ''', conn)
            
            conn.close()
            
            if df.empty:
                return None
            
            # Prepare data
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['win'] = (df['result'] == 'win').astype(int)
            df['date'] = df['timestamp'].dt.date
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Win rate over time
            daily_performance = df.groupby('date')['win'].mean().reset_index()
            daily_performance['win_rate'] = daily_performance['win'] * 100
            
            ax1.plot(daily_performance['date'], daily_performance['win_rate'], 
                    marker='o', linewidth=2, markersize=4)
            ax1.axhline(y=PERFORMANCE_TARGETS['daily_win_rate'], color='r', 
                       linestyle='--', label=f'Target: {PERFORMANCE_TARGETS["daily_win_rate"]}%')
            ax1.set_title('Daily Win Rate')
            ax1.set_ylabel('Win Rate (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Accuracy vs Confidence scatter
            ax2.scatter(df['ai_confidence'], df['accuracy'], 
                       c=df['win'], cmap='RdYlGn', alpha=0.6)
            ax2.set_xlabel('AI Confidence (%)')
            ax2.set_ylabel('Predicted Accuracy (%)')
            ax2.set_title('Accuracy vs Confidence')
            ax2.grid(True, alpha=0.3)
            
            # 3. Win rate by currency pair
            pair_performance = df.groupby('pair')['win'].agg(['mean', 'count']).reset_index()
            pair_performance = pair_performance[pair_performance['count'] >= 3]  # Min 3 trades
            pair_performance['win_rate'] = pair_performance['mean'] * 100
            
            if not pair_performance.empty:
                ax3.bar(range(len(pair_performance)), pair_performance['win_rate'])
                ax3.set_xticks(range(len(pair_performance)))
                ax3.set_xticklabels(pair_performance['pair'], rotation=45)
                ax3.set_title('Win Rate by Currency Pair')
                ax3.set_ylabel('Win Rate (%)')
                ax3.grid(True, alpha=0.3)
            
            # 4. Cumulative performance
            df_sorted = df.sort_values('timestamp')
            df_sorted['cumulative_wins'] = df_sorted['win'].cumsum()
            df_sorted['cumulative_trades'] = range(1, len(df_sorted) + 1)
            df_sorted['cumulative_win_rate'] = df_sorted['cumulative_wins'] / df_sorted['cumulative_trades'] * 100
            
            ax4.plot(df_sorted['cumulative_trades'], df_sorted['cumulative_win_rate'], 
                    linewidth=2, color='blue')
            ax4.axhline(y=PERFORMANCE_TARGETS['daily_win_rate'], color='r', 
                       linestyle='--', label=f'Target: {PERFORMANCE_TARGETS["daily_win_rate"]}%')
            ax4.set_title('Cumulative Win Rate')
            ax4.set_xlabel('Number of Trades')
            ax4.set_ylabel('Win Rate (%)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Adjust layout and save
            plt.tight_layout()
            
            chart_path = '/workspace/data/performance_chart.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            self.logger.error(f"Error generating performance chart: {e}")
            return None
    
    def calculate_model_accuracy(self) -> float:
        """Calculate actual model accuracy vs predictions"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT AVG(
                    CASE 
                        WHEN result = 'win' THEN 100.0 - ABS(accuracy - 100.0)
                        WHEN result = 'loss' THEN 100.0 - ABS(accuracy - 0.0)
                        ELSE 0
                    END
                ) as model_accuracy
                FROM signals 
                WHERE status = 'completed' AND result IS NOT NULL
            ''')
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result and result[0] else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating model accuracy: {e}")
            return 0.0
    
    def get_active_signals(self) -> List[Dict]:
        """Get currently active signals"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, timestamp, pair, direction, accuracy, ai_confidence, 
                       entry_price, expiry_time, duration, status
                FROM signals 
                WHERE status = 'active' AND expiry_time > ?
                ORDER BY timestamp DESC
            ''', (datetime.now().isoformat(),))
            
            active_signals = []
            for row in cursor.fetchall():
                signal_id, timestamp, pair, direction, accuracy, confidence, \
                entry_price, expiry_time, duration, status = row
                
                active_signals.append({
                    'id': signal_id,
                    'timestamp': timestamp,
                    'pair': pair,
                    'direction': direction,
                    'accuracy': accuracy,
                    'ai_confidence': confidence,
                    'entry_price': entry_price,
                    'expiry_time': expiry_time,
                    'duration': duration,
                    'status': status
                })
            
            conn.close()
            return active_signals
            
        except Exception as e:
            self.logger.error(f"Error getting active signals: {e}")
            return []
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            conn.close()
            return True
        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            return False
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to maintain database size"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
            
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            # Archive old completed signals
            cursor.execute('''
                DELETE FROM signals 
                WHERE status = 'completed' AND timestamp < ?
            ''', (cutoff_date,))
            
            # Clean old accuracy tracking
            cursor.execute('''
                DELETE FROM accuracy_tracking 
                WHERE timestamp < ?
            ''', (cutoff_date,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleaned up {deleted_count} old records")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    def export_performance_data(self, filepath: str = None) -> str:
        """Export performance data to CSV"""
        try:
            if filepath is None:
                filepath = f'/workspace/backup/performance_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            
            # Export all signals data
            df = pd.read_sql_query('''
                SELECT timestamp, pair, direction, accuracy, ai_confidence, strength,
                       entry_price, expiry_time, duration, volatility_level, risk_level,
                       status, result, actual_price, pnl
                FROM signals
                ORDER BY timestamp DESC
            ''', conn)
            
            conn.close()
            
            df.to_csv(filepath, index=False)
            self.logger.info(f"Performance data exported to {filepath}")
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error exporting performance data: {e}")
            return ""