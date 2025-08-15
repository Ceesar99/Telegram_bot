#!/usr/bin/env python3
"""
🔍 ADVANCED DATA VALIDATOR - PRODUCTION READY
Comprehensive data quality validation and cleaning for financial market data
Implements institutional-grade data integrity checks and anomaly detection
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import json
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import talib
warnings.filterwarnings('ignore')

from config import DATABASE_CONFIG, TIMEZONE

class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationIssue:
    """Data validation issue"""
    timestamp: datetime
    symbol: str
    issue_type: str
    severity: ValidationSeverity
    description: str
    affected_rows: int
    suggested_action: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataQualityReport:
    """Comprehensive data quality report"""
    symbol: str
    timeframe: str
    total_records: int
    valid_records: int
    invalid_records: int
    data_quality_score: float
    completeness_score: float
    consistency_score: float
    accuracy_score: float
    issues: List[ValidationIssue] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

class MarketDataValidator:
    """Advanced market data validation system"""
    
    def __init__(self):
        self.logger = logging.getLogger('MarketDataValidator')
        self.validation_rules = self._setup_validation_rules()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize validation tracking database"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_validation_issues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    issue_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT NOT NULL,
                    affected_rows INTEGER,
                    suggested_action TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_quality_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    total_records INTEGER,
                    valid_records INTEGER,
                    data_quality_score REAL,
                    completeness_score REAL,
                    consistency_score REAL,
                    accuracy_score REAL,
                    report_date TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, report_date)
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
    
    def _setup_validation_rules(self) -> Dict[str, Dict]:
        """Setup comprehensive validation rules"""
        return {
            'price_validation': {
                'min_price': 0.00001,  # Minimum valid price
                'max_price': 1000000,   # Maximum valid price
                'max_spread_pct': 0.1,  # Maximum spread percentage
                'max_price_change_pct': 0.5,  # Maximum price change between periods
                'ohlc_consistency': True  # Check OHLC consistency
            },
            'volume_validation': {
                'min_volume': 0,
                'max_volume_spike': 100,  # Volume spike threshold (x times average)
                'check_zero_volume': True
            },
            'timestamp_validation': {
                'check_duplicates': True,
                'check_gaps': True,
                'max_gap_minutes': 60,  # Maximum acceptable gap in minutes
                'check_future_dates': True,
                'check_weekends': True  # Check for weekend data in forex
            },
            'statistical_validation': {
                'outlier_threshold': 3,  # Z-score threshold for outliers
                'volatility_threshold': 0.3,  # Maximum acceptable volatility
                'autocorr_threshold': 0.95  # Maximum autocorrelation (detect repeated values)
            }
        }
    
    def validate_dataset(self, 
                        data: pd.DataFrame, 
                        symbol: str, 
                        timeframe: str = None) -> DataQualityReport:
        """Comprehensive dataset validation"""
        
        self.logger.info(f"Starting validation for {symbol} dataset with {len(data)} records")
        
        # Initialize report
        report = DataQualityReport(
            symbol=symbol,
            timeframe=timeframe or 'unknown',
            total_records=len(data),
            valid_records=0,
            invalid_records=0,
            data_quality_score=0.0,
            completeness_score=0.0,
            consistency_score=0.0,
            accuracy_score=0.0
        )
        
        if data.empty:
            report.issues.append(ValidationIssue(
                timestamp=datetime.now(TIMEZONE),
                symbol=symbol,
                issue_type='empty_dataset',
                severity=ValidationSeverity.CRITICAL,
                description='Dataset is empty',
                affected_rows=0,
                suggested_action='Check data source and collection process'
            ))
            return report
        
        # Run all validation checks
        self._validate_schema(data, report)
        self._validate_prices(data, report)
        self._validate_volumes(data, report)
        self._validate_timestamps(data, report)
        self._validate_ohlc_consistency(data, report)
        self._detect_anomalies(data, report)
        self._validate_statistical_properties(data, report)
        self._calculate_quality_scores(data, report)
        
        # Store validation results
        self._store_validation_report(report)
        
        self.logger.info(f"Validation completed for {symbol}: Quality Score {report.data_quality_score:.2%}")
        
        return report
    
    def _validate_schema(self, data: pd.DataFrame, report: DataQualityReport):
        """Validate dataset schema and required columns"""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close']
        optional_columns = ['volume']
        
        missing_required = [col for col in required_columns if col not in data.columns]
        
        if missing_required:
            report.issues.append(ValidationIssue(
                timestamp=datetime.now(TIMEZONE),
                symbol=report.symbol,
                issue_type='missing_columns',
                severity=ValidationSeverity.CRITICAL,
                description=f'Missing required columns: {missing_required}',
                affected_rows=len(data),
                suggested_action='Ensure data source provides all required columns'
            ))
        
        # Check data types
        numeric_columns = ['open', 'high', 'low', 'close']
        if 'volume' in data.columns:
            numeric_columns.append('volume')
        
        for col in numeric_columns:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                report.issues.append(ValidationIssue(
                    timestamp=datetime.now(TIMEZONE),
                    symbol=report.symbol,
                    issue_type='invalid_data_type',
                    severity=ValidationSeverity.ERROR,
                    description=f'Column {col} is not numeric',
                    affected_rows=len(data),
                    suggested_action=f'Convert {col} to numeric type'
                ))
    
    def _validate_prices(self, data: pd.DataFrame, report: DataQualityReport):
        """Validate price data quality"""
        rules = self.validation_rules['price_validation']
        price_columns = ['open', 'high', 'low', 'close']
        
        for col in price_columns:
            if col not in data.columns:
                continue
                
            # Check for negative or zero prices
            invalid_prices = (data[col] <= 0) | (data[col] > rules['max_price'])
            if invalid_prices.any():
                count = invalid_prices.sum()
                report.issues.append(ValidationIssue(
                    timestamp=datetime.now(TIMEZONE),
                    symbol=report.symbol,
                    issue_type='invalid_prices',
                    severity=ValidationSeverity.ERROR,
                    description=f'{count} invalid prices in {col} column',
                    affected_rows=count,
                    suggested_action='Remove or interpolate invalid price records'
                ))
            
            # Check for extreme price movements
            price_changes = data[col].pct_change().abs()
            extreme_changes = price_changes > rules['max_price_change_pct']
            if extreme_changes.any():
                count = extreme_changes.sum()
                report.issues.append(ValidationIssue(
                    timestamp=datetime.now(TIMEZONE),
                    symbol=report.symbol,
                    issue_type='extreme_price_movement',
                    severity=ValidationSeverity.WARNING,
                    description=f'{count} extreme price movements in {col}',
                    affected_rows=count,
                    suggested_action='Review for data errors or genuine market events'
                ))
        
        # Check bid-ask spread if available
        if 'bid' in data.columns and 'ask' in data.columns:
            spreads = ((data['ask'] - data['bid']) / data['bid']).abs()
            extreme_spreads = spreads > rules['max_spread_pct']
            if extreme_spreads.any():
                count = extreme_spreads.sum()
                report.issues.append(ValidationIssue(
                    timestamp=datetime.now(TIMEZONE),
                    symbol=report.symbol,
                    issue_type='extreme_spreads',
                    severity=ValidationSeverity.WARNING,
                    description=f'{count} records with extreme spreads',
                    affected_rows=count,
                    suggested_action='Check for data quality issues in bid/ask data'
                ))
    
    def _validate_volumes(self, data: pd.DataFrame, report: DataQualityReport):
        """Validate volume data"""
        if 'volume' not in data.columns:
            return
        
        rules = self.validation_rules['volume_validation']
        
        # Check for negative volumes
        negative_volumes = data['volume'] < 0
        if negative_volumes.any():
            count = negative_volumes.sum()
            report.issues.append(ValidationIssue(
                timestamp=datetime.now(TIMEZONE),
                symbol=report.symbol,
                issue_type='negative_volumes',
                severity=ValidationSeverity.ERROR,
                description=f'{count} negative volume values',
                affected_rows=count,
                suggested_action='Remove or correct negative volume records'
            ))
        
        # Check for volume spikes
        if len(data) > 10:
            avg_volume = data['volume'].rolling(window=10).mean()
            volume_spikes = data['volume'] > (avg_volume * rules['max_volume_spike'])
            volume_spikes = volume_spikes.fillna(False)
            
            if volume_spikes.any():
                count = volume_spikes.sum()
                report.issues.append(ValidationIssue(
                    timestamp=datetime.now(TIMEZONE),
                    symbol=report.symbol,
                    issue_type='volume_spikes',
                    severity=ValidationSeverity.WARNING,
                    description=f'{count} potential volume spikes detected',
                    affected_rows=count,
                    suggested_action='Review for genuine volume spikes vs data errors'
                ))
    
    def _validate_timestamps(self, data: pd.DataFrame, report: DataQualityReport):
        """Validate timestamp data"""
        if 'timestamp' not in data.columns:
            return
        
        rules = self.validation_rules['timestamp_validation']
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            try:
                timestamps = pd.to_datetime(data['timestamp'])
            except:
                report.issues.append(ValidationIssue(
                    timestamp=datetime.now(TIMEZONE),
                    symbol=report.symbol,
                    issue_type='invalid_timestamps',
                    severity=ValidationSeverity.CRITICAL,
                    description='Cannot convert timestamp column to datetime',
                    affected_rows=len(data),
                    suggested_action='Fix timestamp format in source data'
                ))
                return
        else:
            timestamps = data['timestamp']
        
        # Check for duplicates
        if rules['check_duplicates']:
            duplicates = timestamps.duplicated()
            if duplicates.any():
                count = duplicates.sum()
                report.issues.append(ValidationIssue(
                    timestamp=datetime.now(TIMEZONE),
                    symbol=report.symbol,
                    issue_type='duplicate_timestamps',
                    severity=ValidationSeverity.WARNING,
                    description=f'{count} duplicate timestamps',
                    affected_rows=count,
                    suggested_action='Remove or consolidate duplicate records'
                ))
        
        # Check for future dates
        if rules['check_future_dates']:
            future_dates = timestamps > datetime.now(TIMEZONE)
            if future_dates.any():
                count = future_dates.sum()
                report.issues.append(ValidationIssue(
                    timestamp=datetime.now(TIMEZONE),
                    symbol=report.symbol,
                    issue_type='future_timestamps',
                    severity=ValidationSeverity.ERROR,
                    description=f'{count} future timestamps',
                    affected_rows=count,
                    suggested_action='Remove future timestamp records'
                ))
        
        # Check for gaps
        if rules['check_gaps'] and len(timestamps) > 1:
            sorted_timestamps = timestamps.sort_values()
            time_diffs = sorted_timestamps.diff()
            max_gap = timedelta(minutes=rules['max_gap_minutes'])
            large_gaps = time_diffs > max_gap
            
            if large_gaps.any():
                count = large_gaps.sum()
                report.issues.append(ValidationIssue(
                    timestamp=datetime.now(TIMEZONE),
                    symbol=report.symbol,
                    issue_type='timestamp_gaps',
                    severity=ValidationSeverity.WARNING,
                    description=f'{count} large gaps in timestamps',
                    affected_rows=count,
                    suggested_action='Consider data interpolation for gaps'
                ))
    
    def _validate_ohlc_consistency(self, data: pd.DataFrame, report: DataQualityReport):
        """Validate OHLC price consistency"""
        ohlc_cols = ['open', 'high', 'low', 'close']
        
        if not all(col in data.columns for col in ohlc_cols):
            return
        
        # High should be >= Low
        invalid_high_low = data['high'] < data['low']
        if invalid_high_low.any():
            count = invalid_high_low.sum()
            report.issues.append(ValidationIssue(
                timestamp=datetime.now(TIMEZONE),
                symbol=report.symbol,
                issue_type='invalid_high_low',
                severity=ValidationSeverity.ERROR,
                description=f'{count} records where high < low',
                affected_rows=count,
                suggested_action='Fix or remove inconsistent OHLC records'
            ))
        
        # Open and Close should be within High-Low range
        open_out_of_range = (data['open'] < data['low']) | (data['open'] > data['high'])
        close_out_of_range = (data['close'] < data['low']) | (data['close'] > data['high'])
        
        if open_out_of_range.any():
            count = open_out_of_range.sum()
            report.issues.append(ValidationIssue(
                timestamp=datetime.now(TIMEZONE),
                symbol=report.symbol,
                issue_type='open_out_of_range',
                severity=ValidationSeverity.ERROR,
                description=f'{count} open prices outside high-low range',
                affected_rows=count,
                suggested_action='Fix or remove inconsistent open prices'
            ))
        
        if close_out_of_range.any():
            count = close_out_of_range.sum()
            report.issues.append(ValidationIssue(
                timestamp=datetime.now(TIMEZONE),
                symbol=report.symbol,
                issue_type='close_out_of_range',
                severity=ValidationSeverity.ERROR,
                description=f'{count} close prices outside high-low range',
                affected_rows=count,
                suggested_action='Fix or remove inconsistent close prices'
            ))
    
    def _detect_anomalies(self, data: pd.DataFrame, report: DataQualityReport):
        """Detect anomalies using machine learning"""
        try:
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                return
            
            # Prepare data for anomaly detection
            feature_data = data[numeric_columns].fillna(method='forward').fillna(0)
            
            if len(feature_data) < 10:  # Need minimum samples
                return
            
            # Scale features
            scaled_data = self.scaler.fit_transform(feature_data)
            
            # Detect anomalies
            anomaly_labels = self.anomaly_detector.fit_predict(scaled_data)
            anomalies = anomaly_labels == -1
            
            if anomalies.any():
                count = anomalies.sum()
                anomaly_indices = np.where(anomalies)[0]
                
                report.issues.append(ValidationIssue(
                    timestamp=datetime.now(TIMEZONE),
                    symbol=report.symbol,
                    issue_type='statistical_anomalies',
                    severity=ValidationSeverity.WARNING,
                    description=f'{count} statistical anomalies detected',
                    affected_rows=count,
                    suggested_action='Review anomalous records for data quality issues',
                    metadata={'anomaly_indices': anomaly_indices.tolist()}
                ))
                
        except Exception as e:
            self.logger.error(f"Anomaly detection error: {e}")
    
    def _validate_statistical_properties(self, data: pd.DataFrame, report: DataQualityReport):
        """Validate statistical properties of the data"""
        rules = self.validation_rules['statistical_validation']
        
        if 'close' not in data.columns or len(data) < 10:
            return
        
        # Check for repeated values (potential data feed issues)
        returns = data['close'].pct_change().dropna()
        
        if len(returns) > 0:
            # Check autocorrelation
            if len(returns) > 20:
                autocorr = returns.autocorr(lag=1)
                if abs(autocorr) > rules['autocorr_threshold']:
                    report.issues.append(ValidationIssue(
                        timestamp=datetime.now(TIMEZONE),
                        symbol=report.symbol,
                        issue_type='high_autocorrelation',
                        severity=ValidationSeverity.WARNING,
                        description=f'High autocorrelation in returns: {autocorr:.3f}',
                        affected_rows=len(returns),
                        suggested_action='Check for repeated or stale price data'
                    ))
            
            # Check volatility
            volatility = returns.std()
            if volatility > rules['volatility_threshold']:
                report.issues.append(ValidationIssue(
                    timestamp=datetime.now(TIMEZONE),
                    symbol=report.symbol,
                    issue_type='high_volatility',
                    severity=ValidationSeverity.WARNING,
                    description=f'High volatility detected: {volatility:.3f}',
                    affected_rows=len(returns),
                    suggested_action='Review for market events or data quality issues'
                ))
            
            # Check for outliers in returns
            z_scores = np.abs(stats.zscore(returns))
            outliers = z_scores > rules['outlier_threshold']
            
            if outliers.any():
                count = outliers.sum()
                report.issues.append(ValidationIssue(
                    timestamp=datetime.now(TIMEZONE),
                    symbol=report.symbol,
                    issue_type='return_outliers',
                    severity=ValidationSeverity.WARNING,
                    description=f'{count} outlier returns detected',
                    affected_rows=count,
                    suggested_action='Review outlier returns for data quality issues'
                ))
    
    def _calculate_quality_scores(self, data: pd.DataFrame, report: DataQualityReport):
        """Calculate overall data quality scores"""
        
        # Completeness score (percentage of non-null values)
        total_cells = len(data) * len(data.columns)
        non_null_cells = data.count().sum()
        report.completeness_score = non_null_cells / total_cells if total_cells > 0 else 0
        
        # Consistency score (based on validation issues)
        error_issues = [issue for issue in report.issues if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
        warning_issues = [issue for issue in report.issues if issue.severity == ValidationSeverity.WARNING]
        
        total_affected_rows = sum(issue.affected_rows for issue in report.issues)
        consistency_penalty = min(1.0, total_affected_rows / len(data)) if len(data) > 0 else 1.0
        report.consistency_score = max(0.0, 1.0 - consistency_penalty)
        
        # Accuracy score (based on severity of issues)
        error_penalty = len(error_issues) * 0.1
        warning_penalty = len(warning_issues) * 0.05
        report.accuracy_score = max(0.0, 1.0 - error_penalty - warning_penalty)
        
        # Overall quality score (weighted average)
        report.data_quality_score = (
            report.completeness_score * 0.3 +
            report.consistency_score * 0.4 +
            report.accuracy_score * 0.3
        )
        
        # Calculate valid records
        critical_issues = [issue for issue in report.issues if issue.severity == ValidationSeverity.CRITICAL]
        if critical_issues:
            report.valid_records = 0
        else:
            total_invalid = sum(issue.affected_rows for issue in error_issues)
            report.valid_records = max(0, len(data) - total_invalid)
        
        report.invalid_records = len(data) - report.valid_records
        
        # Store statistics
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            report.statistics = {
                'mean_return': float(returns.mean()) if len(returns) > 0 else 0,
                'volatility': float(returns.std()) if len(returns) > 0 else 0,
                'min_price': float(data['close'].min()),
                'max_price': float(data['close'].max()),
                'price_range': float(data['close'].max() - data['close'].min())
            }
    
    def clean_dataset(self, data: pd.DataFrame, report: DataQualityReport, 
                     aggressive_cleaning: bool = False) -> pd.DataFrame:
        """Clean dataset based on validation results"""
        
        self.logger.info(f"Starting data cleaning for {report.symbol}")
        cleaned_data = data.copy()
        
        # Remove critical issues
        critical_issues = [issue for issue in report.issues if issue.severity == ValidationSeverity.CRITICAL]
        if critical_issues:
            self.logger.warning(f"Critical issues found, returning empty dataset")
            return pd.DataFrame()
        
        # Handle specific issue types
        for issue in report.issues:
            if issue.severity == ValidationSeverity.ERROR or (aggressive_cleaning and issue.severity == ValidationSeverity.WARNING):
                
                if issue.issue_type == 'invalid_prices':
                    # Remove rows with invalid prices
                    price_cols = ['open', 'high', 'low', 'close']
                    for col in price_cols:
                        if col in cleaned_data.columns:
                            cleaned_data = cleaned_data[cleaned_data[col] > 0]
                
                elif issue.issue_type == 'negative_volumes':
                    # Remove rows with negative volumes
                    if 'volume' in cleaned_data.columns:
                        cleaned_data = cleaned_data[cleaned_data['volume'] >= 0]
                
                elif issue.issue_type == 'invalid_high_low':
                    # Remove rows where high < low
                    if all(col in cleaned_data.columns for col in ['high', 'low']):
                        cleaned_data = cleaned_data[cleaned_data['high'] >= cleaned_data['low']]
                
                elif issue.issue_type == 'open_out_of_range':
                    # Remove rows where open is outside high-low range
                    if all(col in cleaned_data.columns for col in ['open', 'high', 'low']):
                        cleaned_data = cleaned_data[
                            (cleaned_data['open'] >= cleaned_data['low']) & 
                            (cleaned_data['open'] <= cleaned_data['high'])
                        ]
                
                elif issue.issue_type == 'close_out_of_range':
                    # Remove rows where close is outside high-low range
                    if all(col in cleaned_data.columns for col in ['close', 'high', 'low']):
                        cleaned_data = cleaned_data[
                            (cleaned_data['close'] >= cleaned_data['low']) & 
                            (cleaned_data['close'] <= cleaned_data['high'])
                        ]
                
                elif issue.issue_type == 'duplicate_timestamps':
                    # Remove duplicate timestamps
                    if 'timestamp' in cleaned_data.columns:
                        cleaned_data = cleaned_data.drop_duplicates(subset=['timestamp'])
                
                elif issue.issue_type == 'future_timestamps':
                    # Remove future timestamps
                    if 'timestamp' in cleaned_data.columns:
                        cleaned_data = cleaned_data[
                            pd.to_datetime(cleaned_data['timestamp']) <= datetime.now(TIMEZONE)
                        ]
        
        # Sort by timestamp if available
        if 'timestamp' in cleaned_data.columns:
            cleaned_data = cleaned_data.sort_values('timestamp')
        
        removed_count = len(data) - len(cleaned_data)
        self.logger.info(f"Data cleaning completed: removed {removed_count} invalid records")
        
        return cleaned_data
    
    def _store_validation_report(self, report: DataQualityReport):
        """Store validation report to database"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            # Store issues
            for issue in report.issues:
                cursor.execute('''
                    INSERT INTO data_validation_issues 
                    (timestamp, symbol, issue_type, severity, description, affected_rows, suggested_action, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    issue.timestamp.isoformat(),
                    issue.symbol,
                    issue.issue_type,
                    issue.severity.value,
                    issue.description,
                    issue.affected_rows,
                    issue.suggested_action,
                    json.dumps(issue.metadata)
                ))
            
            # Store quality report
            cursor.execute('''
                INSERT OR REPLACE INTO data_quality_reports 
                (symbol, timeframe, total_records, valid_records, data_quality_score, 
                 completeness_score, consistency_score, accuracy_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                report.symbol,
                report.timeframe,
                report.total_records,
                report.valid_records,
                report.data_quality_score,
                report.completeness_score,
                report.consistency_score,
                report.accuracy_score
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing validation report: {e}")
    
    def get_validation_summary(self, symbol: str = None, days: int = 7) -> Dict[str, Any]:
        """Get validation summary for recent data"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            cutoff_date = datetime.now(TIMEZONE) - timedelta(days=days)
            
            # Base query
            where_clause = "WHERE created_at > ?"
            params = [cutoff_date.isoformat()]
            
            if symbol:
                where_clause += " AND symbol = ?"
                params.append(symbol)
            
            # Get issue summary
            cursor.execute(f'''
                SELECT issue_type, severity, COUNT(*) as count, SUM(affected_rows) as total_affected
                FROM data_validation_issues 
                {where_clause}
                GROUP BY issue_type, severity
                ORDER BY count DESC
            ''', params)
            
            issues = cursor.fetchall()
            
            # Get quality trends
            cursor.execute(f'''
                SELECT symbol, AVG(data_quality_score) as avg_quality, COUNT(*) as reports
                FROM data_quality_reports 
                {where_clause.replace('created_at', 'report_date')}
                GROUP BY symbol
                ORDER BY avg_quality DESC
            ''', params)
            
            quality_trends = cursor.fetchall()
            
            conn.close()
            
            return {
                'period_days': days,
                'issues_by_type': [
                    {
                        'issue_type': row[0],
                        'severity': row[1],
                        'count': row[2],
                        'total_affected_rows': row[3]
                    }
                    for row in issues
                ],
                'quality_by_symbol': [
                    {
                        'symbol': row[0],
                        'avg_quality_score': row[1],
                        'report_count': row[2]
                    }
                    for row in quality_trends
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting validation summary: {e}")
            return {}

# Example usage and testing
def test_data_validator():
    """Test the advanced data validator"""
    
    # Create sample data with various issues
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    
    # Create test data with intentional issues
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.normal(1.1000, 0.01, 100),
        'high': np.random.normal(1.1020, 0.01, 100),
        'low': np.random.normal(1.0980, 0.01, 100),
        'close': np.random.normal(1.1000, 0.01, 100),
        'volume': np.random.normal(1000, 200, 100)
    })
    
    # Introduce some issues
    test_data.loc[10, 'high'] = test_data.loc[10, 'low'] - 0.001  # high < low
    test_data.loc[20, 'close'] = -1.0  # negative price
    test_data.loc[30, 'volume'] = -100  # negative volume
    test_data.loc[40, 'close'] = test_data.loc[40, 'high'] + 0.001  # close > high
    test_data.loc[50] = test_data.loc[49]  # duplicate row
    
    # Initialize validator
    validator = MarketDataValidator()
    
    print("Testing data validator...")
    print(f"Test dataset: {len(test_data)} records")
    
    # Run validation
    report = validator.validate_dataset(test_data, 'EUR/USD', '1h')
    
    print(f"\nValidation Results:")
    print(f"  Total records: {report.total_records}")
    print(f"  Valid records: {report.valid_records}")
    print(f"  Invalid records: {report.invalid_records}")
    print(f"  Data quality score: {report.data_quality_score:.2%}")
    print(f"  Completeness score: {report.completeness_score:.2%}")
    print(f"  Consistency score: {report.consistency_score:.2%}")
    print(f"  Accuracy score: {report.accuracy_score:.2%}")
    
    print(f"\nIssues found ({len(report.issues)}):")
    for issue in report.issues:
        print(f"  {issue.severity.value.upper()}: {issue.issue_type} - {issue.description}")
    
    # Test data cleaning
    print(f"\nTesting data cleaning...")
    cleaned_data = validator.clean_dataset(test_data, report, aggressive_cleaning=True)
    print(f"  Original records: {len(test_data)}")
    print(f"  Cleaned records: {len(cleaned_data)}")
    print(f"  Removed records: {len(test_data) - len(cleaned_data)}")
    
    # Get validation summary
    print(f"\nValidation Summary:")
    summary = validator.get_validation_summary()
    print(json.dumps(summary, indent=2, default=str))

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_data_validator()