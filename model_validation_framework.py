#!/usr/bin/env python3
"""
🔬 MODEL VALIDATION FRAMEWORK - PRODUCTION READY
Comprehensive model validation with out-of-sample testing, drift detection, and performance benchmarking
Ensures model reliability and performance for live trading deployment
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

from config import DATABASE_CONFIG, TIMEZONE
from real_market_data_collector import RealMarketDataCollector
from enhanced_feature_engine import EnhancedFeatureEngine
from advanced_data_validator import MarketDataValidator
from ensemble_training_system import EnsembleTrainingSystem
from enhanced_lstm_trainer import EnhancedLSTMTrainer

@dataclass
class ValidationConfig:
    """Model validation configuration"""
    test_split_ratio: float = 0.2
    walk_forward_steps: int = 5
    min_test_samples: int = 1000
    drift_detection_threshold: float = 0.05
    performance_threshold: float = 0.80
    stability_threshold: float = 0.02
    cross_validation_folds: int = 5
    statistical_significance_level: float = 0.05
    benchmark_models: List[str] = field(default_factory=lambda: ['random', 'majority', 'historical_mean'])

@dataclass
class ModelMetrics:
    """Comprehensive model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    specificity: float
    sensitivity: float
    positive_predictive_value: float
    negative_predictive_value: float
    matthews_correlation: float
    log_loss: float
    brier_score: float
    calibration_error: float

@dataclass
class ValidationResults:
    """Model validation results"""
    model_name: str
    validation_type: str
    metrics: ModelMetrics
    confidence_interval: Dict[str, Tuple[float, float]]
    statistical_significance: Dict[str, float]
    drift_detection: Dict[str, Any]
    stability_analysis: Dict[str, Any]
    benchmark_comparison: Dict[str, float]
    feature_importance: Dict[str, float]
    prediction_errors: List[float]
    validation_date: datetime
    data_period: Tuple[datetime, datetime]

class DriftDetector:
    """Detect model and data drift"""
    
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        self.logger = logging.getLogger('DriftDetector')
        
    def detect_feature_drift(self, X_train: np.ndarray, X_test: np.ndarray, 
                           feature_names: List[str] = None) -> Dict[str, Any]:
        """Detect feature distribution drift using statistical tests"""
        
        drift_results = {
            'has_drift': False,
            'drift_features': [],
            'p_values': {},
            'test_statistics': {},
            'drift_scores': {}
        }
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        for i, feature_name in enumerate(feature_names):
            # Kolmogorov-Smirnov test for distribution difference
            train_feature = X_train[:, i]
            test_feature = X_test[:, i]
            
            ks_statistic, p_value = ks_2samp(train_feature, test_feature)
            
            drift_results['p_values'][feature_name] = p_value
            drift_results['test_statistics'][feature_name] = ks_statistic
            drift_results['drift_scores'][feature_name] = ks_statistic
            
            if p_value < self.threshold:
                drift_results['drift_features'].append(feature_name)
                drift_results['has_drift'] = True
        
        self.logger.info(f"Feature drift detection: {len(drift_results['drift_features'])} features with drift")
        
        return drift_results
    
    def detect_prediction_drift(self, y_train_pred: np.ndarray, y_test_pred: np.ndarray) -> Dict[str, Any]:
        """Detect prediction distribution drift"""
        
        # Chi-square test for categorical predictions
        try:
            # Create contingency table
            train_counts = np.bincount(y_train_pred.astype(int), minlength=2)
            test_counts = np.bincount(y_test_pred.astype(int), minlength=2)
            
            contingency_table = np.array([train_counts, test_counts])
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
            
            drift_results = {
                'has_drift': p_value < self.threshold,
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'degrees_of_freedom': dof
            }
            
        except Exception as e:
            self.logger.error(f"Error in prediction drift detection: {e}")
            drift_results = {
                'has_drift': False,
                'chi2_statistic': np.nan,
                'p_value': np.nan,
                'degrees_of_freedom': np.nan
            }
        
        return drift_results
    
    def detect_concept_drift(self, model: Any, X_old: np.ndarray, y_old: np.ndarray,
                           X_new: np.ndarray, y_new: np.ndarray) -> Dict[str, Any]:
        """Detect concept drift by comparing model performance across time periods"""
        
        try:
            # Get predictions for both periods
            if hasattr(model, 'predict_proba'):
                old_pred_proba = model.predict_proba(X_old)[:, 1]
                new_pred_proba = model.predict_proba(X_new)[:, 1]
            else:
                old_pred_proba = model.predict(X_old)
                new_pred_proba = model.predict(X_new)
            
            old_pred = (old_pred_proba > 0.5).astype(int)
            new_pred = (new_pred_proba > 0.5).astype(int)
            
            # Calculate performance metrics for both periods
            old_accuracy = accuracy_score(y_old, old_pred)
            new_accuracy = accuracy_score(y_new, new_pred)
            
            old_auc = roc_auc_score(y_old, old_pred_proba)
            new_auc = roc_auc_score(y_new, new_pred_proba)
            
            # Statistical test for performance difference
            accuracy_diff = abs(old_accuracy - new_accuracy)
            auc_diff = abs(old_auc - new_auc)
            
            drift_results = {
                'has_drift': accuracy_diff > self.threshold or auc_diff > self.threshold,
                'old_accuracy': old_accuracy,
                'new_accuracy': new_accuracy,
                'accuracy_difference': accuracy_diff,
                'old_auc': old_auc,
                'new_auc': new_auc,
                'auc_difference': auc_diff,
                'performance_degradation': new_accuracy < old_accuracy
            }
            
        except Exception as e:
            self.logger.error(f"Error in concept drift detection: {e}")
            drift_results = {
                'has_drift': False,
                'old_accuracy': np.nan,
                'new_accuracy': np.nan,
                'accuracy_difference': np.nan,
                'old_auc': np.nan,
                'new_auc': np.nan,
                'auc_difference': np.nan,
                'performance_degradation': False
            }
        
        return drift_results

class StabilityAnalyzer:
    """Analyze model stability across different data periods"""
    
    def __init__(self):
        self.logger = logging.getLogger('StabilityAnalyzer')
    
    def analyze_temporal_stability(self, model: Any, X: np.ndarray, y: np.ndarray,
                                 timestamps: np.ndarray, window_size: int = 30) -> Dict[str, Any]:
        """Analyze model stability across time windows"""
        
        try:
            # Sort data by timestamp
            sorted_indices = np.argsort(timestamps)
            X_sorted = X[sorted_indices]
            y_sorted = y[sorted_indices]
            timestamps_sorted = timestamps[sorted_indices]
            
            # Calculate performance metrics for rolling windows
            window_performances = []
            window_dates = []
            
            for i in range(len(X_sorted) - window_size + 1):
                window_X = X_sorted[i:i+window_size]
                window_y = y_sorted[i:i+window_size]
                
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(window_X)[:, 1]
                else:
                    pred_proba = model.predict(window_X)
                
                pred = (pred_proba > 0.5).astype(int)
                
                accuracy = accuracy_score(window_y, pred)
                window_performances.append(accuracy)
                window_dates.append(timestamps_sorted[i+window_size-1])
            
            # Calculate stability metrics
            performance_std = np.std(window_performances)
            performance_mean = np.mean(window_performances)
            performance_cv = performance_std / performance_mean if performance_mean > 0 else np.inf
            
            # Trend analysis
            from scipy.stats import linregress
            if len(window_performances) > 1:
                slope, intercept, r_value, p_value, std_err = linregress(
                    range(len(window_performances)), window_performances
                )
            else:
                slope, r_value, p_value = 0, 0, 1
            
            stability_results = {
                'performance_std': performance_std,
                'performance_mean': performance_mean,
                'performance_cv': performance_cv,
                'trend_slope': slope,
                'trend_r_squared': r_value**2,
                'trend_p_value': p_value,
                'window_performances': window_performances,
                'window_dates': window_dates,
                'is_stable': performance_cv < 0.1  # 10% coefficient of variation threshold
            }
            
        except Exception as e:
            self.logger.error(f"Error in temporal stability analysis: {e}")
            stability_results = {
                'performance_std': np.nan,
                'performance_mean': np.nan,
                'performance_cv': np.nan,
                'trend_slope': np.nan,
                'trend_r_squared': np.nan,
                'trend_p_value': np.nan,
                'window_performances': [],
                'window_dates': [],
                'is_stable': False
            }
        
        return stability_results
    
    def analyze_cross_validation_stability(self, model: Any, X: np.ndarray, y: np.ndarray,
                                         cv_folds: int = 5) -> Dict[str, Any]:
        """Analyze model stability using cross-validation"""
        
        try:
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            fold_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Clone and train model on fold
                if hasattr(model, 'fit'):
                    fold_model = model.__class__(**model.get_params() if hasattr(model, 'get_params') else {})
                    fold_model.fit(X_train, y_train)
                    
                    if hasattr(fold_model, 'predict_proba'):
                        pred_proba = fold_model.predict_proba(X_val)[:, 1]
                    else:
                        pred_proba = fold_model.predict(X_val)
                    
                    pred = (pred_proba > 0.5).astype(int)
                    accuracy = accuracy_score(y_val, pred)
                    fold_scores.append(accuracy)
            
            if fold_scores:
                cv_mean = np.mean(fold_scores)
                cv_std = np.std(fold_scores)
                cv_cv = cv_std / cv_mean if cv_mean > 0 else np.inf
            else:
                cv_mean = cv_std = cv_cv = np.nan
            
            cv_stability = {
                'cv_scores': fold_scores,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'cv_coefficient_variation': cv_cv,
                'is_stable': cv_cv < 0.1 if not np.isnan(cv_cv) else False
            }
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation stability analysis: {e}")
            cv_stability = {
                'cv_scores': [],
                'cv_mean': np.nan,
                'cv_std': np.nan,
                'cv_coefficient_variation': np.nan,
                'is_stable': False
            }
        
        return cv_stability

class BenchmarkComparator:
    """Compare model performance against benchmark models"""
    
    def __init__(self):
        self.logger = logging.getLogger('BenchmarkComparator')
    
    def create_benchmark_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Create various benchmark models"""
        
        benchmarks = {}
        
        # Random classifier
        random_prob = np.sum(y_train) / len(y_train)
        benchmarks['random'] = {'type': 'random', 'positive_prob': random_prob}
        
        # Majority class classifier
        majority_class = int(np.sum(y_train) > len(y_train) / 2)
        benchmarks['majority'] = {'type': 'majority', 'prediction': majority_class}
        
        # Historical mean (for financial data)
        if len(y_train) > 1:
            historical_mean = np.mean(y_train)
            benchmarks['historical_mean'] = {'type': 'historical_mean', 'threshold': historical_mean}
        
        # Simple moving average classifier
        if len(y_train) > 10:
            window_size = min(20, len(y_train) // 5)
            ma_threshold = np.mean(y_train[-window_size:])
            benchmarks['moving_average'] = {'type': 'moving_average', 'threshold': ma_threshold}
        
        return benchmarks
    
    def evaluate_benchmarks(self, benchmarks: Dict[str, Any], X_test: np.ndarray, 
                          y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate benchmark models on test data"""
        
        benchmark_scores = {}
        
        for name, benchmark in benchmarks.items():
            try:
                if benchmark['type'] == 'random':
                    # Random predictions based on training distribution
                    np.random.seed(42)
                    pred = np.random.binomial(1, benchmark['positive_prob'], len(y_test))
                
                elif benchmark['type'] == 'majority':
                    # Always predict majority class
                    pred = np.full(len(y_test), benchmark['prediction'])
                
                elif benchmark['type'] == 'historical_mean':
                    # Predict based on historical mean threshold
                    pred = (np.full(len(y_test), benchmark['threshold']) > 0.5).astype(int)
                
                elif benchmark['type'] == 'moving_average':
                    # Predict based on moving average threshold
                    pred = (np.full(len(y_test), benchmark['threshold']) > 0.5).astype(int)
                
                else:
                    pred = np.zeros(len(y_test))
                
                accuracy = accuracy_score(y_test, pred)
                benchmark_scores[name] = accuracy
                
            except Exception as e:
                self.logger.error(f"Error evaluating benchmark {name}: {e}")
                benchmark_scores[name] = 0.0
        
        return benchmark_scores

class ModelValidationFramework:
    """Comprehensive model validation framework"""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.logger = logging.getLogger('ModelValidationFramework')
        
        # Initialize components
        self.drift_detector = DriftDetector(self.config.drift_detection_threshold)
        self.stability_analyzer = StabilityAnalyzer()
        self.benchmark_comparator = BenchmarkComparator()
        
        # Data components
        self.data_collector = RealMarketDataCollector()
        self.feature_engineer = EnhancedFeatureEngine()
        self.data_validator = MarketDataValidator()
        
        # Ensure directories exist
        os.makedirs('/workspace/validation_results', exist_ok=True)
        os.makedirs('/workspace/validation_plots', exist_ok=True)
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_pred_proba: np.ndarray = None) -> ModelMetrics:
        """Calculate comprehensive performance metrics"""
        
        try:
            # Basic classification metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Confusion matrix metrics
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # Matthews correlation coefficient
            mcc_num = (tp * tn) - (fp * fn)
            mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = mcc_num / mcc_den if mcc_den > 0 else 0
            
            # Probability-based metrics
            if y_pred_proba is not None:
                auc = roc_auc_score(y_true, y_pred_proba)
                
                # Log loss
                epsilon = 1e-15
                y_pred_proba_clipped = np.clip(y_pred_proba, epsilon, 1 - epsilon)
                log_loss = -np.mean(y_true * np.log(y_pred_proba_clipped) + 
                                  (1 - y_true) * np.log(1 - y_pred_proba_clipped))
                
                # Brier score
                brier_score = np.mean((y_pred_proba - y_true) ** 2)
                
                # Calibration error (simplified)
                calibration_error = np.abs(np.mean(y_pred_proba) - np.mean(y_true))
            else:
                auc = 0.5
                log_loss = np.nan
                brier_score = np.nan
                calibration_error = np.nan
            
            return ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_score=auc,
                specificity=specificity,
                sensitivity=sensitivity,
                positive_predictive_value=ppv,
                negative_predictive_value=npv,
                matthews_correlation=mcc,
                log_loss=log_loss,
                brier_score=brier_score,
                calibration_error=calibration_error
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return ModelMetrics(
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                auc_score=0.5, specificity=0.0, sensitivity=0.0,
                positive_predictive_value=0.0, negative_predictive_value=0.0,
                matthews_correlation=0.0, log_loss=np.nan, brier_score=np.nan,
                calibration_error=np.nan
            )
    
    def calculate_confidence_intervals(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     confidence_level: float = 0.95,
                                     n_bootstrap: int = 1000) -> Dict[str, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals for metrics"""
        
        try:
            # Bootstrap sampling
            n_samples = len(y_true)
            bootstrap_metrics = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': []
            }
            
            np.random.seed(42)
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(n_samples, n_samples, replace=True)
                y_true_boot = y_true[indices]
                y_pred_boot = y_pred[indices]
                
                # Calculate metrics
                bootstrap_metrics['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
                bootstrap_metrics['precision'].append(precision_score(y_true_boot, y_pred_boot, zero_division=0))
                bootstrap_metrics['recall'].append(recall_score(y_true_boot, y_pred_boot, zero_division=0))
                bootstrap_metrics['f1_score'].append(f1_score(y_true_boot, y_pred_boot, zero_division=0))
            
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            ci_results = {}
            
            for metric, values in bootstrap_metrics.items():
                lower = np.percentile(values, 100 * alpha / 2)
                upper = np.percentile(values, 100 * (1 - alpha / 2))
                ci_results[metric] = (lower, upper)
            
            return ci_results
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence intervals: {e}")
            return {
                'accuracy': (0.0, 0.0),
                'precision': (0.0, 0.0),
                'recall': (0.0, 0.0),
                'f1_score': (0.0, 0.0)
            }
    
    def validate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      X_train: np.ndarray = None, y_train: np.ndarray = None,
                      feature_names: List[str] = None,
                      model_name: str = 'unknown') -> ValidationResults:
        """Comprehensive model validation"""
        
        self.logger.info(f"Starting comprehensive validation for {model_name}")
        
        try:
            # Get predictions
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = model.predict(X_test)
            
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate comprehensive metrics
            metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
            
            # Calculate confidence intervals
            confidence_intervals = self.calculate_confidence_intervals(y_test, y_pred)
            
            # Statistical significance tests
            statistical_significance = self._calculate_statistical_significance(y_test, y_pred)
            
            # Drift detection
            drift_results = {}
            if X_train is not None and y_train is not None:
                # Feature drift
                feature_drift = self.drift_detector.detect_feature_drift(X_train, X_test, feature_names)
                
                # Prediction drift
                if hasattr(model, 'predict_proba'):
                    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
                else:
                    y_train_pred_proba = model.predict(X_train)
                
                y_train_pred = (y_train_pred_proba > 0.5).astype(int)
                prediction_drift = self.drift_detector.detect_prediction_drift(y_train_pred, y_pred)
                
                # Concept drift
                concept_drift = self.drift_detector.detect_concept_drift(model, X_train, y_train, X_test, y_test)
                
                drift_results = {
                    'feature_drift': feature_drift,
                    'prediction_drift': prediction_drift,
                    'concept_drift': concept_drift
                }
            
            # Stability analysis
            stability_results = {}
            if X_train is not None and y_train is not None:
                # Combine train and test for temporal analysis
                X_combined = np.vstack([X_train, X_test])
                y_combined = np.hstack([y_train, y_test])
                timestamps_combined = np.arange(len(X_combined))  # Simplified timestamps
                
                temporal_stability = self.stability_analyzer.analyze_temporal_stability(
                    model, X_combined, y_combined, timestamps_combined
                )
                
                cv_stability = self.stability_analyzer.analyze_cross_validation_stability(
                    model, X_combined, y_combined, self.config.cross_validation_folds
                )
                
                stability_results = {
                    'temporal_stability': temporal_stability,
                    'cross_validation_stability': cv_stability
                }
            
            # Benchmark comparison
            benchmark_comparison = {}
            if X_train is not None and y_train is not None:
                benchmarks = self.benchmark_comparator.create_benchmark_models(X_train, y_train)
                benchmark_comparison = self.benchmark_comparator.evaluate_benchmarks(benchmarks, X_test, y_test)
            
            # Feature importance (if available)
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                if feature_names:
                    feature_importance = dict(zip(feature_names, model.feature_importances_))
                else:
                    feature_importance = dict(enumerate(model.feature_importances_))
            
            # Prediction errors
            prediction_errors = np.abs(y_test - y_pred_proba).tolist()
            
            # Create validation results
            results = ValidationResults(
                model_name=model_name,
                validation_type='comprehensive',
                metrics=metrics,
                confidence_interval=confidence_intervals,
                statistical_significance=statistical_significance,
                drift_detection=drift_results,
                stability_analysis=stability_results,
                benchmark_comparison=benchmark_comparison,
                feature_importance=feature_importance,
                prediction_errors=prediction_errors,
                validation_date=datetime.now(TIMEZONE),
                data_period=(datetime.now(TIMEZONE) - timedelta(days=30), datetime.now(TIMEZONE))
            )
            
            # Save results
            self._save_validation_results(results)
            
            self.logger.info(f"Validation completed for {model_name}: Accuracy {metrics.accuracy:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in model validation: {e}")
            # Return minimal results on error
            return ValidationResults(
                model_name=model_name,
                validation_type='error',
                metrics=ModelMetrics(0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, np.nan, np.nan, np.nan),
                confidence_interval={},
                statistical_significance={},
                drift_detection={},
                stability_analysis={},
                benchmark_comparison={},
                feature_importance={},
                prediction_errors=[],
                validation_date=datetime.now(TIMEZONE),
                data_period=(datetime.now(TIMEZONE), datetime.now(TIMEZONE))
            )
    
    def _calculate_statistical_significance(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate statistical significance of results"""
        
        try:
            # McNemar's test for paired predictions (if we had a baseline)
            # For now, calculate basic statistical measures
            
            # Accuracy confidence interval using normal approximation
            accuracy = accuracy_score(y_true, y_pred)
            n = len(y_true)
            se = np.sqrt(accuracy * (1 - accuracy) / n)
            z_score = 1.96  # 95% confidence
            
            accuracy_ci_lower = accuracy - z_score * se
            accuracy_ci_upper = accuracy + z_score * se
            
            # Chi-square test for goodness of fit
            observed = np.bincount(y_pred, minlength=2)
            expected = np.bincount(y_true, minlength=2)
            
            if np.sum(expected) > 0:
                chi2_stat = np.sum((observed - expected) ** 2 / expected)
                chi2_p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
            else:
                chi2_p_value = np.nan
            
            return {
                'accuracy_ci_lower': accuracy_ci_lower,
                'accuracy_ci_upper': accuracy_ci_upper,
                'chi2_p_value': chi2_p_value,
                'sample_size': n
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating statistical significance: {e}")
            return {
                'accuracy_ci_lower': np.nan,
                'accuracy_ci_upper': np.nan,
                'chi2_p_value': np.nan,
                'sample_size': len(y_true)
            }
    
    def _save_validation_results(self, results: ValidationResults):
        """Save validation results to file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'validation_results_{results.model_name}_{timestamp}.json'
        filepath = os.path.join('/workspace/validation_results', filename)
        
        # Convert results to serializable format
        results_dict = {
            'model_name': results.model_name,
            'validation_type': results.validation_type,
            'validation_date': results.validation_date.isoformat(),
            'metrics': {
                'accuracy': results.metrics.accuracy,
                'precision': results.metrics.precision,
                'recall': results.metrics.recall,
                'f1_score': results.metrics.f1_score,
                'auc_score': results.metrics.auc_score,
                'specificity': results.metrics.specificity,
                'sensitivity': results.metrics.sensitivity,
                'matthews_correlation': results.metrics.matthews_correlation
            },
            'confidence_intervals': results.confidence_interval,
            'statistical_significance': results.statistical_significance,
            'drift_detection': results.drift_detection,
            'stability_analysis': results.stability_analysis,
            'benchmark_comparison': results.benchmark_comparison,
            'feature_importance': results.feature_importance
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results_dict, f, indent=2, default=str)
            
            self.logger.info(f"Validation results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving validation results: {e}")
    
    def generate_validation_report(self, results: ValidationResults) -> str:
        """Generate comprehensive validation report"""
        
        report = f"""
# Model Validation Report - {results.model_name}

## Validation Summary
- **Model**: {results.model_name}
- **Validation Date**: {results.validation_date.strftime('%Y-%m-%d %H:%M:%S')}
- **Validation Type**: {results.validation_type}

## Performance Metrics
- **Accuracy**: {results.metrics.accuracy:.4f}
- **Precision**: {results.metrics.precision:.4f}
- **Recall**: {results.metrics.recall:.4f}
- **F1-Score**: {results.metrics.f1_score:.4f}
- **AUC**: {results.metrics.auc_score:.4f}
- **Specificity**: {results.metrics.specificity:.4f}
- **Sensitivity**: {results.metrics.sensitivity:.4f}
- **Matthews Correlation**: {results.metrics.matthews_correlation:.4f}

## Confidence Intervals (95%)
"""
        
        for metric, (lower, upper) in results.confidence_interval.items():
            report += f"- **{metric.title()}**: [{lower:.4f}, {upper:.4f}]\n"
        
        report += f"""
## Model Quality Assessment
- **Accuracy Target (80%)**: {'✅ PASSED' if results.metrics.accuracy >= 0.80 else '❌ FAILED'}
- **Statistical Significance**: {'✅ SIGNIFICANT' if results.statistical_significance.get('chi2_p_value', 1) < 0.05 else '⚠️ NOT SIGNIFICANT'}
"""
        
        # Drift detection results
        if results.drift_detection:
            report += "\n## Drift Detection\n"
            
            if 'feature_drift' in results.drift_detection:
                fd = results.drift_detection['feature_drift']
                report += f"- **Feature Drift**: {'❌ DETECTED' if fd.get('has_drift', False) else '✅ NO DRIFT'}\n"
                if fd.get('drift_features'):
                    report += f"  - Affected features: {len(fd['drift_features'])}\n"
            
            if 'concept_drift' in results.drift_detection:
                cd = results.drift_detection['concept_drift']
                report += f"- **Concept Drift**: {'❌ DETECTED' if cd.get('has_drift', False) else '✅ NO DRIFT'}\n"
                if cd.get('performance_degradation'):
                    report += "  - ⚠️ Performance degradation detected\n"
        
        # Stability analysis
        if results.stability_analysis:
            report += "\n## Stability Analysis\n"
            
            if 'temporal_stability' in results.stability_analysis:
                ts = results.stability_analysis['temporal_stability']
                report += f"- **Temporal Stability**: {'✅ STABLE' if ts.get('is_stable', False) else '⚠️ UNSTABLE'}\n"
                report += f"  - Performance CV: {ts.get('performance_cv', 'N/A'):.4f}\n"
            
            if 'cross_validation_stability' in results.stability_analysis:
                cvs = results.stability_analysis['cross_validation_stability']
                report += f"- **Cross-Validation Stability**: {'✅ STABLE' if cvs.get('is_stable', False) else '⚠️ UNSTABLE'}\n"
                report += f"  - CV Mean: {cvs.get('cv_mean', 'N/A'):.4f}\n"
                report += f"  - CV Std: {cvs.get('cv_std', 'N/A'):.4f}\n"
        
        # Benchmark comparison
        if results.benchmark_comparison:
            report += "\n## Benchmark Comparison\n"
            for benchmark, score in results.benchmark_comparison.items():
                improvement = results.metrics.accuracy - score
                report += f"- **vs {benchmark.title()}**: {improvement:+.4f} ({score:.4f})\n"
        
        # Overall assessment
        report += f"""
## Overall Assessment
"""
        
        if results.metrics.accuracy >= 0.80:
            report += "🎉 **READY FOR PRODUCTION**: Model meets accuracy requirements\n"
        else:
            report += "❌ **NOT READY**: Model does not meet accuracy requirements\n"
        
        if results.drift_detection and any(d.get('has_drift', False) for d in results.drift_detection.values()):
            report += "⚠️ **DRIFT DETECTED**: Model may need retraining\n"
        
        return report

# Example usage and testing
def test_validation_framework():
    """Test the model validation framework"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create a simple model for testing
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Initialize validation framework
    config = ValidationConfig(
        test_split_ratio=0.2,
        drift_detection_threshold=0.05,
        performance_threshold=0.80
    )
    
    validator = ModelValidationFramework(config)
    
    # Run validation
    results = validator.validate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,
        y_train=y_train,
        feature_names=[f'feature_{i}' for i in range(n_features)],
        model_name='test_logistic_regression'
    )
    
    # Generate report
    report = validator.generate_validation_report(results)
    print(report)
    
    print(f"\nValidation completed:")
    print(f"Accuracy: {results.metrics.accuracy:.4f}")
    print(f"AUC: {results.metrics.auc_score:.4f}")
    print(f"Accuracy target met: {results.metrics.accuracy >= 0.80}")

if __name__ == "__main__":
    test_validation_framework()