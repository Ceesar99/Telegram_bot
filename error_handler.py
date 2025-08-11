#!/usr/bin/env python3
"""
Comprehensive Error Handling System
Provides centralized error handling, logging, recovery mechanisms,
and monitoring for the trading bot system.
"""

import logging
import traceback
import sys
import threading
import time
from typing import Dict, List, Callable, Any, Optional
from datetime import datetime, timedelta
from functools import wraps
import sqlite3
import json
from dataclasses import dataclass, asdict
from enum import Enum

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class ErrorReport:
    """Error report data structure"""
    timestamp: str
    error_type: str
    severity: ErrorSeverity
    component: str
    message: str
    traceback: str
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    occurrence_count: int = 1

class ErrorHandler:
    def __init__(self, db_path: str = "/workspace/data/errors.db"):
        self.logger = logging.getLogger('ErrorHandler')
        self.db_path = db_path
        self._error_counts = {}
        self._recovery_strategies = {}
        self._notification_callbacks = []
        self._lock = threading.Lock()
        
        # Initialize error database
        self._init_database()
        
        # Set up global exception handler
        self._setup_global_handler()
    
    def _init_database(self):
        """Initialize error tracking database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS errors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        error_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        component TEXT NOT NULL,
                        message TEXT NOT NULL,
                        traceback TEXT,
                        context TEXT,
                        recovery_attempted BOOLEAN DEFAULT FALSE,
                        recovery_successful BOOLEAN DEFAULT FALSE,
                        occurrence_count INTEGER DEFAULT 1,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON errors(timestamp)
                ''')
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_component ON errors(component)
                ''')
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_severity ON errors(severity)
                ''')
                conn.commit()
                self.logger.info("Error tracking database initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize error database: {e}")
    
    def _setup_global_handler(self):
        """Set up global exception handler"""
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            self.handle_error(
                error=exc_value,
                component="Global",
                severity=ErrorSeverity.CRITICAL,
                context={"exc_type": exc_type.__name__}
            )
        
        sys.excepthook = handle_exception
    
    def handle_error(self, 
                     error: Exception, 
                     component: str,
                     severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                     context: Optional[Dict[str, Any]] = None,
                     attempt_recovery: bool = True) -> bool:
        """
        Handle an error with logging, tracking, and recovery
        
        Args:
            error: The exception/error object
            component: Component where error occurred
            severity: Error severity level
            context: Additional context information
            attempt_recovery: Whether to attempt automatic recovery
            
        Returns:
            bool: True if error was handled successfully
        """
        try:
            with self._lock:
                # Create error report
                error_report = ErrorReport(
                    timestamp=datetime.now().isoformat(),
                    error_type=type(error).__name__,
                    severity=severity,
                    component=component,
                    message=str(error),
                    traceback=traceback.format_exc(),
                    context=context or {}
                )
                
                # Track error frequency
                error_key = f"{component}:{error_report.error_type}"
                self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
                error_report.occurrence_count = self._error_counts[error_key]
                
                # Log error
                self._log_error(error_report)
                
                # Store in database
                self._store_error(error_report)
                
                # Attempt recovery if enabled
                recovery_success = False
                if attempt_recovery and component in self._recovery_strategies:
                    recovery_success = self._attempt_recovery(error_report)
                    error_report.recovery_attempted = True
                    error_report.recovery_successful = recovery_success
                    
                    # Update database with recovery info
                    self._update_recovery_status(error_report)
                
                # Send notifications for critical errors
                if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                    self._send_notifications(error_report)
                
                return recovery_success
                
        except Exception as handler_error:
            # Emergency logging if error handler fails
            self.logger.critical(f"Error handler failed: {handler_error}")
            self.logger.critical(f"Original error: {error}")
            return False
    
    def _log_error(self, error_report: ErrorReport):
        """Log error with appropriate level"""
        log_message = f"[{error_report.component}] {error_report.error_type}: {error_report.message}"
        
        if error_report.occurrence_count > 1:
            log_message += f" (occurred {error_report.occurrence_count} times)"
        
        if error_report.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_report.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_report.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Log context if available
        if error_report.context:
            self.logger.debug(f"Error context: {error_report.context}")
    
    def _store_error(self, error_report: ErrorReport):
        """Store error in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO errors (
                        timestamp, error_type, severity, component, message, 
                        traceback, context, occurrence_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    error_report.timestamp,
                    error_report.error_type,
                    error_report.severity.value,
                    error_report.component,
                    error_report.message,
                    error_report.traceback,
                    json.dumps(error_report.context),
                    error_report.occurrence_count
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to store error in database: {e}")
    
    def _update_recovery_status(self, error_report: ErrorReport):
        """Update recovery status in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE errors 
                    SET recovery_attempted = ?, recovery_successful = ?
                    WHERE timestamp = ? AND component = ? AND error_type = ?
                ''', (
                    error_report.recovery_attempted,
                    error_report.recovery_successful,
                    error_report.timestamp,
                    error_report.component,
                    error_report.error_type
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to update recovery status: {e}")
    
    def _attempt_recovery(self, error_report: ErrorReport) -> bool:
        """Attempt automatic error recovery"""
        try:
            recovery_func = self._recovery_strategies.get(error_report.component)
            if recovery_func:
                self.logger.info(f"Attempting recovery for {error_report.component}")
                result = recovery_func(error_report)
                if result:
                    self.logger.info(f"Recovery successful for {error_report.component}")
                else:
                    self.logger.warning(f"Recovery failed for {error_report.component}")
                return result
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")
        return False
    
    def _send_notifications(self, error_report: ErrorReport):
        """Send error notifications"""
        for callback in self._notification_callbacks:
            try:
                callback(error_report)
            except Exception as e:
                self.logger.error(f"Notification callback failed: {e}")
    
    def register_recovery_strategy(self, component: str, recovery_func: Callable):
        """Register a recovery strategy for a component"""
        self._recovery_strategies[component] = recovery_func
        self.logger.info(f"Registered recovery strategy for {component}")
    
    def register_notification_callback(self, callback: Callable):
        """Register a notification callback"""
        self._notification_callbacks.append(callback)
        self.logger.info("Registered notification callback")
    
    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for the specified time period"""
        try:
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                # Total errors
                total_errors = conn.execute(
                    'SELECT COUNT(*) FROM errors WHERE timestamp >= ?',
                    (cutoff_time,)
                ).fetchone()[0]
                
                # Errors by severity
                severity_stats = {}
                for severity in ErrorSeverity:
                    count = conn.execute(
                        'SELECT COUNT(*) FROM errors WHERE timestamp >= ? AND severity = ?',
                        (cutoff_time, severity.value)
                    ).fetchone()[0]
                    severity_stats[severity.value] = count
                
                # Errors by component
                component_stats = {}
                component_results = conn.execute('''
                    SELECT component, COUNT(*) 
                    FROM errors 
                    WHERE timestamp >= ? 
                    GROUP BY component
                ''', (cutoff_time,)).fetchall()
                
                for component, count in component_results:
                    component_stats[component] = count
                
                # Recovery success rate
                recovery_attempts = conn.execute(
                    'SELECT COUNT(*) FROM errors WHERE timestamp >= ? AND recovery_attempted = 1',
                    (cutoff_time,)
                ).fetchone()[0]
                
                recovery_successes = conn.execute(
                    'SELECT COUNT(*) FROM errors WHERE timestamp >= ? AND recovery_successful = 1',
                    (cutoff_time,)
                ).fetchone()[0]
                
                recovery_rate = (recovery_successes / recovery_attempts * 100) if recovery_attempts > 0 else 0
                
                return {
                    'total_errors': total_errors,
                    'severity_breakdown': severity_stats,
                    'component_breakdown': component_stats,
                    'recovery_attempts': recovery_attempts,
                    'recovery_successes': recovery_successes,
                    'recovery_success_rate': round(recovery_rate, 2)
                }
        except Exception as e:
            self.logger.error(f"Failed to get error statistics: {e}")
            return {}
    
    def get_recent_errors(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent errors"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM errors 
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (limit,))
                
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Failed to get recent errors: {e}")
            return []

def error_handler_decorator(component: str, 
                           severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                           suppress: bool = False,
                           retry_count: int = 0,
                           retry_delay: float = 1.0):
    """
    Decorator for automatic error handling
    
    Args:
        component: Component name for error tracking
        severity: Error severity level
        suppress: Whether to suppress the exception after handling
        retry_count: Number of retries to attempt
        retry_delay: Delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retry_count + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    context = {
                        'function': func.__name__,
                        'attempt': attempt + 1,
                        'max_attempts': retry_count + 1,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    }
                    
                    # Handle the error
                    global_error_handler.handle_error(
                        error=e,
                        component=component,
                        severity=severity,
                        context=context
                    )
                    
                    # Retry if attempts remaining
                    if attempt < retry_count:
                        time.sleep(retry_delay)
                        continue
                    
                    # Re-raise if not suppressing
                    if not suppress:
                        raise
                    
                    return None
            
        return wrapper
    return decorator

# Global error handler instance
global_error_handler = ErrorHandler()

# Convenience functions
def handle_error(error: Exception, 
                component: str, 
                severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                context: Optional[Dict[str, Any]] = None) -> bool:
    """Convenience function for error handling"""
    return global_error_handler.handle_error(error, component, severity, context)

def register_recovery_strategy(component: str, recovery_func: Callable):
    """Register a recovery strategy"""
    global_error_handler.register_recovery_strategy(component, recovery_func)

def register_notification_callback(callback: Callable):
    """Register a notification callback"""
    global_error_handler.register_notification_callback(callback)