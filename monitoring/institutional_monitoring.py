import asyncio
import psutil
import time
import json
import smtplib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import sqlite3
import pandas as pd
import numpy as np
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from institutional_config import MONITORING_CONFIG

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class MetricType(Enum):
    SYSTEM = "system"
    TRADING = "trading"
    RISK = "risk"
    PERFORMANCE = "performance"
    DATA_QUALITY = "data_quality"

@dataclass
class SystemMetric:
    """System performance metric"""
    timestamp: datetime
    metric_type: MetricType
    name: str
    value: float
    unit: str
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    description: str = ""

@dataclass
class Alert:
    """Alert notification"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    metric_type: MetricType
    title: str
    message: str
    current_value: float
    threshold_value: float
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    acknowledgement: bool = False
    escalated: bool = False

@dataclass
class HealthCheck:
    """Component health check result"""
    component: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    timestamp: datetime
    response_time_ms: float
    error_message: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)

class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger('SystemMonitor')
        self.thresholds = MONITORING_CONFIG['metrics']['system_health']
        
    def get_cpu_usage(self) -> SystemMetric:
        """Get CPU usage percentage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        return SystemMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.SYSTEM,
            name="cpu_usage",
            value=cpu_percent,
            unit="percent",
            threshold_warning=self.thresholds['cpu_threshold'] * 0.8,
            threshold_critical=self.thresholds['cpu_threshold'],
            description="CPU usage percentage"
        )
    
    def get_memory_usage(self) -> SystemMetric:
        """Get memory usage percentage"""
        memory = psutil.virtual_memory()
        return SystemMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.SYSTEM,
            name="memory_usage",
            value=memory.percent,
            unit="percent",
            threshold_warning=self.thresholds['memory_threshold'] * 0.8,
            threshold_critical=self.thresholds['memory_threshold'],
            description="Memory usage percentage"
        )
    
    def get_disk_usage(self) -> SystemMetric:
        """Get disk usage percentage"""
        disk = psutil.disk_usage('/')
        usage_percent = (disk.used / disk.total) * 100
        return SystemMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.SYSTEM,
            name="disk_usage",
            value=usage_percent,
            unit="percent",
            threshold_warning=self.thresholds['disk_threshold'] * 0.8,
            threshold_critical=self.thresholds['disk_threshold'],
            description="Disk usage percentage"
        )
    
    def get_network_latency(self, host: str = "8.8.8.8") -> SystemMetric:
        """Get network latency to a host"""
        try:
            import subprocess
            result = subprocess.run(
                ['ping', '-c', '1', host], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            if result.returncode == 0:
                # Extract latency from ping output
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'time=' in line:
                        latency_str = line.split('time=')[1].split(' ')[0]
                        latency_ms = float(latency_str)
                        break
                else:
                    latency_ms = 999.0  # Default high latency
            else:
                latency_ms = 999.0  # Failed ping
                
        except Exception as e:
            self.logger.error(f"Error measuring network latency: {e}")
            latency_ms = 999.0
        
        return SystemMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.SYSTEM,
            name="network_latency",
            value=latency_ms,
            unit="milliseconds",
            threshold_warning=self.thresholds['network_latency'] * 2,
            threshold_critical=self.thresholds['network_latency'] * 5,
            description=f"Network latency to {host}"
        )
    
    def get_process_count(self) -> SystemMetric:
        """Get number of running processes"""
        process_count = len(psutil.pids())
        return SystemMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.SYSTEM,
            name="process_count",
            value=process_count,
            unit="count",
            threshold_warning=1000,
            threshold_critical=1500,
            description="Number of running processes"
        )
    
    def get_all_metrics(self) -> List[SystemMetric]:
        """Get all system metrics"""
        return [
            self.get_cpu_usage(),
            self.get_memory_usage(),
            self.get_disk_usage(),
            self.get_network_latency(),
            self.get_process_count()
        ]

class TradingMonitor:
    """Trading system monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger('TradingMonitor')
        self.thresholds = MONITORING_CONFIG['metrics']['trading_metrics']
        self.signal_times = deque(maxlen=100)
        self.execution_times = deque(maxlen=100)
        
    def record_signal_generation_time(self, duration_ms: float):
        """Record signal generation timing"""
        self.signal_times.append({
            'timestamp': datetime.now(),
            'duration_ms': duration_ms
        })
    
    def record_execution_time(self, duration_ms: float):
        """Record trade execution timing"""
        self.execution_times.append({
            'timestamp': datetime.now(),
            'duration_ms': duration_ms
        })
    
    def get_signal_generation_metric(self) -> SystemMetric:
        """Get signal generation performance metric"""
        if not self.signal_times:
            avg_time = 0
        else:
            recent_times = [entry['duration_ms'] for entry in list(self.signal_times)[-10:]]
            avg_time = np.mean(recent_times)
        
        return SystemMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.TRADING,
            name="signal_generation_time",
            value=avg_time,
            unit="milliseconds",
            threshold_warning=self.thresholds['signal_generation_time'] * 0.8,
            threshold_critical=self.thresholds['signal_generation_time'],
            description="Average signal generation time"
        )
    
    def get_execution_time_metric(self) -> SystemMetric:
        """Get execution performance metric"""
        if not self.execution_times:
            avg_time = 0
        else:
            recent_times = [entry['duration_ms'] for entry in list(self.execution_times)[-10:]]
            avg_time = np.mean(recent_times)
        
        return SystemMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.TRADING,
            name="execution_time",
            value=avg_time,
            unit="milliseconds",
            threshold_warning=self.thresholds['execution_time'] * 0.8,
            threshold_critical=self.thresholds['execution_time'],
            description="Average execution time"
        )
    
    def get_data_staleness_metric(self, last_data_update: datetime) -> SystemMetric:
        """Get data staleness metric"""
        staleness_ms = (datetime.now() - last_data_update).total_seconds() * 1000
        
        return SystemMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.TRADING,
            name="data_staleness",
            value=staleness_ms,
            unit="milliseconds",
            threshold_warning=self.thresholds['data_staleness'] * 0.8,
            threshold_critical=self.thresholds['data_staleness'],
            description="Data staleness since last update"
        )

class HealthChecker:
    """Component health checking"""
    
    def __init__(self):
        self.logger = logging.getLogger('HealthChecker')
        
    async def check_database_health(self) -> HealthCheck:
        """Check database connectivity and performance"""
        start_time = time.time()
        
        try:
            # Test database connection
            conn = sqlite3.connect('/workspace/data/risk_management.db', timeout=5)
            cursor = conn.cursor()
            
            # Simple query to test responsiveness
            cursor.execute('SELECT COUNT(*) FROM sqlite_master')
            result = cursor.fetchone()
            
            conn.close()
            
            response_time = (time.time() - start_time) * 1000
            
            if response_time > 1000:  # > 1 second
                status = "degraded"
            else:
                status = "healthy"
            
            return HealthCheck(
                component="database",
                status=status,
                timestamp=datetime.now(),
                response_time_ms=response_time,
                additional_info={'tables_count': result[0] if result else 0}
            )
            
        except Exception as e:
            return HealthCheck(
                component="database",
                status="unhealthy",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def check_data_feed_health(self) -> HealthCheck:
        """Check data feed connectivity"""
        start_time = time.time()
        
        try:
            # Test connection to a data provider (using Alpha Vantage as example)
            test_url = "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey=demo"
            
            response = requests.get(test_url, timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                if 'Global Quote' in data:
                    status = "healthy"
                else:
                    status = "degraded"
            else:
                status = "unhealthy"
            
            return HealthCheck(
                component="data_feed",
                status=status,
                timestamp=datetime.now(),
                response_time_ms=response_time,
                additional_info={'status_code': response.status_code}
            )
            
        except Exception as e:
            return HealthCheck(
                component="data_feed",
                status="unhealthy",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def check_risk_engine_health(self) -> HealthCheck:
        """Check risk engine health"""
        start_time = time.time()
        
        try:
            # Simple test of risk calculations
            test_returns = pd.Series([0.01, -0.005, 0.008, -0.003, 0.012])
            
            # Test VaR calculation
            var_95 = np.percentile(test_returns, 5)
            
            response_time = (time.time() - start_time) * 1000
            status = "healthy"
            
            return HealthCheck(
                component="risk_engine",
                status=status,
                timestamp=datetime.now(),
                response_time_ms=response_time,
                additional_info={'test_var': float(var_95)}
            )
            
        except Exception as e:
            return HealthCheck(
                component="risk_engine",
                status="unhealthy",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def check_all_components(self) -> List[HealthCheck]:
        """Check all component health"""
        health_checks = await asyncio.gather(
            self.check_database_health(),
            self.check_data_feed_health(),
            self.check_risk_engine_health(),
            return_exceptions=True
        )
        
        return [check for check in health_checks if isinstance(check, HealthCheck)]

class AlertManager:
    """Alert management and notification system"""
    
    def __init__(self):
        self.logger = logging.getLogger('AlertManager')
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels = {
            'email': self._setup_email_notifier(),
            'slack': self._setup_slack_notifier(),
            'pagerduty': self._setup_pagerduty_notifier()
        }
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize alerts database"""
        try:
            conn = sqlite3.connect('/workspace/data/monitoring.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT,
                    current_value REAL,
                    threshold_value REAL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_time TEXT,
                    acknowledgement BOOLEAN DEFAULT FALSE
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing alerts database: {e}")
    
    def _setup_email_notifier(self) -> Optional[Dict]:
        """Setup email notification configuration"""
        email_config = MONITORING_CONFIG['alerting']['email']
        if email_config.get('smtp_server') and email_config.get('username'):
            return email_config
        return None
    
    def _setup_slack_notifier(self) -> Optional[Dict]:
        """Setup Slack notification configuration"""
        slack_config = MONITORING_CONFIG['alerting']['slack']
        if slack_config.get('webhook_url'):
            return slack_config
        return None
    
    def _setup_pagerduty_notifier(self) -> Optional[Dict]:
        """Setup PagerDuty notification configuration"""
        pagerduty_config = MONITORING_CONFIG['alerting']['pagerduty']
        if pagerduty_config.get('api_key'):
            return pagerduty_config
        return None
    
    def create_alert(self, metric: SystemMetric, severity: AlertSeverity, 
                    threshold_breached: float) -> Alert:
        """Create a new alert"""
        alert_id = f"{metric.name}_{int(time.time())}"
        
        alert = Alert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            severity=severity,
            metric_type=metric.metric_type,
            title=f"{metric.name.replace('_', ' ').title()} Alert",
            message=f"{metric.description}: {metric.value:.2f} {metric.unit} exceeds threshold {threshold_breached:.2f} {metric.unit}",
            current_value=metric.value,
            threshold_value=threshold_breached
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Save to database
        self._save_alert(alert)
        
        # Send notifications
        asyncio.create_task(self._send_notifications(alert))
        
        self.logger.warning(f"Alert created: {alert.title} - {alert.message}")
        
        return alert
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = datetime.now()
            
            # Update database
            self._update_alert(alert)
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            self.logger.info(f"Alert resolved: {alert.title}")
            return True
        
        return False
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledgement = True
            
            # Update database
            self._update_alert(alert)
            
            self.logger.info(f"Alert acknowledged: {alert.title}")
            return True
        
        return False
    
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications through configured channels"""
        try:
            # Email notification
            if self.notification_channels['email'] and alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                await self._send_email_notification(alert)
            
            # Slack notification
            if self.notification_channels['slack']:
                await self._send_slack_notification(alert)
            
            # PagerDuty notification
            if self.notification_channels['pagerduty'] and alert.severity == AlertSeverity.EMERGENCY:
                await self._send_pagerduty_notification(alert)
                
        except Exception as e:
            self.logger.error(f"Error sending notifications: {e}")
    
    async def _send_email_notification(self, alert: Alert):
        """Send email notification"""
        try:
            config = self.notification_channels['email']
            
            msg = MIMEMultipart()
            msg['From'] = config['username']
            msg['To'] = ', '.join(config['recipients'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
            Alert Details:
            
            Severity: {alert.severity.value.upper()}
            Metric: {alert.title}
            Message: {alert.message}
            Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            Current Value: {alert.current_value:.2f}
            Threshold: {alert.threshold_value:.2f}
            
            Please investigate and take appropriate action.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            text = msg.as_string()
            server.sendmail(config['username'], config['recipients'], text)
            server.quit()
            
            self.logger.info(f"Email notification sent for alert: {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Error sending email notification: {e}")
    
    async def _send_slack_notification(self, alert: Alert):
        """Send Slack notification"""
        try:
            config = self.notification_channels['slack']
            
            # Color coding for severity
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9500",
                AlertSeverity.CRITICAL: "#ff0000",
                AlertSeverity.EMERGENCY: "#8b0000"
            }
            
            payload = {
                "channel": config['channel'],
                "username": "Trading Bot Monitor",
                "icon_emoji": ":warning:",
                "attachments": [{
                    "color": color_map.get(alert.severity, "#ff0000"),
                    "title": f"[{alert.severity.value.upper()}] {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Current Value",
                            "value": f"{alert.current_value:.2f}",
                            "short": True
                        },
                        {
                            "title": "Threshold",
                            "value": f"{alert.threshold_value:.2f}",
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                            "short": True
                        }
                    ]
                }]
            }
            
            response = requests.post(config['webhook_url'], json=payload, timeout=10)
            
            if response.status_code == 200:
                self.logger.info(f"Slack notification sent for alert: {alert.alert_id}")
            else:
                self.logger.error(f"Slack notification failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error sending Slack notification: {e}")
    
    async def _send_pagerduty_notification(self, alert: Alert):
        """Send PagerDuty notification"""
        try:
            config = self.notification_channels['pagerduty']
            
            payload = {
                "routing_key": config['service_key'],
                "event_action": "trigger",
                "dedup_key": alert.alert_id,
                "payload": {
                    "summary": alert.title,
                    "severity": "critical",
                    "source": "trading-bot-monitor",
                    "custom_details": {
                        "message": alert.message,
                        "current_value": alert.current_value,
                        "threshold": alert.threshold_value,
                        "metric_type": alert.metric_type.value
                    }
                }
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Token token={config['api_key']}"
            }
            
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 202:
                self.logger.info(f"PagerDuty notification sent for alert: {alert.alert_id}")
            else:
                self.logger.error(f"PagerDuty notification failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error sending PagerDuty notification: {e}")
    
    def _save_alert(self, alert: Alert):
        """Save alert to database"""
        try:
            conn = sqlite3.connect('/workspace/data/monitoring.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alerts 
                (id, timestamp, severity, metric_type, title, message, 
                 current_value, threshold_value, resolved, acknowledgement)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id,
                alert.timestamp.isoformat(),
                alert.severity.value,
                alert.metric_type.value,
                alert.title,
                alert.message,
                alert.current_value,
                alert.threshold_value,
                alert.resolved,
                alert.acknowledgement
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving alert: {e}")
    
    def _update_alert(self, alert: Alert):
        """Update alert in database"""
        try:
            conn = sqlite3.connect('/workspace/data/monitoring.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE alerts 
                SET resolved = ?, resolution_time = ?, acknowledgement = ?
                WHERE id = ?
            ''', (
                alert.resolved,
                alert.resolution_time.isoformat() if alert.resolution_time else None,
                alert.acknowledgement,
                alert.alert_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error updating alert: {e}")

class InstitutionalMonitoringSystem:
    """Comprehensive institutional monitoring system"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.system_monitor = SystemMonitor()
        self.trading_monitor = TradingMonitor()
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager()
        
        self.is_running = False
        self.monitoring_thread = None
        self.metrics_history: List[SystemMetric] = []
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('InstitutionalMonitoringSystem')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('/workspace/logs/institutional_monitoring.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start the monitoring system"""
        if not self.is_running:
            self.is_running = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(interval_seconds,),
                daemon=True
            )
            self.monitoring_thread.start()
            self.logger.info("Monitoring system started")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Monitoring system stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect all metrics
                system_metrics = self.system_monitor.get_all_metrics()
                trading_metrics = [
                    self.trading_monitor.get_signal_generation_metric(),
                    self.trading_monitor.get_execution_time_metric(),
                ]
                
                all_metrics = system_metrics + trading_metrics
                
                # Store metrics in history
                self.metrics_history.extend(all_metrics)
                
                # Keep only recent metrics (last 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.metrics_history = [
                    metric for metric in self.metrics_history 
                    if metric.timestamp > cutoff_time
                ]
                
                # Check for alerts
                for metric in all_metrics:
                    self._check_metric_thresholds(metric)
                
                # Save metrics to database
                self._save_metrics(all_metrics)
                
                # Perform health checks every 5th iteration (5 minutes if interval is 1 minute)
                if len(self.metrics_history) % 5 == 0:
                    asyncio.run(self._perform_health_checks())
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def _check_metric_thresholds(self, metric: SystemMetric):
        """Check if metric exceeds thresholds and create alerts"""
        try:
            # Check critical threshold
            if metric.threshold_critical and metric.value >= metric.threshold_critical:
                # Check if alert already exists for this metric
                existing_alert = any(
                    alert.title.lower().replace(' ', '_') == metric.name 
                    for alert in self.alert_manager.active_alerts.values()
                )
                
                if not existing_alert:
                    self.alert_manager.create_alert(
                        metric, AlertSeverity.CRITICAL, metric.threshold_critical
                    )
            
            # Check warning threshold
            elif metric.threshold_warning and metric.value >= metric.threshold_warning:
                # Check if alert already exists for this metric
                existing_alert = any(
                    alert.title.lower().replace(' ', '_') == metric.name 
                    for alert in self.alert_manager.active_alerts.values()
                )
                
                if not existing_alert:
                    self.alert_manager.create_alert(
                        metric, AlertSeverity.WARNING, metric.threshold_warning
                    )
            
            else:
                # Check if we need to resolve any existing alerts for this metric
                alerts_to_resolve = [
                    alert_id for alert_id, alert in self.alert_manager.active_alerts.items()
                    if alert.title.lower().replace(' ', '_') == metric.name
                ]
                
                for alert_id in alerts_to_resolve:
                    self.alert_manager.resolve_alert(alert_id)
                    
        except Exception as e:
            self.logger.error(f"Error checking metric thresholds: {e}")
    
    async def _perform_health_checks(self):
        """Perform component health checks"""
        try:
            health_checks = await self.health_checker.check_all_components()
            
            for check in health_checks:
                self.logger.info(f"Health check - {check.component}: {check.status} "
                               f"({check.response_time_ms:.1f}ms)")
                
                # Create alerts for unhealthy components
                if check.status == "unhealthy":
                    # Check if alert already exists
                    existing_alert = any(
                        alert.title.lower().find(check.component) != -1
                        for alert in self.alert_manager.active_alerts.values()
                    )
                    
                    if not existing_alert:
                        alert = Alert(
                            alert_id=f"{check.component}_health_{int(time.time())}",
                            timestamp=datetime.now(),
                            severity=AlertSeverity.CRITICAL,
                            metric_type=MetricType.SYSTEM,
                            title=f"{check.component.title()} Health Alert",
                            message=f"{check.component} is unhealthy: {check.error_message or 'Unknown error'}",
                            current_value=check.response_time_ms,
                            threshold_value=5000  # 5 second threshold
                        )
                        
                        self.alert_manager.active_alerts[alert.alert_id] = alert
                        self.alert_manager.alert_history.append(alert)
                        await self.alert_manager._send_notifications(alert)
                        
        except Exception as e:
            self.logger.error(f"Error performing health checks: {e}")
    
    def _save_metrics(self, metrics: List[SystemMetric]):
        """Save metrics to database"""
        try:
            conn = sqlite3.connect('/workspace/data/monitoring.db')
            cursor = conn.cursor()
            
            for metric in metrics:
                cursor.execute('''
                    INSERT INTO system_metrics 
                    (timestamp, metric_type, name, value, unit)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    metric.timestamp.isoformat(),
                    metric.metric_type.value,
                    metric.name,
                    metric.value,
                    metric.unit
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
    
    def get_current_status(self) -> Dict:
        """Get current system status"""
        try:
            # Get latest metrics
            latest_metrics = {}
            for metric in self.metrics_history[-20:]:  # Last 20 metrics
                latest_metrics[metric.name] = {
                    'value': metric.value,
                    'unit': metric.unit,
                    'timestamp': metric.timestamp.isoformat(),
                    'status': 'healthy'
                }
                
                # Determine status based on thresholds
                if metric.threshold_critical and metric.value >= metric.threshold_critical:
                    latest_metrics[metric.name]['status'] = 'critical'
                elif metric.threshold_warning and metric.value >= metric.threshold_warning:
                    latest_metrics[metric.name]['status'] = 'warning'
            
            # Get active alerts
            active_alerts = [
                {
                    'id': alert.alert_id,
                    'severity': alert.severity.value,
                    'title': alert.title,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in self.alert_manager.active_alerts.values()
            ]
            
            # Overall system health
            overall_status = 'healthy'
            if any(alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] 
                   for alert in self.alert_manager.active_alerts.values()):
                overall_status = 'critical'
            elif any(alert.severity == AlertSeverity.WARNING 
                     for alert in self.alert_manager.active_alerts.values()):
                overall_status = 'warning'
            
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': overall_status,
                'metrics': latest_metrics,
                'active_alerts': active_alerts,
                'monitoring_active': self.is_running
            }
            
        except Exception as e:
            self.logger.error(f"Error getting current status: {e}")
            return {'error': str(e)}
    
    def get_metrics_history(self, hours: int = 24) -> Dict:
        """Get metrics history for specified hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Filter metrics by time
            recent_metrics = [
                metric for metric in self.metrics_history
                if metric.timestamp > cutoff_time
            ]
            
            # Group by metric name
            metrics_by_name = {}
            for metric in recent_metrics:
                if metric.name not in metrics_by_name:
                    metrics_by_name[metric.name] = []
                
                metrics_by_name[metric.name].append({
                    'timestamp': metric.timestamp.isoformat(),
                    'value': metric.value,
                    'unit': metric.unit
                })
            
            # Sort by timestamp
            for metric_name in metrics_by_name:
                metrics_by_name[metric_name].sort(key=lambda x: x['timestamp'])
            
            return {
                'time_range_hours': hours,
                'metrics': metrics_by_name,
                'total_data_points': len(recent_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting metrics history: {e}")
            return {'error': str(e)}

# Example usage and testing
async def main():
    """Test the monitoring system"""
    
    # Create monitoring system
    monitor = InstitutionalMonitoringSystem()
    
    print("Starting institutional monitoring system...")
    monitor.start_monitoring(interval_seconds=10)  # 10 second intervals for testing
    
    # Let it run for a bit
    await asyncio.sleep(30)
    
    # Get current status
    status = monitor.get_current_status()
    print(f"\nCurrent Status: {status['overall_status']}")
    print(f"Active Alerts: {len(status['active_alerts'])}")
    
    if status['active_alerts']:
        print("\nActive Alerts:")
        for alert in status['active_alerts']:
            print(f"  - {alert['severity'].upper()}: {alert['title']}")
    
    # Get metrics history
    history = monitor.get_metrics_history(hours=1)
    print(f"\nMetrics History: {history['total_data_points']} data points")
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("\nMonitoring system stopped")

if __name__ == "__main__":
    asyncio.run(main())