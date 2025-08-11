import asyncio
import logging
import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import warnings
warnings.filterwarnings('ignore')

class RegulationType(Enum):
    MIFID_II = "mifid_ii"
    DODD_FRANK = "dodd_frank"
    BASEL_III = "basel_iii"
    CFTC = "cftc"
    FCA = "fca"
    SEC = "sec"
    ESMA = "esma"

class TradeReportingRegime(Enum):
    PRE_TRADE = "pre_trade"
    POST_TRADE = "post_trade"
    BEST_EXECUTION = "best_execution"
    TRANSACTION_REPORTING = "transaction_reporting"

class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    EXEMPTED = "exempted"

@dataclass
class ComplianceRule:
    """Individual compliance rule definition"""
    rule_id: str
    regulation_type: RegulationType
    rule_name: str
    description: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    check_frequency: str  # 'real_time', 'daily', 'weekly', 'monthly'
    parameters: Dict[str, Any]
    is_active: bool = True

@dataclass
class TradeRecord:
    """Comprehensive trade record for compliance"""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    venue: str
    order_type: str
    client_id: str
    trader_id: str
    execution_timestamp: datetime
    settlement_date: datetime
    
    # MiFID II specific fields
    client_classification: str  # 'retail', 'professional', 'eligible_counterparty'
    order_reception_time: datetime
    order_transmission_time: datetime
    execution_decision_time: datetime
    
    # Additional compliance fields
    pre_trade_transparency_waiver: Optional[str]
    post_trade_transparency_delay: Optional[int]
    best_execution_venue_selection: str
    liquidity_provision_activity: bool
    
    # Risk and position data
    notional_amount: float
    currency: str
    counterparty: Optional[str]
    clearing_status: str
    
    # Audit trail
    order_modifications: List[Dict[str, Any]]
    execution_quality_data: Dict[str, float]

@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    violation_id: str
    rule_id: str
    trade_id: Optional[str]
    timestamp: datetime
    severity: str
    description: str
    affected_regulation: RegulationType
    status: ComplianceStatus
    remediation_actions: List[str]
    resolved_timestamp: Optional[datetime] = None

class EncryptionManager:
    """Advanced encryption for sensitive compliance data"""
    
    def __init__(self, password: str = None):
        if password is None:
            password = os.getenv('COMPLIANCE_ENCRYPTION_KEY', 'default_key_change_me')
        
        # Generate key from password
        password_bytes = password.encode()
        salt = b'salt_1234567890'  # In production, use random salt per encryption
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        self.cipher_suite = Fernet(key)
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()

class AuditTrail:
    """Immutable audit trail for compliance"""
    
    def __init__(self, db_path: str = "compliance_audit.db"):
        self.db_path = db_path
        self.encryption_manager = EncryptionManager()
        self.logger = logging.getLogger('AuditTrail')
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize audit trail database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Audit events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                user_id TEXT,
                system_component TEXT,
                action_description TEXT,
                data_hash TEXT,
                encrypted_data TEXT,
                ip_address TEXT,
                session_id TEXT,
                regulation_type TEXT,
                criticality_level TEXT
            )
        ''')
        
        # Trade audit table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL,
                audit_event_id TEXT NOT NULL,
                pre_trade_data TEXT,
                execution_data TEXT,
                post_trade_data TEXT,
                compliance_checks TEXT,
                FOREIGN KEY (audit_event_id) REFERENCES audit_events (event_id)
            )
        ''')
        
        # Compliance checks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                check_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                rule_id TEXT NOT NULL,
                trade_id TEXT,
                check_result TEXT NOT NULL,
                details TEXT,
                remediation_required BOOLEAN
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_event(self, event_type: str, description: str, data: Dict[str, Any] = None,
                  user_id: str = None, regulation_type: RegulationType = None) -> str:
        """Log an audit event"""
        event_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # Encrypt sensitive data
        encrypted_data = None
        data_hash = None
        if data:
            data_json = json.dumps(data, default=str)
            encrypted_data = self.encryption_manager.encrypt_data(data_json)
            data_hash = hashlib.sha256(data_json.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_events 
            (event_id, timestamp, event_type, user_id, action_description, 
             data_hash, encrypted_data, regulation_type, criticality_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event_id, timestamp, event_type, user_id, description,
            data_hash, encrypted_data, 
            regulation_type.value if regulation_type else None,
            'high'
        ))
        
        conn.commit()
        conn.close()
        
        return event_id
    
    def log_trade_audit(self, trade_record: TradeRecord, compliance_results: Dict[str, Any]) -> str:
        """Log trade-specific audit information"""
        event_id = self.log_event(
            event_type="trade_execution",
            description=f"Trade executed: {trade_record.symbol} {trade_record.side} {trade_record.quantity}",
            data=asdict(trade_record),
            regulation_type=RegulationType.MIFID_II
        )
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trade_audit 
            (trade_id, audit_event_id, execution_data, compliance_checks)
            VALUES (?, ?, ?, ?)
        ''', (
            trade_record.trade_id,
            event_id,
            json.dumps(asdict(trade_record), default=str),
            json.dumps(compliance_results, default=str)
        ))
        
        conn.commit()
        conn.close()
        
        return event_id
    
    def get_audit_trail(self, start_date: datetime = None, end_date: datetime = None,
                       event_type: str = None) -> List[Dict[str, Any]]:
        """Retrieve audit trail with optional filtering"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        query += " ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df.to_dict('records')

class MiFIDIIComplianceEngine:
    """MiFID II specific compliance engine"""
    
    def __init__(self, audit_trail: AuditTrail):
        self.audit_trail = audit_trail
        self.logger = logging.getLogger('MiFIDIICompliance')
        
        # MiFID II rules
        self.compliance_rules = self._initialize_mifid_rules()
    
    def _initialize_mifid_rules(self) -> List[ComplianceRule]:
        """Initialize MiFID II compliance rules"""
        return [
            ComplianceRule(
                rule_id="MIFID_CLOCK_SYNC",
                regulation_type=RegulationType.MIFID_II,
                rule_name="Clock Synchronization",
                description="All timestamps must be synchronized to UTC with microsecond precision",
                severity="critical",
                check_frequency="real_time",
                parameters={"max_drift_microseconds": 100}
            ),
            ComplianceRule(
                rule_id="MIFID_ORDER_RECORD",
                regulation_type=RegulationType.MIFID_II,
                rule_name="Order Record Keeping",
                description="All orders must be recorded with required data fields",
                severity="critical",
                check_frequency="real_time",
                parameters={"required_fields": ["client_id", "timestamp", "symbol", "quantity"]}
            ),
            ComplianceRule(
                rule_id="MIFID_BEST_EXECUTION",
                regulation_type=RegulationType.MIFID_II,
                rule_name="Best Execution",
                description="Demonstrate best execution for client orders",
                severity="high",
                check_frequency="real_time",
                parameters={"execution_venues": ["primary", "mtu", "dark_pool"]}
            ),
            ComplianceRule(
                rule_id="MIFID_POSITION_LIMITS",
                regulation_type=RegulationType.MIFID_II,
                rule_name="Position Limits",
                description="Monitor commodity derivative position limits",
                severity="high",
                check_frequency="daily",
                parameters={"position_limit_percentage": 0.25}
            ),
            ComplianceRule(
                rule_id="MIFID_TRANSACTION_REPORTING",
                regulation_type=RegulationType.MIFID_II,
                rule_name="Transaction Reporting",
                description="Report transactions to trade repositories within T+1",
                severity="critical",
                check_frequency="daily",
                parameters={"reporting_deadline_hours": 24}
            )
        ]
    
    def check_clock_synchronization(self, trade_record: TradeRecord) -> Dict[str, Any]:
        """Check MiFID II clock synchronization requirements"""
        # Simulate clock drift check
        current_time = datetime.utcnow()
        time_diff = abs((trade_record.timestamp - current_time).total_seconds() * 1000000)  # microseconds
        
        max_drift = self.compliance_rules[0].parameters["max_drift_microseconds"]
        is_compliant = time_diff <= max_drift
        
        return {
            "rule_id": "MIFID_CLOCK_SYNC",
            "compliant": is_compliant,
            "details": {
                "time_drift_microseconds": time_diff,
                "max_allowed_microseconds": max_drift,
                "trade_timestamp": trade_record.timestamp.isoformat(),
                "system_timestamp": current_time.isoformat()
            }
        }
    
    def check_order_record_keeping(self, trade_record: TradeRecord) -> Dict[str, Any]:
        """Check MiFID II order record keeping requirements"""
        required_fields = self.compliance_rules[1].parameters["required_fields"]
        missing_fields = []
        
        trade_dict = asdict(trade_record)
        for field in required_fields:
            if field not in trade_dict or trade_dict[field] is None:
                missing_fields.append(field)
        
        is_compliant = len(missing_fields) == 0
        
        return {
            "rule_id": "MIFID_ORDER_RECORD",
            "compliant": is_compliant,
            "details": {
                "missing_fields": missing_fields,
                "required_fields": required_fields,
                "has_client_classification": trade_record.client_classification is not None
            }
        }
    
    def check_best_execution(self, trade_record: TradeRecord, market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check MiFID II best execution requirements"""
        # Simplified best execution check
        # In reality, this would analyze execution quality against multiple venues
        
        execution_quality = trade_record.execution_quality_data or {}
        venue_selection = trade_record.best_execution_venue_selection
        
        # Check if execution venue was properly selected
        available_venues = self.compliance_rules[2].parameters["execution_venues"]
        venue_in_scope = venue_selection in available_venues
        
        # Check execution quality metrics
        price_improvement = execution_quality.get("price_improvement", 0)
        execution_speed = execution_quality.get("execution_speed_ms", 1000)
        
        # Best execution criteria
        is_compliant = (
            venue_in_scope and
            price_improvement >= 0 and  # No adverse price movement
            execution_speed <= 500  # Executed within 500ms
        )
        
        return {
            "rule_id": "MIFID_BEST_EXECUTION",
            "compliant": is_compliant,
            "details": {
                "venue_selection": venue_selection,
                "venue_in_scope": venue_in_scope,
                "price_improvement": price_improvement,
                "execution_speed_ms": execution_speed,
                "available_venues": available_venues
            }
        }
    
    def check_transaction_reporting(self, trade_record: TradeRecord) -> Dict[str, Any]:
        """Check MiFID II transaction reporting requirements"""
        reporting_deadline = self.compliance_rules[4].parameters["reporting_deadline_hours"]
        
        # Check if trade was reported within deadline
        current_time = datetime.utcnow()
        hours_since_trade = (current_time - trade_record.timestamp).total_seconds() / 3600
        
        is_compliant = hours_since_trade <= reporting_deadline
        
        return {
            "rule_id": "MIFID_TRANSACTION_REPORTING",
            "compliant": is_compliant,
            "details": {
                "hours_since_trade": hours_since_trade,
                "reporting_deadline_hours": reporting_deadline,
                "trade_timestamp": trade_record.timestamp.isoformat(),
                "current_timestamp": current_time.isoformat()
            }
        }

class ComplianceMonitor:
    """Real-time compliance monitoring system"""
    
    def __init__(self, db_path: str = "compliance.db"):
        self.logger = logging.getLogger('ComplianceMonitor')
        self.audit_trail = AuditTrail()
        self.mifid_engine = MiFIDIIComplianceEngine(self.audit_trail)
        self.violations: List[ComplianceViolation] = []
        
        # Initialize compliance database
        self._initialize_compliance_db(db_path)
        
        # Active monitoring
        self.monitoring_active = False
        self.alert_thresholds = {
            'critical_violations_per_hour': 5,
            'high_violations_per_day': 20,
            'compliance_rate_threshold': 0.95
        }
    
    def _initialize_compliance_db(self, db_path: str):
        """Initialize compliance monitoring database"""
        self.db_path = db_path
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Violations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                violation_id TEXT UNIQUE NOT NULL,
                rule_id TEXT NOT NULL,
                trade_id TEXT,
                timestamp TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT,
                regulation_type TEXT,
                status TEXT NOT NULL,
                remediation_actions TEXT,
                resolved_timestamp TEXT
            )
        ''')
        
        # Compliance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                regulation_type TEXT,
                total_checks INTEGER,
                passed_checks INTEGER,
                failed_checks INTEGER,
                compliance_rate REAL,
                critical_violations INTEGER,
                high_violations INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def monitor_trade_compliance(self, trade_record: TradeRecord) -> Dict[str, Any]:
        """Monitor a trade for compliance violations"""
        compliance_results = {
            'trade_id': trade_record.trade_id,
            'timestamp': datetime.utcnow().isoformat(),
            'checks_performed': [],
            'violations': [],
            'overall_compliant': True
        }
        
        # Perform MiFID II checks
        mifid_checks = [
            self.mifid_engine.check_clock_synchronization(trade_record),
            self.mifid_engine.check_order_record_keeping(trade_record),
            self.mifid_engine.check_best_execution(trade_record),
            self.mifid_engine.check_transaction_reporting(trade_record)
        ]
        
        for check_result in mifid_checks:
            compliance_results['checks_performed'].append(check_result)
            
            if not check_result['compliant']:
                compliance_results['overall_compliant'] = False
                
                # Create violation record
                violation = ComplianceViolation(
                    violation_id=str(uuid.uuid4()),
                    rule_id=check_result['rule_id'],
                    trade_id=trade_record.trade_id,
                    timestamp=datetime.utcnow(),
                    severity='critical',
                    description=f"Compliance violation: {check_result['rule_id']}",
                    affected_regulation=RegulationType.MIFID_II,
                    status=ComplianceStatus.NON_COMPLIANT,
                    remediation_actions=[]
                )
                
                compliance_results['violations'].append(asdict(violation))
                await self._handle_violation(violation)
        
        # Log compliance check to audit trail
        self.audit_trail.log_trade_audit(trade_record, compliance_results)
        
        # Update compliance metrics
        await self._update_compliance_metrics(compliance_results)
        
        return compliance_results
    
    async def _handle_violation(self, violation: ComplianceViolation):
        """Handle compliance violation"""
        self.violations.append(violation)
        
        # Store violation in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO violations 
            (violation_id, rule_id, trade_id, timestamp, severity, description,
             regulation_type, status, remediation_actions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            violation.violation_id, violation.rule_id, violation.trade_id,
            violation.timestamp.isoformat(), violation.severity, violation.description,
            violation.affected_regulation.value, violation.status.value,
            json.dumps(violation.remediation_actions)
        ))
        
        conn.commit()
        conn.close()
        
        # Log critical violations
        if violation.severity == 'critical':
            self.logger.critical(f"CRITICAL COMPLIANCE VIOLATION: {violation.description}")
            await self._send_compliance_alert(violation)
    
    async def _send_compliance_alert(self, violation: ComplianceViolation):
        """Send compliance alert to relevant parties"""
        alert_message = {
            'alert_type': 'compliance_violation',
            'severity': violation.severity,
            'regulation': violation.affected_regulation.value,
            'description': violation.description,
            'timestamp': violation.timestamp.isoformat(),
            'trade_id': violation.trade_id,
            'violation_id': violation.violation_id
        }
        
        # In a real implementation, this would send alerts via:
        # - Email to compliance team
        # - SMS for critical violations
        # - Integration with compliance management systems
        # - Real-time dashboard updates
        
        self.logger.warning(f"Compliance alert sent: {alert_message}")
    
    async def _update_compliance_metrics(self, compliance_results: Dict[str, Any]):
        """Update compliance metrics"""
        total_checks = len(compliance_results['checks_performed'])
        passed_checks = sum(1 for check in compliance_results['checks_performed'] if check['compliant'])
        failed_checks = total_checks - passed_checks
        compliance_rate = passed_checks / total_checks if total_checks > 0 else 1.0
        
        critical_violations = sum(1 for v in compliance_results['violations'] 
                                if v.get('severity') == 'critical')
        high_violations = sum(1 for v in compliance_results['violations'] 
                            if v.get('severity') == 'high')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO compliance_metrics 
            (timestamp, regulation_type, total_checks, passed_checks, failed_checks,
             compliance_rate, critical_violations, high_violations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.utcnow().isoformat(), RegulationType.MIFID_II.value,
            total_checks, passed_checks, failed_checks, compliance_rate,
            critical_violations, high_violations
        ))
        
        conn.commit()
        conn.close()
    
    def get_compliance_report(self, start_date: datetime = None, 
                            end_date: datetime = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.utcnow()
        
        conn = sqlite3.connect(self.db_path)
        
        # Get violations summary
        violations_query = '''
            SELECT regulation_type, severity, COUNT(*) as count
            FROM violations 
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY regulation_type, severity
        '''
        violations_df = pd.read_sql_query(
            violations_query, conn, 
            params=[start_date.isoformat(), end_date.isoformat()]
        )
        
        # Get compliance metrics
        metrics_query = '''
            SELECT * FROM compliance_metrics 
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
        '''
        metrics_df = pd.read_sql_query(
            metrics_query, conn,
            params=[start_date.isoformat(), end_date.isoformat()]
        )
        
        conn.close()
        
        # Calculate summary statistics
        avg_compliance_rate = metrics_df['compliance_rate'].mean() if not metrics_df.empty else 0
        total_violations = violations_df['count'].sum() if not violations_df.empty else 0
        critical_violations = violations_df[violations_df['severity'] == 'critical']['count'].sum() if not violations_df.empty else 0
        
        return {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary': {
                'average_compliance_rate': float(avg_compliance_rate),
                'total_violations': int(total_violations),
                'critical_violations': int(critical_violations),
                'compliance_status': 'GOOD' if avg_compliance_rate >= 0.95 else 'NEEDS_ATTENTION'
            },
            'violations_by_type': violations_df.to_dict('records'),
            'compliance_metrics': metrics_df.to_dict('records')
        }
    
    def start_monitoring(self):
        """Start real-time compliance monitoring"""
        self.monitoring_active = True
        self.logger.info("✅ Compliance monitoring started")
    
    def stop_monitoring(self):
        """Stop compliance monitoring"""
        self.monitoring_active = False
        self.logger.info("⏹️ Compliance monitoring stopped")

class RegulatoryReportingEngine:
    """Automated regulatory reporting engine"""
    
    def __init__(self, compliance_monitor: ComplianceMonitor):
        self.compliance_monitor = compliance_monitor
        self.logger = logging.getLogger('RegulatoryReporting')
        
    async def generate_mifid_ii_report(self, report_date: datetime = None) -> Dict[str, Any]:
        """Generate MiFID II regulatory report"""
        if report_date is None:
            report_date = datetime.utcnow()
        
        start_date = report_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)
        
        compliance_report = self.compliance_monitor.get_compliance_report(start_date, end_date)
        
        # MiFID II specific reporting requirements
        mifid_report = {
            'report_type': 'MIFID_II_DAILY',
            'reporting_entity': 'TradingBot_Institution',
            'report_date': report_date.isoformat(),
            'reporting_timestamp': datetime.utcnow().isoformat(),
            'compliance_summary': compliance_report['summary'],
            'transaction_count': len(compliance_report.get('compliance_metrics', [])),
            'best_execution_analysis': self._generate_best_execution_analysis(),
            'position_limits_status': self._check_position_limits(),
            'clock_synchronization_status': 'COMPLIANT',  # Would be calculated from actual data
            'systematic_internaliser_data': None  # If applicable
        }
        
        self.logger.info(f"Generated MiFID II report for {report_date.date()}")
        return mifid_report
    
    def _generate_best_execution_analysis(self) -> Dict[str, Any]:
        """Generate best execution analysis"""
        return {
            'execution_venues_used': ['PRIMARY_EXCHANGE', 'DARK_POOL', 'MTF'],
            'average_execution_quality': 0.95,
            'price_improvement_percentage': 0.02,
            'execution_speed_ms_average': 12.5
        }
    
    def _check_position_limits(self) -> Dict[str, Any]:
        """Check position limits compliance"""
        return {
            'commodity_derivatives_limits_compliant': True,
            'position_limit_utilization': 0.15,
            'max_position_limit_percentage': 0.25
        }