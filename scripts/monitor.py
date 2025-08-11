#!/usr/bin/env python3
"""
System Health Monitoring Script
Monitors trading bot system health and sends alerts for critical issues
"""

import psutil
import sys
import time
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# Add workspace to path
sys.path.append('/workspace')

try:
    from error_handler import global_error_handler
    from config_manager import config_manager
except ImportError as e:
    print(f"Warning: Could not import enhanced modules: {e}")
    global_error_handler = None
    config_manager = None

class SystemMonitor:
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger('SystemMonitor')
        self.alerts = []
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/workspace/logs/monitor.log'),
                logging.StreamHandler()
            ]
        )
    
    def check_system_resources(self):
        """Check system resource usage"""
        alerts = []
        
        # Memory check
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            alerts.append({
                'type': 'CRITICAL',
                'component': 'Memory',
                'message': f'High memory usage: {memory.percent}%',
                'value': memory.percent,
                'threshold': 90
            })
        elif memory.percent > 80:
            alerts.append({
                'type': 'WARNING',
                'component': 'Memory',
                'message': f'Elevated memory usage: {memory.percent}%',
                'value': memory.percent,
                'threshold': 80
            })
        
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 85:
            alerts.append({
                'type': 'CRITICAL',
                'component': 'CPU',
                'message': f'High CPU usage: {cpu_percent}%',
                'value': cpu_percent,
                'threshold': 85
            })
        elif cpu_percent > 70:
            alerts.append({
                'type': 'WARNING',
                'component': 'CPU',
                'message': f'Elevated CPU usage: {cpu_percent}%',
                'value': cpu_percent,
                'threshold': 70
            })
        
        # Disk space check
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        if disk_percent > 90:
            alerts.append({
                'type': 'CRITICAL',
                'component': 'Disk',
                'message': f'Low disk space: {disk_percent:.1f}% used',
                'value': disk_percent,
                'threshold': 90
            })
        elif disk_percent > 80:
            alerts.append({
                'type': 'WARNING',
                'component': 'Disk',
                'message': f'Disk space warning: {disk_percent:.1f}% used',
                'value': disk_percent,
                'threshold': 80
            })
        
        return alerts
    
    def check_trading_processes(self):
        """Check if trading bot processes are running"""
        alerts = []
        
        # Check for Python processes related to trading
        trading_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python3' and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if any(keyword in cmdline for keyword in 
                          ['unified_trading_system', 'telegram_bot', 'signal_engine']):
                        trading_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if not trading_processes:
            alerts.append({
                'type': 'CRITICAL',
                'component': 'Process',
                'message': 'No trading bot processes found running',
                'value': 0,
                'threshold': 1
            })
        else:
            self.logger.info(f"Found {len(trading_processes)} trading processes running")
        
        return alerts
    
    def check_error_rates(self):
        """Check error rates from error handler"""
        alerts = []
        
        if global_error_handler:
            try:
                stats = global_error_handler.get_error_statistics(1)  # Last 1 hour
                total_errors = stats.get('total_errors', 0)
                
                if total_errors > 20:
                    alerts.append({
                        'type': 'CRITICAL',
                        'component': 'Errors',
                        'message': f'High error rate: {total_errors} errors in last hour',
                        'value': total_errors,
                        'threshold': 20
                    })
                elif total_errors > 10:
                    alerts.append({
                        'type': 'WARNING',
                        'component': 'Errors',
                        'message': f'Elevated error rate: {total_errors} errors in last hour',
                        'value': total_errors,
                        'threshold': 10
                    })
                
                # Check recovery rate
                recovery_rate = stats.get('recovery_success_rate', 0)
                if recovery_rate < 50 and stats.get('recovery_attempts', 0) > 0:
                    alerts.append({
                        'type': 'WARNING',
                        'component': 'Recovery',
                        'message': f'Low recovery success rate: {recovery_rate}%',
                        'value': recovery_rate,
                        'threshold': 50
                    })
                    
            except Exception as e:
                alerts.append({
                    'type': 'WARNING',
                    'component': 'Monitoring',
                    'message': f'Failed to get error statistics: {e}',
                    'value': 0,
                    'threshold': 0
                })
        
        return alerts
    
    def check_database_health(self):
        """Check database connectivity and integrity"""
        alerts = []
        
        databases = [
            '/workspace/data/signals.db',
            '/workspace/data/performance.db',
            '/workspace/data/monitoring.db',
            '/workspace/data/risk_management.db'
        ]
        
        for db_path in databases:
            if Path(db_path).exists():
                try:
                    with sqlite3.connect(db_path) as conn:
                        conn.execute('PRAGMA integrity_check').fetchone()
                except Exception as e:
                    alerts.append({
                        'type': 'CRITICAL',
                        'component': 'Database',
                        'message': f'Database integrity check failed for {Path(db_path).name}: {e}',
                        'value': 0,
                        'threshold': 1
                    })
            else:
                alerts.append({
                    'type': 'WARNING',
                    'component': 'Database',
                    'message': f'Database file missing: {Path(db_path).name}',
                    'value': 0,
                    'threshold': 1
                })
        
        return alerts
    
    def check_log_files(self):
        """Check log file sizes and recent activity"""
        alerts = []
        
        log_dir = Path('/workspace/logs')
        if not log_dir.exists():
            alerts.append({
                'type': 'WARNING',
                'component': 'Logs',
                'message': 'Log directory does not exist',
                'value': 0,
                'threshold': 1
            })
            return alerts
        
        # Check for very large log files (>100MB)
        for log_file in log_dir.glob('*.log'):
            try:
                size_mb = log_file.stat().st_size / (1024 * 1024)
                if size_mb > 100:
                    alerts.append({
                        'type': 'WARNING',
                        'component': 'Logs',
                        'message': f'Large log file: {log_file.name} ({size_mb:.1f}MB)',
                        'value': size_mb,
                        'threshold': 100
                    })
            except Exception as e:
                self.logger.warning(f"Could not check log file {log_file}: {e}")
        
        return alerts
    
    def generate_health_report(self):
        """Generate comprehensive health report"""
        self.logger.info("Starting system health check...")
        
        all_alerts = []
        
        # Run all checks
        checks = [
            ('System Resources', self.check_system_resources),
            ('Trading Processes', self.check_trading_processes),
            ('Error Rates', self.check_error_rates),
            ('Database Health', self.check_database_health),
            ('Log Files', self.check_log_files)
        ]
        
        for check_name, check_func in checks:
            try:
                alerts = check_func()
                all_alerts.extend(alerts)
                self.logger.info(f"{check_name}: {len(alerts)} alerts")
            except Exception as e:
                self.logger.error(f"Check {check_name} failed: {e}")
                all_alerts.append({
                    'type': 'CRITICAL',
                    'component': 'Monitoring',
                    'message': f'Health check failed: {check_name} - {e}',
                    'value': 0,
                    'threshold': 0
                })
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_alerts': len(all_alerts),
            'critical_alerts': len([a for a in all_alerts if a['type'] == 'CRITICAL']),
            'warning_alerts': len([a for a in all_alerts if a['type'] == 'WARNING']),
            'alerts': all_alerts,
            'system_info': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100,
                'uptime': time.time() - psutil.boot_time()
            }
        }
        
        return report
    
    def save_report(self, report):
        """Save health report to file"""
        report_file = f"/workspace/logs/health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Health report saved: {report_file}")
        except Exception as e:
            self.logger.error(f"Failed to save health report: {e}")
    
    def print_summary(self, report):
        """Print health check summary"""
        print("\n" + "="*60)
        print("🏥 TRADING BOT SYSTEM HEALTH CHECK")
        print("="*60)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Total Alerts: {report['total_alerts']}")
        print(f"Critical: {report['critical_alerts']}")
        print(f"Warnings: {report['warning_alerts']}")
        
        # System overview
        sys_info = report['system_info']
        print(f"\n📊 System Overview:")
        print(f"CPU Usage: {sys_info['cpu_percent']:.1f}%")
        print(f"Memory Usage: {sys_info['memory_percent']:.1f}%")
        print(f"Disk Usage: {sys_info['disk_percent']:.1f}%")
        print(f"Uptime: {sys_info['uptime']/3600:.1f} hours")
        
        # Show alerts
        if report['alerts']:
            print(f"\n🚨 Alerts:")
            for alert in report['alerts']:
                icon = "🔴" if alert['type'] == 'CRITICAL' else "🟡"
                print(f"{icon} [{alert['type']}] {alert['component']}: {alert['message']}")
        else:
            print("\n✅ All systems healthy - no alerts!")
        
        print("="*60)
    
    def run_health_check(self, save_report=True, print_summary=True):
        """Run complete health check"""
        try:
            report = self.generate_health_report()
            
            if save_report:
                self.save_report(report)
            
            if print_summary:
                self.print_summary(report)
            
            # Return exit code based on alerts
            if report['critical_alerts'] > 0:
                return 2  # Critical issues
            elif report['warning_alerts'] > 0:
                return 1  # Warnings
            else:
                return 0  # All good
                
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            print(f"❌ Health check failed: {e}")
            return 3  # System error

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading Bot System Health Monitor')
    parser.add_argument('--no-save', action='store_true', help='Do not save report to file')
    parser.add_argument('--quiet', action='store_true', help='Suppress summary output')
    parser.add_argument('--continuous', type=int, metavar='SECONDS', 
                       help='Run continuously with specified interval')
    
    args = parser.parse_args()
    
    monitor = SystemMonitor()
    
    if args.continuous:
        print(f"Starting continuous monitoring (interval: {args.continuous}s)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                exit_code = monitor.run_health_check(
                    save_report=not args.no_save,
                    print_summary=not args.quiet
                )
                
                if exit_code >= 2:
                    print("🚨 Critical issues detected!")
                
                time.sleep(args.continuous)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            return 0
    else:
        # Single run
        exit_code = monitor.run_health_check(
            save_report=not args.no_save,
            print_summary=not args.quiet
        )
        return exit_code

if __name__ == "__main__":
    sys.exit(main())