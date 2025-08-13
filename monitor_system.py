#!/usr/bin/env python3
"""
🔍 ULTIMATE AI TRADING SYSTEM MONITOR
Real-time monitoring and status checker for the trading system
"""

import os
import sys
import time
import psutil
import json
from datetime import datetime, timedelta
import subprocess
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/logs/system_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('SystemMonitor')

class UltimateSystemMonitor:
    """🎯 Ultimate Trading System Monitor"""
    
    def __init__(self):
        self.workspace = '/workspace'
        self.logs_dir = os.path.join(self.workspace, 'logs')
        self.processes = {}
        self.start_time = datetime.now()
        
    def check_processes(self):
        """Check if trading system processes are running"""
        processes = {}
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
            try:
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                
                if 'python' in proc.info['name'] and any(script in cmdline for script in [
                    'main.py', 'ultimate_ai_trading_bot.py', 'working_telegram_bot.py',
                    'universal_trading_launcher.py', 'unified_trading_system.py'
                ]):
                    processes[proc.info['pid']] = {
                        'name': proc.info['name'],
                        'cmdline': cmdline,
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent'],
                        'status': 'running'
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return processes
    
    def check_logs(self):
        """Check recent log activity"""
        log_status = {}
        
        important_logs = [
            'trading_system.log',
            'telegram_bot.log',
            'signal_engine.log',
            'ultimate_ai_system.log',
            'pocket_option_api.log'
        ]
        
        for log_file in important_logs:
            log_path = os.path.join(self.logs_dir, log_file)
            if os.path.exists(log_path):
                try:
                    stat = os.stat(log_path)
                    last_modified = datetime.fromtimestamp(stat.st_mtime)
                    size = stat.st_size
                    
                    # Check if log was updated in last 5 minutes
                    is_active = (datetime.now() - last_modified) < timedelta(minutes=5)
                    
                    log_status[log_file] = {
                        'exists': True,
                        'last_modified': last_modified.isoformat(),
                        'size_bytes': size,
                        'is_active': is_active
                    }
                except Exception as e:
                    log_status[log_file] = {
                        'exists': True,
                        'error': str(e)
                    }
            else:
                log_status[log_file] = {'exists': False}
                
        return log_status
    
    def check_telegram_bot(self):
        """Check Telegram bot connectivity"""
        try:
            # Try to import and test bot config
            sys.path.append(self.workspace)
            from config import TELEGRAM_BOT_TOKEN, TELEGRAM_USER_ID
            
            return {
                'token_configured': bool(TELEGRAM_BOT_TOKEN),
                'user_id_configured': bool(TELEGRAM_USER_ID),
                'status': 'configured'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def check_system_resources(self):
        """Check system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'status': 'healthy' if cpu_percent < 80 and memory.percent < 80 else 'warning'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def generate_report(self):
        """Generate comprehensive system status report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'uptime_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
            'processes': self.check_processes(),
            'logs': self.check_logs(),
            'telegram_bot': self.check_telegram_bot(),
            'system_resources': self.check_system_resources()
        }
        
        # Count active processes
        active_processes = len(report['processes'])
        active_logs = sum(1 for log in report['logs'].values() if log.get('is_active', False))
        
        # Determine overall status
        if active_processes > 0 and active_logs > 0:
            report['overall_status'] = 'running'
        elif active_processes > 0:
            report['overall_status'] = 'partially_running'
        else:
            report['overall_status'] = 'stopped'
            
        return report
    
    def print_status(self, report):
        """Print formatted status report"""
        print("\n" + "="*80)
        print("🚀 ULTIMATE AI TRADING SYSTEM - STATUS REPORT")
        print("="*80)
        
        print(f"📊 Overall Status: {report['overall_status'].upper()}")
        print(f"⏱️  System Uptime: {report['uptime_minutes']:.1f} minutes")
        print(f"🕐 Report Time: {report['timestamp']}")
        
        print(f"\n🔧 PROCESSES ({len(report['processes'])} running):")
        for pid, proc in report['processes'].items():
            print(f"  ✅ PID {pid}: {proc['cmdline'][:60]}...")
            print(f"     CPU: {proc['cpu_percent']:.1f}% | Memory: {proc['memory_percent']:.1f}%")
        
        print(f"\n📋 LOG FILES:")
        for log_file, status in report['logs'].items():
            if status['exists']:
                active_indicator = "🟢" if status.get('is_active', False) else "🟡"
                print(f"  {active_indicator} {log_file}: {status.get('size_bytes', 0)} bytes")
            else:
                print(f"  ❌ {log_file}: Not found")
        
        print(f"\n📱 TELEGRAM BOT:")
        tg_status = report['telegram_bot']
        if tg_status['status'] == 'configured':
            print("  ✅ Bot configured and ready")
        else:
            print(f"  ❌ Bot error: {tg_status.get('error', 'Unknown')}")
        
        print(f"\n💻 SYSTEM RESOURCES:")
        res = report['system_resources']
        if res['status'] == 'healthy':
            print(f"  ✅ CPU: {res['cpu_percent']:.1f}%")
            print(f"  ✅ Memory: {res['memory_percent']:.1f}% ({res['memory_available_gb']:.1f}GB free)")
            print(f"  ✅ Disk: {res['disk_percent']:.1f}% ({res['disk_free_gb']:.1f}GB free)")
        else:
            print(f"  ⚠️  Resource warning - check system load")
        
        print("="*80)
    
    def monitor_continuous(self, interval=30):
        """Run continuous monitoring"""
        logger.info("🔍 Starting continuous system monitoring...")
        
        try:
            while True:
                report = self.generate_report()
                self.print_status(report)
                
                # Save report to file
                report_file = os.path.join(self.logs_dir, 'system_status.json')
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2)
                
                logger.info(f"📊 Status: {report['overall_status']} | Processes: {len(report['processes'])} | Active logs: {sum(1 for log in report['logs'].values() if log.get('is_active', False))}")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")

def main():
    """Main monitoring function"""
    monitor = UltimateSystemMonitor()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        monitor.monitor_continuous()
    else:
        report = monitor.generate_report()
        monitor.print_status(report)

if __name__ == "__main__":
    main()