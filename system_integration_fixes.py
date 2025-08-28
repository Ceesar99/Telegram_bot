#!/usr/bin/env python3
"""
🔧 SYSTEM INTEGRATION FIXES - AUTOMATED REPAIRS
Comprehensive fixes for critical system integration issues
"""

import os
import sys
import subprocess
import logging
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemIntegrationFixer:
    """Automated system integration fixes"""
    
    def __init__(self):
        self.fixes_applied = []
        self.errors_found = []
        
    def check_dependencies(self):
        """Check and install missing dependencies"""
        logger.info("🔍 Checking dependencies...")
        
        try:
            # Check if requirements are installed
            result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                                  capture_output=True, text=True)
            installed_packages = result.stdout.lower()
            
            missing_packages = []
            critical_packages = [
                'aiohttp', 'torch', 'gym', 'dash', 'h5py', 'ta', 
                'redis', 'asyncpg', 'tensorflow', 'xgboost'
            ]
            
            for package in critical_packages:
                if package not in installed_packages:
                    missing_packages.append(package)
            
            if missing_packages:
                logger.warning(f"❌ Missing packages: {missing_packages}")
                self.errors_found.append(f"Missing dependencies: {missing_packages}")
                return False
            else:
                logger.info("✅ All critical dependencies are installed")
                self.fixes_applied.append("Dependencies verified")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error checking dependencies: {e}")
            self.errors_found.append(f"Dependency check failed: {e}")
            return False
    
    def install_missing_dependencies(self):
        """Install missing dependencies"""
        logger.info("📦 Installing missing dependencies...")
        
        try:
            # Install from requirements.txt
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Dependencies installed successfully")
                self.fixes_applied.append("Dependencies installed")
                return True
            else:
                logger.error(f"❌ Failed to install dependencies: {result.stderr}")
                self.errors_found.append(f"Dependency installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error installing dependencies: {e}")
            self.errors_found.append(f"Installation error: {e}")
            return False
    
    def validate_model_files(self):
        """Validate model files exist and are accessible"""
        logger.info("🧠 Validating model files...")
        
        model_files = [
            '/workspace/models/production_lstm_trained.h5',
            '/workspace/models/feature_scaler.pkl'
        ]
        
        valid_models = 0
        for model_file in model_files:
            if os.path.exists(model_file):
                try:
                    size = os.path.getsize(model_file)
                    if size > 0:
                        logger.info(f"✅ Valid model: {model_file} ({size/1024:.1f} KB)")
                        valid_models += 1
                    else:
                        logger.warning(f"⚠️ Empty model file: {model_file}")
                        self.errors_found.append(f"Empty model file: {model_file}")
                except Exception as e:
                    logger.error(f"❌ Error reading {model_file}: {e}")
                    self.errors_found.append(f"Model file error: {e}")
            else:
                logger.warning(f"❌ Missing model file: {model_file}")
                self.errors_found.append(f"Missing model: {model_file}")
        
        if valid_models > 0:
            self.fixes_applied.append(f"Model validation: {valid_models}/{len(model_files)} valid")
            return True
        else:
            return False
    
    def test_database_connections(self):
        """Test database connections and fix issues"""
        logger.info("🗄️ Testing database connections...")
        
        import sqlite3
        from config import DATABASE_CONFIG
        
        databases = [
            DATABASE_CONFIG.get('signals_db', '/workspace/data/signals.db'),
            DATABASE_CONFIG.get('performance_db', '/workspace/data/performance.db'),
        ]
        
        valid_dbs = 0
        for db_path in databases:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                
                # Test connection
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    logger.info(f"✅ Database connection OK: {db_path}")
                    valid_dbs += 1
                    
            except Exception as e:
                logger.error(f"❌ Database error {db_path}: {e}")
                self.errors_found.append(f"Database error: {e}")
        
        if valid_dbs > 0:
            self.fixes_applied.append(f"Database validation: {valid_dbs}/{len(databases)} valid")
            return True
        else:
            return False
    
    def test_import_system(self):
        """Test critical system imports"""
        logger.info("📦 Testing system imports...")
        
        critical_imports = [
            'config',
            'lstm_model',
            'ensemble_models',
            'enhanced_signal_engine',
            'enhanced_risk_manager',
            'paper_trading_engine'
        ]
        
        successful_imports = 0
        for module in critical_imports:
            try:
                __import__(module)
                logger.info(f"✅ Import successful: {module}")
                successful_imports += 1
            except Exception as e:
                logger.error(f"❌ Import failed {module}: {e}")
                self.errors_found.append(f"Import error {module}: {e}")
        
        if successful_imports >= len(critical_imports) * 0.8:  # 80% success rate
            self.fixes_applied.append(f"Import validation: {successful_imports}/{len(critical_imports)} successful")
            return True
        else:
            return False
    
    def create_missing_directories(self):
        """Create missing directories"""
        logger.info("📁 Creating missing directories...")
        
        required_dirs = [
            '/workspace/data',
            '/workspace/models',
            '/workspace/logs',
            '/workspace/backup'
        ]
        
        created_dirs = 0
        for directory in required_dirs:
            try:
                os.makedirs(directory, exist_ok=True)
                if os.path.exists(directory):
                    logger.info(f"✅ Directory ready: {directory}")
                    created_dirs += 1
                else:
                    logger.error(f"❌ Failed to create: {directory}")
                    self.errors_found.append(f"Directory creation failed: {directory}")
            except Exception as e:
                logger.error(f"❌ Error creating {directory}: {e}")
                self.errors_found.append(f"Directory error: {e}")
        
        self.fixes_applied.append(f"Directory creation: {created_dirs}/{len(required_dirs)} ready")
        return created_dirs == len(required_dirs)
    
    def generate_fix_report(self):
        """Generate comprehensive fix report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            "timestamp": timestamp,
            "fixes_applied": self.fixes_applied,
            "errors_found": self.errors_found,
            "success_rate": len(self.fixes_applied) / (len(self.fixes_applied) + len(self.errors_found)) * 100 if (self.fixes_applied or self.errors_found) else 0,
            "overall_status": "READY" if len(self.errors_found) == 0 else "NEEDS_ATTENTION"
        }
        
        # Save report
        report_path = f'/workspace/system_integration_report_{timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def run_comprehensive_fixes(self):
        """Run all system integration fixes"""
        logger.info("🚀 Starting comprehensive system integration fixes...")
        
        # Run all fix methods
        fixes = [
            self.create_missing_directories,
            self.check_dependencies,
            self.validate_model_files,
            self.test_database_connections,
            self.test_import_system
        ]
        
        success_count = 0
        for fix_method in fixes:
            try:
                if fix_method():
                    success_count += 1
            except Exception as e:
                logger.error(f"❌ Fix method failed: {fix_method.__name__}: {e}")
                self.errors_found.append(f"Fix method error: {e}")
        
        # Generate final report
        report = self.generate_fix_report()
        
        logger.info("=" * 60)
        logger.info("🔧 SYSTEM INTEGRATION FIXES COMPLETE")
        logger.info(f"✅ Fixes Applied: {len(self.fixes_applied)}")
        logger.info(f"❌ Errors Found: {len(self.errors_found)}")
        logger.info(f"📊 Success Rate: {report['success_rate']:.1f}%")
        logger.info(f"🎯 Overall Status: {report['overall_status']}")
        logger.info("=" * 60)
        
        return report

if __name__ == "__main__":
    fixer = SystemIntegrationFixer()
    report = fixer.run_comprehensive_fixes()
    
    # Print summary
    print(f"\n🎯 SYSTEM INTEGRATION SUMMARY:")
    print(f"Status: {report['overall_status']}")
    print(f"Success Rate: {report['success_rate']:.1f}%")
    
    if report['errors_found']:
        print(f"\n❌ Issues requiring attention:")
        for error in report['errors_found']:
            print(f"  • {error}")
    
    if report['fixes_applied']:
        print(f"\n✅ Fixes successfully applied:")
        for fix in report['fixes_applied']:
            print(f"  • {fix}")