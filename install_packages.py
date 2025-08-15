#!/usr/bin/env python3
"""
Install Required Python Packages for Ultimate Trading System
"""

import subprocess
import sys
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PackageInstaller')

def run_command(command, description):
    """Run a command and handle errors"""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info(f"✅ {description} completed successfully")
            if result.stdout.strip():
                logger.info(f"Output: {result.stdout.strip()}")
            return True
        else:
            logger.error(f"❌ {description} failed")
            logger.error(f"Error: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {description} timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"❌ {description} failed with exception: {e}")
        return False

def check_python_version():
    """Check Python version compatibility"""
    logger.info("Checking Python version...")
    
    version = sys.version_info
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        logger.info("✅ Python version is compatible (3.8+)")
        return True
    else:
        logger.error("❌ Python 3.8+ required")
        return False

def install_pip_upgrade():
    """Upgrade pip to latest version"""
    logger.info("Upgrading pip...")
    
    commands = [
        ("python3 -m pip install --upgrade pip", "Pip upgrade"),
        ("python3 -m pip install --upgrade setuptools wheel", "Setuptools and wheel upgrade")
    ]
    
    success_count = 0
    for command, description in commands:
        if run_command(command, description):
            success_count += 1
    
    return success_count == len(commands)

def install_core_packages():
    """Install core numerical and ML packages"""
    logger.info("Installing core packages...")
    
    core_packages = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0"
    ]
    
    success_count = 0
    for package in core_packages:
        command = f"python3 -m pip install {package}"
        if run_command(command, f"Install {package}"):
            success_count += 1
    
    logger.info(f"Core packages: {success_count}/{len(core_packages)} installed successfully")
    return success_count >= len(core_packages) * 0.8  # 80% success threshold

def install_tensorflow():
    """Install TensorFlow with GPU support if available"""
    logger.info("Installing TensorFlow...")
    
    # Check if CUDA is available
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, timeout=10)
        if result.returncode == 0:
            logger.info("✅ CUDA detected, installing TensorFlow with GPU support")
            command = "python3 -m pip install tensorflow[gpu]>=2.16.0"
        else:
            logger.info("ℹ️  CUDA not detected, installing CPU-only TensorFlow")
            command = "python3 -m pip install tensorflow>=2.16.0"
    except:
        logger.info("ℹ️  CUDA check failed, installing CPU-only TensorFlow")
        command = "python3 -m pip install tensorflow>=2.16.0"
    
    return run_command(command, "Install TensorFlow")

def install_pytorch():
    """Install PyTorch with appropriate backend"""
    logger.info("Installing PyTorch...")
    
    # Check if CUDA is available
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, timeout=10)
        if result.returncode == 0:
            logger.info("✅ CUDA detected, installing PyTorch with GPU support")
            command = "python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        else:
            logger.info("ℹ️  CUDA not detected, installing CPU-only PyTorch")
            command = "python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    except:
        logger.info("ℹ️  CUDA check failed, installing CPU-only PyTorch")
        command = "python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    return run_command(command, "Install PyTorch")

def install_specialized_packages():
    """Install specialized packages for trading and ML"""
    logger.info("Installing specialized packages...")
    
    specialized_packages = [
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        "optuna>=3.0.0",
        "gym>=0.21.0",
        "stable-baselines3>=1.5.0",
        "plotly>=5.0.0",
        "dash>=2.0.0",
        "ta-lib>=0.4.0",  # Technical analysis
        "yfinance>=0.1.70",  # Yahoo Finance data
        "ccxt>=2.0.0",  # Cryptocurrency exchange
        "pytz>=2021.1",  # Timezone handling
        "python-dotenv>=0.19.0"  # Environment variables
    ]
    
    success_count = 0
    for package in specialized_packages:
        command = f"python3 -m pip install {package}"
        if run_command(command, f"Install {package}"):
            success_count += 1
        else:
            logger.warning(f"⚠️  Failed to install {package}, continuing...")
    
    logger.info(f"Specialized packages: {success_count}/{len(specialized_packages)} installed successfully")
    return success_count >= len(specialized_packages) * 0.7  # 70% success threshold

def verify_installations():
    """Verify that key packages can be imported"""
    logger.info("Verifying package installations...")
    
    packages_to_test = [
        'numpy', 'pandas', 'scipy', 'sklearn', 'tensorflow', 'torch', 
        'xgboost', 'optuna', 'gym', 'plotly'
    ]
    
    success_count = 0
    for package in packages_to_test:
        try:
            __import__(package)
            logger.info(f"✅ {package}: Import successful")
            success_count += 1
        except ImportError as e:
            logger.error(f"❌ {package}: Import failed - {e}")
    
    logger.info(f"Package verification: {success_count}/{len(packages_to_test)} successful")
    return success_count >= len(packages_to_test) * 0.8  # 80% success threshold

def create_requirements_installed():
    """Create a requirements file with actually installed versions"""
    logger.info("Creating requirements file with installed versions...")
    
    try:
        result = subprocess.run("python3 -m pip freeze", shell=True, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            with open('/workspace/requirements-installed.txt', 'w') as f:
                f.write("# Installed package versions\n")
                f.write(f"# Generated at: {datetime.now().isoformat()}\n")
                f.write("# Command: python3 -m pip freeze\n\n")
                f.write(result.stdout)
            
            logger.info("✅ requirements-installed.txt created successfully")
            return True
        else:
            logger.error("❌ Failed to create requirements file")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to create requirements file: {e}")
        return False

def main():
    """Main installation function"""
    logger.info("🚀 Starting Package Installation for Ultimate Trading System...")
    
    # Check Python version
    if not check_python_version():
        logger.error("❌ Python version check failed. Exiting.")
        return False
    
    # Installation steps
    installation_steps = [
        ("Pip Upgrade", install_pip_upgrade),
        ("Core Packages", install_core_packages),
        ("TensorFlow", install_tensorflow),
        ("PyTorch", install_pytorch),
        ("Specialized Packages", install_specialized_packages),
        ("Verification", verify_installations),
        ("Requirements File", create_requirements_installed)
    ]
    
    success_count = 0
    total_steps = len(installation_steps)
    
    for step_name, step_func in installation_steps:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {step_name}")
        logger.info(f"{'='*50}")
        
        if step_func():
            success_count += 1
            logger.info(f"✅ {step_name} PASSED")
        else:
            logger.error(f"❌ {step_name} FAILED")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("INSTALLATION SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Passed: {success_count}/{total_steps}")
    logger.info(f"Success Rate: {(success_count/total_steps)*100:.1f}%")
    
    if success_count >= total_steps * 0.8:  # 80% threshold
        logger.info("🎉 Package installation completed successfully!")
        logger.info("✅ System is ready for the next steps")
        return True
    else:
        logger.warning(f"⚠️  {total_steps-success_count} step(s) failed. Check logs above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)