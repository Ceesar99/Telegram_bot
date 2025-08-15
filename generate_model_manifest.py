#!/usr/bin/env python3
"""
Generate comprehensive model manifest with version hashes and dependency information
"""

import hashlib
import json
import os
import sys
from datetime import datetime
import importlib
import subprocess

def get_package_version(package_name):
    """Get package version, handling various import scenarios"""
    try:
        # Try direct import first
        mod = importlib.import_module(package_name)
        version = getattr(mod, '__version__', None)
        if version:
            return version
        
        # Try alternative names
        alt_names = {
            'tensorflow': 'tf',
            'torch': 'torch',
            'sklearn': 'scikit-learn',
            'cv2': 'opencv-python'
        }
        
        if package_name in alt_names:
            alt_mod = importlib.import_module(alt_names[package_name])
            version = getattr(alt_mod, '__version__', None)
            if version:
                return version
                
    except ImportError:
        pass
    
    # Try pip show as fallback
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'show', package_name], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':', 1)[1].strip()
    except:
        pass
    
    return 'unknown'

def hash_file(filepath):
    """Generate SHA256 hash of a file"""
    if not os.path.exists(filepath):
        return None
    
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception:
        return None

def generate_manifest():
    """Generate comprehensive model manifest"""
    
    # Key ML/AI dependencies to track
    key_dependencies = [
        'tensorflow', 'torch', 'xgboost', 'scikit-learn', 'pandas', 'numpy',
        'scipy', 'matplotlib', 'seaborn', 'plotly', 'talib', 'optuna',
        'gym', 'stable-baselines3', 'transformers', 'datasets'
    ]
    
    # System information
    system_info = {
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'platform': sys.platform,
        'architecture': '64bit' if sys.maxsize > 2**32 else '32bit'
    }
    
    # Dependency versions
    dependencies = {}
    for dep in key_dependencies:
        version = get_package_version(dep)
        dependencies[dep] = version
        print(f"✓ {dep}: {version}")
    
    # Model artifacts (if they exist)
    model_artifacts = {}
    models_dir = '/workspace/models'
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            if item.endswith(('.h5', '.pkl', '.pth', '.ts', '.json')):
                filepath = os.path.join(models_dir, item)
                file_hash = hash_file(filepath)
                if file_hash:
                    model_artifacts[item] = {
                        'hash': file_hash,
                        'size_bytes': os.path.getsize(filepath),
                        'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                    }
    
    # Configuration files
    config_files = {}
    config_paths = [
        '/workspace/config.py',
        '/workspace/requirements.txt',
        '/workspace/requirements-constraints.txt'
    ]
    
    for config_path in config_paths:
        if os.path.exists(config_path):
            config_hash = hash_file(config_path)
            if config_hash:
                config_files[os.path.basename(config_path)] = {
                    'hash': config_hash,
                    'size_bytes': os.path.getsize(config_path)
                }
    
    # Create manifest
    manifest = {
        'generated_at': datetime.utcnow().isoformat(),
        'system_info': system_info,
        'dependencies': dependencies,
        'model_artifacts': model_artifacts,
        'config_files': config_files,
        'workspace_info': {
            'models_dir': models_dir,
            'logs_dir': '/workspace/logs',
            'data_dir': '/workspace/data'
        }
    }
    
    # Save manifest
    manifest_path = '/workspace/model_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Model manifest generated: {manifest_path}")
    print(f"📊 Dependencies tracked: {len(dependencies)}")
    print(f"🔧 Model artifacts: {len(model_artifacts)}")
    print(f"⚙️  Config files: {len(config_files)}")
    
    return manifest

if __name__ == "__main__":
    print("🚀 Generating Model Manifest...")
    manifest = generate_manifest()
    print("\n📋 Manifest Summary:")
    print(f"   Generated: {manifest['generated_at']}")
    print(f"   Python: {manifest['system_info']['python_version']}")
    print(f"   Platform: {manifest['system_info']['platform']}")