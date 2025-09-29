#!/usr/bin/env python3
"""
Pre-Server Check Script

This script runs on your Mac to validate that your code is ready for the server.
It checks for common issues that could cause problems on the server.

Usage:
    python scripts/pre_server_check.py
"""

import sys
import os
import time
import subprocess
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_file_structure():
    """Check that all required files exist."""
    print("\n=== File Structure Check ===")
    
    project_root = Path(__file__).parent.parent.parent
    required_files = [
        'scripts/gpu/gpu_utils.py',
        'scripts/optimization/optimize_hbt_parameters_gpu.py',
        'notebooks/trimmed_HBT_analysis.py',
        'notebooks/untrimmed_HBT_analysis.py',
        'requirements_gpu.txt',
        'docs/GPU_SETUP_GUIDE.md',
        'scripts/validation/test_gpu_setup.py',
        'scripts/validation/server_validation.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - missing")
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files

def check_imports():
    """Check that all imports work correctly."""
    print("\n=== Import Check ===")
    
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    sys.path.append(str(project_root / 'scripts'))
    
    import_tests = [
        ('tensorflow', 'import tensorflow as tf'),
        ('numpy', 'import numpy as np'),
        ('pandas', 'import pandas as pd'),
        ('matplotlib', 'import matplotlib.pyplot as plt'),
        ('PIL', 'from PIL import Image'),
        ('sklearn', 'from sklearn.metrics import mean_absolute_error'),
    ]
    
    failed_imports = []
    
    for name, import_code in import_tests:
        try:
            exec(import_code)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name} - {e}")
            failed_imports.append(name)
    
    return len(failed_imports) == 0, failed_imports

def check_script_syntax():
    """Check syntax of key scripts."""
    print("\n=== Script Syntax Check ===")
    
    project_root = Path(__file__).parent.parent.parent
    scripts_to_check = [
        'scripts/gpu/gpu_utils.py',
        'scripts/optimization/optimize_hbt_parameters_gpu.py',
        'notebooks/trimmed_HBT_analysis.py',
        'notebooks/untrimmed_HBT_analysis.py',
        'scripts/validation/test_gpu_setup.py',
        'scripts/validation/server_validation.py'
    ]
    
    syntax_errors = []
    
    for script in scripts_to_check:
        script_path = project_root / script
        if script_path.exists():
            try:
                # Check syntax by compiling
                with open(script_path, 'r') as f:
                    compile(f.read(), str(script_path), 'exec')
                print(f"✓ {script}")
            except SyntaxError as e:
                print(f"✗ {script} - Syntax error: {e}")
                syntax_errors.append(script)
            except Exception as e:
                print(f"✗ {script} - Error: {e}")
                syntax_errors.append(script)
        else:
            print(f"✗ {script} - File not found")
            syntax_errors.append(script)
    
    return len(syntax_errors) == 0, syntax_errors

def check_data_directories():
    """Check that data directories exist and have expected structure."""
    print("\n=== Data Directory Check ===")
    
    project_root = Path(__file__).parent.parent.parent
    data_dirs = [
        'data/shots/new',
        'data/shots/old',
        'data/predictions',
        'data/optimization_results'
    ]
    
    missing_dirs = []
    for data_dir in data_dirs:
        full_path = project_root / data_dir
        if full_path.exists():
            print(f"✓ {data_dir}")
        else:
            print(f"✗ {data_dir} - missing")
            missing_dirs.append(data_dir)
    
    return len(missing_dirs) == 0, missing_dirs

def check_gpu_utils_compatibility():
    """Check that GPU utils will work on both CPU and GPU."""
    print("\n=== GPU Utils Compatibility Check ===")
    
    try:
        project_root = Path(__file__).parent.parent.parent.parent
        sys.path.append(str(project_root / 'scripts'))
        
        from scripts.gpu.gpu_utils import (
            configure_gpu_memory,
            get_optimal_batch_size,
            monitor_gpu_usage,
            create_gpu_optimized_model
        )
        
        # Test GPU configuration (should work on CPU)
        gpu_config = configure_gpu_memory()
        print(f"✓ GPU configuration: {gpu_config}")
        
        # Test model creation
        model = create_gpu_optimized_model(
            input_shape=(32, 32, 1),
            conv2d_neurons=[32, 16],
            conv2d_size=[(3, 3), (3, 3)],
            dense_layer_neurons=[64, 32],
            num_conv2d_layers=2,
            num_dense_layers=2,
            max_pooling_size=(2, 2)
        )
        print("✓ Model creation successful")
        
        # Test batch size optimization
        optimal_batch = get_optimal_batch_size(model, (32, 32, 1))
        print(f"✓ Batch size optimization: {optimal_batch}")
        
        return True, []
        
    except Exception as e:
        print(f"✗ GPU utils compatibility failed: {e}")
        return False, [str(e)]

def check_requirements_file():
    """Check that requirements file is properly formatted."""
    print("\n=== Requirements File Check ===")
    
    project_root = Path(__file__).parent.parent.parent
    req_file = project_root / 'requirements_gpu.txt'
    
    if not req_file.exists():
        print("✗ requirements_gpu.txt not found")
        return False, ["requirements_gpu.txt not found"]
    
    try:
        with open(req_file, 'r') as f:
            lines = f.readlines()
        
        # Check for essential packages
        essential_packages = [
            'tensorflow',
            'numpy',
            'pandas',
            'matplotlib',
            'PIL',
            'scipy'
        ]
        
        missing_packages = []
        for package in essential_packages:
            if not any(package in line for line in lines):
                missing_packages.append(package)
        
        if missing_packages:
            print(f"✗ Missing packages in requirements: {missing_packages}")
            return False, missing_packages
        else:
            print("✓ Requirements file looks good")
            return True, []
            
    except Exception as e:
        print(f"✗ Error reading requirements file: {e}")
        return False, [str(e)]

def check_path_consistency():
    """Check that all paths are consistent and relative."""
    print("\n=== Path Consistency Check ===")
    
    project_root = Path(__file__).parent.parent.parent
    
    # Check for hardcoded paths
    problematic_files = []
    
    files_to_check = [
        'scripts/gpu_utils.py',
        'scripts/optimization/optimize_hbt_parameters_gpu.py',
        'notebooks/trimmed_HBT_analysis.py',
        'notebooks/untrimmed_HBT_analysis.py'
    ]
    
    for file_path in files_to_check:
        full_path = project_root / file_path
        if full_path.exists():
            with open(full_path, 'r') as f:
                content = f.read()
                
                # Check for hardcoded paths
                if '/Users/' in content or 'C:\\' in content:
                    print(f"⚠ {file_path} - contains hardcoded paths")
                    problematic_files.append(file_path)
                else:
                    print(f"✓ {file_path} - paths look good")
    
    return len(problematic_files) == 0, problematic_files

def generate_deployment_package():
    """Generate a deployment package for the server."""
    print("\n=== Generating Deployment Package ===")
    
    project_root = Path(__file__).parent.parent.parent
    
    # Create deployment script
    deploy_script = project_root / 'deploy_to_server.sh'
    with open(deploy_script, 'w') as f:
        f.write("""#!/bin/bash
# Deployment script for HBT Analysis on lab server

echo "Deploying HBT Analysis to server..."

# Install GPU requirements
echo "Installing GPU requirements..."
pip install -r requirements_gpu.txt

# Run validation
echo "Running server validation..."
python scripts/server_validation.py

# Test GPU setup
echo "Testing GPU setup..."
python test_gpu_setup.py

echo "Deployment complete!"
echo "Run: python scripts/optimization/optimize_hbt_parameters_gpu.py --use_gpu"
""")
    
    # Make executable
    os.chmod(deploy_script, 0o755)
    print(f"✓ Created deployment script: {deploy_script}")
    
    return True

def main():
    """Run all pre-server checks."""
    print("HBT Analysis Pre-Server Check")
    print("="*40)
    print("This script validates your code before deploying to the server.\n")
    
    # Run all checks
    checks = [
        ("File Structure", check_file_structure),
        ("Imports", check_imports),
        ("Script Syntax", check_script_syntax),
        ("Data Directories", check_data_directories),
        ("GPU Utils Compatibility", check_gpu_utils_compatibility),
        ("Requirements File", check_requirements_file),
        ("Path Consistency", check_path_consistency)
    ]
    
    results = {}
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            passed, issues = check_func()
            results[check_name] = passed
            if not passed:
                all_passed = False
                print(f"  Issues: {issues}")
        except Exception as e:
            print(f"✗ {check_name} failed with exception: {e}")
            results[check_name] = False
            all_passed = False
    
    # Generate deployment package
    generate_deployment_package()
    
    # Summary
    print("\n" + "="*60)
    print("PRE-SERVER CHECK SUMMARY")
    print("="*60)
    
    passed_checks = sum(results.values())
    total_checks = len(results)
    
    print(f"Checks passed: {passed_checks}/{total_checks}")
    
    if all_passed:
        print("\nALL CHECKS PASSED! Your code is ready for the server.")
        print("\nNext steps:")
        print("1. Upload your code to the server")
        print("2. Run: bash deploy_to_server.sh")
        print("3. Run: python scripts/validation/server_validation.py")
        print("4. Start GPU optimization: python scripts/optimization/optimize_hbt_parameters_gpu.py --use_gpu")
    else:
        print(f"\n{total_checks - passed_checks} checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install missing dependencies: pip install -r requirements_gpu.txt")
        print("- Check file paths and structure")
        print("- Fix any syntax errors")
    
    # Save results only if there are issues or if explicitly requested
    if not all_passed or os.environ.get('SAVE_VALIDATION_REPORT', '').lower() == 'true':
        project_root = Path(__file__).parent.parent.parent
        report_file = project_root / 'pre_server_check_report.json'
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'results': results,
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'all_passed': all_passed
            }, f, indent=2)
        
        if not all_passed:
            print(f"\nDetailed report saved to: {report_file}")
        else:
            print(f"\nReport saved to: {report_file} (SAVE_VALIDATION_REPORT=true)")

if __name__ == "__main__":
    main()
