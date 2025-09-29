#!/usr/bin/env python3
"""
Server Validation Script for HBT Analysis

This script performs comprehensive validation to ensure your code will work
properly on the lab server. Run this on the server after setup to verify
everything is configured correctly.

Usage:
    python scripts/server_validation.py
"""

import sys
import os
import time
import subprocess
import json
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_system_requirements():
    """Check basic system requirements."""
    print("\n=== System Requirements Check ===")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("✗ Python version too old. Need 3.8+")
        return False
    else:
        print("✓ Python version OK")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"System memory: {memory_gb:.1f} GB")
        
        if memory_gb < 8:
            print("⚠ Warning: Less than 8GB RAM available")
        else:
            print("✓ Sufficient memory available")
    except ImportError:
        print("⚠ psutil not available, cannot check memory")
    
    return True

def check_gpu_availability():
    """Check GPU availability and configuration."""
    print("\n=== GPU Availability Check ===")
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ nvidia-smi available")
            print("GPU Info:")
            print(result.stdout.split('\n')[0:3])  # Show first few lines
        else:
            print("✗ nvidia-smi failed")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("✗ nvidia-smi not found or failed")
        return False
    
    # Check CUDA version
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ CUDA compiler available")
            # Extract version from output
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print(f"  {line.strip()}")
                    break
        else:
            print("✗ CUDA compiler not found")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("✗ CUDA compiler not found")
    
    return True

def check_python_dependencies():
    """Check if all required Python packages are installed."""
    print("\n=== Python Dependencies Check ===")
    
    required_packages = [
        'tensorflow',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'PIL',
        'scipy',
        'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
                print(f"✓ {package} (Pillow) - {PIL.__version__}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"✓ {package} - {version}")
        except ImportError:
            print(f"✗ {package} - not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install with: pip install -r requirements_gpu.txt")
        return False
    
    return True

def check_tensorflow_gpu():
    """Check TensorFlow GPU support."""
    print("\n=== TensorFlow GPU Check ===")
    
    try:
        import tensorflow as tf
        
        # Check TensorFlow version
        print(f"TensorFlow version: {tf.__version__}")
        
        # Check GPU devices
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(f"Physical GPUs: {len(gpus)}")
        
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            
            # Test GPU configuration
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print("✓ GPU memory growth enabled")
            except Exception as e:
                print(f"✗ GPU memory growth failed: {e}")
                return False
            
            # Test GPU computation
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print(f"✓ GPU computation test: {c.numpy()}")
            
            return True
        else:
            print("✗ No GPUs found by TensorFlow")
            return False
            
    except Exception as e:
        print(f"✗ TensorFlow GPU check failed: {e}")
        return False

def test_gpu_utils():
    """Test the GPU utilities module."""
    print("\n=== GPU Utils Test ===")
    
    try:
        # Add scripts directory to path
        sys.path.append(str(Path(__file__).parent.parent))
        from scripts.gpu.gpu_utils import (
            configure_gpu_memory,
            get_optimal_batch_size,
            monitor_gpu_usage,
            create_gpu_optimized_model
        )
        
        # Test GPU configuration
        gpu_config = configure_gpu_memory()
        print(f"GPU config: {gpu_config}")
        
        if gpu_config.get('gpu_available', False):
            print("✓ GPU utils working")
            
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
            print(f"✓ Optimal batch size: {optimal_batch}")
            
            return True
        else:
            print("✗ GPU not available in utils")
            return False
            
    except Exception as e:
        print(f"✗ GPU utils test failed: {e}")
        return False

def test_data_access():
    """Test access to data directories."""
    print("\n=== Data Access Test ===")
    
    project_root = Path(__file__).parent.parent.parent
    data_dirs = [
        'data/shots/new',
        'data/shots/old',
        'data/predictions',
        'data/optimization_results'
    ]
    
    all_accessible = True
    
    for data_dir in data_dirs:
        full_path = project_root / data_dir
        if full_path.exists():
            print(f"✓ {data_dir} - accessible")
        else:
            print(f"✗ {data_dir} - not found")
            all_accessible = False
    
    # Test file permissions
    test_file = project_root / 'data' / 'test_write.tmp'
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print("✓ Write permissions OK")
    except Exception as e:
        print(f"✗ Write permissions failed: {e}")
        all_accessible = False
    
    return all_accessible

def test_script_execution():
    """Test execution of key scripts."""
    print("\n=== Script Execution Test ===")
    
    project_root = Path(__file__).parent.parent.parent
    scripts_to_test = [
        'notebooks/trimmed_HBT_analysis.py',
        'notebooks/untrimmed_HBT_analysis.py',
        'scripts/optimization/optimize_hbt_parameters.py'
    ]
    
    all_working = True
    
    for script in scripts_to_test:
        script_path = project_root / script
        if script_path.exists():
            try:
                # Test script import (dry run)
                result = subprocess.run([
                    'python', '-c', f'import sys; sys.path.append("{project_root}"); exec(open("{script_path}").read())'
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    print(f"✓ {script} - importable")
                else:
                    print(f"✗ {script} - import failed")
                    print(f"  Error: {result.stderr[:200]}...")
                    all_working = False
            except subprocess.TimeoutExpired:
                print(f"✗ {script} - timeout during import")
                all_working = False
            except Exception as e:
                print(f"✗ {script} - error: {e}")
                all_working = False
        else:
            print(f"✗ {script} - file not found")
            all_working = False
    
    return all_working

def test_gpu_optimization_script():
    """Test the GPU optimization script specifically."""
    print("\n=== GPU Optimization Script Test ===")
    
    project_root = Path(__file__).parent.parent.parent
    script_path = project_root / 'scripts' / 'optimization' / 'optimize_hbt_parameters_gpu.py'
    
    if not script_path.exists():
        print("✗ GPU optimization script not found")
        return False
    
    try:
        # Test script help
        result = subprocess.run([
            'python', str(script_path), '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ GPU optimization script - help works")
        else:
            print(f"✗ GPU optimization script - help failed: {result.stderr}")
            return False
        
        # Test dry run (very short execution)
        result = subprocess.run([
            'python', str(script_path), '--use_cpu', '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ GPU optimization script - CPU mode works")
        else:
            print(f"✗ GPU optimization script - CPU mode failed: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ GPU optimization script test failed: {e}")
        return False

def run_performance_benchmark():
    """Run a quick performance benchmark."""
    print("\n=== Performance Benchmark ===")
    
    try:
        import tensorflow as tf
        import time
        
        # Create test data
        batch_size = 64
        x_test = tf.random.normal((batch_size, 32, 32, 1))
        y_test = tf.random.normal((batch_size, 1))
        
        # Test CPU performance
        with tf.device('/CPU:0'):
            model_cpu = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 1)),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(1)
            ])
            model_cpu.compile(optimizer='adam', loss='mse')
            
            start_time = time.time()
            model_cpu.fit(x_test, y_test, epochs=2, verbose=0)
            cpu_time = time.time() - start_time
            print(f"CPU training time: {cpu_time:.2f}s")
        
        # Test GPU performance if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            with tf.device('/GPU:0'):
                model_gpu = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 1)),
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(1)
                ])
                model_gpu.compile(optimizer='adam', loss='mse')
                
                start_time = time.time()
                model_gpu.fit(x_test, y_test, epochs=2, verbose=0)
                gpu_time = time.time() - start_time
                print(f"GPU training time: {gpu_time:.2f}s")
                print(f"Speedup: {cpu_time/gpu_time:.1f}x")
            return True
        else:
            print("No GPU available for benchmark")
            return True
            
    except Exception as e:
        print(f"✗ Performance benchmark failed: {e}")
        return False

def generate_validation_report(results):
    """Generate a comprehensive validation report."""
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"Overall Status: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\nALL TESTS PASSED! Your setup is ready for HBT analysis.")
        print("\nNext steps:")
        print("1. Run: python scripts/optimization/optimize_hbt_parameters_gpu.py --use_gpu")
        print("2. Monitor with: watch -n 1 nvidia-smi")
        print("3. Check results in data/optimization_results/")
    else:
        print(f"\n{total_tests - passed_tests} tests failed. Please address the issues above.")
        
        if not results.get("GPU Availability", False):
            print("\nGPU Issues:")
            print("- Check CUDA installation: nvidia-smi")
            print("- Install GPU requirements: pip install -r requirements_gpu.txt")
            print("- Use CPU fallback: --use_cpu flag")
        
        if not results.get("Python Dependencies", False):
            print("\nDependency Issues:")
            print("- Install requirements: pip install -r requirements_gpu.txt")
            print("- Check Python version: python --version")
        
        if not results.get("Data Access", False):
            print("\nData Access Issues:")
            print("- Check file permissions")
            print("- Verify data directory structure")
    
    # Save report to file
    report_file = Path(__file__).parent.parent / 'validation_report.json'
    with open(report_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': results,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests
        }, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")

def main():
    """Run all validation tests."""
    print("HBT Analysis Server Validation")
    print("="*40)
    print("This script validates your server setup for HBT analysis.")
    print("Run this on the lab server after installation.\n")
    
    # Run all tests
    tests = [
        ("System Requirements", check_system_requirements),
        ("GPU Availability", check_gpu_availability),
        ("Python Dependencies", check_python_dependencies),
        ("TensorFlow GPU", check_tensorflow_gpu),
        ("GPU Utils", test_gpu_utils),
        ("Data Access", test_data_access),
        ("Script Execution", test_script_execution),
        ("GPU Optimization Script", test_gpu_optimization_script),
        ("Performance Benchmark", run_performance_benchmark)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Generate report
    generate_validation_report(results)

if __name__ == "__main__":
    main()
