#!/usr/bin/env python3
"""
GPU Setup Test Script

This script tests your GPU setup and provides recommendations for optimal configuration.
Run this on the lab server to verify everything is working correctly.

Usage:
    python test_gpu_setup.py
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / 'scripts'))

try:
    import tensorflow as tf
    from scripts.gpu_utils import (
        configure_gpu_memory, 
        get_optimal_batch_size, 
        monitor_gpu_usage,
        create_gpu_optimized_model,
        setup_gpu_environment
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please install requirements: pip install -r requirements_gpu.txt")
    sys.exit(1)

def test_gpu_availability():
    """Test if GPU is available and configured."""
    print("\n=== GPU Availability Test ===")
    
    # Check physical devices
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"Physical GPUs found: {len(gpus)}")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    
    # Check logical devices
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(f"Logical GPUs available: {len(logical_gpus)}")
    
    return len(gpus) > 0

def test_gpu_configuration():
    """Test GPU configuration and memory management."""
    print("\n=== GPU Configuration Test ===")
    
    try:
        gpu_config = configure_gpu_memory(
            gpu_memory_limit=None,  # Use all available memory
            allow_memory_growth=True
        )
        
        print(f"GPU Configuration: {gpu_config}")
        
        if gpu_config.get('gpu_available', False):
            print("✓ GPU configuration successful")
            return True
        else:
            print("✗ GPU configuration failed")
            return False
            
    except Exception as e:
        print(f"✗ GPU configuration error: {e}")
        return False

def test_model_creation():
    """Test creating a GPU-optimized model."""
    print("\n=== Model Creation Test ===")
    
    try:
        # Create a simple model
        model = create_gpu_optimized_model(
            input_shape=(32, 32, 1),
            conv2d_neurons=[32, 16],
            conv2d_size=[(3, 3), (3, 3)],
            dense_layer_neurons=[64, 32],
            num_conv2d_layers=2,
            num_dense_layers=2,
            max_pooling_size=(2, 2)
        )
        
        print("✓ Model created successfully")
        print(f"Model summary:")
        model.summary()
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False

def test_batch_size_optimization():
    """Test batch size optimization."""
    print("\n=== Batch Size Optimization Test ===")
    
    try:
        # Create a test model
        model = create_gpu_optimized_model(
            input_shape=(32, 32, 1),
            conv2d_neurons=[32, 16],
            conv2d_size=[(3, 3), (3, 3)],
            dense_layer_neurons=[64, 32],
            num_conv2d_layers=2,
            num_dense_layers=2,
            max_pooling_size=(2, 2)
        )
        
        # Test batch size optimization
        optimal_batch = get_optimal_batch_size(model, (32, 32, 1), max_batch_size=256)
        print(f"✓ Optimal batch size: {optimal_batch}")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch size optimization error: {e}")
        return False

def test_training_performance():
    """Test training performance with dummy data."""
    print("\n=== Training Performance Test ===")
    
    try:
        # Create model
        model = create_gpu_optimized_model(
            input_shape=(32, 32, 1),
            conv2d_neurons=[32, 16],
            conv2d_size=[(3, 3), (3, 3)],
            dense_layer_neurons=[64, 32],
            num_conv2d_layers=2,
            num_dense_layers=2,
            max_pooling_size=(2, 2)
        )
        
        # Create dummy data
        batch_size = 64
        x_train = np.random.random((batch_size, 32, 32, 1)).astype(np.float32)
        y_train = np.random.random((batch_size, 1)).astype(np.float32)
        
        # Test training
        start_time = time.time()
        history = model.fit(
            x_train, y_train,
            epochs=2,
            batch_size=batch_size,
            verbose=0
        )
        training_time = time.time() - start_time
        
        print(f"✓ Training successful")
        print(f"  Batch size: {batch_size}")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Final loss: {history.history['loss'][-1]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training performance error: {e}")
        return False

def test_memory_monitoring():
    """Test GPU memory monitoring."""
    print("\n=== Memory Monitoring Test ===")
    
    try:
        gpu_info = monitor_gpu_usage()
        print(f"GPU Info: {gpu_info}")
        
        if gpu_info.get('gpu_available', False):
            print("✓ GPU monitoring successful")
            return True
        else:
            print("✗ GPU monitoring failed")
            return False
            
    except Exception as e:
        print(f"✗ Memory monitoring error: {e}")
        return False

def main():
    """Run all tests and provide recommendations."""
    print("HBT Analysis GPU Setup Test")
    print("=" * 40)
    
    # Run tests
    tests = [
        ("GPU Availability", test_gpu_availability),
        ("GPU Configuration", test_gpu_configuration),
        ("Model Creation", test_model_creation),
        ("Batch Size Optimization", test_batch_size_optimization),
        ("Training Performance", test_training_performance),
        ("Memory Monitoring", test_memory_monitoring)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    # Recommendations
    print("\n" + "=" * 40)
    print("RECOMMENDATIONS")
    print("=" * 40)
    
    if results["GPU Availability"]:
        print("✓ GPU is available and ready for use")
        print("  - You can run GPU-optimized scripts")
        print("  - Expected 5-10x speedup over CPU")
    else:
        print("✗ GPU not available")
        print("  - Check CUDA installation: nvidia-smi")
        print("  - Install GPU requirements: pip install -r requirements_gpu.txt")
        print("  - Use CPU fallback: --use_cpu flag")
    
    if results["Model Creation"] and results["Training Performance"]:
        print("✓ Model training is working correctly")
        print("  - Ready for HBT analysis")
        print("  - Use: python scripts/optimization/optimize_hbt_parameters_gpu.py --use_gpu")
    else:
        print("✗ Model training issues detected")
        print("  - Check TensorFlow installation")
        print("  - Verify GPU memory availability")
    
    if results["Batch Size Optimization"]:
        print("✓ Batch size optimization working")
        print("  - Scripts will automatically optimize batch sizes")
    else:
        print("✗ Batch size optimization issues")
        print("  - May need to manually set batch sizes")
    
    print("\nNext steps:")
    if all(results.values()):
        print("1. Run: python scripts/optimization/optimize_hbt_parameters_gpu.py --use_gpu")
        print("2. Monitor with: watch -n 1 nvidia-smi")
        print("3. Check results in data/optimization_results/")
    else:
        print("1. Fix the failing tests above")
        print("2. Re-run this test script")
        print("3. Use CPU fallback if GPU issues persist")

if __name__ == "__main__":
    main()
