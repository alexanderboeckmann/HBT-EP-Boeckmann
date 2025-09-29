# GPU Setup Guide for HBT Analysis

This guide will help you prepare your HBT analysis code for GPU execution on the lab server.

## Prerequisites

### 1. Server Requirements
- CUDA-compatible GPU (NVIDIA)
- CUDA Toolkit (version 11.2 or 11.8 recommended)
- cuDNN library
- Python 3.8-3.10
- Sufficient GPU memory (8GB+ recommended)

### 2. Check GPU Availability
```bash
# Check if GPU is available
nvidia-smi

# Check CUDA version
nvcc --version

# Check Python GPU support
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

## Installation Steps

### 1. Install GPU Dependencies
```bash
# Install GPU-optimized requirements
pip install -r requirements_gpu.txt

# Alternative: Install TensorFlow with CUDA support
pip install tensorflow[and-cuda]
```

### 2. Verify Installation
```bash
# Test GPU configuration
python scripts/gpu_utils.py

# Expected output:
# GPU Configuration: {'gpu_available': True, 'device_count': 1, ...}
# Model created successfully: [model summary]
```

## Usage

### 1. Basic GPU Execution
```bash
# Run trimmed analysis with GPU
python notebooks/trimmed_HBT_analysis_gpu.py --state 2 --RESERVED_SHOT 114458 --use_gpu

# Run untrimmed analysis with GPU
python notebooks/untrimmed_HBT_analysis_gpu.py --state 2 --RESERVED_SHOT 114458 --use_gpu
```

### 2. GPU-Optimized Optimization
```bash
# Run genetic algorithm with GPU acceleration
python scripts/optimization/optimize_hbt_parameters_gpu.py --use_gpu

# Specify GPU memory limit (in MB)
python scripts/optimization/optimize_hbt_parameters_gpu.py --use_gpu --gpu_memory_limit 8192

# Force CPU execution
python scripts/optimization/optimize_hbt_parameters_gpu.py --use_cpu
```

### 3. Batch Size Optimization
The GPU scripts automatically optimize batch sizes for your specific GPU. You can also manually specify:
```bash
python notebooks/trimmed_HBT_analysis_gpu.py --state 2 --BATCH_SIZE 128 --use_gpu
```

## Performance Optimizations

### 1. Memory Management
- **Mixed Precision**: Automatically enabled for better GPU utilization
- **Memory Growth**: Allows TensorFlow to allocate memory as needed
- **Batch Size Optimization**: Automatically finds optimal batch size for your GPU

### 2. Expected Performance Gains
- **Training Speed**: 5-10x faster than CPU
- **Batch Processing**: Larger batch sizes (64-256 vs 32)
- **Memory Efficiency**: Better utilization of GPU memory

### 3. Monitoring GPU Usage
```bash
# Monitor GPU usage during execution
watch -n 1 nvidia-smi

# Check memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## Troubleshooting

### 1. Common Issues

**Issue**: `CUDA out of memory`
```bash
# Solution: Reduce batch size or GPU memory limit
python scripts/optimization/optimize_hbt_parameters_gpu.py --use_gpu --gpu_memory_limit 4096
```

**Issue**: `No GPU devices found`
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall TensorFlow with CUDA support
pip uninstall tensorflow
pip install tensorflow[and-cuda]
```

**Issue**: `TensorFlow not using GPU`
```bash
# Check TensorFlow GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TF_GPU_THREAD_MODE=gpu_private
```

### 2. Fallback to CPU
If GPU issues persist, the scripts automatically fall back to CPU:
```bash
# Force CPU execution
python scripts/optimization/optimize_hbt_parameters_gpu.py --use_cpu
```

## File Structure

```
HBT-EP-Boeckmann/
├── scripts/
│   ├── gpu_utils.py                    # GPU utilities and configuration
│   └── optimization/
│       └── optimize_hbt_parameters_gpu.py  # GPU-optimized genetic algorithm
├── notebooks/
│   ├── trimmed_HBT_analysis_gpu.py     # GPU-optimized trimmed analysis
│   └── untrimmed_HBT_analysis_gpu.py   # GPU-optimized untrimmed analysis
├── requirements_gpu.txt                # GPU-specific dependencies
└── GPU_SETUP_GUIDE.md                 # This guide
```

## Best Practices

### 1. Resource Management
- Monitor GPU memory usage during execution
- Use appropriate batch sizes for your GPU memory
- Close other GPU-intensive applications

### 2. Code Modifications
- The GPU scripts are drop-in replacements for CPU versions
- All original parameters are supported
- Automatic optimization of batch sizes and memory usage

### 3. Performance Tuning
- Start with default settings and let the script optimize
- Increase batch size if you have more GPU memory
- Use mixed precision for better performance

## Expected Results

With GPU acceleration, you should see:
- **Faster training**: 5-10x speedup for model training
- **Larger batches**: 2-4x larger batch sizes
- **Better convergence**: More stable training with larger batches
- **Memory efficiency**: Better utilization of available resources

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify GPU and CUDA installation
3. Test with the provided GPU utilities
4. Fall back to CPU execution if needed

The GPU-optimized scripts maintain full compatibility with your existing workflow while providing significant performance improvements.
