# Server Deployment Guide

This guide ensures your HBT analysis code will work properly on the lab server.

## Pre-Deployment Checklist (Run on Mac)

### 1. Run Pre-Server Check
```bash
# Validate your code before uploading
python scripts/pre_server_check.py
```

This will check:
- ✅ All required files exist
- ✅ Python imports work correctly
- ✅ Script syntax is valid
- ✅ Data directories are present
- ✅ GPU utilities are compatible
- ✅ Requirements file is complete
- ✅ No hardcoded paths

### 2. Fix Any Issues
If the pre-server check finds issues, fix them before uploading:
- Install missing dependencies: `pip install -r requirements_gpu.txt`
- Fix syntax errors in scripts
- Ensure all data directories exist

## Server Deployment Steps

### 1. Upload Your Code
```bash
# Upload your entire project to the server
scp -r HBT-EP-Boeckmann/ username@server:/path/to/destination/
```

### 2. Connect to Server
```bash
ssh username@server
cd /path/to/HBT-EP-Boeckmann/
```

### 3. Run Deployment Script
```bash
# Make deployment script executable
chmod +x deploy_to_server.sh

# Run deployment
bash deploy_to_server.sh
```

### 4. Validate Server Setup
```bash
# Run comprehensive server validation
python scripts/server_validation.py
```

This will check:
- ✅ System requirements (Python version, memory)
- ✅ GPU availability (nvidia-smi, CUDA)
- ✅ Python dependencies
- ✅ TensorFlow GPU support
- ✅ GPU utilities functionality
- ✅ Data access permissions
- ✅ Script execution
- ✅ Performance benchmarks

### 5. Test GPU Setup
```bash
# Test GPU configuration
python test_gpu_setup.py
```

## Expected Results

### Successful Deployment
If everything works, you should see:
```
🎉 ALL TESTS PASSED! Your setup is ready for HBT analysis.

Next steps:
1. Run: python scripts/optimization/optimize_hbt_parameters_gpu.py --use_gpu
2. Monitor with: watch -n 1 nvidia-smi
3. Check results in data/optimization_results/
```

### Performance Expectations
- **GPU Training**: 5-10x faster than CPU
- **Batch Sizes**: 64-256 (vs 32 on CPU)
- **Memory Usage**: Optimized with mixed precision
- **Convergence**: More stable with larger batches

## Troubleshooting

### Common Issues and Solutions

#### 1. GPU Not Detected
```bash
# Check GPU availability
nvidia-smi

# Check CUDA installation
nvcc --version

# Reinstall TensorFlow with CUDA
pip uninstall tensorflow
pip install tensorflow[and-cuda]
```

#### 2. CUDA Out of Memory
```bash
# Use smaller batch sizes
python scripts/optimization/optimize_hbt_parameters_gpu.py --use_gpu --gpu_memory_limit 4096

# Or force CPU execution
python scripts/optimization/optimize_hbt_parameters_gpu.py --use_cpu
```

#### 3. Import Errors
```bash
# Install all requirements
pip install -r requirements_gpu.txt

# Check Python version
python --version  # Should be 3.8+

# Check virtual environment
which python
```

#### 4. Permission Errors
```bash
# Fix data directory permissions
chmod -R 755 data/
chmod -R 755 scripts/
chmod -R 755 notebooks/

# Check write permissions
touch data/test_write.tmp && rm data/test_write.tmp
```

#### 5. Script Execution Errors
```bash
# Check script syntax
python -m py_compile scripts/gpu_utils.py

# Test individual components
python -c "from scripts.gpu_utils import configure_gpu_memory; print(configure_gpu_memory())"
```

## Monitoring and Maintenance

### 1. Monitor GPU Usage
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### 2. Check Logs
```bash
# View optimization logs
tail -f data/optimization_results/run_gpu_*/optimization.log

# Check for errors
grep -i error data/optimization_results/run_gpu_*/*.log
```

### 3. Performance Tuning
```bash
# Test different batch sizes
python test_gpu_setup.py

# Run performance benchmark
python scripts/server_validation.py
```

## File Structure Verification

After deployment, verify this structure exists on the server:
```
HBT-EP-Boeckmann/
├── scripts/
│   ├── gpu_utils.py
│   ├── pre_server_check.py
│   ├── server_validation.py
│   └── optimization/
│       └── optimize_hbt_parameters_gpu.py
├── notebooks/
│   ├── trimmed_HBT_analysis.py
│   └── untrimmed_HBT_analysis.py
├── data/
│   ├── shots/new/
│   ├── shots/old/
│   ├── predictions/
│   └── optimization_results/
├── requirements_gpu.txt
├── GPU_SETUP_GUIDE.md
├── test_gpu_setup.py
└── deploy_to_server.sh
```

## Success Criteria

Your deployment is successful when:
1. ✅ Pre-server check passes on Mac
2. ✅ Server validation passes on server
3. ✅ GPU test passes on server
4. ✅ Can run GPU optimization script
5. ✅ Can monitor GPU usage with nvidia-smi
6. ✅ Results are saved to data/optimization_results/

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Run the validation scripts for detailed error messages
3. Check server logs and GPU status
4. Use CPU fallback if GPU issues persist
5. Contact system administrator for server-specific issues

## Quick Start Commands

```bash
# 1. Pre-deployment check (on Mac)
python scripts/pre_server_check.py

# 2. Upload to server
scp -r HBT-EP-Boeckmann/ username@server:/path/to/destination/

# 3. Deploy on server
ssh username@server
cd /path/to/HBT-EP-Boeckmann/
bash deploy_to_server.sh

# 4. Validate setup
python scripts/server_validation.py

# 5. Start GPU optimization
python scripts/optimization/optimize_hbt_parameters_gpu.py --use_gpu
```

This comprehensive approach ensures your code will work reliably on the server with full GPU acceleration!
