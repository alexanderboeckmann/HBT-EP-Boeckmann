# HBT-EP-Boeckmann Project Structure

## Organized File Structure

```
HBT-EP-Boeckmann/
├── data/                          # Data directories
│   ├── shots/
│   │   ├── new/                   # New shot data
│   │   └── old/                   # Old shot data
│   ├── predictions/               # Model predictions
│   ├── optimization_results/      # Optimization results
│   └── models/                    # Saved models
│
├── notebooks/                     # Analysis notebooks
│   ├── trimmed_HBT_analysis.py       # Original trimmed analysis
│   ├── untrimmed_HBT_analysis.py     # Original untrimmed analysis
│   └── trimmed_HBT_analysis_gpu.py   # GPU-optimized trimmed analysis
│
├── scripts/                       # All scripts organized by function
│   ├── __init__.py
│   ├── gpu/                       # GPU utilities and optimization
│   │   ├── __init__.py
│   │   └── gpu_utils.py           # Core GPU utilities
│   │
│   ├── optimization/              # Genetic algorithm optimization
│   │   ├── optimize_hbt_parameters.py           # Original CPU version
│   │   ├── optimize_hbt_parameters_parallel.py  # Parallel CPU version
│   │   └── optimize_hbt_parameters_gpu.py       # GPU-optimized version
│   │
│   ├── validation/                # Testing and validation
│   │   ├── __init__.py
│   │   ├── pre_server_check.py    # Pre-deployment validation
│   │   ├── server_validation.py   # Server setup validation
│   │   └── test_gpu_setup.py      # GPU functionality testing
│   │
│   ├── analysis/                  # Data analysis scripts
│   │   ├── compare_hbt_predictions.py
│   │   ├── genetic_manual_comparison.py
│   │   └── missing_results_analysis.py
│   │
│   ├── preprocessing/             # Data preprocessing
│   │   └── basic_ip_trim.py
│   │
│   └── visualization/             # Plotting and visualization
│       ├── find_camera_frame.py
│       └── optimization_plots.py
│
├── docs/                          # Documentation
│   ├── GPU_SETUP_GUIDE.md        # GPU setup instructions
│   ├── SERVER_DEPLOYMENT_GUIDE.md # Server deployment guide
│   └── README_GPU_DEPLOYMENT.md  # GPU deployment summary
│
├── archive/                       # Archived files
│   ├── extra_photos/
│   ├── old_files/
│   └── potential_models/
│
├── outputs/                       # Generated plots and outputs
│   └── *.png
│
├── Configuration Files
│   ├── requirements_gpu.txt       # GPU-specific dependencies
│   ├── deploy_to_server.sh        # Server deployment script
│   └── PROJECT_STRUCTURE.md       # This file
│
└── README.md                      # Main project README
```

## Key Improvements

### 1. Logical Organization
- **`scripts/gpu/`** - All GPU-related utilities
- **`scripts/validation/`** - All testing and validation scripts
- **`scripts/optimization/`** - All optimization algorithms
- **`docs/`** - All documentation in one place

### 2. Clear Separation of Concerns
- **GPU utilities** are isolated and reusable
- **Validation scripts** are grouped together
- **Documentation** is centralized
- **Original scripts** remain unchanged

### 3. Easy Navigation
- Related files are grouped together
- Clear naming conventions
- Logical directory structure
- Easy to find what you need

## Usage Examples

### GPU Optimization
```bash
# Run GPU-optimized genetic algorithm
python scripts/optimization/optimize_hbt_parameters_gpu.py --use_gpu
```

### Validation and Testing
```bash
# Pre-deployment check (on Mac)
python scripts/validation/pre_server_check.py

# Server validation (on server)
python scripts/validation/server_validation.py

# GPU testing
python scripts/validation/test_gpu_setup.py
```

### GPU Utilities
```python
# Import GPU utilities
from scripts.gpu.gpu_utils import configure_gpu_memory, create_gpu_optimized_model
```

## File Descriptions

### GPU Scripts (`scripts/gpu/`)
- **`gpu_utils.py`** - Core GPU configuration, memory management, and model creation

### Validation Scripts (`scripts/validation/`)
- **`pre_server_check.py`** - Validates code before server deployment
- **`server_validation.py`** - Comprehensive server setup validation
- **`test_gpu_setup.py`** - Tests GPU functionality and performance

### Optimization Scripts (`scripts/optimization/`)
- **`optimize_hbt_parameters.py`** - Original CPU genetic algorithm
- **`optimize_hbt_parameters_parallel.py`** - Parallel CPU version
- **`optimize_hbt_parameters_gpu.py`** - GPU-optimized version

### Documentation (`docs/`)
- **`GPU_SETUP_GUIDE.md`** - Complete GPU setup instructions
- **`SERVER_DEPLOYMENT_GUIDE.md`** - Step-by-step deployment guide
- **`README_GPU_DEPLOYMENT.md`** - Quick deployment summary

## Migration Benefits

### Before (Disorganized)
```
scripts/
├── gpu_utils.py
├── pre_server_check.py
├── server_validation.py
├── test_gpu_setup.py
└── optimization/
    └── optimize_hbt_parameters_gpu.py
```

### After (Organized)
```
scripts/
├── gpu/                    # GPU utilities
├── validation/             # Testing scripts
├── optimization/           # Optimization algorithms
├── analysis/               # Analysis scripts
├── preprocessing/          # Data preprocessing
└── visualization/          # Plotting scripts
```

## Data Types Supported

The analysis scripts support both mode amplitude and mode phase data:

### Mode Amplitude (ma1-ma4)
- `ma1`: Mode amplitude 1
- `ma2`: Mode amplitude 2 (default)
- `ma3`: Mode amplitude 3
- `ma4`: Mode amplitude 4

### Mode Phase (mp1-mp4)
- `mp1`: Mode phase 1
- `mp2`: Mode phase 2
- `mp3`: Mode phase 3
- `mp4`: Mode phase 4

### Usage
All analysis scripts accept a `--selected_data_type` parameter:
```bash
# Use mode amplitude 2 (default)
python trimmed_HBT_analysis.py --state 2 --selected_data_type ma2

# Use mode phase 3
python trimmed_HBT_analysis.py --state 2 --selected_data_type mp3
```

## Benefits

1. **Cleaner Structure** - Related files grouped together
2. **Easy Navigation** - Find files quickly by function
3. **Better Maintainability** - Clear separation of concerns
4. **Centralized Docs** - All documentation in one place
5. **Scalable** - Easy to add new features in appropriate directories
6. **Flexible Data Types** - Support for both amplitude and phase analysis

This organization makes your project much more professional and maintainable!