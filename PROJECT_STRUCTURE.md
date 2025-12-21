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
├── archive/notebooks/             # Legacy analysis notebooks (archived)
│   ├── trimmed_HBT_analysis.py       # Original trimmed analysis
│   ├── untrimmed_HBT_analysis.py     # Original untrimmed analysis
│
├── scripts/                       # All scripts organized by function
│   ├── __init__.py
│   ├── optimization/              # Genetic algorithm optimization
│   │   ├── optimize_hbt_parameters.py           # Original CPU version
│   │   ├── optimize_hbt_parameters_parallel.py  # Parallel CPU version
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
│   └── (currently unused)
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
│   └── PROJECT_STRUCTURE.md       # This file
│
└── README.md                      # Main project README
```

## Key Improvements

### 1. Logical Organization
- **`scripts/optimization/`** - All optimization algorithms
- **`docs/`** - Documentation (currently minimal)

### 2. Clear Separation of Concerns
- **Documentation** is centralized
- **Original scripts** remain unchanged

### 3. Easy Navigation
- Related files are grouped together
- Clear naming conventions
- Logical directory structure
- Easy to find what you need

## Usage Examples

### Optimization (CPU)
```bash
# Standard genetic algorithm optimization
python scripts/optimization/optimize_hbt_parameters.py --data_type ma2 --state 2

# Parallel CPU optimization
python scripts/optimization/optimize_hbt_parameters_parallel.py --data_type ma2 --max_workers 4
```

## File Descriptions

### Optimization Scripts (`scripts/optimization/`)
- **`optimize_hbt_parameters.py`** - Original CPU genetic algorithm
- **`optimize_hbt_parameters_parallel.py`** - Parallel CPU version

## Migration Benefits

This repository is now maintained as a CPU-only, local workflow. Older server/GPU deployment materials have been removed to keep the project clean and cross-platform.

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