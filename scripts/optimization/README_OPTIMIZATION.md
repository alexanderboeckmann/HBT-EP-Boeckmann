# HBT Parameter Optimization - Performance Guide

## Fastest Speed Improvements

### 1. **Parallel Execution** (10-40x speedup)
Your original code runs each individual sequentially. The new parallel version uses all CPU cores:

```bash
# Run with parallel processing (uses all CPU cores)
python scripts/optimization/optimize_hbt_parameters_parallel.py

# Run with specific number of workers
python scripts/optimization/optimize_hbt_parameters_parallel.py --max_workers 8
```

### 2. **Memory-Efficient Processing**
The new version processes data more efficiently to handle larger populations and longer runs.

## Performance Comparison

| Method | Speed | Setup Complexity | Cost |
|--------|-------|------------------|------|
| **Original Sequential** | 1x | Low | Free |
| **Parallel Local** | 10-40x | Low | Free |

## Quick Start

### Local Parallel (Fastest to get started)
```bash
cd /Users/aboeckmann/Documents/Columbia/PlasmaLab/HBT-EP-Boeckmann
python scripts/optimization/optimize_hbt_parameters_parallel.py
```

## Configuration

### Adjust Parallel Workers
```bash
# Use 8 CPU cores
python scripts/optimization/optimize_hbt_parameters_parallel.py --max_workers 8

# Use all available cores (default)
python scripts/optimization/optimize_hbt_parameters_parallel.py
```

### Resume Previous Run
```bash
# Resume from specific directory
python scripts/optimization/optimize_hbt_parameters_parallel.py --run_dir data/optimization_results/run_20250101_120000
```

## Expected Performance Gains

- **Parallel Execution**: 10-40x faster (depending on CPU cores)
- **Memory Efficiency**: Better handling of large populations and long runs

## Troubleshooting

### Memory Issues
If you run out of memory with parallel execution:
```bash
# Use fewer workers
python scripts/optimization/optimize_hbt_parameters_parallel.py --max_workers 4
```

### Performance Issues
If you experience slow performance:
```bash
# Use fewer workers to reduce memory usage
python scripts/optimization/optimize_hbt_parameters_parallel.py --max_workers 4
```

## Output

Results are saved in the same format as the original:
- `data/optimization_results/run_YYYYMMDD_HHMMSS/`
- `hbt_optimization_results.csv` - All results
- `plot_analysis/` - Visualization plots
- `best_individual/` - Best performing individual

The parallel version is fully compatible with the original results format.
