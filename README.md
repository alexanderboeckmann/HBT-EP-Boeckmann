# HBT-EP-Boeckmann

A repository for HBT analysis using machine learning (CPU-only), focused on parameter optimization workflows.

## Features

- **Parameter Optimization**: Genetic algorithm-based hyperparameter optimization
- **Parallel Processing**: Multi-core parallel optimization runs
- **Flexible Data Types**: Support for mode amplitude (ma1-ma4) and mode phase (mp1-mp4) analysis
- **Multiple Analysis Modes**: Trimmed and untrimmed data analysis with crossover validation

## Quick Start

### Parameter Optimization (Main Use Case)
```bash
# Standard optimization
python run_optimization.py --data_type ma2 --state 2

# Parallel optimization
python run_optimization.py --data_type ma2 --parallel --num_workers 4

# Custom parameters
python run_optimization.py --data_type ma2 --epochs 20 --generations 200 --population_size 100
```

### Direct Optimization Scripts
```bash
# Standard optimization
python scripts/optimization/optimize_hbt_parameters.py --data_type ma2 --state 2

# Parallel optimization
python scripts/optimization/optimize_hbt_parameters_parallel.py --data_type ma2 --num_workers 4
```

### Analysis Classes (For Custom Workflows)
```python
from hbt_analysis import HBTAnalysisTrimmed, HBTAnalysisUntrimmed

# Create analysis instance
config = {
    'state': 2,
    'selected_data_type': 'ma2',
    'reserved_shot': 114458,
}

# Run analysis
analysis = HBTAnalysisTrimmed(config)
results = analysis.run_analysis()
```

## Data Types

- **Mode Amplitude**: `ma1`, `ma2`, `ma3`, `ma4` (default: `ma2`)
- **Mode Phase**: `mp1`, `mp2`, `mp3`, `mp4`

## Project Structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed information about the codebase organization.

## Documentation

- [Analysis Package](hbt_analysis/README.md) - New package system documentation

## Installation

```bash
# Install dependencies (CPU-only)
pip install -r requirements.txt

# Install the package in development mode (optional, for imports like `from hbt_analysis import ...`)
pip install -e .
```

## Performance Notes (Local Optimization)

Optimization speed is dominated by repeatedly reading thousands of TIFF frames from disk.
This repo now includes a simple on-disk cache of center-cropped 32x32 frames under `data/cache/`
to accelerate repeated genetic algorithm evaluations. The first run will populate the cache; subsequent
runs should be significantly faster.

## Project Structure

```
scripts/optimization/          # Main optimization scripts
├── optimize_hbt_parameters.py        # Standard optimization
├── optimize_hbt_parameters_parallel.py # Parallel optimization
└── README_OPTIMIZATION.md            # Performance notes / usage

hbt_analysis/                  # Core analysis classes
├── core/                     # Analysis implementations
│   ├── base.py              # Base analysis class
│   ├── trimmed.py           # Trimmed data analysis
│   ├── untrimmed.py         # Untrimmed data analysis
│   └── crossover.py         # Crossover validation
└── utils/                   # Utility functions

run_optimization.py          # Main optimization runner
``` 
