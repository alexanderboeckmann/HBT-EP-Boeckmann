# HBT Analysis Package

Core analysis classes for HBT data processing and optimization.

## Package Structure

```
hbt_analysis/                  # Main package
├── __init__.py               # Package initialization
├── core/                     # Core analysis classes
│   ├── base.py              # HBTAnalysisBase - shared functionality
│   ├── trimmed.py           # HBTAnalysisTrimmed - IP-based cutoff detection
│   ├── untrimmed.py         # HBTAnalysisUntrimmed - raw data processing
│   └── crossover.py         # Crossover validation variants
└── utils/                   # Utility functions
    └── gpu.py               # GPU optimization utilities
```

## Core Analysis Classes

- **`HBTAnalysisBase`** - Base class with shared functionality
- **`HBTAnalysisTrimmed`** - IP-based cutoff detection and outlier removal
- **`HBTAnalysisUntrimmed`** - Raw data with minimal preprocessing
- **`HBTAnalysisTrimmedCrossover`** - Crossover validation for trimmed data
- **`HBTAnalysisUntrimmedCrossover`** - Crossover validation for untrimmed data

## Usage

```python
from hbt_analysis import HBTAnalysisTrimmed, HBTAnalysisUntrimmed

# Create analysis instance
config = {
    'state': 2,
    'selected_data_type': 'ma2',
    'reserved_shot': 114458,
    'epoch_num': 15,
    # ... other parameters
}

# Run trimmed analysis
analysis = HBTAnalysisTrimmed(config)
results = analysis.run_analysis()

# Run untrimmed analysis
analysis = HBTAnalysisUntrimmed(config)
results = analysis.run_analysis()
```

## Key Features

- **Shared Functionality** - Common logic in base class
- **Flexible Configuration** - Configurable model architecture and training parameters
- **Data Processing** - Specialized processing for trimmed vs untrimmed data
- **Crossover Validation** - Cross-state validation capabilities