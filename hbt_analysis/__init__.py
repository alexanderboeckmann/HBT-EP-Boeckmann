"""
HBT Analysis Package

Core analysis classes for HBT data processing and optimization.
"""

# Import main analysis classes for easy access
from .core.base import HBTAnalysisBase
from .core.trimmed import HBTAnalysisTrimmed
from .core.untrimmed import HBTAnalysisUntrimmed
from .core.crossover import HBTAnalysisTrimmedCrossover, HBTAnalysisUntrimmedCrossover

__all__ = [
    'HBTAnalysisBase',
    'HBTAnalysisTrimmed', 
    'HBTAnalysisUntrimmed',
    'HBTAnalysisTrimmedCrossover',
    'HBTAnalysisUntrimmedCrossover'
]
