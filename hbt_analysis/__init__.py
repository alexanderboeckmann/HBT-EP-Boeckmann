"""
HBT Analysis Package

Core analysis classes for HBT data processing and optimization.
"""

from __future__ import annotations

# NOTE: Keep this module lightweight.
#
# Historically this file eagerly imported the full analysis stack, which meant *any*
# `import hbt_analysis` (including `import hbt_analysis.utils...`) would import
# everything and execute any import-time side effects.
#
# We now use lazy attribute loading so:
# - basic imports are fast and reliable
# - downstream tools can import utilities without pulling in heavy native stacks

from importlib import import_module
from typing import Any, Dict, Tuple

__all__ = [
    "HBTAnalysisBase",
    "HBTAnalysisTrimmed",
    "HBTAnalysisUntrimmed",
    "HBTAnalysisTrimmedCrossover",
    "HBTAnalysisUntrimmedCrossover",
]

_LAZY_IMPORTS: Dict[str, Tuple[str, str]] = {
    "HBTAnalysisBase": ("hbt_analysis.core.base", "HBTAnalysisBase"),
    "HBTAnalysisTrimmed": ("hbt_analysis.core.trimmed", "HBTAnalysisTrimmed"),
    "HBTAnalysisUntrimmed": ("hbt_analysis.core.untrimmed", "HBTAnalysisUntrimmed"),
    "HBTAnalysisTrimmedCrossover": ("hbt_analysis.core.crossover", "HBTAnalysisTrimmedCrossover"),
    "HBTAnalysisUntrimmedCrossover": ("hbt_analysis.core.crossover", "HBTAnalysisUntrimmedCrossover"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))
