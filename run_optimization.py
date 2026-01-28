#!/usr/bin/env python
"""
HBT Parameter Optimization Runner

Main script for running HBT parameter optimization with various configurations.
This is the primary entry point for the optimization workflow.

Usage:
    python run_optimization.py --data_type ma2 --parallel --num_workers 4
    python run_optimization.py --data_type ma2 --state 2 --epochs 20
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is importable so we can import `scripts.*` as a normal package.
sys.path.insert(0, str(Path(__file__).parent))

"""
Note on imports:
We import optimization entrypoints lazily inside `main()` so that the parallel optimizer
doesn't run import-time side effects (logging/resource probing) when the user selected
the standard optimizer.
"""


def create_parser():
    """Create argument parser for optimization runner."""
    parser = argparse.ArgumentParser(description='HBT Parameter Optimization Runner')
    
    # Core parameters
    parser.add_argument('--data_type', type=str, default='ma2',
                       help='Data type: ma1-ma4 (mode amplitude 1-4), mp1-mp4 (mode phase 1-4), '
                            'mps1-mps4 (sin(phase) for modes 1-4), mpc1-mpc4 (cos(phase) for modes 1-4) '
                            'mp_sc1-mp_sc4 (combined sin+cos -> phase MAPE fitness) '
                            '(default: ma2)')
    parser.add_argument('--state', type=int, default=2, 
                       help='State number (1, 2, or 3) (default: 2)')
    parser.add_argument(
        '--states',
        type=str,
        default='',
        help="Comma-separated list of states to include (e.g. '1,2,3'). "
             "If provided, overrides --state."
    )
    # Epochs are treated as a maximum; training typically ends earlier via early stopping.
    parser.add_argument('--epochs', type=int, default=35,
                       help='Max epochs for each optimization run (default: 35; early stopping usually ends earlier)')
    parser.add_argument(
        '--notebook_types',
        type=str,
        choices=['trimmed', 'untrimmed', 'both'],
        default=None,
        help="Notebook types to search: trimmed, untrimmed, or both. "
             "If omitted, defaults to untrimmed for mp_sc* and both otherwise."
    )
    
    # Optimization method
    # Default to parallel since it's typically much faster; allow opting out with --no-parallel.
    parser.add_argument(
        '--parallel',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use parallel optimization (default: enabled; pass --no-parallel to disable)'
    )
    # Default to auto worker selection in the parallel optimizer, but never below 2 unless
    # the user explicitly requests it. (The parallel optimizer implements the min=2 clamp.)
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of parallel workers (0 = auto; default: 0)')
    
    # Genetic algorithm parameters
    parser.add_argument('--population_size', type=int, default=50, 
                       help='Population size for genetic algorithm (default: 50)')
    parser.add_argument('--generations', type=int, default=5,
                       help='Number of generations for genetic algorithm (default: 5)')
    parser.add_argument('--mutation_rate', type=float, default=0.1, 
                       help='Mutation rate for genetic algorithm (default: 0.1)')
    parser.add_argument('--crossover_rate', type=float, default=0.8, 
                       help='Crossover rate for genetic algorithm (default: 0.8)')
    
    # Output
    parser.add_argument('--output_dir', type=str, 
                       help='Output directory for optimization results')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose output')
    
    return parser


def main():
    """Main optimization runner."""
    parser = create_parser()
    args = parser.parse_args()
    
    print(f"Starting HBT parameter optimization...")
    print(f"Data type: {args.data_type}")
    print(f"States: {args.states if args.states else args.state}")
    print(f"Epochs: {args.epochs}")
    
    # Prepare arguments for the specific optimization script
    optimization_args = [
        '--data_type', args.data_type,
        '--state', str(args.state),
        '--epochs', str(args.epochs),
        '--population_size', str(args.population_size),
        '--generations', str(args.generations),
        '--mutation_rate', str(args.mutation_rate),
        '--crossover_rate', str(args.crossover_rate),
    ]

    if args.states:
        optimization_args.extend(['--states', args.states])

    if args.notebook_types:
        optimization_args.extend(['--notebook_types', args.notebook_types])
    
    if args.output_dir:
        optimization_args.extend(['--output_dir', args.output_dir])
    
    if args.verbose:
        optimization_args.append('--verbose')
    
    # Choose optimization method
    if args.parallel:
        print("Using parallel optimization...")
        # Only force a specific worker count when the user explicitly requests it.
        if args.num_workers and args.num_workers > 0:
            optimization_args.extend(['--max_workers', str(args.num_workers)])
        sys.argv = ['optimize_hbt_parameters_parallel.py'] + optimization_args
        import scripts.optimization.optimize_hbt_parameters_parallel as optimize_parallel
        optimize_parallel.main()
    else:
        print("Using standard optimization...")
        sys.argv = ['optimize_hbt_parameters.py'] + optimization_args
        import scripts.optimization.optimize_hbt_parameters as optimize_std
        optimize_std.main()


if __name__ == "__main__":
    main()
