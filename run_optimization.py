#!/usr/bin/env python
"""
HBT Parameter Optimization Runner

Main script for running HBT parameter optimization with various configurations.
This is the primary entry point for the optimization workflow.

Usage:
    python run_optimization.py --data_type ma2 --use_gpu
    python run_optimization.py --data_type ma2 --parallel --num_workers 4
    python run_optimization.py --data_type ma2 --state 2 --epochs 20
"""

import argparse
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / 'scripts'))

# Import the optimization modules
import optimization.optimize_hbt_parameters as optimize_std
import optimization.optimize_hbt_parameters_gpu as optimize_gpu
import optimization.optimize_hbt_parameters_parallel as optimize_parallel


def create_parser():
    """Create argument parser for optimization runner."""
    parser = argparse.ArgumentParser(description='HBT Parameter Optimization Runner')
    
    # Core parameters
    parser.add_argument('--data_type', type=str, default='ma2', 
                       help='Data type: ma1-ma4 (mode amplitude 1-4) or mp1-mp4 (mode phase 1-4) (default: ma2)')
    parser.add_argument('--state', type=int, default=2, 
                       help='State number (1, 2, or 3) (default: 2)')
    parser.add_argument('--epochs', type=int, default=15, 
                       help='Number of epochs for each optimization run (default: 15)')
    
    # Optimization method
    parser.add_argument('--use_gpu', action='store_true', 
                       help='Use GPU-optimized optimization')
    parser.add_argument('--parallel', action='store_true', 
                       help='Use parallel optimization')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of parallel workers (default: 4)')
    
    # Genetic algorithm parameters
    parser.add_argument('--population_size', type=int, default=50, 
                       help='Population size for genetic algorithm (default: 50)')
    parser.add_argument('--generations', type=int, default=100, 
                       help='Number of generations for genetic algorithm (default: 100)')
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
    print(f"State: {args.state}")
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
    
    if args.output_dir:
        optimization_args.extend(['--output_dir', args.output_dir])
    
    if args.verbose:
        optimization_args.append('--verbose')
    
    # Choose optimization method
    if args.parallel:
        print("Using parallel optimization...")
        optimization_args.extend(['--max_workers', str(args.num_workers)])
        sys.argv = ['optimize_hbt_parameters_parallel.py'] + optimization_args
        optimize_parallel.main()
    elif args.use_gpu:
        print("Using GPU-optimized optimization...")
        sys.argv = ['optimize_hbt_parameters_gpu.py'] + optimization_args
        optimize_gpu.main()
    else:
        print("Using standard optimization...")
        sys.argv = ['optimize_hbt_parameters.py'] + optimization_args
        import optimization.optimize_hbt_parameters as opt
        opt.main()


if __name__ == "__main__":
    main()
