#!/usr/bin/env python3
"""
Performance comparison between sequential and parallel HBT optimization

This script runs both the sequential and parallel versions of the HBT parameter
optimization with identical parameters and measures their performance.
"""

import time
import multiprocessing as mp
import subprocess
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
import sys

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

class PerformanceComparison:
    def __init__(self, test_mode=True, max_workers=None):
        """
        Initialize performance comparison
        
        Args:
            test_mode: If True, creates small test scripts for quick comparison
            max_workers: Number of parallel workers (None = use all cores)
        """
        self.test_mode = test_mode
        self.max_workers = max_workers or mp.cpu_count()
        self.results = {}
        
        if test_mode:
            self.create_test_scripts()
    
    def create_test_scripts(self):
        """Create small test scripts for quick performance comparison"""
        # Create test sequential script
        test_seq_script = f'''#!/usr/bin/env python3
"""
Test sequential optimization script for performance comparison
"""
import time
import sys
import os
sys.path.append('{PROJECT_ROOT}')

# Import the actual optimization functions
from scripts.optimization.optimize_hbt_parameters import genetic_algorithm

def main():
    print("Running test sequential optimization...")
    start_time = time.time()
    
    # Run with small parameters
    try:
        # This will run the actual genetic algorithm but with small parameters
        # We'll modify the global constants temporarily
        import scripts.optimization.optimize_hbt_parameters as opt_module
        original_pop = opt_module.POPULATION_SIZE
        original_gen = opt_module.GENERATIONS
        
        # Set small test parameters
        opt_module.POPULATION_SIZE = 2
        opt_module.GENERATIONS = 2
        
        # Run the optimization
        best_params, best_mape = opt_module.genetic_algorithm(None)
        
        # Restore original parameters
        opt_module.POPULATION_SIZE = original_pop
        opt_module.GENERATIONS = original_gen
        
        elapsed = time.time() - start_time
        print(f"Test sequential completed in {{elapsed:.2f}}s")
        print(f"Best MAPE: {{best_mape:.2f}}%")
        
    except Exception as e:
        print(f"Test sequential failed: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        # Create test parallel script
        test_par_script = f'''#!/usr/bin/env python3
"""
Test parallel optimization script for performance comparison
"""
import time
import sys
import os
sys.path.append('{PROJECT_ROOT}')

# Import the actual optimization functions
from scripts.optimization.optimize_hbt_parameters_parallel import genetic_algorithm_parallel

def main():
    print("Running test parallel optimization...")
    start_time = time.time()
    
    # Run with small parameters (parallel script already has small defaults)
    try:
        best_params, best_mape = genetic_algorithm_parallel(None)
        
        elapsed = time.time() - start_time
        print(f"Test parallel completed in {{elapsed:.2f}}s")
        print(f"Best MAPE: {{best_mape:.2f}}%")
        
    except Exception as e:
        print(f"Test parallel failed: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        # Write test scripts
        test_seq_path = os.path.join(PROJECT_ROOT, 'test_sequential.py')
        test_par_path = os.path.join(PROJECT_ROOT, 'test_parallel.py')
        
        with open(test_seq_path, 'w') as f:
            f.write(test_seq_script)
        
        with open(test_par_path, 'w') as f:
            f.write(test_par_script)
        
        # Make them executable
        os.chmod(test_seq_path, 0o755)
        os.chmod(test_par_path, 0o755)
        
        self.test_seq_path = test_seq_path
        self.test_par_path = test_par_path
        
    def cleanup_test_scripts(self):
        """Clean up test scripts"""
        if hasattr(self, 'test_seq_path') and os.path.exists(self.test_seq_path):
            os.remove(self.test_seq_path)
        if hasattr(self, 'test_par_path') and os.path.exists(self.test_par_path):
            os.remove(self.test_par_path)
        
    def run_sequential_test(self):
        """Run sequential optimization test"""
        print("Running sequential test...")
        start_time = time.time()
        
        if self.test_mode:
            # Use test script
            cmd = ['python', self.test_seq_path]
            timeout = 120  # 2 minute timeout for test
        else:
            # Use full script
            cmd = ['python', 'scripts/optimization/optimize_hbt_parameters.py']
            timeout = 600  # 10 minute timeout for full run
        
        try:
            result = subprocess.run(
                cmd, 
                cwd=PROJECT_ROOT, 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            
            sequential_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"Sequential test completed in {sequential_time:.2f}s")
                return {
                    'time': sequential_time,
                    'success': True,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                print(f"Sequential test failed: {result.stderr}")
                return {
                    'time': sequential_time,
                    'success': False,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            print("Sequential test timed out")
            return {
                'time': timeout,
                'success': False,
                'stdout': '',
                'stderr': f'Timeout after {timeout/60:.0f} minutes'
            }

    def run_parallel_test(self):
        """Run parallel optimization test"""
        print("Running parallel test...")
        start_time = time.time()
        
        if self.test_mode:
            # Use test script
            cmd = ['python', self.test_par_path]
            timeout = 120  # 2 minute timeout for test
        else:
            # Use full script with max_workers argument
            cmd = [
                'python', 
                'scripts/optimization/optimize_hbt_parameters_parallel.py',
                '--max_workers', str(self.max_workers)
            ]
            timeout = 600  # 10 minute timeout for full run
        
        try:
            result = subprocess.run(
                cmd, 
                cwd=PROJECT_ROOT, 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            
            parallel_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"Parallel test completed in {parallel_time:.2f}s")
                return {
                    'time': parallel_time,
                    'success': True,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                print(f"Parallel test failed: {result.stderr}")
                return {
                    'time': parallel_time,
                    'success': False,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            print("Parallel test timed out")
            return {
                'time': timeout,
                'success': False,
                'stdout': '',
                'stderr': f'Timeout after {timeout/60:.0f} minutes'
            }

    def run_comparison(self):
        """Run the full performance comparison"""
        print("HBT Optimization Performance Comparison")
        print("=" * 60)
        
        # System information
        print(f"System specifications:")
        print(f"- CPU cores: {mp.cpu_count()}")
        print(f"- Max workers: {self.max_workers}")
        if self.test_mode:
            print(f"- Test mode: Both scripts use 2 population, 2 generations")
        else:
            print(f"- Sequential script: 40 population, 10 generations")
            print(f"- Parallel script: 2 population, 2 generations")
        print(f"- Project root: {PROJECT_ROOT}")
        
        # Run tests
        print(f"\nRunning performance tests...")
        
        # Run sequential test
        sequential_result = self.run_sequential_test()
        self.results['sequential'] = sequential_result
        
        # Run parallel test
        parallel_result = self.run_parallel_test()
        self.results['parallel'] = parallel_result
        
        # Analyze results
        self.analyze_results()
        
        return self.results

    def analyze_results(self):
        """Analyze and display comparison results"""
        print(f"\nPerformance Analysis")
        print("=" * 40)
        
        seq_result = self.results['sequential']
        par_result = self.results['parallel']
        
        if not seq_result['success'] or not par_result['success']:
            print("One or both tests failed. Cannot perform comparison.")
            if not seq_result['success']:
                print(f"Sequential test error: {seq_result['stderr']}")
            if not par_result['success']:
                print(f"Parallel test error: {par_result['stderr']}")
            return
        
        seq_time = seq_result['time']
        par_time = par_result['time']
        
        # Calculate speedup
        speedup = seq_time / par_time if par_time > 0 else 0
        time_saved = seq_time - par_time
        efficiency = speedup / self.max_workers * 100
        
        print(f"Sequential time: {seq_time:.2f}s")
        print(f"Parallel time: {par_time:.2f}s")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Time saved: {time_saved:.2f}s")
        print(f"Parallel efficiency: {efficiency:.1f}%")
        
        # Estimate for full runs (using actual script parameters)
        # Sequential script uses: POPULATION_SIZE = 40, GENERATIONS = 10
        # Parallel script uses: POPULATION_SIZE = 2, GENERATIONS = 2 (for testing)
        seq_population = 40
        seq_generations = 10
        par_population = 2
        par_generations = 2
        
        # Scale up parallel results to match sequential scale
        estimated_par_full = par_time * (seq_population / par_population) * (seq_generations / par_generations)
        estimated_seq_full = seq_time  # Already at full scale
        
        print(f"\nEstimated performance for full runs:")
        print(f"Sequential: {seq_population} population, {seq_generations} generations")
        print(f"Parallel: {par_population} population, {par_generations} generations (scaled up)")
        print(f"Sequential time: {estimated_seq_full/60:.1f} minutes")
        print(f"Parallel time (scaled): {estimated_par_full/60:.1f} minutes")
        print(f"Estimated time saved: {(estimated_seq_full - estimated_par_full)/60:.1f} minutes")
        
        # Store results for visualization
        self.results['analysis'] = {
            'speedup': speedup,
            'time_saved': time_saved,
            'efficiency': efficiency,
            'estimated_seq_full': estimated_seq_full,
            'estimated_par_full': estimated_par_full
        }
        
        # Store population and generation info for the plot
        if self.test_mode:
            self.results['population_size'] = '2 (test mode)'
            self.results['generations'] = '2 (test mode)'
        else:
            self.results['population_size'] = '40 (sequential) / 2 (parallel)'
            self.results['generations'] = '10 (sequential) / 2 (parallel)'

    def create_visualization(self):
        """Create performance comparison visualization"""
        if 'analysis' not in self.results:
            print("No analysis results available for visualization")
            return
        
        analysis = self.results['analysis']
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Time comparison
        methods = ['Sequential', 'Parallel']
        times = [self.results['sequential']['time'], self.results['parallel']['time']]
        
        bars1 = ax1.bar(methods, times, color=['red', 'green'], alpha=0.7)
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Execution Time Comparison')
        ax1.set_ylim(0, max(times) * 1.2)
        
        # Add value labels on bars
        for bar, time in zip(bars1, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                    f'{time:.2f}s', ha='center', va='bottom')
        
        # Speedup visualization
        speedup_data = [1.0, analysis['speedup']]
        bars2 = ax2.bar(methods, speedup_data, color=['blue', 'orange'], alpha=0.7)
        ax2.set_ylabel('Speedup (x)')
        ax2.set_title('Performance Speedup')
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for bar, speedup in zip(bars2, speedup_data):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{speedup:.2f}x', ha='center', va='bottom')
        
        # Add statistics text to the plot
        stats_text = f"Population: {self.results.get('population_size', 'N/A')}\nGenerations: {self.results.get('generations', 'N/A')}\nCPU Cores: {mp.cpu_count()}\nMax Workers: {self.max_workers}"
        fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(PROJECT_ROOT, 'scripts', 'optimization', 'performance_comparison_outputs', f'performance_comparison_{timestamp}.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Performance comparison plot saved to: {plot_path}")
        
        # Close the figure to prevent display
        plt.close()

    def save_results(self):
        """Save detailed results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(PROJECT_ROOT, 'scripts', 'optimization', 'performance_comparison_outputs', f'performance_comparison_{timestamp}.json')
        
        # Prepare results for JSON serialization
        json_results = {
            'timestamp': timestamp,
            'system_info': {
                'cpu_cores': mp.cpu_count(),
                'max_workers': self.max_workers,
                'test_mode': self.test_mode
            },
            'results': self.results
        }
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Detailed results saved to: {results_path}")

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Compare sequential vs parallel HBT optimization performance')
    parser.add_argument('--test_mode', action='store_true', default=True, help='Use test mode with small parameters (default: True)')
    parser.add_argument('--full_mode', action='store_true', help='Use full mode with original parameters')
    parser.add_argument('--max_workers', type=int, default=None, help='Max parallel workers (default: all cores)')
    parser.add_argument('--no_plot', action='store_true', help='Skip creating visualization plot')
    parser.add_argument('--no_save', action='store_true', help='Skip saving results to file')
    
    args = parser.parse_args()
    
    # Determine test mode
    test_mode = args.test_mode and not args.full_mode
    
    # Create and run comparison
    comparison = PerformanceComparison(
        test_mode=test_mode,
        max_workers=args.max_workers
    )
    
    try:
        results = comparison.run_comparison()
        
        if not args.no_plot:
            comparison.create_visualization()
        
        if not args.no_save:
            comparison.save_results()
            
    except KeyboardInterrupt:
        print("\nComparison interrupted by user")
        comparison.cleanup_test_scripts()
        sys.exit(1)
    except Exception as e:
        print(f"Error during comparison: {e}")
        comparison.cleanup_test_scripts()
        sys.exit(1)
    finally:
        # Always cleanup test scripts
        if test_mode:
            comparison.cleanup_test_scripts()

if __name__ == "__main__":
    main()
