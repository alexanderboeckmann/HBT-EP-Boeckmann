"""
GPU-Optimized HBT Parameter Optimization Script

This script extends the parallel optimization to work efficiently on GPU servers.
Key improvements for GPU execution:

1. GPU memory management and batch size optimization
2. Mixed precision training for better GPU utilization
3. GPU-aware parallel processing (CPU for genetic algorithm, GPU for model training)
4. Automatic fallback to CPU if GPU unavailable
5. Memory monitoring and optimization

Usage:
    # GPU execution (automatically detects and uses available GPUs)
    python optimize_hbt_parameters_gpu.py --use_gpu
    
    # CPU fallback
    python optimize_hbt_parameters_gpu.py --use_cpu
    
    # Specify GPU memory limit
    python optimize_hbt_parameters_gpu.py --gpu_memory_limit 8192
"""

import os
import sys
import time
import uuid
import copy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import random
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from typing import Dict, List, Tuple, Optional, Any
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from scripts.gpu.gpu_utils import (
    configure_gpu_memory, 
    get_optimal_batch_size, 
    monitor_gpu_usage,
    create_gpu_optimized_model,
    setup_gpu_environment
)

# Centralized project root and path utilities
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

def project_path(*parts):
    filtered_parts = [part for part in parts if part is not None]
    return os.path.join(PROJECT_ROOT, *filtered_parts)

# Configuration
POPULATION_SIZE = 8  # Reduced for GPU memory efficiency
GENERATIONS = 5
TOP_PERCENT = 0.125
MUTATION_RATE = 0.1
OUTPUT_DIR = project_path('data', 'optimization_results')
CSV_FILENAME = 'hbt_optimization_results_gpu.csv'
PLOT_FILENAME = 'hbt_optimization_progress_gpu.png'

# GPU-specific settings
GPU_BATCH_SIZE_MULTIPLIER = 2  # Use larger batches on GPU
GPU_MEMORY_LIMIT = None  # Use all available GPU memory by default

# Hyperparameter space (same as original but with GPU-optimized defaults)
PARAM_SPACE = {
    'notebook_type': ['trimmed', 'untrimmed'],
    'state': [1, 2, 3],
    'epochs': list(range(15, 51, 5)),  # Slightly higher epochs for GPU efficiency
    'validation_split': [0.1, 0.15, 0.2, 0.25, 0.3],
    'activation_func': ['relu', 'sigmoid', 'tanh'],
    'loss_func': ['mse', 'mae'],
    'optimizer_func': ['adam', 'sgd', 'rmsprop'],
    'outlier_cutoff': list(range(80, 101, 2)),
    'num_conv2d_layers': [1, 2, 3],
    'num_dense_layers': [1, 2, 3],
    'conv2d_neurons': [16, 32, 64, 128],  # Larger networks for GPU
    'conv2d_size': [(3, 3), (4, 4), (5, 5), (7, 7), (8, 8)],
    'dense_layer_neurons': [32, 64, 128, 256],  # Larger dense layers
    'max_pooling_size': [(2, 2), (3, 3), (4, 4)],
    'early_stopping_patience': [10, 15, 20, 25, 30],
    'early_stopping_min_delta': [0.001, 0.005, 0.01, 0.02, 0.05],
    'batch_size': [32, 64, 128, 256]  # GPU-optimized batch sizes
}

RESERVED_SHOTS = {
    1: [119671],
    2: [114458],
    3: [119671, 114458]
}

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPUOptimizationManager:
    """Manages GPU resources and optimization for parallel execution."""
    
    def __init__(self, use_gpu: bool = True, gpu_memory_limit: Optional[int] = None):
        self.use_gpu = use_gpu
        self.gpu_memory_limit = gpu_memory_limit
        self.gpu_config = None
        self.optimal_batch_sizes = {}
        
        if use_gpu:
            self._setup_gpu()
    
    def _setup_gpu(self):
        """Initialize GPU environment."""
        try:
            self.gpu_config = configure_gpu_memory(
                gpu_memory_limit=self.gpu_memory_limit,
                allow_memory_growth=True
            )
            
            if self.gpu_config.get('gpu_available', False):
                logger.info(f"GPU setup successful: {self.gpu_config['device_count']} GPU(s) available")
                setup_gpu_environment()
            else:
                logger.warning("GPU not available, falling back to CPU")
                self.use_gpu = False
                
        except Exception as e:
            logger.error(f"GPU setup failed: {e}")
            self.use_gpu = False
    
    def get_optimal_batch_size(self, model_config: Dict) -> int:
        """Get optimal batch size for given model configuration."""
        if not self.use_gpu:
            return 32  # Default CPU batch size
        
        # Create a temporary model to test batch size
        try:
            model = create_gpu_optimized_model(
                input_shape=(32, 32, 1),
                conv2d_neurons=model_config.get('conv2d_neurons', [32, 32, 16]),
                conv2d_size=model_config.get('conv2d_size', [(8, 8), (8, 8), (4, 4)]),
                dense_layer_neurons=model_config.get('dense_layer_neurons', [64, 32]),
                num_conv2d_layers=model_config.get('num_conv2d_layers', 3),
                num_dense_layers=model_config.get('num_dense_layers', 2),
                max_pooling_size=model_config.get('max_pooling_size', (2, 2))
            )
            
            optimal_batch = get_optimal_batch_size(
                model, 
                input_shape=(32, 32, 1),
                max_batch_size=512
            )
            
            # Scale up for GPU efficiency
            return min(optimal_batch * GPU_BATCH_SIZE_MULTIPLIER, 512)
            
        except Exception as e:
            logger.warning(f"Could not determine optimal batch size: {e}")
            return 128  # Conservative GPU batch size
    
    def monitor_resources(self) -> Dict[str, Any]:
        """Monitor GPU and system resources."""
        if self.use_gpu:
            gpu_info = monitor_gpu_usage()
            return {
                'gpu_available': gpu_info.get('gpu_available', False),
                'gpu_devices': gpu_info.get('device_count', 0),
                'gpu_config': self.gpu_config
            }
        else:
            return {'gpu_available': False, 'cpu_cores': mp.cpu_count()}

def generate_individual():
    """Generate a random individual with GPU-optimized parameters."""
    state = int(np.random.choice(PARAM_SPACE['state']))
    reserved_shot = int(np.random.choice(RESERVED_SHOTS[state]))
    num_conv2d = int(np.random.choice(PARAM_SPACE['num_conv2d_layers']))
    num_dense = int(np.random.choice(PARAM_SPACE['num_dense_layers']))
    
    individual = {
        'notebook_type': np.random.choice(PARAM_SPACE['notebook_type']),
        'state': state,
        'reserved_shot': reserved_shot,
        'epochs': int(np.random.choice(PARAM_SPACE['epochs'])),
        'validation_split': float(np.random.choice(PARAM_SPACE['validation_split'])),
        'activation_func': np.random.choice(PARAM_SPACE['activation_func']),
        'loss_func': np.random.choice(PARAM_SPACE['loss_func']),
        'optimizer_func': np.random.choice(PARAM_SPACE['optimizer_func']),
        'outlier_cutoff': float(np.random.choice(PARAM_SPACE['outlier_cutoff'])),
        'num_conv2d_layers': num_conv2d,
        'num_dense_layers': num_dense,
        'conv2d_neurons': [int(np.random.choice(PARAM_SPACE['conv2d_neurons'])) for _ in range(num_conv2d)],
        'conv2d_size': [random.choice(PARAM_SPACE['conv2d_size']) for _ in range(num_conv2d)],
        'dense_layer_neurons': [int(np.random.choice(PARAM_SPACE['dense_layer_neurons'])) for _ in range(num_dense)],
        'max_pooling_size': random.choice(PARAM_SPACE['max_pooling_size']),
        'early_stopping_patience': int(np.random.choice(PARAM_SPACE['early_stopping_patience'])),
        'early_stopping_min_delta': float(np.random.choice(PARAM_SPACE['early_stopping_min_delta'])),
        'batch_size': int(np.random.choice(PARAM_SPACE['batch_size'])),
        'id': str(uuid.uuid4())
    }
    
    return individual

def validate_individual(individual):
    """Validate individual parameters."""
    return individual['state'] in RESERVED_SHOTS and individual['reserved_shot'] in RESERVED_SHOTS[individual['state']]

def prepare_parameters(individual, gpu_manager: GPUOptimizationManager):
    """Prepare parameters for script execution with GPU optimization."""
    # Get optimal batch size for this individual
    optimal_batch = gpu_manager.get_optimal_batch_size(individual)
    
    params = {
        'individual_id': individual['id'],
        'state': int(individual['state']),
        'selected_data_type': 'ma2',
        'RESERVED_SHOT': int(individual['reserved_shot']),
        'EPOCH_NUM': int(individual['epochs']),
        'VALIDATION_SPLIT': float(individual['validation_split']),
        'ACTIVATION_FUNC': individual['activation_func'],
        'LOSS_FUNC': individual['loss_func'],
        'OPTIMIZER_FUNC': individual['optimizer_func'],
        'OUTLIER_CUTOFF': float(individual['outlier_cutoff']),
        'NUM_CONV2D_LAYERS': int(individual['num_conv2d_layers']),
        'NUM_DENSE_LAYERS': int(individual['num_dense_layers']),
        'CONV2D_NEURONS': individual['conv2d_neurons'],
        'CONV2D_SIZE': individual['conv2d_size'],
        'DENSE_LAYER_NEURONS': individual['dense_layer_neurons'],
        'MAX_POOLING_SIZE': individual['max_pooling_size'],
        'EARLY_STOPPING_PATIENCE': int(individual['early_stopping_patience']),
        'EARLY_STOPPING_MIN_DELTA': float(individual['early_stopping_min_delta']),
        'BATCH_SIZE': optimal_batch,  # GPU-optimized batch size
        'USE_GPU': gpu_manager.use_gpu,
        'GPU_MEMORY_LIMIT': gpu_manager.gpu_memory_limit
    }
    
    return params

def execute_script_gpu(input_script, parameters, individual_dir, gpu_manager: GPUOptimizationManager):
    """Execute the analysis script with GPU optimization."""
    os.makedirs(individual_dir, exist_ok=True)
    
    try:
        cmd = ['python', input_script]
        for key, value in parameters.items():
            if key != 'individual_id':
                cmd.extend([f'--{key}', str(value)])
        cmd.extend(['--output_dir', individual_dir])
        
        # Set environment variables for GPU execution
        env = os.environ.copy()
        if gpu_manager.use_gpu:
            env['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
            env['TF_GPU_THREAD_MODE'] = 'gpu_private'
        
        result = subprocess.run(
            cmd,
            cwd=individual_dir,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout for GPU training
            env=env
        )
        
        if result.returncode != 0:
            logger.error(f"Script execution failed for ID {parameters.get('individual_id', 'unknown')}: {result.stderr}")
            raise RuntimeError(f"Script execution failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error(f"Script execution timed out for ID {parameters.get('individual_id', 'unknown')}")
        raise
    except Exception as e:
        logger.error(f"Script execution failed for ID {parameters.get('individual_id', 'unknown')}: {e}")
        raise

def evaluate_individual_worker_gpu(args):
    """Worker function for parallel evaluation with GPU support."""
    individual, run_dir, existing_ids, gpu_manager = args
    
    start = time.time()
    if not validate_individual(individual):
        logger.warning(f"Invalid state or shot for {individual['id']}")
        return individual, None
    
    try:
        # Create individual directory
        individual_dir = os.path.join(run_dir, f"individual_{individual['id']}")
        os.makedirs(individual_dir, exist_ok=True)
        
        # Prepare parameters
        params = prepare_parameters(individual, gpu_manager)
        
        # Save parameters
        params_path = os.path.join(individual_dir, "parameters.json")
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=4)
        
        # Execute script
        input_script = project_path('notebooks', f"{individual['notebook_type']}_HBT_analysis.py")
        if not os.path.exists(input_script):
            logger.error(f"Notebook not found: {input_script}")
            return individual, None
        
        execute_script_gpu(input_script, params, individual_dir, gpu_manager)
        
        # Load and validate results
        npy_files = {f: os.path.join(individual_dir, f) for f in os.listdir(individual_dir) if f.endswith('.npy')}
        true_file = None
        pred_file = None
        
        for name, path in npy_files.items():
            if 'true' in name.lower():
                true_file = path
            elif 'pred' in name.lower():
                pred_file = path
        
        if true_file is None or pred_file is None:
            logger.error(f"Missing output files: true={true_file is not None}, pred={pred_file is not None}")
            return individual, None
        
        # Load and validate data
        true_data = np.load(true_file) if os.path.exists(true_file) else None
        pred_data = np.load(pred_file) if os.path.exists(pred_file) else None
        
        if true_data is None or pred_data is None:
            logger.warning(f"Missing result files for {individual['id']}")
            return individual, None
        
        if np.any(np.isnan(true_data)) or np.any(np.isinf(true_data)) or np.any(np.isnan(pred_data)) or np.any(np.isinf(pred_data)):
            logger.warning(f"Invalid data for {individual['id']}")
            return individual, None
        
        # Compute MAPE
        true_flat = true_data.flatten()
        pred_flat = pred_data.flatten()
        min_len = min(len(true_flat), len(pred_flat))
        errors = np.abs(true_flat[:min_len] - pred_flat[:min_len]) / (np.max(np.abs(true_flat[:min_len])) + 1e-8)
        mape = np.mean(errors) * 100
        
        elapsed = time.time() - start
        
        logger.info(f"SUCCESS {individual['id'][:8]}... | {individual['notebook_type']} state{individual['state']} | "
                   f"epochs: {individual['epochs']} | batch: {params['BATCH_SIZE']} | "
                   f"MAPE: {mape:.2f}% | {elapsed:.1f}s")
        
        return individual, mape
        
    except Exception as e:
        logger.error(f"Exception for {individual['id']}: {e}")
        return individual, None

def genetic_algorithm_gpu(use_gpu: bool = True, gpu_memory_limit: Optional[int] = None, run_dir: Optional[str] = None):
    """GPU-optimized genetic algorithm."""
    
    # Initialize GPU manager
    gpu_manager = GPUOptimizationManager(use_gpu=use_gpu, gpu_memory_limit=gpu_memory_limit)
    
    # Monitor resources
    resource_info = gpu_manager.monitor_resources()
    logger.info(f"Resource configuration: {resource_info}")
    
    if run_dir is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(OUTPUT_DIR, f'run_gpu_{timestamp}')
    
    os.makedirs(run_dir, exist_ok=True)
    plot_dir = os.path.join(run_dir, 'plot_analysis')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Initialize population
    population = [generate_individual() for _ in range(POPULATION_SIZE)]
    existing_ids = set()
    
    results_df = pd.DataFrame(columns=[
        'generation', 'individual_id', 'notebook_type', 'state', 'reserved_shot',
        'epochs', 'validation_split', 'activation_func', 'loss_func',
        'optimizer_func', 'outlier_cutoff', 'num_conv2d_layers',
        'num_dense_layers', 'conv2d_neurons', 'conv2d_size',
        'dense_layer_neurons', 'max_pooling_size', 'early_stopping_patience',
        'early_stopping_min_delta', 'batch_size', 'mape'
    ], dtype=object)
    
    best_mape = float('inf')
    best_params = None
    mape_history = []
    
    for generation in range(1, GENERATIONS + 1):
        logger.info(f"\nGeneration {generation}/{GENERATIONS} ({time.strftime('%H:%M:%S')})")
        generation_start = time.time()
        
        # Evaluate population
        fitness = []
        for individual in population:
            individual, mape = evaluate_individual_worker_gpu((individual, run_dir, existing_ids, gpu_manager))
            if mape is not None:
                fitness.append({'individual': individual, 'mape': mape})
                existing_ids.add(individual['id'])
                
                # Update results dataframe
                new_row = pd.DataFrame([{
                    'generation': generation,
                    'individual_id': individual['id'],
                    'notebook_type': individual['notebook_type'],
                    'state': individual['state'],
                    'reserved_shot': individual['reserved_shot'],
                    'epochs': individual['epochs'],
                    'validation_split': individual['validation_split'],
                    'activation_func': individual['activation_func'],
                    'loss_func': individual['loss_func'],
                    'optimizer_func': individual['optimizer_func'],
                    'outlier_cutoff': individual['outlier_cutoff'],
                    'num_conv2d_layers': individual['num_conv2d_layers'],
                    'num_dense_layers': int(individual['num_dense_layers']),
                    'conv2d_neurons': individual['conv2d_neurons'],
                    'conv2d_size': individual['conv2d_size'],
                    'dense_layer_neurons': individual['dense_layer_neurons'],
                    'max_pooling_size': individual['max_pooling_size'],
                    'early_stopping_patience': individual['early_stopping_patience'],
                    'early_stopping_min_delta': individual['early_stopping_min_delta'],
                    'batch_size': individual.get('batch_size', 32),
                    'mape': mape
                }])
                
                if results_df.empty:
                    results_df = new_row
                else:
                    results_df = pd.concat([results_df, new_row], ignore_index=True)
                
                if mape < best_mape:
                    best_mape = mape
                    best_params = copy.deepcopy(individual)
                    logger.info(f"New best MAPE: {best_mape:.4f}%")
        
        if not fitness:
            logger.error("No valid models. Terminating.")
            break
        
        # Save results
        results_df.to_csv(os.path.join(run_dir, CSV_FILENAME), index=False)
        
        avg_mape = np.mean([f['mape'] for f in fitness])
        mape_history.append(avg_mape)
        
        generation_time = time.time() - generation_start
        logger.info(f"Generation {generation} | Average MAPE: {avg_mape:.2f}% | Best MAPE: {best_mape:.2f}% | Time: {generation_time:.1f}s")
        
        # Create next generation (simplified for GPU efficiency)
        if generation < GENERATIONS:
            top_n = max(2, int(POPULATION_SIZE * TOP_PERCENT))
            fitness.sort(key=lambda x: x['mape'])
            top_individuals = [f['individual'] for f in fitness[:min(top_n, len(fitness))]]
            
            new_population = []
            for individual in top_individuals:
                new_individual = copy.deepcopy(individual)
                new_individual['id'] = str(uuid.uuid4())
                new_population.append(new_individual)
            
            while len(new_population) < POPULATION_SIZE:
                new_individual = generate_individual()
                new_population.append(new_individual)
            
            population = new_population
    
    # Create final progress plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(mape_history) + 1), mape_history, '-o')
    plt.xlabel('Generation')
    plt.ylabel('Average MAPE (%)')
    plt.title('GPU Optimization Progress')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, PLOT_FILENAME))
    plt.close()
    
    logger.info(f"\nGPU Optimization Complete!")
    logger.info(f"Best MAPE: {best_mape:.2f}%")
    logger.info(f"Results saved in: {run_dir}")
    
    return best_params, best_mape

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='GPU-Optimized HBT Parameter Optimization')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--use_cpu', action='store_true', help='Force CPU execution')
    parser.add_argument('--gpu_memory_limit', type=int, help='GPU memory limit in MB')
    parser.add_argument('--run_dir', help='Specific run directory to resume')
    
    args = parser.parse_args()
    
    # Determine execution mode
    use_gpu = args.use_gpu or (not args.use_cpu)  # Default to GPU if available
    
    logger.info(f"Starting GPU-optimized optimization (GPU: {use_gpu})")
    
    best_params, best_mape = genetic_algorithm_gpu(
        use_gpu=use_gpu,
        gpu_memory_limit=args.gpu_memory_limit,
        run_dir=args.run_dir
    )
    
    logger.info(f"Optimization completed successfully. Best MAPE: {best_mape:.2f}%")

if __name__ == "__main__":
    main()
