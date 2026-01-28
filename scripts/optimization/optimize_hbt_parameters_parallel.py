"""
Parallel HBT Parameter Optimization Script

This is an optimized version of the genetic algorithm that runs individuals in parallel
for local execution. Key improvements:

1. Parallel execution of genetic algorithm individuals (10-40x speedup)
2. Memory-efficient data processing
3. Progress tracking and resumable execution

Usage:
    # Local parallel execution (uses all CPU cores)
    python optimize_hbt_parameters_parallel.py
"""

import os
import time
import uuid
import copy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm
import psutil
import sys
import contextlib

# Allow importing the package when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hbt_analysis import HBTAnalysisTrimmed, HBTAnalysisUntrimmed


# Centralized project root and path utilities
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

def project_path(*parts):
    filtered_parts = [part for part in parts if part is not None]
    return os.path.join(PROJECT_ROOT, *filtered_parts)

def pretty_path(path: str) -> str:
    """Render repo-local paths as '<repo_name>/<relative>' for cleaner logs."""
    try:
        p = Path(path).resolve()
        root = Path(PROJECT_ROOT).resolve()
        rel = p.relative_to(root)
        return f"{root.name}/{rel.as_posix()}"
    except Exception:
        return str(path)

# Configuration (defaults, can be overridden by command line)
POPULATION_SIZE = 50
GENERATIONS = 10
EPOCHS = 50  # Default epochs, can be overridden by command line
TOP_PERCENT = 0.125
MUTATION_RATE = 0.1
OUTPUT_DIR = project_path('data', 'optimization_results')
CSV_FILENAME = 'hbt_optimization_results.csv'
PLOT_FILENAME = 'hbt_optimization_progress.png'

# Parallel execution settings
def get_optimal_workers():
    """Calculate optimal number of workers based on system resources"""
    import psutil
    cpu_count = mp.cpu_count()
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    # Conservative approach: limit workers based on available memory
    # Assume each worker needs ~2GB of memory
    memory_limited_workers = max(1, int(available_memory_gb / 2))
    
    # Don't use more than 80% of available cores to avoid system overload
    cpu_limited_workers = max(1, int(cpu_count * 0.8))
    
    # Use the more restrictive limit
    optimal_workers = min(cpu_limited_workers, memory_limited_workers, POPULATION_SIZE)
    
    return optimal_workers

def monitor_memory_usage():
    """Monitor current memory usage"""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024**3)
    return memory_gb


# Hyperparameter space (same as original)
PARAM_SPACE = {
    'notebook_type': ['trimmed', 'untrimmed'],
    'state': [1, 2, 3],
    'epochs': list(range(10, 51, 5)),
    'validation_split': [0.1, 0.15, 0.2, 0.25, 0.3],
    'activation_func': ['relu', 'sigmoid', 'tanh'],
    'loss_func': ['mse', 'mae'],
    'optimizer_func': ['adam', 'sgd', 'rmsprop'],
    'outlier_cutoff': list(range(80, 101, 2)),
    'num_conv2d_layers': [1, 2, 3],
    'num_dense_layers': [1, 2, 3],
    'conv2d_neurons': [8, 16, 32, 64],
    'conv2d_size': [(3, 3), (4, 4), (5, 5), (7, 7), (8, 8)],
    'dense_layer_neurons': [8, 16, 32, 64],
    'max_pooling_size': [(2, 2), (3, 3), (4, 4)],
    'early_stopping_patience': [5, 10, 15, 20, 25, 30],
    'early_stopping_min_delta': [0.001, 0.005, 0.01, 0.02, 0.05]
}

RESERVED_SHOTS = {
    1: [119671],
    2: [114458],
    3: [119671, 114458]
}

# Setup logging
logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = True):
    """
    Configure logging.

    Important: do NOT configure logging at import time because macOS ProcessPoolExecutor uses
    the 'spawn' start method, which re-imports this module in each worker.
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')


@contextlib.contextmanager
def redirect_worker_output(log_path: str):
    """
    Redirect this process's stdout/stderr to a file.

    This prevents multiple parallel workers (and TensorFlow/Keras progress output) from
    interleaving on the console.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # Line-buffered text output for Python prints.
    f = open(log_path, "a", buffering=1)
    # Also redirect the underlying OS-level file descriptors so C/C++ (TF) output is captured.
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    try:
        os.dup2(f.fileno(), stdout_fd)
        os.dup2(f.fileno(), stderr_fd)
        yield
    finally:
        try:
            os.dup2(saved_stdout, stdout_fd)
            os.dup2(saved_stderr, stderr_fd)
        finally:
            os.close(saved_stdout)
            os.close(saved_stderr)
            f.close()


def generate_individual():
    """Generate a random individual (same as original)"""
    state = int(np.random.choice(PARAM_SPACE['state']))
    reserved_shot = int(np.random.choice(RESERVED_SHOTS[state]))
    num_conv2d = int(np.random.choice(PARAM_SPACE['num_conv2d_layers']))
    num_dense = int(np.random.choice(PARAM_SPACE['num_dense_layers']))
    return {
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
        'id': str(uuid.uuid4())
    }

def validate_individual(individual):
    """Validate individual parameters (same as original)"""
    return individual['state'] in RESERVED_SHOTS and individual['reserved_shot'] in RESERVED_SHOTS[individual['state']]

def ensure_unique_id(individual, existing_ids):
    """Ensure individual has a unique ID, generating new one if needed"""
    while individual['id'] in existing_ids:
        logger.info(f"ID conflict for {individual['id']}. Generating new ID.")
        individual['id'] = str(uuid.uuid4())
    existing_ids.add(individual['id'])
    return individual

def construct_paths(individual, run_dir, existing_ids):
    """Construct file paths for an individual (same as original)"""
    individual_dir = os.path.join(run_dir, f"individual_{individual['id']}")
    if individual['id'] in existing_ids and os.path.exists(individual_dir):
        logger.info(f"ID conflict detected for {individual['id']}. Generating new ID.")
        individual['id'] = str(uuid.uuid4())
        # Update existing_ids with the new ID to prevent future conflicts
        existing_ids.add(individual['id'])
        individual_dir = os.path.join(run_dir, f"individual_{individual['id']}")
    os.makedirs(individual_dir, exist_ok=True)
    params_path = os.path.join(individual_dir, "parameters.json")
    return individual_dir, params_path

def prepare_parameters(individual, data_type='ma2'):
    """Prepare parameters for script execution (same as original)"""
    return {
        'individual_id': individual['id'],
        'state': int(individual['state']),
        'selected_data_type': data_type,
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
        'EARLY_STOPPING_MIN_DELTA': float(individual['early_stopping_min_delta'])
    }

def run_analysis_for_individual(individual, data_type: str, output_dir: str):
    """Run analysis in-process (no subprocess/notebook runner)."""
    config = {
        'state': int(individual['state']),
        'selected_data_type': data_type,
        'reserved_shot': int(individual['reserved_shot']),
        'epoch_num': int(individual['epochs']),
        'validation_split': float(individual['validation_split']),
        'activation_func': individual['activation_func'],
        'loss_func': individual['loss_func'],
        'optimizer_func': individual['optimizer_func'],
        'outlier_cutoff': float(individual['outlier_cutoff']),
        'num_conv2d_layers': int(individual['num_conv2d_layers']),
        'num_dense_layers': int(individual['num_dense_layers']),
        'conv2d_neurons': individual['conv2d_neurons'],
        'conv2d_size': individual['conv2d_size'],
        'dense_layer_neurons': individual['dense_layer_neurons'],
        'max_pooling_size': individual['max_pooling_size'],
        'batch_size': 32,
        'early_stopping_patience': int(individual.get('early_stopping_patience', 20)),
        'early_stopping_min_delta': float(individual.get('early_stopping_min_delta', 0.01)),
        # Reduce noisy output in optimization workers.
        'fit_verbose': 0,
        'model_summary': False,
        # Ensure normalization defaults to writing within this individual's output dir.
        'output_dir': output_dir,
    }

    if individual['notebook_type'] == 'trimmed':
        analysis = HBTAnalysisTrimmed(config)
    elif individual['notebook_type'] == 'untrimmed':
        analysis = HBTAnalysisUntrimmed(config)
    else:
        raise ValueError(f"Unknown notebook_type: {individual['notebook_type']}")

    analysis.run_analysis(output_dir=output_dir)

def load_result_arrays(true_path, pred_path):
    """Load result arrays (same as original)"""
    true = np.load(true_path) if os.path.exists(true_path) else None
    pred = np.load(pred_path) if os.path.exists(pred_path) else None
    return true, pred

def validate_result_data(true, pred, individual_id):
    """Validate result data (same as original)"""
    if true is None or pred is None:
        logger.warning(f"Missing result files for {individual_id}")
        return False
    if np.any(np.isnan(true)) or np.any(np.isinf(true)):
        logger.warning(f"Invalid true values for {individual_id}: NaN={np.any(np.isnan(true))}, Inf={np.any(np.isinf(true))}")
        return False
    if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
        logger.warning(f"Invalid pred values for {individual_id}: NaN={np.any(np.isnan(pred))}, Inf={np.any(np.isinf(pred))}")
        return False
    return True

def compute_mape(true, pred):
    """Compute MAPE (same as original)"""
    true = true.flatten()
    pred = pred.flatten()
    min_len = min(len(true), len(pred))
    errors = np.abs(true[:min_len] - pred[:min_len]) / (np.max(np.abs(true[:min_len])) + 1e-8)
    return np.mean(errors) * 100

def evaluate_individual_worker(args):
    """Worker function for parallel evaluation of individuals"""
    individual, run_dir, existing_ids, data_type = args
    
    start = time.time()
    if not validate_individual(individual):
        logger.warning(f"Invalid state or shot for {individual['id']}")
        return individual, None
    
    try:
        individual_dir, params_path = construct_paths(individual, run_dir, existing_ids)
        params = prepare_parameters(individual, data_type)
        
        worker_log = os.path.join(individual_dir, "worker.log")
        with redirect_worker_output(worker_log):
            with open(params_path, 'w') as f:
                json.dump(params, f, indent=4)

            # Keep TF C++ logs down where possible (must be set before TF import to be fully effective,
            # but still helps for some subcomponents).
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

            run_analysis_for_individual(individual, data_type=data_type, output_dir=individual_dir)
        
            # Find result files
            npy_files = {f: os.path.join(individual_dir, f) for f in os.listdir(individual_dir) if f.endswith('.npy')}
            true_file = None
            pred_file = None
            for name, path in npy_files.items():
                if 'true' in name.lower():
                    true_file = path
                elif 'pred' in name.lower():
                    pred_file = path
            
            if true_file is None or pred_file is None:
                raise RuntimeError(f"Missing output files: true={true_file is not None}, pred={pred_file is not None}")
            
            # Validate state consistency
            base_name = os.path.basename(true_file).replace('_true.npy', '')
            parts = base_name.split('_')
            state_idx = parts.index('state') + 1
            state = int(parts[state_idx]) if state_idx < len(parts) else None
            
            if state is None or state != individual['state']:
                raise RuntimeError(f"Mismatched state in {individual_dir}: expected {individual['state']}, found {state}")
            
            true_data, pred_data = load_result_arrays(true_file, pred_file)
            if not validate_result_data(true_data, pred_data, individual['id']):
                raise RuntimeError("Invalid result data (NaN/Inf or missing).")
            
            mape = compute_mape(true_data, pred_data)
            elapsed = time.time() - start
            return individual, mape, elapsed, None, worker_log
        
    except Exception as e:
        # Return error message to main process; details will be in worker.log if created.
        worker_log = None
        try:
            worker_log = os.path.join(run_dir, f"individual_{individual['id']}", "worker.log")
            os.makedirs(os.path.dirname(worker_log), exist_ok=True)
            with open(worker_log, "a", buffering=1) as f:
                f.write(f"\n[worker exception] {repr(e)}\n")
        except Exception:
            worker_log = None
        return individual, None, None, str(e), worker_log

def evaluate_population_parallel(population, run_dir, existing_ids, max_workers=None, data_type='ma2'):
    """Evaluate population in parallel with progress tracking"""
    if max_workers is None:
        max_workers = get_optimal_workers()
    
    logger.info(f"Evaluating {len(population)} individuals in parallel using {max_workers} workers")
    
    # Prepare arguments for workers
    worker_args = [(individual, run_dir, existing_ids, data_type) for individual in population]
    
    fitness = []
    successful_count = 0
    failed_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_individual = {executor.submit(evaluate_individual_worker, args): args[0] 
                               for args in worker_args}
        
        # Process completed jobs with progress bar
        with tqdm(total=len(population), desc="Evaluating individuals", unit="individual") as pbar:
            for future in as_completed(future_to_individual):
                individual, mape, elapsed, err, worker_log = future.result()
                
                # Always add the individual's ID to existing_ids, regardless of success/failure
                # This prevents ID conflicts in future generations
                existing_ids.add(individual['id'])
                
                if mape is not None:
                    fitness.append({'individual': individual, 'mape': mape})
                    successful_count += 1
                    pbar.write(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - "
                        f"DONE {individual['id'][:8]}... | {individual['notebook_type']} state{individual['state']} | "
                        f"epochs: {individual['epochs']} | patience: {individual.get('early_stopping_patience')} | "
                        f"MAPE: {mape:.2f}% | {elapsed:.1f}s"
                    )
                    pbar.set_postfix({
                        'Success': successful_count, 
                        'Failed': failed_count,
                        'Best MAPE': f"{min([f['mape'] for f in fitness]):.2f}%" if fitness else "N/A"
                    })
                else:
                    failed_count += 1
                    msg = f"FAIL {individual['id'][:8]}... | {individual['notebook_type']} state{individual['state']}"
                    if worker_log:
                        msg += f" | see {pretty_path(worker_log)}"
                    if err:
                        msg += f" | {err}"
                    pbar.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - {msg}")
                    pbar.set_postfix({
                        'Success': successful_count, 
                        'Failed': failed_count,
                        'Best MAPE': f"{min([f['mape'] for f in fitness]):.2f}%" if fitness else "N/A"
                    })
                
                pbar.update(1)
    
    logger.info(f"Evaluation complete: {successful_count} successful, {failed_count} failed")
    return fitness

def crossover(parent1, parent2, existing_ids=None):
    """Crossover function (same as original)"""
    child = {}
    for key in ['notebook_type', 'state', 'reserved_shot', 'epochs', 'validation_split',
                'activation_func', 'loss_func', 'optimizer_func', 'outlier_cutoff',
                'num_conv2d_layers', 'num_dense_layers', 'early_stopping_patience', 'early_stopping_min_delta']:
        child[key] = np.random.choice([parent1[key], parent2[key]])
    child['max_pooling_size'] = random.choice([parent1['max_pooling_size'], parent2['max_pooling_size']])
    child['conv2d_neurons'] = []
    child['conv2d_size'] = []
    for i in range(child['num_conv2d_layers']):
        p = np.random.choice([parent1, parent2])
        child['conv2d_neurons'].append(p['conv2d_neurons'][i] if i < len(p['conv2d_neurons']) else int(np.random.choice(PARAM_SPACE['conv2d_neurons'])))
        child['conv2d_size'].append(p['conv2d_size'][i] if i < len(p['conv2d_size']) else random.choice(PARAM_SPACE['conv2d_size']))
    child['dense_layer_neurons'] = []
    for i in range(child['num_dense_layers']):
        p = np.random.choice([parent1, parent2])
        child['dense_layer_neurons'].append(p['dense_layer_neurons'][i] if i < len(p['dense_layer_neurons']) else int(np.random.choice(PARAM_SPACE['dense_layer_neurons'])))
    
    # Generate unique ID, checking against existing_ids if provided
    child['id'] = str(uuid.uuid4())
    if existing_ids is not None:
        child = ensure_unique_id(child, existing_ids)
    
    if child['state'] in RESERVED_SHOTS and child['reserved_shot'] not in RESERVED_SHOTS[child['state']]:
        child['reserved_shot'] = int(np.random.choice(RESERVED_SHOTS[child['state']]))
    return child

def mutate(individual, existing_ids=None):
    """Mutation function (same as original)"""
    mutated = copy.deepcopy(individual)
    if np.random.random() < MUTATION_RATE:
        param = np.random.choice(['notebook_type', 'state', 'epochs', 'validation_split',
                                  'activation_func', 'loss_func', 'optimizer_func', 'outlier_cutoff',
                                  'num_conv2d_layers', 'num_dense_layers', 'conv2d_neurons',
                                  'conv2d_size', 'dense_layer_neurons', 'max_pooling_size',
                                  'early_stopping_patience', 'early_stopping_min_delta'])
        if param == 'state':
            mutated['state'] = int(np.random.choice(PARAM_SPACE['state']))
            mutated['reserved_shot'] = int(np.random.choice(RESERVED_SHOTS[mutated['state']]))
        elif param == 'validation_split' or param == 'outlier_cutoff' or param == 'early_stopping_min_delta':
            mutated[param] = float(np.random.choice(PARAM_SPACE[param]))
        elif param == 'num_conv2d_layers':
            mutated[param] = int(np.random.choice(PARAM_SPACE[param]))
            mutated['conv2d_neurons'] = [int(np.random.choice(PARAM_SPACE['conv2d_neurons'])) for _ in range(mutated[param])]
            mutated['conv2d_size'] = [random.choice(PARAM_SPACE['conv2d_size']) for _ in range(mutated[param])]
        elif param == 'num_dense_layers':
            mutated[param] = int(np.random.choice(PARAM_SPACE[param]))
            mutated['dense_layer_neurons'] = [int(np.random.choice(PARAM_SPACE['dense_layer_neurons'])) for _ in range(mutated[param])]
        elif param == 'conv2d_neurons':
            idx = np.random.randint(0, len(mutated['conv2d_neurons']))
            mutated[param][idx] = int(np.random.choice(PARAM_SPACE[param]))
        elif param == 'conv2d_size':
            idx = np.random.randint(0, len(mutated['conv2d_size']))
            mutated[param][idx] = random.choice(PARAM_SPACE[param])
        elif param == 'dense_layer_neurons':
            idx = np.random.randint(0, len(mutated['dense_layer_neurons']))
            mutated[param][idx] = int(np.random.choice(PARAM_SPACE[param]))
        elif param == 'max_pooling_size':
            mutated[param] = random.choice(PARAM_SPACE[param])
        elif param == 'early_stopping_patience':
            mutated[param] = int(np.random.choice(PARAM_SPACE[param]))
        else:
            mutated[param] = np.random.choice(PARAM_SPACE[param])
        if param == 'epochs':
            mutated[param] = int(mutated[param])
        
        # Generate new ID if mutation occurred and existing_ids is provided
        if existing_ids is not None:
            mutated['id'] = str(uuid.uuid4())
            mutated = ensure_unique_id(mutated, existing_ids)
    
    return mutated

def load_previous_population(run_dir):
    """Load previous population (same as original)"""
    csv_path = os.path.join(run_dir, CSV_FILENAME)
    if not os.path.exists(csv_path):
        logger.info(f"No previous results found at {pretty_path(csv_path)}. Starting fresh.")
        return None, 0, None, float('inf'), None
    
    df = pd.read_csv(csv_path)
    if df.empty:
        logger.info("Previous CSV is empty. Starting fresh.")
        return None, 0, None, float('inf'), None
    
    # Check for duplicate individual_ids
    duplicate_ids = df[df['individual_id'].duplicated()]['individual_id'].tolist()
    if duplicate_ids:
        logger.warning(f"Duplicate individual_ids found in CSV: {duplicate_ids}. Removing duplicates.")
        df = df.drop_duplicates(subset='individual_id', keep='last')
    
    last_gen = df['generation'].max()
    last_gen_df = df[df['generation'] == last_gen]
    population = []
    existing_ids = set(df['individual_id'])
    top_n = max(2, int(POPULATION_SIZE * TOP_PERCENT))
    top_individuals = last_gen_df[last_gen_df['mape'].notna()].sort_values('mape').head(top_n)
    
    for _, row in top_individuals.iterrows():
        individual = {
            'notebook_type': row['notebook_type'],
            'state': int(row['state']),
            'reserved_shot': int(row['reserved_shot']),
            'epochs': int(row['epochs']),
            'validation_split': float(row['validation_split']),
            'activation_func': row['activation_func'],
            'loss_func': row['loss_func'],
            'optimizer_func': row['optimizer_func'],
            'outlier_cutoff': float(row['outlier_cutoff']),
            'num_conv2d_layers': int(row['num_conv2d_layers']),
            'num_dense_layers': int(row['num_dense_layers']),
            'conv2d_neurons': eval(row['conv2d_neurons']),
            'conv2d_size': eval(row['conv2d_size']),
            'dense_layer_neurons': eval(row['dense_layer_neurons']),
            'max_pooling_size': eval(row['max_pooling_size']),
            'early_stopping_patience': int(row.get('early_stopping_patience', 20)),  # Default to 20 if not present
            'early_stopping_min_delta': float(row.get('early_stopping_min_delta', 0.01)),  # Default to 0.01 if not present
            'id': row['individual_id']
        }
        if validate_individual(individual):
            population.append(individual)
    
    for _, row in last_gen_df.iterrows():
        if pd.notna(row['mape']) and row['individual_id'] not in [ind['id'] for ind in population]:
            individual = {
                'notebook_type': row['notebook_type'],
                'state': int(row['state']),
                'reserved_shot': int(row['reserved_shot']),
                'epochs': int(row['epochs']),
                'validation_split': float(row['validation_split']),
                'activation_func': row['activation_func'],
                'loss_func': row['loss_func'],
                'optimizer_func': row['optimizer_func'],
                'outlier_cutoff': float(row['outlier_cutoff']),
                'num_conv2d_layers': int(row['num_conv2d_layers']),
                'num_dense_layers': int(row['num_dense_layers']),
                'conv2d_neurons': eval(row['conv2d_neurons']),
                'conv2d_size': eval(row['conv2d_size']),
                'dense_layer_neurons': eval(row['dense_layer_neurons']),
                'max_pooling_size': eval(row['max_pooling_size']),
                'early_stopping_patience': int(row.get('early_stopping_patience', 20)),  # Default to 20 if not present
                'early_stopping_min_delta': float(row.get('early_stopping_min_delta', 0.01)),  # Default to 0.01 if not present
                'id': row['individual_id']
            }
            if validate_individual(individual):
                population.append(individual)
    
    logger.info(f"Loaded {len(population)} individuals for generation {last_gen + 1}, including {len(top_individuals)} top performers")
    
    while len(population) < POPULATION_SIZE:
        new_individual = generate_individual()
        new_individual = ensure_unique_id(new_individual, existing_ids)
        population.append(new_individual)
    
    best_mape = df['mape'].min()
    best_row = df[df['mape'] == best_mape].iloc[0]
    best_params = {
        'notebook_type': best_row['notebook_type'],
        'state': int(best_row['state']),
        'reserved_shot': int(best_row['reserved_shot']),
        'epochs': int(best_row['epochs']),
        'validation_split': float(best_row['validation_split']),
        'activation_func': best_row['activation_func'],
        'loss_func': best_row['loss_func'],
        'optimizer_func': best_row['optimizer_func'],
        'outlier_cutoff': float(best_row['outlier_cutoff']),
        'num_conv2d_layers': int(best_row['num_conv2d_layers']),
        'num_dense_layers': int(best_row['num_dense_layers']),
        'conv2d_neurons': eval(best_row['conv2d_neurons']),
        'conv2d_size': eval(best_row['conv2d_size']),
        'dense_layer_neurons': eval(best_row['dense_layer_neurons']),
        'max_pooling_size': eval(best_row['max_pooling_size']),
        'early_stopping_patience': int(best_row.get('early_stopping_patience', 20)),  # Default to 20 if not present
        'early_stopping_min_delta': float(best_row.get('early_stopping_min_delta', 0.01)),  # Default to 0.01 if not present
        'id': best_row['individual_id']
    }
    best_individual_dir = os.path.join(run_dir, f"individual_{best_params['id']}")
    return population, last_gen, best_params, best_mape, best_individual_dir

def create_analysis_plots(results_df, plot_dir):
    """Create analysis plots (same as original)"""
    os.makedirs(plot_dir, exist_ok=True)
    plot_params = ['epochs', 'validation_split', 'outlier_cutoff', 'num_conv2d_layers', 'num_dense_layers']
    for param in plot_params:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=results_df, x=param, y='mape', hue='generation', palette='viridis', size='generation', sizes=(50, 200))
        plt.xlabel(param.replace('_', ' ').title())
        plt.ylabel('MAPE (%)')
        plt.title(f'MAPE vs {param.replace("_", " ").title()} by Generation')
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f'mape_vs_{param}.png'))
        plt.close()

def genetic_algorithm_parallel(run_dir=None, max_workers=None, data_type='ma2'):
    """Parallel genetic algorithm"""
    if run_dir is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(OUTPUT_DIR, f'run_parallel_{data_type}_{timestamp}')
    
    os.makedirs(run_dir, exist_ok=True)
    plot_dir = os.path.join(run_dir, 'plot_analysis')
    os.makedirs(plot_dir, exist_ok=True)
    
    population, start_gen, best_params, best_mape, best_individual_dir = load_previous_population(run_dir)
    
    if population is None:
        population = [generate_individual() for _ in range(POPULATION_SIZE)]
        start_gen = 0
        best_mape = float('inf')
        best_params = None
        best_individual_dir = None
        results_df = pd.DataFrame(columns=['generation', 'individual_id', 'notebook_type', 'state', 'reserved_shot',
                                           'epochs', 'validation_split', 'activation_func', 'loss_func',
                                           'optimizer_func', 'outlier_cutoff', 'num_conv2d_layers',
                                           'num_dense_layers', 'conv2d_neurons', 'conv2d_size',
                                           'dense_layer_neurons', 'max_pooling_size', 'early_stopping_patience',
                                           'early_stopping_min_delta', 'mape'], dtype=object)
    else:
        results_df = pd.read_csv(os.path.join(run_dir, CSV_FILENAME))
        logger.info(f"Resuming from generation {start_gen} with {len(population)} individuals")
    
    mape_history = []
    if start_gen > 0:
        mape_history = [results_df[results_df['generation'] == g]['mape'].mean() for g in range(1, start_gen + 1)]
    
    existing_ids = set(results_df['individual_id']) if not results_df.empty else set()
    
    for generation in range(start_gen + 1, GENERATIONS + 1):
        logger.info(f"\nGeneration {generation}/{GENERATIONS} ({time.strftime('%H:%M:%S')})")
        generation_start = time.time()
        
        # Monitor memory before evaluation
        memory_before = monitor_memory_usage()
        logger.info(f"Memory usage before evaluation: {memory_before:.2f}GB")
        
        # Evaluate population in parallel
        fitness = evaluate_population_parallel(population, run_dir, existing_ids, max_workers, data_type)
        
        # Monitor memory after evaluation
        memory_after = monitor_memory_usage()
        logger.info(f"Memory usage after evaluation: {memory_after:.2f}GB (Δ: {memory_after - memory_before:+.2f}GB)")
        
        if not fitness:
            logger.error("No valid models. Terminating.")
            break
        
        # Update results dataframe
        for f in fitness:
            individual = f['individual']
            mape = f['mape']
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
                'mape': mape
            }])
            if results_df.empty:
                results_df = new_row
            else:
                results_df = pd.concat([results_df, new_row], ignore_index=True)
            
            if mape < best_mape:
                best_mape = mape
                best_params = copy.deepcopy(individual)
                best_individual_dir = os.path.join(run_dir, f"individual_{individual['id']}")
        
        results_df.to_csv(os.path.join(run_dir, CSV_FILENAME), index=False)
        avg_mape = np.mean([f['mape'] for f in fitness])
        mape_history.append(avg_mape)
        
        generation_time = time.time() - generation_start
        logger.info(f"Generation {generation} | Average MAPE: {avg_mape:.2f}% | Best MAPE: {best_mape:.2f}% | Time: {generation_time:.1f}s")
        
        create_analysis_plots(results_df, plot_dir)
        
        # Create next generation
        top_n = max(2, int(POPULATION_SIZE * TOP_PERCENT))
        fitness.sort(key=lambda x: x['mape'])
        top_individuals = [f['individual'] for f in fitness[:min(top_n, len(fitness))]]
        
        if len(top_individuals) < 2:
            logger.warning(f"Only {len(top_individuals)} valid individuals. Regenerating population.")
            population = []
            for _ in range(POPULATION_SIZE):
                new_individual = generate_individual()
                new_individual = ensure_unique_id(new_individual, existing_ids)
                population.append(new_individual)
            continue
        
        # Create new population starting with top individuals (with new IDs)
        new_population = []
        for individual in top_individuals:
            # Create a copy and give it a new ID to avoid conflicts
            new_individual = copy.deepcopy(individual)
            new_individual['id'] = str(uuid.uuid4())
            new_individual = ensure_unique_id(new_individual, existing_ids)
            new_population.append(new_individual)
        
        while len(new_population) < POPULATION_SIZE:
            p1, p2 = np.random.choice(top_individuals, 2, replace=False)
            child = crossover(p1, p2, existing_ids)
            child = mutate(child, existing_ids)
            # ID conflicts are already handled in crossover and mutate functions
            new_population.append(child)
        
        population = new_population
    
    # Create final progress plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(mape_history) + 1), mape_history, '-o')
    plt.xlabel('Generation')
    plt.ylabel('Average MAPE (%)')
    plt.title('Optimization Progress')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, PLOT_FILENAME))
    plt.close()
    
    
    logger.info(f"\nOptimization Complete!")
    logger.info(f"Best MAPE: {best_mape:.2f}%")
    logger.info(f"Results saved in: {pretty_path(run_dir)}")
    logger.info(f"Best individual: {pretty_path(best_individual_dir) if best_individual_dir else best_individual_dir}")
    
    return best_params, best_mape


def main():
    """Main function with command-line interface"""
    global POPULATION_SIZE, GENERATIONS, MUTATION_RATE, EPOCHS
    
    parser = argparse.ArgumentParser(description='Parallel HBT Parameter Optimization')
    parser.add_argument('--data_type', type=str, default='ma2',
                       help='Data type: ma1-ma4 (mode amplitude 1-4), mp1-mp4 (mode phase 1-4), '
                            'mps1-mps4 (sin(phase) for modes 1-4), mpc1-mpc4 (cos(phase) for modes 1-4) '
                            '(default: ma2)')
    parser.add_argument('--max_workers', type=int, help='Maximum number of parallel workers')
    parser.add_argument('--run_dir', help='Specific run directory to resume')
    parser.add_argument('--population_size', type=int, default=POPULATION_SIZE,
                       help=f'Population size for genetic algorithm (default: {POPULATION_SIZE})')
    parser.add_argument('--generations', type=int, default=GENERATIONS,
                       help=f'Number of generations for genetic algorithm (default: {GENERATIONS})')
    parser.add_argument('--mutation_rate', type=float, default=MUTATION_RATE,
                       help=f'Mutation rate for genetic algorithm (default: {MUTATION_RATE})')
    parser.add_argument('--crossover_rate', type=float, default=0.8,
                       help='Crossover rate for genetic algorithm (default: 0.8)')
    parser.add_argument('--state', type=int, default=2,
                       help='State number (1, 2, or 3) (default: 2)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs for each optimization run (default: 50)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging output')
    
    args = parser.parse_args()

    setup_logging(verbose=bool(args.verbose))
    
    # Update global configuration with command line arguments
    POPULATION_SIZE = args.population_size
    GENERATIONS = args.generations
    MUTATION_RATE = args.mutation_rate
    EPOCHS = args.epochs

    # Honor the requested state: constrain the search space so individuals don't randomly pick other states.
    PARAM_SPACE['state'] = [int(args.state)]
    
    # Update PARAM_SPACE to cap epochs at the specified value
    PARAM_SPACE['epochs'] = list(range(10, args.epochs + 1, 5))
    
    # Local execution only
    recommended = get_optimal_workers()
    if args.max_workers is None:
        # Auto mode: take the recommended value, but never go below 2.
        # (If the user explicitly wants 1, they can pass --max_workers 1.)
        max_workers = max(2, recommended)
    else:
        max_workers = args.max_workers
    logger.info(f"System resources: {mp.cpu_count()} CPUs, {psutil.virtual_memory().available / (1024**3):.1f}GB RAM")
    logger.info(f"Recommended workers: {recommended} | Using: {max_workers}")
    logger.info(f"Starting parallel optimization with {max_workers} workers (available CPUs: {mp.cpu_count()})")
    
    best_params, best_mape = genetic_algorithm_parallel(
        run_dir=args.run_dir,
        max_workers=max_workers,
        data_type=args.data_type
    )
    
    logger.info(f"Optimization completed successfully. Best MAPE: {best_mape:.2f}%")


if __name__ == "__main__":
    main()
