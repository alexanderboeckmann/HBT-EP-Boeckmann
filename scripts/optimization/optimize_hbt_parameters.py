"""
HBT Parameter Optimization Script

This script implements a genetic algorithm to optimize hyperparameters for HBT
neural network models. It searches through a parameter space including model architecture, training parameters,
and data processing settings to find configurations that minimize prediction error (MAPE).

The optimization process:
1. Generates a population of random parameter combinations
2. Evaluates each individual by training and testing HBT models
3. Selects top performers and creates new generations through crossover/mutation
4. Tracks progress and saves results for analysis

Supports both trimmed and untrimmed data analysis across different plasma states.
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
import shutil
import random
from pathlib import Path
import sys

# Allow importing the package when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hbt_analysis import HBTAnalysisTrimmed, HBTAnalysisUntrimmed

# Centralized project root and path utilities
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

def project_path(*parts):
    # Filter out None values before joining
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

# Configuration
POPULATION_SIZE = 50
GENERATIONS = 10
EPOCHS = 50  # Default epochs, can be overridden by command line
TOP_PERCENT = 0.125
MUTATION_RATE = 0.1
OUTPUT_DIR = project_path('data', 'optimization_results')
CSV_FILENAME = 'hbt_optimization_results.csv'
PLOT_FILENAME = 'hbt_optimization_progress.png'

# Hyperparameter space
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
    'max_pooling_size': [(2, 2), (3, 3), (4, 4)]
}

# Define RESERVED_SHOTS with multiple options for state 3
RESERVED_SHOTS = {
    1: [119671],
    2: [114458],
    3: [119671, 114458]
}


def generate_individual():
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
        'id': str(uuid.uuid4())
    }

def validate_individual(individual):
    return individual['state'] in RESERVED_SHOTS and individual['reserved_shot'] in RESERVED_SHOTS[individual['state']]

def construct_paths(individual, run_dir, existing_ids, data_type='ma2'):
    individual_dir = os.path.join(run_dir, f"individual_{individual['id']}")
    # Check for ID conflict
    if individual['id'] in existing_ids and os.path.exists(individual_dir):
        print(f"ID conflict detected for {individual['id']}. Generating new ID.")
        individual['id'] = str(uuid.uuid4())
        individual_dir = os.path.join(run_dir, f"individual_{individual['id']}")
    os.makedirs(individual_dir, exist_ok=True)
    params_path = os.path.join(individual_dir, "parameters.json")
    return individual_dir, params_path

def prepare_parameters(individual, data_type='ma2'):
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
        'MAX_POOLING_SIZE': individual['max_pooling_size']
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
        # Keep defaults consistent with the analysis classes
        'batch_size': 32,
        # Stop quickly once validation loss stops improving meaningfully
        'early_stopping_patience': 8,
        'early_stopping_min_delta': 0.001,
    }

    if individual['notebook_type'] == 'trimmed':
        analysis = HBTAnalysisTrimmed(config)
    elif individual['notebook_type'] == 'untrimmed':
        analysis = HBTAnalysisUntrimmed(config)
    else:
        raise ValueError(f"Unknown notebook_type: {individual['notebook_type']}")

    analysis.run_analysis(output_dir=output_dir)

def load_result_arrays(true_path, pred_path):
    true = np.load(true_path) if os.path.exists(true_path) else None
    pred = np.load(pred_path) if os.path.exists(pred_path) else None
    return true, pred

def validate_result_data(true, pred, individual_id):
    if true is None or pred is None:
        print(f"Missing result files for {individual_id}")
        return False
    if np.any(np.isnan(true)) or np.any(np.isinf(true)):
        print(f"Invalid true values for {individual_id}: NaN={np.any(np.isnan(true))}, Inf={np.any(np.isinf(true))}")
        return False
    if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
        print(f"Invalid pred values for {individual_id}: NaN={np.any(np.isnan(pred))}, Inf={np.any(np.isinf(pred))}")
        return False
    return True

def compute_mape(true, pred):
    true = true.flatten()
    pred = pred.flatten()
    min_len = min(len(true), len(pred))
    errors = np.abs(true[:min_len] - pred[:min_len]) / (np.max(np.abs(true[:min_len])) + 1e-8)
    return np.mean(errors) * 100

def print_summary(individual, mape, elapsed):
    print(f"SUCCESS {individual['id'][:8]}... | {individual['notebook_type']} state{individual['state']} | "
          f"MAPE: {mape:.2f}% | {elapsed:.1f}s")

def evaluate_individual(individual, run_dir, existing_ids, data_type='ma2'):
    start = time.time()
    if not validate_individual(individual):
        print(f"Invalid state or shot for {individual['id']}")
        return None, None
    individual_dir, params_path = construct_paths(individual, run_dir, existing_ids, data_type)
    params = prepare_parameters(individual, data_type)
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
    try:
        run_analysis_for_individual(individual, data_type=data_type, output_dir=individual_dir)
        npy_files = {f: os.path.join(individual_dir, f) for f in os.listdir(individual_dir) if f.endswith('.npy')}
        true_file = None
        pred_file = None
        for name, path in npy_files.items():
            if 'true' in name.lower():
                true_file = path
            elif 'pred' in name.lower():
                pred_file = path
        if true_file is None or pred_file is None:
            print(f"ERROR {individual['id'][:8]}... | Missing output files: true={true_file is not None}, pred={pred_file is not None}")
            return None, None
        base_name = os.path.basename(true_file).replace('_true.npy', '')
        parts = base_name.split('_')
        notebook_type = parts[1]
        state_idx = parts.index('state') + 1
        state = int(parts[state_idx]) if state_idx < len(parts) else None
        if state is None or state != individual['state']:
            print(f"Mismatched state in {individual_dir}: expected {individual['state']}, found {state}, file={true_file}")
            return None, None
        true_data, pred_data = load_result_arrays(true_file, pred_file)
        if not validate_result_data(true_data, pred_data, individual['id']):
            return None, None
        mape = compute_mape(true_data, pred_data)
        print_summary(individual, mape, time.time() - start)
        return None, mape
    except Exception as e:
        print(f"Exception for {individual['id']}: {e}")
        return None, None
    finally:
        pass

def crossover(parent1, parent2):
    child = {}
    for key in ['notebook_type', 'state', 'reserved_shot', 'epochs', 'validation_split',
                'activation_func', 'loss_func', 'optimizer_func', 'outlier_cutoff',
                'num_conv2d_layers', 'num_dense_layers']:
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
    child['id'] = str(uuid.uuid4())
    if child['state'] in RESERVED_SHOTS and child['reserved_shot'] not in RESERVED_SHOTS[child['state']]:
        child['reserved_shot'] = int(np.random.choice(RESERVED_SHOTS[child['state']]))
    return child

def mutate(individual):
    mutated = copy.deepcopy(individual)
    if np.random.random() < MUTATION_RATE:
        param = np.random.choice(['notebook_type', 'state', 'epochs', 'validation_split',
                                  'activation_func', 'loss_func', 'optimizer_func', 'outlier_cutoff',
                                  'num_conv2d_layers', 'num_dense_layers', 'conv2d_neurons',
                                  'conv2d_size', 'dense_layer_neurons', 'max_pooling_size'])
        if param == 'state':
            mutated['state'] = int(np.random.choice(PARAM_SPACE['state']))
            mutated['reserved_shot'] = int(np.random.choice(RESERVED_SHOTS[mutated['state']]))
        elif param == 'validation_split' or param == 'outlier_cutoff':
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
            mutated[param][idx] = int(np.random.choice(PARAM_SPACE['dense_layer_neurons']))
        elif param == 'max_pooling_size':
            mutated[param] = random.choice(PARAM_SPACE[param])
        else:
            mutated[param] = np.random.choice(PARAM_SPACE[param])
        if param == 'epochs':
            mutated[param] = int(mutated[param])
    return mutated

def load_previous_population(run_dir):
    csv_path = os.path.join(run_dir, CSV_FILENAME)
    if not os.path.exists(csv_path):
        print(f"No previous results found at {pretty_path(csv_path)}. Starting fresh.")
        return None, 0, None, float('inf'), None
    df = pd.read_csv(csv_path)
    if df.empty:
        print("Previous CSV is empty. Starting fresh.")
        return None, 0, None, float('inf'), None
    # Check for duplicate individual_ids
    duplicate_ids = df[df['individual_id'].duplicated()]['individual_id'].tolist()
    if duplicate_ids:
        print(f"Warning: Duplicate individual_ids found in CSV: {duplicate_ids}. Removing duplicates.")
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
                'id': row['individual_id']
            }
            if validate_individual(individual):
                population.append(individual)
    print(f"Loaded {len(population)} individuals for generation {last_gen + 1}, including {len(top_individuals)} top performers")
    while len(population) < POPULATION_SIZE:
        new_individual = generate_individual()
        # Ensure new ID doesn't conflict with existing ones
        while new_individual['id'] in existing_ids:
            print(f"ID conflict for {new_individual['id']}. Generating new ID.")
            new_individual['id'] = str(uuid.uuid4())
        existing_ids.add(new_individual['id'])
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
        'id': best_row['individual_id']
    }
    best_individual_dir = os.path.join(run_dir, f"individual_{best_params['id']}")
    return population, last_gen, best_params, best_mape, best_individual_dir

def create_analysis_plots(results_df, plot_dir):
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

def genetic_algorithm(run_dir=None, data_type='ma2'):
    if run_dir is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(OUTPUT_DIR, f'run_{data_type}_{timestamp}')
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
                                           'dense_layer_neurons', 'max_pooling_size', 'mape'], dtype=object)
    else:
        results_df = pd.read_csv(os.path.join(run_dir, CSV_FILENAME))
        print(f"Resuming from generation {start_gen} with {len(population)} individuals")
    mape_history = []
    if start_gen > 0:
        mape_history = [results_df[results_df['generation'] == g]['mape'].mean() for g in range(1, start_gen + 1)]
    # Track existing IDs to prevent conflicts
    existing_ids = set(results_df['individual_id']) if not results_df.empty else set()
    for generation in range(start_gen + 1, GENERATIONS + 1):
        print(f"\nGeneration {generation}/{GENERATIONS} ({time.strftime('%H:%M:%S')})")
        generation_start = time.time()
        fitness = []
        for i, individual in enumerate(population, 1):
            print(f"  [{i:2d}/{POPULATION_SIZE}] {individual['id'][:8]}... | {individual['notebook_type']} state{individual['state']} ({individual['epochs']}ep) | ", end="", flush=True)
            try:
                _, mape = evaluate_individual(individual, run_dir, existing_ids, data_type)
            except Exception as e:
                print(f"ERROR: {e}")
                continue
            if mape is None:
                continue
            fitness.append({'individual': individual, 'mape': mape})
            existing_ids.add(individual['id'])
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
                print(f"New best MAPE: {best_mape:.4f}%")
        if not fitness:
            print("No valid models. Terminating.")
            break
        results_df.to_csv(os.path.join(run_dir, CSV_FILENAME), index=False)
        avg_mape = np.mean([f['mape'] for f in fitness])
        mape_history.append(avg_mape)
        print(f"  Gen {generation} | Avg: {avg_mape:.2f}% | Best: {best_mape:.2f}% | Time: {time.time() - generation_start:.1f}s")
        create_analysis_plots(results_df, plot_dir)
        top_n = max(2, int(POPULATION_SIZE * TOP_PERCENT))
        fitness.sort(key=lambda x: x['mape'])
        top_individuals = [f['individual'] for f in fitness[:min(top_n, len(fitness))]]
        if len(top_individuals) < 2:
            print(f"Only {len(top_individuals)} valid individuals. Regenerating population.")
            population = [generate_individual() for _ in range(POPULATION_SIZE)]
            existing_ids.update([ind['id'] for ind in population])
            continue
        new_population = top_individuals.copy()
        while len(new_population) < POPULATION_SIZE:
            p1, p2 = np.random.choice(top_individuals, 2, replace=False)
            child = crossover(p1, p2)
            child = mutate(child)
            # Ensure new ID doesn't conflict
            while child['id'] in existing_ids:
                print(f"ID conflict for {child['id']}. Generating new ID.")
                child['id'] = str(uuid.uuid4())
            existing_ids.add(child['id'])
            new_population.append(child)
        population = new_population
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(mape_history) + 1), mape_history, '-o')
    plt.xlabel('Generation')
    plt.ylabel('Average MAPE (%)')
    plt.title('Optimization Progress')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, PLOT_FILENAME))
    plt.close()
    if best_individual_dir:
        best_link = os.path.join(run_dir, 'best_individual')
        try:
            if os.path.exists(best_link):
                if os.path.islink(best_link) or os.path.isdir(best_link):
                    os.remove(best_link)
                else:
                    shutil.rmtree(best_link)
            os.symlink(best_individual_dir, best_link)
            print(f"Created symbolic link to best individual: {pretty_path(best_link)}")
        except OSError:
            if os.path.exists(best_link):
                shutil.rmtree(best_link)
            shutil.copytree(best_individual_dir, best_link)
            print(f"Copied best individual directory to: {pretty_path(best_link)}")
    print(f"\nOptimization Complete!")
    print(f"Best MAPE: {best_mape:.2f}%")
    print(f"Results saved in: {pretty_path(run_dir)}")
    print(f"Best individual: {pretty_path(best_individual_dir) if best_individual_dir else best_individual_dir}")
    return best_params, best_mape

def main():
    """Main function with command-line interface"""
    global POPULATION_SIZE, GENERATIONS, MUTATION_RATE, OUTPUT_DIR, EPOCHS
    
    import argparse
    
    parser = argparse.ArgumentParser(description='HBT Parameter Optimization')
    parser.add_argument('--data_type', type=str, default='ma2',
                       help='Data type: ma1-ma4 (mode amplitude 1-4), mp1-mp4 (mode phase 1-4), '
                            'mps1-mps4 (sin(phase) for modes 1-4), mpc1-mpc4 (cos(phase) for modes 1-4) '
                            '(default: ma2)')
    parser.add_argument('--state', type=int, default=2, 
                       help='State number (1, 2, or 3) (default: 2)')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help=f'Max epochs for each optimization run (default: {EPOCHS}; early stopping usually ends earlier)')
    parser.add_argument('--population_size', type=int, default=POPULATION_SIZE, 
                       help=f'Population size for genetic algorithm (default: {POPULATION_SIZE})')
    parser.add_argument('--generations', type=int, default=GENERATIONS, 
                       help=f'Number of generations for genetic algorithm (default: {GENERATIONS})')
    parser.add_argument('--mutation_rate', type=float, default=0.1, 
                       help='Mutation rate for genetic algorithm (default: 0.1)')
    parser.add_argument('--crossover_rate', type=float, default=0.8, 
                       help='Crossover rate for genetic algorithm (default: 0.8)')
    parser.add_argument('--output_dir', type=str, 
                       help='Output directory for optimization results')
    
    args = parser.parse_args()
    
    # Update global configuration
    POPULATION_SIZE = args.population_size
    GENERATIONS = args.generations
    MUTATION_RATE = args.mutation_rate
    EPOCHS = args.epochs

    # Honor the requested state: constrain the search space so individuals don't randomly pick other states.
    PARAM_SPACE['state'] = [int(args.state)]
    
    # Update PARAM_SPACE to cap epochs at the specified value
    PARAM_SPACE['epochs'] = list(range(10, args.epochs + 1, 5))
    
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    
    # Pass None to create a new timestamped run directory
    best_params, best_mape = genetic_algorithm(None, args.data_type)
    
    return best_params, best_mape


if __name__ == "__main__":
    main()