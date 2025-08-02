import os
import time
import uuid
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import papermill as pm
from jupyter_client.kernelspec import find_kernel_specs

# Configuration
POPULATION_SIZE = 2  # Reduced for testing
GENERATIONS = 2
TOP_PERCENT = 0.1
MUTATION_RATE = 0.1
OUTPUT_DIR = 'optimization_results'
CSV_FILENAME = 'hbt_optimization_results.csv'
PLOT_FILENAME = 'hbt_optimization_progress.png'

# Hyperparameter space
PARAM_SPACE = {
    'notebook_type': ['trimmed', 'untrimmed'],
    'state': {
        'mode1': [1, 3],
        'mode2': [2, 3],
    },
    'epochs': list(range(10, 51, 5)),
    'validation_split': [0.1, 0.15, 0.2, 0.25, 0.3],
    'activation_func': ['relu', 'sigmoid', 'tanh'],
    'loss_func': ['mse', 'mae'],
    'optimizer_func': ['adam', 'sgd', 'rmsprop'],
    'outlier_cutoff': list(range(80, 101, 2))
}

RESERVED_SHOTS = {
    'mode1': {1: 119671, 3: 119671},
    'mode2': {2: 114458, 3: 114458}
}

def get_available_kernels():
    return find_kernel_specs().keys()

def generate_individual(mode):
    return {
        'notebook_type': np.random.choice(PARAM_SPACE['notebook_type']),
        'state': int(np.random.choice(PARAM_SPACE['state'][mode])),
        'epochs': int(np.random.choice(PARAM_SPACE['epochs'])),
        'validation_split': float(np.random.choice(PARAM_SPACE['validation_split'])),
        'activation_func': np.random.choice(PARAM_SPACE['activation_func']),
        'loss_func': np.random.choice(PARAM_SPACE['loss_func']),
        'optimizer_func': np.random.choice(PARAM_SPACE['optimizer_func']),
        'outlier_cutoff': float(np.random.choice(PARAM_SPACE['outlier_cutoff'])),
        'id': str(uuid.uuid4())
    }

# ==== Refactored Evaluation Logic ====

def validate_individual(individual, mode):
    return individual['state'] in RESERVED_SHOTS[mode]

def construct_paths(individual):
    base = f"results_{individual['notebook_type']}_state_{individual['state']}_ma2"
    return (
        f"{individual['notebook_type']}_HBT_analysis.ipynb",
        os.path.join(OUTPUT_DIR, f"{individual['id']}_output.ipynb"),
        f"{base}_true.npy",
        f"{base}_pred.npy"
    )

def prepare_parameters(individual, mode):
    return {
        'state': int(individual['state']),
        'selected_data_type': 'ma2',
        'RESERVED_SHOT': int(RESERVED_SHOTS[mode][individual['state']]),
        'EPOCH_NUM': int(individual['epochs']),
        'VALIDATION_SPLIT': float(individual['validation_split']),
        'ACTIVATION_FUNC': individual['activation_func'],
        'LOSS_FUNC': individual['loss_func'],
        'OPTIMIZER_FUNC': individual['optimizer_func'],
        'OUTLIER_CUTOFF': float(individual['outlier_cutoff'])
    }

def execute_notebook(input_nb, output_nb, parameters, kernel_name):
    pm.execute_notebook(
        input_path=input_nb,
        output_path=output_nb,
        parameters=parameters,
        kernel_name=kernel_name,
        cwd=OUTPUT_DIR
    )

def load_result_arrays(true_path, pred_path):
    true = np.load(true_path) if os.path.exists(true_path) else None
    pred = np.load(pred_path) if os.path.exists(pred_path) else None
    return true, pred

def validate_result_data(true, pred, individual_id):
    if true is None or pred is None:
        print(f"Missing result files for {individual_id}")
        return False
    if np.any(np.isnan(true)) or np.any(np.isnan(pred)) or \
       np.any(np.isinf(true)) or np.any(np.isinf(pred)) or \
       np.any(true < 0) or np.any(pred < 0):
        print(f"Invalid values in data for {individual_id}")
        return False
    return True

def compute_mape(true, pred):
    true = true.flatten()
    pred = pred.flatten()
    min_len = min(len(true), len(pred))
    errors = np.abs(true[:min_len] - pred[:min_len]) / (np.max(np.abs(true[:min_len])) + 1e-8)
    return np.mean(errors) * 100

def print_summary(individual, mape, elapsed):
    print(f"Success for {individual['id']}: "
          f"type={individual['notebook_type']} state={individual['state']} "
          f"epochs={individual['epochs']} split={individual['validation_split']} "
          f"act={individual['activation_func']} loss={individual['loss_func']} opt={individual['optimizer_func']} "
          f"cutoff={individual['outlier_cutoff']} MAPE={mape:.4f}% time={elapsed:.2f}s")

def evaluate_individual(individual, mode='mode1', kernel_name='python3'):
    start = time.time()

    if not validate_individual(individual, mode):
        print(f"Invalid state for {individual['id']}")
        return None, None

    input_nb, output_nb, path_true, path_pred = construct_paths(individual)
    params = prepare_parameters(individual, mode)

    try:
        if not os.path.exists(input_nb):
            print(f"Notebook not found: {input_nb}")
            return None, None

        execute_notebook(input_nb, output_nb, params, kernel_name)
        true_data, pred_data = load_result_arrays(path_true, path_pred)

        if not validate_result_data(true_data, pred_data, individual['id']):
            return None, None

        mape = compute_mape(true_data, pred_data)
        print_summary(individual, mape, time.time() - start)
        return None, mape

    except Exception as e:
        print(f"Exception for {individual['id']}: {e}")
        return None, None
    finally:
        if os.path.exists(output_nb):
            os.remove(output_nb)

# ==== Genetic Algorithm ====

def crossover(parent1, parent2):
    child = {}
    for key in ['notebook_type', 'state', 'epochs', 'validation_split',
                'activation_func', 'loss_func', 'optimizer_func', 'outlier_cutoff']:
        child[key] = np.random.choice([parent1[key], parent2[key]])
    child['id'] = str(uuid.uuid4())
    return child

def mutate(individual, mode):
    mutated = copy.deepcopy(individual)
    if np.random.random() < MUTATION_RATE:
        param = np.random.choice(['notebook_type', 'state', 'epochs', 'validation_split',
                                  'activation_func', 'loss_func', 'optimizer_func', 'outlier_cutoff'])
        if param == 'state':
            mutated[param] = int(np.random.choice(PARAM_SPACE['state'][mode]))
        elif param in ['validation_split', 'outlier_cutoff']:
            mutated[param] = float(np.random.choice(PARAM_SPACE[param]))
        else:
            mutated[param] = np.random.choice(PARAM_SPACE[param])
        if param == 'epochs':
            mutated[param] = int(mutated[param])
    return mutated

def genetic_algorithm(mode='mode1'):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df = pd.DataFrame(columns=['generation', 'individual_id', 'notebook_type', 'state', 'epochs',
                                       'validation_split', 'activation_func', 'loss_func', 'optimizer_func',
                                       'outlier_cutoff', 'mape'], dtype=object)

    population = [generate_individual(mode) for _ in range(POPULATION_SIZE)]
    best_mape = float('inf')
    best_params = None
    mape_history = []

    kernel_name = 'conda-base-py' if 'conda-base-py' in get_available_kernels() else 'python3'
    print(f"Using kernel: {kernel_name}")

    for generation in range(GENERATIONS):
        print(f"\nGeneration {generation + 1}/{GENERATIONS} ({time.strftime('%H:%M:%S')})")
        generation_start = time.time()
        fitness = []

        for i, individual in enumerate(population, 1):
            print(f"Evaluating {i}/{POPULATION_SIZE} (ID: {individual['id']})")
            _, mape = evaluate_individual(individual, mode, kernel_name)
            if mape is None:
                continue
            fitness.append({'individual': individual, 'mape': mape})
            results_df = pd.concat([results_df, pd.DataFrame([{
                'generation': generation + 1,
                'individual_id': individual['id'],
                'notebook_type': individual['notebook_type'],
                'state': individual['state'],
                'epochs': individual['epochs'],
                'validation_split': individual['validation_split'],
                'activation_func': individual['activation_func'],
                'loss_func': individual['loss_func'],
                'optimizer_func': individual['optimizer_func'],
                'outlier_cutoff': individual['outlier_cutoff'],
                'mape': mape
            }])], ignore_index=True)

            if mape < best_mape:
                best_mape = mape
                best_params = copy.deepcopy(individual)
                print(f"New best MAPE: {best_mape:.4f}%")

        if not fitness:
            print("No valid models. Terminating.")
            break

        results_df.to_csv(os.path.join(OUTPUT_DIR, CSV_FILENAME), index=False)
        avg_mape = np.mean([f['mape'] for f in fitness])
        mape_history.append(avg_mape)
        print(f"Generation {generation + 1} summary: avg MAPE={avg_mape:.4f}, best so far={best_mape:.4f}, time={time.time() - generation_start:.2f}s")

        top_n = max(2, int(POPULATION_SIZE * TOP_PERCENT))
        fitness.sort(key=lambda x: x['mape'])
        top_individuals = [f['individual'] for f in fitness[:min(top_n, len(fitness))]]

        if len(top_individuals) < 2:
            print(f"Only {len(top_individuals)} valid individuals. Regenerating population.")
            population = [generate_individual(mode) for _ in range(POPULATION_SIZE)]
            continue

        new_population = top_individuals.copy()
        while len(new_population) < POPULATION_SIZE:
            p1, p2 = np.random.choice(top_individuals, 2, replace=False)
            child = crossover(p1, p2)
            child = mutate(child, mode)
            new_population.append(child)

        population = new_population

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(mape_history) + 1), mape_history, '-o')
    plt.xlabel('Generation')
    plt.ylabel('Average MAPE (%)')
    plt.title('Optimization Progress')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, PLOT_FILENAME))
    plt.close()

    print(f"\nOptimization complete. Best MAPE: {best_mape:.2f}%")
    print(f"Best Parameters: {best_params}")
    return best_params, best_mape

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optimize the prediction hyperparameters using genetic algorithm.")
    parser.add_argument('--mode', choices=['mode1', 'mode2'], default='mode1',
                        help='Execution mode: mode1 (states 1, 3 with shot 119671), mode2 (states 2, 3 with shot 114458)')
    args = parser.parse_args()
    
    best_params, best_mape = genetic_algorithm(args.mode)