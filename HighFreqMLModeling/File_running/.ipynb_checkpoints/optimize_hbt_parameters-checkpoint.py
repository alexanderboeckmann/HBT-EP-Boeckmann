import os
import time
import uuid
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import papermill as pm
from jupyter_client.kernelspec import find_kernel_specs
import shutil

# Configuration
POPULATION_SIZE = 2  # Reduced for testing
GENERATIONS = 3
TOP_PERCENT = 0.1
MUTATION_RATE = 0.1
OUTPUT_DIR = 'optimization_results'
CSV_FILENAME = 'hbt_optimization_results.csv'
PLOT_FILENAME = 'hbt_optimization_progress.png'

# Hyperparameter space
PARAM_SPACE = {
    'notebook_type': ['trimmed', 'untrimmed'],
    'state': [1, 2, 3],  # All possible states
    'epochs': list(range(10, 26, 5)),
    'validation_split': [0.1, 0.15, 0.2, 0.25, 0.3],
    'activation_func': ['relu', 'sigmoid', 'tanh'],
    'loss_func': ['mse', 'mae'],
    'optimizer_func': ['adam', 'sgd', 'rmsprop'],
    'outlier_cutoff': list(range(80, 101, 2))
}

# Define RESERVED_SHOTS with multiple options for state 3
RESERVED_SHOTS = {
    1: [119671],
    2: [114458],
    3: [119671, 114458]  # State 3 can use either shot number
}

def get_available_kernels():
    return find_kernel_specs().keys()

def generate_individual():
    state = int(np.random.choice(PARAM_SPACE['state']))
    # Randomly choose a RESERVED_SHOT based on the state
    reserved_shot = int(np.random.choice(RESERVED_SHOTS[state]))
    return {
        'notebook_type': np.random.choice(PARAM_SPACE['notebook_type']),
        'state': state,
        'reserved_shot': reserved_shot,  # Store the chosen shot
        'epochs': int(np.random.choice(PARAM_SPACE['epochs'])),
        'validation_split': float(np.random.choice(PARAM_SPACE['validation_split'])),
        'activation_func': np.random.choice(PARAM_SPACE['activation_func']),
        'loss_func': np.random.choice(PARAM_SPACE['loss_func']),
        'optimizer_func': np.random.choice(PARAM_SPACE['optimizer_func']),
        'outlier_cutoff': float(np.random.choice(PARAM_SPACE['outlier_cutoff'])),
        'id': str(uuid.uuid4())
    }

def validate_individual(individual):
    # Check if the state and reserved_shot combination is valid
    return individual['state'] in RESERVED_SHOTS and individual['reserved_shot'] in RESERVED_SHOTS[individual['state']]

def construct_paths(individual):
    individual_dir = os.path.join(OUTPUT_DIR, f"individual_{individual['id']}")
    base = f"results_{individual['notebook_type']}_state_{individual['state']}_ma2"
    return (
        f"{individual['notebook_type']}_HBT_analysis.ipynb",
        os.path.join(individual_dir, f"{individual['id']}_output.ipynb"),
        os.path.join(individual_dir, f"{base}_true.npy"),
        os.path.join(individual_dir, f"{base}_pred.npy"),
        individual_dir
    )

def prepare_parameters(individual):
    return {
        'state': int(individual['state']),
        'selected_data_type': 'ma2',
        'RESERVED_SHOT': int(individual['reserved_shot']),  # Use the stored shot
        'EPOCH_NUM': int(individual['epochs']),
        'VALIDATION_SPLIT': float(individual['validation_split']),
        'ACTIVATION_FUNC': individual['activation_func'],
        'LOSS_FUNC': individual['loss_func'],
        'OPTIMIZER_FUNC': individual['optimizer_func'],
        'OUTLIER_CUTOFF': float(individual['outlier_cutoff'])
    }

def execute_notebook(input_nb, output_nb, parameters, kernel_name, individual_dir):
    os.makedirs(individual_dir, exist_ok=True)
    pm.execute_notebook(
        input_path=input_nb,
        output_path=output_nb,
        parameters=parameters,
        kernel_name=kernel_name,
        cwd=individual_dir
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
          f"shot={individual['reserved_shot']} epochs={individual['epochs']} "
          f"split={individual['validation_split']} act={individual['activation_func']} "
          f"loss={individual['loss_func']} opt={individual['optimizer_func']} "
          f"cutoff={individual['outlier_cutoff']} MAPE={mape:.4f}% time={elapsed:.2f}s")

def evaluate_individual(individual, kernel_name='python3'):
    start = time.time()

    if not validate_individual(individual):
        print(f"Invalid state or shot for {individual['id']}")
        return None, None

    input_nb, output_nb, path_true, path_pred, individual_dir = construct_paths(individual)
    params = prepare_parameters(individual)

    try:
        if not os.path.exists(input_nb):
            print(f"Notebook not found: {input_nb}")
            return None, None

        execute_notebook(input_nb, output_nb, params, kernel_name, individual_dir)
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
        pass

def crossover(parent1, parent2):
    child = {}
    for key in ['notebook_type', 'state', 'reserved_shot', 'epochs', 'validation_split',
                'activation_func', 'loss_func', 'optimizer_func', 'outlier_cutoff']:
        child[key] = np.random.choice([parent1[key], parent2[key]])
    child['id'] = str(uuid.uuid4())
    # Ensure reserved_shot is valid for the chosen state
    if child['state'] in RESERVED_SHOTS and child['reserved_shot'] not in RESERVED_SHOTS[child['state']]:
        child['reserved_shot'] = int(np.random.choice(RESERVED_SHOTS[child['state']]))
    return child

def mutate(individual):
    mutated = copy.deepcopy(individual)
    if np.random.random() < MUTATION_RATE:
        param = np.random.choice(['notebook_type', 'state', 'epochs', 'validation_split',
                                  'activation_func', 'loss_func', 'optimizer_func', 'outlier_cutoff'])
        if param == 'state':
            mutated['state'] = int(np.random.choice(PARAM_SPACE['state']))
            # Update reserved_shot for the new state
            mutated['reserved_shot'] = int(np.random.choice(RESERVED_SHOTS[mutated['state']]))
        elif param == 'validation_split' or param == 'outlier_cutoff':
            mutated[param] = float(np.random.choice(PARAM_SPACE[param]))
        else:
            mutated[param] = np.random.choice(PARAM_SPACE[param])
        if param == 'epochs':
            mutated[param] = int(mutated[param])
    return mutated

def genetic_algorithm():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df = pd.DataFrame(columns=['generation', 'individual_id', 'notebook_type', 'state', 'reserved_shot',
                                       'epochs', 'validation_split', 'activation_func', 'loss_func',
                                       'optimizer_func', 'outlier_cutoff', 'mape'], dtype=object)

    population = [generate_individual() for _ in range(POPULATION_SIZE)]
    best_mape = float('inf')
    best_params = None
    best_individual_dir = None
    mape_history = []

    kernel_name = 'conda-base-py' if 'conda-base-py' in get_available_kernels() else 'python3'
    print(f"Using kernel: {kernel_name}")

    for generation in range(GENERATIONS):
        print(f"\nGeneration {generation + 1}/{GENERATIONS} ({time.strftime('%H:%M:%S')})")
        generation_start = time.time()
        fitness = []

        for i, individual in enumerate(population, 1):
            print(f"Evaluating {i}/{POPULATION_SIZE} (ID: {individual['id']})")
            _, mape = evaluate_individual(individual, kernel_name)
            if mape is None:
                continue
            fitness.append({'individual': individual, 'mape': mape})
            new_row = pd.DataFrame([{
                'generation': generation + 1,
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
                'mape': mape
            }])
            if results_df.empty:
                results_df = new_row
            else:
                results_df = pd.concat([results_df, new_row], ignore_index=True)

            if mape < best_mape:
                best_mape = mape
                best_params = copy.deepcopy(individual)
                best_individual_dir = os.path.join(OUTPUT_DIR, f"individual_{individual['id']}")
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
            population = [generate_individual() for _ in range(POPULATION_SIZE)]
            continue

        new_population = top_individuals.copy()
        while len(new_population) < POPULATION_SIZE:
            p1, p2 = np.random.choice(top_individuals, 2, replace=False)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

        for individual in population:
            individual_dir = os.path.join(OUTPUT_DIR, f"individual_{individual['id']}")
            if individual_dir != best_individual_dir and os.path.exists(individual_dir):
                shutil.rmtree(individual_dir)

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
    print(f"Best individual files retained in: {best_individual_dir}")
    return best_params, best_mape

if __name__ == "__main__":
    best_params, best_mape = genetic_algorithm()