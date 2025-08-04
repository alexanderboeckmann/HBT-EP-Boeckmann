import os
import time
import uuid
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import papermill as pm
from jupyter_client.kernelspec import find_kernel_specs
import shutil
import random 

# Configuration
POPULATION_SIZE = 100
GENERATIONS = 10
TOP_PERCENT = 0.1
MUTATION_RATE = 0.1
OUTPUT_DIR = 'optimization_results'
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

def get_available_kernels():
    return find_kernel_specs().keys()

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

def construct_paths(individual, run_dir):
    individual_dir = os.path.join(run_dir, f"individual_{individual['id']}")
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

def execute_notebook(input_nb, output_nb, parameters, kernel_name, individual_dir):
    os.makedirs(individual_dir, exist_ok=True)
    try:
        print(f"Executing notebook with parameters: {parameters}")
        pm.execute_notebook(
            input_path=input_nb,
            output_path=output_nb,
            parameters=parameters,
            kernel_name=kernel_name,
            cwd=individual_dir
        )
    except pm.PapermillExecutionError as e:
        print(f"Exception for {parameters['individual_id']}: {str(e)}")
        raise

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
          f"cutoff={individual['outlier_cutoff']} "
          f"conv2d_layers={individual['num_conv2d_layers']} "
          f"dense_layers={individual['num_dense_layers']} "
          f"conv2d_neurons={individual['conv2d_neurons']} "
          f"conv2d_size={individual['conv2d_size']} "
          f"dense_neurons={individual['dense_layer_neurons']} "
          f"pool_size={individual['max_pooling_size']} "
          f"MAPE={mape:.4f}% time={elapsed:.2f}s")

def evaluate_individual(individual, kernel_name, run_dir):
    start = time.time()

    if not validate_individual(individual):
        print(f"Invalid state or shot for {individual['id']}")
        return None, None

    input_nb, output_nb, path_true, path_pred, individual_dir = construct_paths(individual, run_dir)
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
        print(f"No previous results found at {csv_path}. Starting fresh.")
        return None, 0, None, float('inf')

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Previous CSV is empty. Starting fresh.")
        return None, 0, None, float('inf')

    # Get the last generation
    last_gen = df['generation'].max()
    last_gen_df = df[df['generation'] == last_gen]

    # Reconstruct population
    population = []
    for _, row in last_gen_df.iterrows():
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
            'conv2d_neurons': eval(row['conv2d_neurons']),  # Convert string representation to list
            'conv2d_size': eval(row['conv2d_size']),        # Convert string representation to list of tuples
            'dense_layer_neurons': eval(row['dense_layer_neurons']),
            'max_pooling_size': eval(row['max_pooling_size']),
            'id': row['individual_id']
        }
        if validate_individual(individual):
            population.append(individual)

    # Ensure population size
    while len(population) < POPULATION_SIZE:
        new_individual = generate_individual()
        population.append(new_individual)

    # Get best MAPE and parameters
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

def genetic_algorithm(run_dir=None):
    # Create or use existing run directory
    if run_dir is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(OUTPUT_DIR, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    plot_dir = os.path.join(run_dir, 'plot_analysis')
    os.makedirs(plot_dir, exist_ok=True)

    # Load previous population if available
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

    kernel_name = 'conda-base-py' if 'conda-base-py' in get_available_kernels() else 'python3'
    print(f"Using kernel: {kernel_name}")

    for generation in range(start_gen + 1, GENERATIONS + 1):
        print(f"\nGeneration {generation}/{GENERATIONS} ({time.strftime('%H:%M:%S')})")
        generation_start = time.time()
        fitness = []

        for i, individual in enumerate(population, 1):
            print(f"Evaluating {i}/{POPULATION_SIZE} (ID: {individual['id']})")
            _, mape = evaluate_individual(individual, kernel_name, run_dir)
            if mape is None:
                continue
            fitness.append({'individual': individual, 'mape': mape})
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
                'num_dense_layers': individual['num_dense_layers'],
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
        print(f"Generation {generation} summary: avg MAPE={avg_mape:.4f}, best so far={best_mape:.4f}, time={time.time() - generation_start:.2f}s")

        create_analysis_plots(results_df, plot_dir)

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
            print(f"Created symbolic link to best individual: {best_link}")
        except OSError:
            if os.path.exists(best_link):
                shutil.rmtree(best_link)
            shutil.copytree(best_individual_dir, best_link)
            print(f"Copied best individual directory to: {best_link}")

    print(f"\nOptimization complete. Best MAPE: {best_mape:.2f}%")
    print(f"Best Parameters: {best_params}")
    print(f"All run files (individuals, CSV, plots) saved in: {run_dir}")
    print(f"Analysis plots saved in: {plot_dir}")
    print(f"Best individual directory: {best_individual_dir}")
    return best_params, best_mape

if __name__ == "__main__":
    # Example: Specify the run directory to resume (replace with your actual run directory)
    run_dir = 'optimization_results/run_20250803_232855'  # Set to 'optimization_results/run_YYYYMMDD_HHMMSS' to resume a specific run
    best_params, best_mape = genetic_algorithm(run_dir)