import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess
from scipy.stats import chisquare
import uuid
import argparse
import copy

# Configuration
POPULATION_SIZE = 50  # Reduced for testing; increase to 100+ for full runs
GENERATIONS = 10
TOP_PERCENT = 0.1  # Top 10% for selection
MUTATION_RATE = 0.1  # 10% chance of mutation
OUTPUT_DIR = 'optimization_results'
CSV_FILENAME = 'hbt_optimization_results.csv'
PLOT_FILENAME = 'hbt_optimization_progress.png'

# Hyperparameter search space
PARAM_SPACE = {
    'notebook_type': ['trimmed', 'untrimmed'],
    'state': [1, 2, 3],
    'epochs': list(range(10, 51, 5))  # 10 to 50, step of 5
}

# Reserved shots (unchanged from original)
RESERVED_SHOTS = {
    'mode1': {1: 119671, 3: 119671},
    'mode2': {2: 114458, 3: 114458}
}

def generate_individual():
    """Generate a random individual (set of hyperparameters)."""
    return {
        'notebook_type': np.random.choice(PARAM_SPACE['notebook_type']),
        'state': np.random.choice(PARAM_SPACE['state']),
        'epochs': np.random.choice(PARAM_SPACE['epochs']),
        'id': str(uuid.uuid4())
    }

def run_model(individual, mode='mode1'):
    """Run the original script with given hyperparameters and return metrics."""
    # Prepare output file paths
    output_true = f"results_{individual['notebook_type']}_state_{individual['state']}_ma2_true.npy"
    output_pred = f"results_{individual['notebook_type']}_state_{individual['state']}_ma2_pred.npy"
    output_notebook = os.path.join(OUTPUT_DIR, f"{individual['id']}_output.ipynb")
    
    # Construct command to run the original script
    cmd = [
        'python', 'compare_hbt_predictions.py',
        '--mode', mode,
        # Pass parameters via environment variables or modify script to accept them
    ]
    
    # Create a temporary script with modified parameters
    with open('compare_hbt_predictions.py', 'r') as f:
        original_code = f.read()
    
    # Modify parameters in the script
    modified_code = original_code
    modified_code = modified_code.replace(
        f"NOTEBOOKS = {{",
        f"NOTEBOOKS = {{\n    'trimmed': '/Users/aboeckmann/Documents/Columbia/PlasmaLab/HBT-EP-Boeckmann/HighFreqMLModeling/File_running/trimmed_HBT_analysis.ipynb',\n    'untrimmed': '/Users/aboeckmann/Documents/Columbia/PlasmaLab/HBT-EP-Boeckmann/HighFreqMLModeling/File_running/untrimmed_HBT_analysis.ipynb'\n}}\nEPOCHS = {individual['epochs']}\n"
    )
    modified_code = modified_code.replace(
        f"STATES = STATES_MODE1 if args.mode == 'mode1' else STATES_MODE2",
        f"STATES = [{individual['state']}]"
    )
    
    temp_script = f"temp_{individual['id']}.py"
    with open(temp_script, 'w') as f:
        f.write(modified_code)
    
    try:
        # Run the modified script
        result = subprocess.run(['python', temp_script], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running model for individual {individual['id']}: {result.stderr}")
            return None, None
        
        # Load results
        true_data = np.load(output_true) if os.path.exists(output_true) else None
        pred_data = np.load(output_pred) if os.path.exists(output_pred) else None
        
        if true_data is None or pred_data is None:
            return None, None
        
        # Calculate chi-squared
        true_data = true_data.flatten()
        pred_data = pred_data.flatten()
        if len(true_data) != len(pred_data):
            return None, None
        
        # Normalize data for chi-squared
        true_data = true_data + 1e-8  # Avoid division by zero
        expected = true_data / np.sum(true_data)
        observed = pred_data / np.sum(pred_data)
        chi2, _ = chisquare(observed, expected)
        
        # Calculate MAPE
        errors = np.abs(true_data - pred_data) / (np.max(np.abs(true_data)) + 1e-8) * 100
        mape = np.mean(errors)
        
        return chi2, mape
    
    except Exception as e:
        print(f"Exception in run_model for {individual['id']}: {str(e)}")
        return None, None
    finally:
        # Clean up temporary script
        if os.path.exists(temp_script):
            os.remove(temp_script)

def crossover(parent1, parent2):
    """Perform crossover between two parents."""
    child = {}
    for key in ['notebook_type', 'state', 'epochs']:
        child[key] = np.random.choice([parent1[key], parent2[key]])
    child['id'] = str(uuid.uuid4())
    return child

def mutate(individual):
    """Apply mutation to an individual."""
    mutated = copy.deepcopy(individual)
    if np.random.random() < MUTATION_RATE:
        param_to_mutate = np.random.choice(['notebook_type', 'state', 'epochs'])
        mutated[param_to_mutate] = np.random.choice(PARAM_SPACE[param_to_mutate])
    return mutated

def genetic_algorithm(mode='mode1'):
    """Run genetic algorithm to optimize hyperparameters."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df = pd.DataFrame(columns=['generation', 'individual_id', 'notebook_type', 'state', 'epochs', 'chi2', 'mape'])
    
    # Initialize population
    population = [generate_individual() for _ in range(POPULATION_SIZE)]
    best_mape = float('inf')
    best_params = None
    mape_history = []
    
    for generation in range(GENERATIONS):
        print(f"\nGeneration {generation + 1}/{GENERATIONS}")
        
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            chi2, mape = run_model(individual, mode)
            if chi2 is None or mape is None:
                continue
            
            fitness_scores.append({
                'individual': individual,
                'chi2': chi2,
                'mape': mape
            })
            
            # Save to CSV
            results_df = results_df.append({
                'generation': generation + 1,
                'individual_id': individual['id'],
                'notebook_type': individual['notebook_type'],
                'state': individual['state'],
                'epochs': individual['epochs'],
                'chi2': chi2,
                'mape': mape
            }, ignore_index=True)
            
            if mape < best_mape:
                best_mape = mape
                best_params = copy.deepcopy(individual)
        
        if not fitness_scores:
            print("No valid models in this generation. Terminating.")
            break
        
        # Save results to CSV
        results_df.to_csv(os.path.join(OUTPUT_DIR, CSV_FILENAME), index=False)
        
        # Select top performers
        fitness_scores.sort(key=lambda x: x['mape'])  # Sort by MAPE
        n_top = max(1, int(POPULATION_SIZE * TOP_PERCENT))
        top_individuals = [fs['individual'] for fs in fitness_scores[:n_top]]
        
        # Track average MAPE for plotting
        mape_history.append(np.mean([fs['mape'] for fs in fitness_scores]))
        
        # Generate new population
        new_population = top_individuals.copy()
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = np.random.choice(top_individuals, 2, replace=False)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
    
    # Plot optimization progress
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(mape_history) + 1), mape_history, '-o')
    plt.xlabel('Generation')
    plt.ylabel('Average MAPE (%)')
    plt.title('Optimization Progress')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, PLOT_FILENAME))
    plt.close()
    
    print(f"\nBest parameters found: {best_params}")
    print(f"Best MAPE: {best_mape:.2f}%")
    return best_params, best_mape

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize HBT prediction hyperparameters using genetic algorithm.")
    parser.add_argument('--mode', choices=['mode1', 'mode2'], default='mode1',
                        help='Execution mode: mode1 (states 1, 3 with shot 119671), mode2 (states 2, 3 with shot 114412)')
    args = parser.parse_args()
    
    best_params, best_mape = genetic_algorithm(args.mode)