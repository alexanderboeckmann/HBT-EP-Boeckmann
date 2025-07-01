import papermill as pm
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run HBT predictions for different state configurations.")
parser.add_argument('--mode', choices=['mode1', 'mode2'], default='mode1',
                    help='Execution mode: mode1 (states 1, 3 with shot 119671), mode2 (states 2, 3 with shot 114412)')
args = parser.parse_args()

# Configuration
STATES_MODE1 = [1, 3]  # States for mode1
STATES_MODE2 = [2, 3]  # States for mode2
RESERVED_SHOTS = {
    'mode1': {
        1: 119671,  # Shot for state 1 in mode1
        3: 119671   # Shot for state 3 in mode1
    },
    'mode2': {
        2: 114412,  # Shot for state 2 in mode2
        3: 114412   # Shot for state 3 in mode2
    }
}
SELECTED_DATA_TYPE = 'ma2'
NOTEBOOKS = {
    'trimmed': '/Users/aboeckmann/Documents/Columbia/PlasmaLab/HBT-EP-Boeckmann/HighFreqMLModeling/File_running/trimmed_HBT_analysis.ipynb',
    'untrimmed': '/Users/aboeckmann/Documents/Columbia/PlasmaLab/HBT-EP-Boeckmann/HighFreqMLModeling/File_running/untrimmed_HBT_analysis.ipynb'
}
OUTPUT_DIR = 'output_notebooks'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Select states and shots based on mode
STATES = STATES_MODE1 if args.mode == 'mode1' else STATES_MODE2
shots = RESERVED_SHOTS[args.mode]
print(f"Running in {args.mode} with states {STATES} and shots {shots}")

# Dictionary to store results
results = {
    'trimmed': {state: {'true': None, 'pred': None} for state in STATES},
    'untrimmed': {state: {'true': None, 'pred': None} for state in STATES}
}

# Execute notebooks for each configuration
for notebook_type in NOTEBOOKS:
    for state in STATES:
        input_notebook = NOTEBOOKS[notebook_type]
        output_notebook = os.path.join(OUTPUT_DIR, f'{notebook_type}_state_{state}_output.ipynb')
        output_true = f'results_{notebook_type}_state_{state}_{SELECTED_DATA_TYPE}_true.npy'
        output_pred = f'results_{notebook_type}_state_{state}_{SELECTED_DATA_TYPE}_pred.npy'
        
        # Parameters to pass to the notebook
        parameters = {
            'state': state,
            'selected_data_type': SELECTED_DATA_TYPE,
            'RESERVED_SHOT': shots[state],
            'EPOCH_NUM': 20
        }
        
        # Execute the notebook
        try:
            print(f"Executing {notebook_type} notebook for state {state} with reserved shot {shots[state]}...")
            pm.execute_notebook(
                input_notebook,
                output_notebook,
                parameters=parameters,
                kernel_name='python3'
            )
        except Exception as e:
            print(f"Error executing {notebook_type} notebook for state {state}: {str(e)}")
            continue
        
        # Load results
        try:
            true_data = np.load(output_true)
            pred_data = np.load(output_pred)
            print(f"Loaded true data shape for {notebook_type} state {state}: {true_data.shape}")
            results[notebook_type][state]['true'] = true_data
            results[notebook_type][state]['pred'] = pred_data
            print(f"Loaded results for {notebook_type} state {state}")
        except FileNotFoundError:
            print(f"Result files not found for {notebook_type} state {state}...")
        except Exception as e:
            print(f"Error loading results for {notebook_type} state {state}: {str(e)}")

# Populate true_data (only one true value needed per state)
true_data = {state: None for state in STATES}
for state in STATES:
    for notebook_type in results:
        if results[notebook_type][state]['true'] is not None:
            print(f"Assigning true data for state {state} from {notebook_type} (shape: {results[notebook_type][state]['true'].shape})")
            true_data[state] = results[notebook_type][state]['true']
            break

# Generate comparison plot
plt.figure(figsize=(12, 8))
colors = {
    'trimmed_1': 'b',
    'trimmed_2': 'g',
    'trimmed_3': 'r',
    'untrimmed_1': 'c',
    'untrimmed_2': 'm',
    'untrimmed_3': 'y'
}
labels = {
    'trimmed_1': 'Trimmed State 1',
    'trimmed_2': 'Trimmed State 2',
    'trimmed_3': 'Trimmed State 3',
    'untrimmed_1': 'Untrimmed State 1',
    'untrimmed_2': 'Untrimmed State 2',
    'untrimmed_3': 'Untrimmed State 3'
}

# Plot true and predicted data for each state
for state in STATES:
    if true_data[state] is None:
        print(f"Warning: No true data found for state {state} with reserved shot {shots[state]}. Skipping plot for this state.")
        continue
    plt.plot(true_data[state], 'k-', label=f'True ma2 (State {state}, Shot {shots[state]})', linewidth=2, alpha=0.5)
    for notebook_type in results:
        if results[notebook_type][state]['pred'] is not None:
            key = f"{notebook_type}_{state}"
            plt.plot(results[notebook_type][state]['pred'], '--', color=colors[key], label=labels[key])

plt.xlabel('Frame Index')
plt.ylabel('ma2 (Original Scale)')
plt.title(f'Actual vs Predicted ma2 for Reserved Shots {shots} (Mode: {args.mode})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print summary metrics
for notebook_type in results:
    for state in results[notebook_type]:
        if results[notebook_type][state]['true'] is not None and results[notebook_type][state]['pred'] is not None:
            errors = np.abs(results[notebook_type][state]['true'] - results[notebook_type][state]['pred']) / np.max(np.abs(results[notebook_type][state]['true'])) * 100
            print(f"{notebook_type} State {state} (Shot {shots[state]}) - Mean absolute percentage error: {np.mean(errors):.2f}%")