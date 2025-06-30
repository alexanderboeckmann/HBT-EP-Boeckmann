import papermill as pm
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
RESERVED_SHOT = 119671  # Fixed reserved shot (valid for states 1 and 3, will check for state 2)
SELECTED_DATA_TYPE = 'ma2'
STATES = [1, 2, 3]
NOTEBOOKS = {
    'trimmed': '/Users/aboeckmann/Documents/Columbia/PlasmaLab/HBT-EP-Boeckmann/HighFreqMLModeling/File_running/trimmed_HBT_analysis.ipynb',
    'untrimmed': '/Users/aboeckmann/Documents/Columbia/PlasmaLab/HBT-EP-Boeckmann/HighFreqMLModeling/File_running/untrimmed_HBT_analysis.ipynb'
}
OUTPUT_DIR = 'output_notebooks'
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
            'RESERVED_SHOT': RESERVED_SHOT
        }
        
        # Execute the notebook
        print(f"Executing {notebook_type} notebook for state {state}...")
        pm.execute_notebook(
            input_notebook,
            output_notebook,
            parameters=parameters,
            kernel_name='python3'
        )
        
        # Assume the notebook saves results to files (requires the save cell to be added)
        try:
            results[notebook_type][state]['true'] = np.load(output_true)
            results[notebook_type][state]['pred'] = np.load(output_pred)
            print(f"Loaded results for {notebook_type} state {state}")
        except FileNotFoundError:
            print(f"Result files not found for {notebook_type} state {state}. Ensure the notebook saves 'hbt_data' and 'predictions'.")

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

# Use the true data from one configuration (should be the same across all if RESERVED_SHOT is consistent)
true_data = None
for notebook_type in results:
    for state in results[notebook_type]:
        if results[notebook_type][state]['true'] is not None:
            true_data = results[notebook_type][state]['true']
            break
    if true_data is not None:
        break

if true_data is None:
    raise ValueError("No true data found for the reserved shot.")

plt.plot(true_data, 'k-', label='True ma2', linewidth=2)

# Plot predictions for each configuration
for notebook_type in results:
    for state in results[notebook_type]:
        if results[notebook_type][state]['pred'] is not None:
            key = f"{notebook_type}_{state}"
            plt.plot(results[notebook_type][state]['pred'], '--', color=colors[key], label=labels[key])

plt.xlabel('Frame Index')
plt.ylabel('ma2 (Original Scale)')
plt.title(f'Actual vs Predicted ma2 for Reserved Shot {RESERVED_SHOT}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print summary metrics
for notebook_type in results:
    for state in results[notebook_type]:
        if results[notebook_type][state]['true'] is not None and results[notebook_type][state]['pred'] is not None:
            errors = np.abs(results[notebook_type][state]['true'] - results[notebook_type][state]['pred']) / np.max(np.abs(results[notebook_type][state]['true'])) * 100
            print(f"{notebook_type} State {state} - Mean absolute percentage error: {np.mean(errors):.2f}%")