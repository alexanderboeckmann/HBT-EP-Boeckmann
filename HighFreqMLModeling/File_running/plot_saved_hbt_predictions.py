import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
STATES = [1, 3]
SELECTED_DATA_TYPE = 'ma2'
OUTPUT_DIR = 'output_notebooks'

# File name templates
def get_result_file(notebook_type, state, result_type):
    return f'results_{notebook_type}_state_{state}_{SELECTED_DATA_TYPE}_{result_type}.npy'

results = {
    'trimmed': {state: {'true': None, 'pred': None} for state in STATES},
    'untrimmed': {state: {'true': None, 'pred': None} for state in STATES}
}

# Load all available .npy files
for notebook_type in results:
    for state in STATES:
        try:
            true_path = get_result_file(notebook_type, state, 'true')
            pred_path = get_result_file(notebook_type, state, 'pred')
            if os.path.exists(true_path):
                results[notebook_type][state]['true'] = np.load(true_path)
            if os.path.exists(pred_path):
                results[notebook_type][state]['pred'] = np.load(pred_path)
            print(f"Loaded {notebook_type} state {state} data")
        except Exception as e:
            print(f"Error loading files for {notebook_type} state {state}: {e}")

# Use state 1 trimmed true data to normalize
true_data = results['trimmed'][1]['true']
if true_data is None:
    raise ValueError("Missing true data for trimmed state 1")

true_min = np.min(true_data)
true_max = np.max(true_data)
true_range = true_max - true_min

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(true_data, 'k-', label='True ma2 (State 1, Trimmed)')

for notebook_type in results:
    for state in STATES:
        pred = results[notebook_type][state]['pred']
        if pred is not None:
            # Normalize
            norm_pred = true_min + (pred - np.min(pred)) * true_range / (np.max(pred) - np.min(pred))
            linestyle = '--' if notebook_type == 'trimmed' else ':'
            color = {'trimmed': 'b', 'untrimmed': 'r'}[notebook_type] if state == 1 else {'trimmed': 'g', 'untrimmed': 'm'}[notebook_type]
            label = f"{notebook_type.capitalize()} State {state}"
            plt.plot(norm_pred, linestyle, color=color, label=label)

plt.legend()
plt.title('HBT ma2: Saved Predictions vs True Data (Normalized to State 1 True Range)')
plt.xlabel('Frame Index')
plt.ylabel('Normalized Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()

# Print summary metrics
for notebook_type in results:
    for state in STATES:
        pred = results[notebook_type][state]['pred']
        true = results[notebook_type][state]['true']
        if pred is not None and true is not None:
            mape = np.mean(np.abs(true - pred) / np.max(np.abs(true))) * 100
            print(f"{notebook_type.capitalize()} State {state} - Mean absolute percentage error: {mape:.2f}%")
