import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# ------------------------
# Command-line arguments
# ------------------------
parser = argparse.ArgumentParser(description="Plot HBT predictions from saved .npy files.")
parser.add_argument('--mode', choices=['mode1', 'mode2'], default='mode1',
                    help='Choose mode1 (states 1, 3) or mode2 (states 2, 3)')
args = parser.parse_args()

# ------------------------
# Configuration
# ------------------------
STATES = [1, 3] if args.mode == 'mode1' else [2, 3]
SELECTED_DATA_TYPE = 'ma2'
FIGURE_FILENAME = f"hbt_prediction_comparison_{args.mode}.png"

def get_result_file(notebook_type, state, result_type):
    return f'results_{notebook_type}_state_{state}_{SELECTED_DATA_TYPE}_{result_type}.npy'

results = {
    'trimmed': {state: {'true': None, 'pred': None} for state in STATES},
    'untrimmed': {state: {'true': None, 'pred': None} for state in STATES}
}

# ------------------------
# Load .npy files
# ------------------------
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

# ------------------------
# Plotting
# ------------------------
plt.figure(figsize=(12, 8))

for notebook_type in results:
    for state in STATES:
        true = results[notebook_type][state]['true']
        pred = results[notebook_type][state]['pred']
        if true is None or pred is None:
            print(f"Skipping {notebook_type} state {state} due to missing data")
            continue

        # Normalize prediction to match the range of its own true data
        true_min, true_max = np.min(true), np.max(true)
        true_range = true_max - true_min
        norm_pred = true_min + (pred - np.min(pred)) * true_range / (np.max(pred) - np.min(pred))

        # Plot true and predicted
        color = {
            (1, 'trimmed'): 'b', (1, 'untrimmed'): 'c',
            (2, 'trimmed'): 'g', (2, 'untrimmed'): 'm',
            (3, 'trimmed'): 'r', (3, 'untrimmed'): 'y'
        }.get((state, notebook_type), 'k')

        label = f"{notebook_type.capitalize()} State {state}"
        linestyle = '-' if 'true' in label else '--'

        plt.plot(true, '-', color=color, label=f"True State {state} ({notebook_type})")
        plt.plot(norm_pred, '--', color=color, label=f"Pred State {state} ({notebook_type})")

plt.title(f'HBT {SELECTED_DATA_TYPE}: Saved Predictions vs True Data (Mode: {args.mode})')
plt.xlabel('Frame Index')
plt.ylabel('Amplitude (Normalized per state)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_FILENAME)
plt.show()

print(f"\nFigure saved as '{FIGURE_FILENAME}'")

# ------------------------
# Metrics
# ------------------------
print("\nSummary (Mean Absolute Percentage Error):")
for notebook_type in results:
    for state in STATES:
        true = results[notebook_type][state]['true']
        pred = results[notebook_type][state]['pred']
        if true is not None and pred is not None:
            mape = np.mean(np.abs(true - pred) / np.max(np.abs(true))) * 100
            print(f"{notebook_type.capitalize()} State {state} - MAPE: {mape:.2f}%")
