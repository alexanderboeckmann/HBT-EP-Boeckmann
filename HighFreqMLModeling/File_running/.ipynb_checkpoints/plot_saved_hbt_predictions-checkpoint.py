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

def get_result_file(notebook_type, state, suffix):
    return f'results_{notebook_type}_state_{state}_{SELECTED_DATA_TYPE}_{suffix}.npy'

results = {
    'trimmed': {state: {'true': None, 'pred': None, 'time': None} for state in STATES},
    'untrimmed': {state: {'true': None, 'pred': None, 'time': None} for state in STATES}
}

# ------------------------
# Load .npy files
# ------------------------
for notebook_type in results:
    for state in STATES:
        try:
            true_path = get_result_file(notebook_type, state, 'true')
            pred_path = get_result_file(notebook_type, state, 'pred')
            time_path = get_result_file(notebook_type, state, 'time')

            if os.path.exists(true_path):
                results[notebook_type][state]['true'] = np.load(true_path)
            if os.path.exists(pred_path):
                results[notebook_type][state]['pred'] = np.load(pred_path)
            if os.path.exists(time_path):
                results[notebook_type][state]['time'] = np.load(time_path)

            print(f"Loaded {notebook_type} state {state} data")
        except Exception as e:
            print(f"Error loading files for {notebook_type} state {state}: {e}")

# ------------------------
# Plotting
# ------------------------
plt.figure(figsize=(12, 8))

# ---------- 1. Plot only one instance of unnormalized true data ----------
DEFAULT_FRAME_COUNT = 800
true_plotted = False
original_true = None
for state in STATES:
    if results['untrimmed'][state]['true'] is not None:
        true = results['untrimmed'][state]['true']
        original_true = true
        downsampled_time = results['untrimmed'][state]['time'] # come in at 800
        
        if downsampled_time is None or len(downsampled_time) != len(true):
            print(f"⚠️  Missing or mismatched time for TRUE data; using indices")
            downsampled_time = np.arange(len(true))
        plt.plot(downsampled_time, true, '-', color='black', label='True Signal (Untrimmed)')

        # Store scale for prediction normalization
        y_min, y_max = np.min(true), np.max(true)
        print(y_min)
        print(y_max)
        y_range = y_max - y_min if y_max != y_min else 1.0
        print(y_range)
        true_plotted = True
        break

if not true_plotted:
    print("⚠️  No untrimmed true data found. Skipping true signal plot.")
    y_min, y_range = 0.0, 1.0  # fallback values

# ---------- 2. Plot predictions (adjusted for no normalization) ----------
color_table = {
    ('trimmed', 1): 'blue',
    ('untrimmed', 1): 'magenta',
    ('trimmed', 2): 'green',
    ('untrimmed', 2): 'cyan',
    ('trimmed', 3): 'red',
    ('untrimmed', 3): 'orange',
}

for notebook_type in results:
    for state in STATES:
        pred = results[notebook_type][state]['pred']
        pred_time = results[notebook_type][state]['time']

        if pred is None:
            continue

        # Debugging: Print prediction range
        print(f"{notebook_type} State {state} - Pred min: {np.min(pred):.4f}, max: {np.max(pred):.4f}")

        if notebook_type == 'untrimmed':
            # Untrimmed predictions are close to true signal scale, plot as-is
            pred_denorm = pred
        else:
            # Trimmed predictions: Scale using original_true signal's range or recompute ma_norm
            if original_true is not None and len(original_true) > 0:
                # Compute true_range using the 95th percentile for the max
                true_min = np.min(original_true)
                true_max = np.percentile(original_true, 95)  # Use 95th percentile instead of max
                true_range = true_max - true_min if true_max != true_min else 3.0
                pred_range = np.max(pred) - np.min(pred) if np.max(pred) != np.min(pred) else 1.0
                #print(true_range)
                ma_norm = true_range / pred_range
                pred_denorm = pred * ma_norm
                # Shift pred_denorm to have a floor at 0
                pred_denorm = pred_denorm - np.min(pred_denorm)
                print(f"Computed ma_norm={ma_norm:.4f} for {notebook_type} state {state}")
            else:
                # Fallback: Use true signal's range from untrimmed true data
                pred_denorm = pred * y_range if true_plotted else pred
                # Shift pred_denorm to have a floor at 0
                pred_denorm = pred_denorm - np.min(pred_denorm)
                print(f"⚠️ No true data for {notebook_type} state {state}, using {'true range' if true_plotted else 'raw prediction'}")

        color = color_table.get((notebook_type, state), 'gray')
        label = f"Data Set {state} ({notebook_type})"

        plt.plot(pred_time, pred_denorm, '-', color=color, label=label)



# ---------- 3. Final touches ----------
plt.title(f'HBT {SELECTED_DATA_TYPE}: Predictions vs True (Mode: {args.mode})')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_FILENAME, dpi=300)
plt.show()

print(f"\n✅ Figure saved as '{FIGURE_FILENAME}'")


# ------------------------
# Metrics
# ------------------------
print("\nSummary (Mean Absolute Percentage Error):")
for notebook_type in results:
    for state in STATES:
        true = results[notebook_type][state]['true']
        pred = results[notebook_type][state]['pred']
        if true is not None and pred is not None:
            min_len = min(len(true), len(pred))
            mape = np.mean(np.abs(true[:min_len] - pred[:min_len]) / (np.max(np.abs(true[:min_len])) + 1e-8)) * 100
            print(f"{notebook_type.capitalize()} State {state} - MAPE: {mape:.2f}%")
