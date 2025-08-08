import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# ------------------------
# Command-line arguments
# ------------------------
parser = argparse.ArgumentParser(description="Plot HBT predictions from saved .npy files.")
parser.add_argument('--data_type', choices=['trimmed', 'untrimmed'], default='untrimmed',
                    help='Choose trimmed or untrimmed data')
args = parser.parse_args()

# ------------------------
# Configuration
# ------------------------
STATES = [1, 2, 3]
SELECTED_DATA_TYPE = 'ma2'
FIGURE_FILENAME = f"hbt_prediction_comparison_{args.data_type}.png"

def get_result_file(data_type, state, suffix):
    return f'results_{data_type}_state_{state}_{SELECTED_DATA_TYPE}_{suffix}.npy'

results = {
    args.data_type: {state: {'true': None, 'pred': None, 'time': None} for state in STATES}
}

# ------------------------
# Load .npy files
# ------------------------
for state in STATES:
    try:
        true_path = get_result_file(args.data_type, state, 'true')
        pred_path = get_result_file(args.data_type, state, 'pred')
        time_path = get_result_file(args.data_type, state, 'time')

        if os.path.exists(true_path):
            results[args.data_type][state]['true'] = np.load(true_path)
        if os.path.exists(pred_path):
            results[args.data_type][state]['pred'] = np.load(pred_path)
        if os.path.exists(time_path):
            results[args.data_type][state]['time'] = np.load(time_path)

        print(f"Loaded {args.data_type} state {state} data")
    except Exception as e:
        print(f"Error loading files for {args.data_type} state {state}: {e}")

# ------------------------
# Plotting
# ------------------------
plt.figure(figsize=(12, 8))

# ---------- 1. Plot true data from state 1 ----------
DEFAULT_FRAME_COUNT = 800
true_plotted = False
original_true = None
state = 1  # Changed to state 1 for true data
if results[args.data_type][state]['true'] is not None:
    true = results[args.data_type][state]['true']
    original_true = true
    downsampled_time = results[args.data_type][state]['time']
    
    if downsampled_time is None or len(downsampled_time) != len(true):
        print(f"⚠️  Missing or mismatched time for TRUE data (state {state}); using indices")
        downsampled_time = np.arange(len(true))
    plt.plot(downsampled_time, true, '-', color='black', label='Untrimmed ma2 Data', zorder=3)

    # Store scale for prediction normalization
    y_min, y_max = np.min(true), np.max(true)
    print(y_min)
    print(y_max)
    y_range = y_max - y_min if y_max != y_min else 1.0
    print(y_range)
    true_plotted = True
else:
    print(f"⚠️  No true data found for state {state}. Skipping true signal plot.")
    y_min, y_range = 0.0, 1.0  # fallback values

# ---------- 2. Plot predictions ----------
color_table = {
    1: 'blue',
    2: 'green',
    3: 'red',
}

for state in STATES:
    pred = results[args.data_type][state]['pred']
    pred_time = results[args.data_type][state]['time']

    if pred is None:
        continue

    # Debugging: Print prediction range
    print(f"State {state} - Pred min: {np.min(pred):.4f}, max: {np.max(pred):.4f}")

    if args.data_type == 'untrimmed':
        # Untrimmed predictions are close to true signal scale, plot as-is
        pred_denorm = pred
    else:
        # Trimmed predictions: Scale using state 1 true signal's range
        if original_true is not None and len(original_true) > 0:
            true_min = np.min(original_true)
            true_max = np.percentile(original_true, 95)  # Use 95th percentile
            true_range = true_max - true_min if true_max != true_min else 3.0
            pred_range = np.max(pred) - np.min(pred) if np.max(pred) != np.min(pred) else 1.0
            ma_norm = true_range / pred_range
            pred_denorm = pred * ma_norm
            pred_denorm = pred_denorm - np.min(pred_denorm)
            print(f"Computed ma_norm={ma_norm:.4f} for state {state}")
        else:
            pred_denorm = pred * y_range if true_plotted else pred
            pred_denorm = pred_denorm - np.min(pred_denorm)
            print(f"⚠️ No true data for state {state}, using {'true range' if true_plotted else 'raw prediction'}")

    color = color_table.get(state, 'gray')
    if state==3:
        label = "Data Set 1 and 2"
    else:
        label = f"Data Set {state}"

    plt.plot(pred_time, pred_denorm, '-', color=color, label=label)

# ---------- 3. Final touches ----------
#plt.title(f'HBT {SELECTED_DATA_TYPE}: Predictions vs True (Data: {args.data_type}, True: State 1)')
plt.title("Untrimmed Model Prediction Comparison")
plt.xlabel('Time (s)')
plt.ylabel('Mode Amplitude (G)')
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
for state in STATES:
    true = results[args.data_type][state]['true']
    pred = results[args.data_type][state]['pred']
    if true is not None and pred is not None:
        min_len = min(len(true), len(pred))
        mape = np.mean(np.abs(true[:min_len] - pred[:min_len]) / (np.max(np.abs(true[:min_len])) + 1e-8)) * 100
        print(f"State {state} - MAPE: {mape:.2f}%")