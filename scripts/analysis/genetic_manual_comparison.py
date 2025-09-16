"""
Genetic vs Manual Model Comparison Script

This script compares the performance of genetically optimized models against manually
configured models for HBT prediction tasks. It loads results from both optimization
runs and manual configurations to create comparative visualizations.

Features:
- Compares genetic algorithm optimized models vs manual parameter selection
- Loads results from optimization directories and manual prediction files
- Creates side-by-side comparison plots
- Calculates and displays performance metrics (MAPE, etc.)
- Supports both trimmed and untrimmed data analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
from pathlib import Path

# Centralized project root and path utilities
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)

# ------------------------
# Command-line arguments
# ------------------------
parser = argparse.ArgumentParser(description="Plot best HBT predictions: true, manual, and genetic models.")
parser.add_argument('--data_type', choices=['trimmed', 'untrimmed'], default='untrimmed',
                    help='Choose trimmed or untrimmed data')
parser.add_argument('--optimization_dir', default=project_path('data', 'optimization_results'),
                    help='Directory containing optimization results')
args = parser.parse_args()

# ------------------------
# Configuration
# ------------------------
STATES = [1]  # Only need state 1 (Data Set 1) for true and manual data
SELECTED_DATA_TYPE = 'ma2'
FIGURE_FILENAME = project_path('outputs', f"hbt_best_model_comparison_{args.data_type}.png")
CSV_FILENAME = 'hbt_optimization_results.csv'

# Initialize results dictionary
results = {
    args.data_type: {state: {'true': None, 'pred': None, 'time': None} for state in STATES},
    'genetic': {'pred': None, 'time': None, 'state': None, 'individual': None}  # For best genetic model with dynamic state and individual
}

# ------------------------
# Find best genetic model
# ------------------------
def find_best_genetic_model(optimization_dir):
    """Find the best genetic model from the latest run directory."""
    # Find latest run directory
    run_dirs = [d for d in os.listdir(optimization_dir) if d.startswith('run_') and os.path.isdir(os.path.join(optimization_dir, d))]
    if not run_dirs:
        print(f"No run directories found in {optimization_dir}")
        return None

    run_dirs.sort(key=lambda x: os.path.getctime(os.path.join(optimization_dir, x)), reverse=True)
    for run_dir in run_dirs:
        csv_path = os.path.join(optimization_dir, run_dir, CSV_FILENAME)
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    # Find row with lowest MAPE
                    best_row = df.loc[df['mape'].idxmin()]
                    state = int(best_row.get('state', 3))  # Default to 3 if 'state' column not found
                    individual_id = best_row.get('individual_id', None)  # Try to get individual_id from CSV
                    if individual_id is None:
                        # Fallback: Use the only individual directory if one exists
                        individual_dir = os.path.join(optimization_dir, run_dir)
                        individual_subdirs = [d for d in os.listdir(individual_dir) if d.startswith('individual_')]
                        if len(individual_subdirs) == 1:
                            individual_id = individual_subdirs[0].replace('individual_', '')  # Extract UUID
                        else:
                            print(f"Multiple or no individual directories found in {individual_dir}. Please ensure 'individual_id' is in CSV.")
                            return None
                    else:
                        individual_id = f"individual_{individual_id}"  # Prefix with 'individual_' to match directory format
                    print(f"Best genetic model from {run_dir}: Individual {individual_id}, State {state}, MAPE = {best_row['mape']:.2f}%")
                    return os.path.join(optimization_dir, run_dir), best_row, state, individual_id
            except Exception as e:
                print(f"Error reading CSV in {run_dir}: {e}")
                continue
    print(f"No valid CSV files found in {optimization_dir}")
    return None

# Load best genetic model predictions based on naming scheme
best_run_dir, best_row, genetic_state, individual_id = find_best_genetic_model(args.optimization_dir) if os.path.exists(args.optimization_dir) else (None, None, None, None)
if best_run_dir:
    # Use the naming scheme with dynamic state and individual subdirectory
    individual_dir = os.path.join(best_run_dir, individual_id)
    genetic_pred_path = os.path.join(individual_dir, f'results_{args.data_type}_state_{genetic_state}_{SELECTED_DATA_TYPE}_pred.npy')
    genetic_time_path = os.path.join(individual_dir, f'results_{args.data_type}_state_{genetic_state}_{SELECTED_DATA_TYPE}_time.npy')
    
    if os.path.exists(genetic_pred_path):
        results['genetic']['pred'] = np.load(genetic_pred_path)
        results['genetic']['state'] = genetic_state
        results['genetic']['individual'] = individual_id
        print(f"Loaded best genetic model predictions from {genetic_pred_path}")
    if os.path.exists(genetic_time_path):
        results['genetic']['time'] = np.load(genetic_time_path)
        print(f"Loaded best genetic model time from {genetic_time_path}")
else:
    print("⚠️ No best genetic model found. Plotting only true and manual models.")

# ------------------------
# Load true and manual model data
# ------------------------
for state in STATES:
    try:
        true_path = project_path('data', 'predictions', f'results_{args.data_type}_state_{state}_{SELECTED_DATA_TYPE}_true.npy')
        pred_path = project_path('data', 'predictions', f'results_{args.data_type}_state_{state}_{SELECTED_DATA_TYPE}_pred.npy')
        time_path = project_path('data', 'predictions', f'results_{args.data_type}_state_{state}_{SELECTED_DATA_TYPE}_time.npy')

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
state = 1
true_plotted = False
original_true = None
reference_time = None  # Store reference time for alignment
if results[args.data_type][state]['true'] is not None:
    true = results[args.data_type][state]['true']
    original_true = true
    true_time = results[args.data_type][state]['time']
    
    if true_time is None or len(true_time) != len(true):
        print(f"⚠️ Missing or mismatched time for TRUE data (state {state}); using indices")
        true_time = np.arange(len(true))
    
    # Store reference time for aligning other plots
    reference_time = true_time
    
    plt.plot(true_time, true, '-', color='black', label='Untrimmed ma2, Data Set 1', zorder=3)
    
    # Store scale for prediction normalization
    y_min, y_max = np.min(true), np.max(true)
    y_range = y_max - y_min if y_max != y_min else 1.0
    true_plotted = True
else:
    print(f"⚠️ No true data found for state {state}. Skipping true signal plot.")
    y_min, y_range = 0.0, 1.0  # Fallback values

# ---------- 2. Plot manual model predictions (state 1, untrimmed) ----------
if results[args.data_type][state]['pred'] is not None:
    pred = results[args.data_type][state]['pred']
    pred_time = results[args.data_type][state]['time']
    
    if pred_time is None or len(pred_time) != len(pred):
        print(f"⚠️ Missing or mismatched time for manual model (state {state}); using indices")
        pred_time = np.arange(len(pred))
    
    # Align time arrays with reference time if available
    if reference_time is not None and pred_time is not None:
        # Scale prediction time to match reference time range
        if len(pred_time) == len(reference_time):
            # If same length, use reference time directly for proper alignment
            pred_time = reference_time
        else:
            # Scale to match reference time range
            ref_min, ref_max = reference_time[0], reference_time[-1]
            pred_min, pred_max = pred_time[0], pred_time[-1]
            if pred_max != pred_min:
                pred_time = ref_min + (pred_time - pred_min) * (ref_max - ref_min) / (pred_max - pred_min)
            else:
                pred_time = np.linspace(ref_min, ref_max, len(pred_time))
    
    # Untrimmed predictions are close to true signal scale, plot as-is
    pred_denorm = pred
    plt.plot(pred_time, pred_denorm, '-', color='blue', label='Manual Model, Data Set 1')
else:
    print(f"⚠️ No manual model predictions found for state {state}.")

# ---------- 3. Plot best genetic model predictions ----------
if results['genetic']['pred'] is not None:
    pred = results['genetic']['pred']
    pred_time = results['genetic']['time']
    genetic_state = results['genetic']['state']
    individual_id = results['genetic']['individual']
    
    if pred_time is None or len(pred_time) != len(pred):
        print(f"⚠️ Missing or mismatched time for genetic model (state {genetic_state}); using indices")
        pred_time = np.arange(len(pred))
    
    # Align time arrays with reference time if available
    if reference_time is not None and pred_time is not None:
        # Scale prediction time to match reference time range
        if len(pred_time) == len(reference_time):
            # If same length, use reference time directly for proper alignment
            pred_time = reference_time
        else:
            # Scale to match reference time range
            ref_min, ref_max = reference_time[0], reference_time[-1]
            pred_min, pred_max = pred_time[0], pred_time[-1]
            if pred_max != pred_min:
                pred_time = ref_min + (pred_time - pred_min) * (ref_max - ref_min) / (pred_max - pred_min)
            else:
                pred_time = np.linspace(ref_min, ref_max, len(pred_time))
    
    # Assume genetic model predictions are untrimmed and follow same scaling
    pred_denorm = pred
    plt.plot(pred_time, pred_denorm, '-', color='green', label=f'Best Genetic Model, Data Set {genetic_state}, (Individual {individual_id})')
else:
    print(f"⚠️ No genetic model predictions found. Check paths: {genetic_pred_path}, {genetic_time_path}")

# ---------- 4. Final touches ----------
plt.title("Best Model Comparison: True ma2, Manual, and Genetic Models")
plt.xlabel('Time (s)')
plt.ylabel('Mode Amplitude (G)')
plt.grid(True)
plt.legend(loc='upper left')  # Legend in top-left corner
plt.tight_layout()
plt.savefig(FIGURE_FILENAME, dpi=300)
plt.show()

print(f"\n✅ Figure saved as '{FIGURE_FILENAME}'")

# ------------------------
# Metrics
# ------------------------
print("\nSummary (Mean Absolute Percentage Error):")
# Manual model MAPE
if results[args.data_type][state]['true'] is not None and results[args.data_type][state]['pred'] is not None:
    true = results[args.data_type][state]['true']
    pred = results[args.data_type][state]['pred']
    min_len = min(len(true), len(pred))
    mape = np.mean(np.abs(true[:min_len] - pred[:min_len]) / (np.max(np.abs(true[:min_len])) + 1e-8)) * 100
    print(f"Manual Model (State {state}) - MAPE: {mape:.2f}%")

# Genetic model MAPE
if results[args.data_type][state]['true'] is not None and results['genetic']['pred'] is not None:
    true = results[args.data_type][state]['true']
    pred = results['genetic']['pred']
    min_len = min(len(true), len(pred))
    mape = np.mean(np.abs(true[:min_len] - pred[:min_len]) / (np.max(np.abs(true[:min_len])) + 1e-8)) * 100
    print(f"Best Genetic Model (State {genetic_state}, Individual {individual_id}) - MAPE: {mape:.2f}%")