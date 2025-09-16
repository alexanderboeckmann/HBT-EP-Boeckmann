"""
HBT Prediction Comparison Script

This script compares HBT predictions between trimmed and untrimmed 
data analysis methods across different plasma states. It executes both analysis types for 
specified states and generates comparison plots.

Modes:
- mode1: States 1 & 3 with shot 119671
- mode2: States 2 & 3 with shot 114458

The script runs both trimmed and untrimmed analysis for each state, loads the results,
and creates visualization plots showing predictions vs true data with proper normalization.
"""

import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import tempfile
import json
from pathlib import Path

# Centralized project root and path utilities
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)

# Currently only doing mode amplitude 2 (ma2) analysis

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run HBT predictions for different state configurations.")
parser.add_argument('--mode', choices=['mode1', 'mode2'], default='mode1',
                    help='Execution mode: mode1 (states 1, 3 with shot 119671), mode2 (states 2, 3 with shot 114412)')
args = parser.parse_args()

# Configuration
STATES_MODE1 = [1, 3]  # States for mode1
STATES_MODE2 = [2, 3]  # States for mode2
EPOCHS = 20
RESERVED_SHOTS = {
    'mode1': {
        1: 119671,  # Shot for state 1 in mode1
        3: 119671   # Shot for state 3 in mode1
    },
    'mode2': {
        2: 114458,  # Shot for state 2 in mode2
        3: 114458   # Shot for state 3 in mode2
    }
}
SELECTED_DATA_TYPE = 'ma2'
NOTEBOOKS = {
    'trimmed': project_path('notebooks', 'trimmed_HBT_analysis.py'),
    'untrimmed': project_path('notebooks', 'untrimmed_HBT_analysis_cleaned.py')
}
OUTPUT_DIR = project_path('outputs', 'notebooks')
FIGURE_FILENAME = project_path('outputs', f"hbt_prediction_comparison_{args.mode}.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Select states and shots based on mode
STATES = STATES_MODE1 if args.mode == 'mode1' else STATES_MODE2
shots = RESERVED_SHOTS[args.mode]
print(f"Running in {args.mode} with states {STATES} and shots {shots}")

# Dictionary to store results
results = {
    'trimmed': {state: {'true': None, 'pred': None, 'time': None} for state in STATES},
    'untrimmed': {state: {'true': None, 'pred': None, 'time': None} for state in STATES}
}

# Execute notebooks for each configuration
for notebook_type in NOTEBOOKS:
    for state in STATES:
        input_notebook = NOTEBOOKS[notebook_type]
        output_notebook = os.path.join(OUTPUT_DIR, f'{notebook_type}_state_{state}_output.ipynb')
        output_true = project_path('data', 'predictions', f'results_{notebook_type}_state_{state}_{SELECTED_DATA_TYPE}_true.npy')
        output_pred = project_path('data', 'predictions', f'results_{notebook_type}_state_{state}_{SELECTED_DATA_TYPE}_pred.npy')
        output_time = project_path('data', 'predictions', f'results_{notebook_type}_state_{state}_{SELECTED_DATA_TYPE}_time.npy')
        
        # Parameters to pass to the notebook
        parameters = {
            'state': state,
            'selected_data_type': SELECTED_DATA_TYPE,
            'RESERVED_SHOT': shots[state],
            'EPOCH_NUM': EPOCHS
        }
        
        # Execute the Python script
        try:
            print(f"Executing {notebook_type} script for state {state} with reserved shot {shots[state]}...")
            
            # Create a temporary parameters file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(parameters, f, indent=2)
                params_file = f.name
            
            # Execute the Python script with parameters
            result = subprocess.run([
                'python', input_notebook,
                '--state', str(state),
                '--selected_data_type', parameters['selected_data_type'],
                '--RESERVED_SHOT', str(parameters['RESERVED_SHOT']),
                '--EPOCH_NUM', str(parameters['EPOCH_NUM'])
            ], capture_output=True, text=True, cwd=PROJECT_ROOT)
            
            if result.returncode != 0:
                print(f"Error executing {notebook_type} script for state {state}: {result.stderr}")
                continue
            else:
                print(f"Successfully executed {notebook_type} script for state {state}")
            
            # Clean up temporary file
            os.unlink(params_file)
            
        except Exception as e:
            print(f"Error executing {notebook_type} script for state {state}: {str(e)}")
            continue
        
        # Load results
        try:
            if os.path.exists(output_true):
                results[notebook_type][state]['true'] = np.load(output_true)
                print(f"Loaded true data shape for {notebook_type} state {state}: {results[notebook_type][state]['true'].shape}")
            if os.path.exists(output_pred):
                results[notebook_type][state]['pred'] = np.load(output_pred)
                print(f"Loaded pred data shape for {notebook_type} state {state}: {results[notebook_type][state]['pred'].shape}")
            if os.path.exists(output_time):
                results[notebook_type][state]['time'] = np.load(output_time)
                print(f"Loaded time data shape for {notebook_type} state {state}: {results[notebook_type][state]['time'].shape}")
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

# Color table for plotting
color_table = {
    ('trimmed', 1): 'blue',
    ('untrimmed', 1): 'cyan',
    ('trimmed', 2): 'green',
    ('untrimmed', 2): 'lime',
    ('trimmed', 3): 'red',
    ('untrimmed', 3): 'orange',
}

# ---------- 1. Plot only one instance of unnormalized true data ----------
true_plotted = False
original_true = None
reference_time = None  # Store reference time for alignment
for state in STATES:
    if results['untrimmed'][state]['true'] is not None:
        true = results['untrimmed'][state]['true']
        original_true = true
        true_time = results['untrimmed'][state]['time']
        
        if true_time is None or len(true_time) != len(true):
            print(f"⚠️ Missing or mismatched time for true data (state {state}); using indices")
            true_time = np.arange(len(true))
        
        # Store reference time for aligning other plots
        reference_time = true_time
        
        plt.plot(true_time, true, '-', color='black', label=f'True ma2 (State {state}, Shot {shots[state]})')

        # Store scale for prediction normalization
        y_min, y_max = np.min(true), np.max(true)
        print(f"True signal (state {state}) - min: {y_min:.4f}, max: {y_max:.4f}")
        y_range = y_max - y_min if y_max != y_min else 1.0
        print(f"True signal range: {y_range:.4f}")
        true_plotted = True
        break

if not true_plotted:
    print("⚠️ No untrimmed true data found. Skipping true signal plot.")
    y_min, y_range = 0.0, 1.0  # Fallback values

# ---------- 2. Plot predictions (adjusted for no normalization) ----------
for notebook_type in results:
    for state in STATES:
        pred = results[notebook_type][state]['pred']
        pred_time = results[notebook_type][state]['time']

        if pred is None:
            continue

        # Handle dimension mismatch between time and prediction arrays
        if pred_time is not None and len(pred_time) != len(pred):
            print(f"⚠️ Time/prediction dimension mismatch for {notebook_type} state {state}: time={len(pred_time)}, pred={len(pred)}")
            # Downsample time array to match prediction array length
            if len(pred_time) > len(pred):
                # Downsample time array to match prediction length
                original_length = len(pred_time)
                indices = np.linspace(0, len(pred_time) - 1, len(pred), dtype=int)
                pred_time = pred_time[indices]
                print(f"   Downsampled time array from {original_length} to {len(pred)} points")
            else:
                # If time is shorter, create indices
                pred_time = np.arange(len(pred))
                print(f"   Created time indices for {len(pred)} points")
        elif pred_time is None:
            # Create time array if missing
            pred_time = np.arange(len(pred))
            print(f"⚠️ Missing time data for {notebook_type} state {state}; using indices")
        
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
                print(true_range)
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
        label = f"Pred State {state} ({notebook_type})"

        plt.plot(pred_time, pred_denorm, '--', color=color, label=label)

# ---------- 3. Final touches ----------
plt.title(f'HBT {SELECTED_DATA_TYPE}: Predictions vs True (Mode: {args.mode}, Shots: {shots})')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_FILENAME)
plt.show()

print(f"\n✅ Figure saved as '{FIGURE_FILENAME}'")

# Print summary metrics
for notebook_type in results:
    for state in results[notebook_type]:
        if results[notebook_type][state]['true'] is not None and results[notebook_type][state]['pred'] is not None:
            errors = np.abs(results[notebook_type][state]['true'] - results[notebook_type][state]['pred']) / (np.max(np.abs(results[notebook_type][state]['true'])) + 1e-8) * 100
            print(f"{notebook_type.capitalize()} State {state} (Shot {shots[state]}) - MAPE: {np.mean(errors):.2f}%")