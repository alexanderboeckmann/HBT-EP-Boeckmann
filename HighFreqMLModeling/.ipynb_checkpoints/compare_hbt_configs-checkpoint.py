#!/usr/bin/env python
# coding: utf-8

import importlib.util
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from contextlib import contextmanager
import io

@contextmanager
def suppress_stdout():
    """Temporarily suppress stdout to reduce clutter during script execution."""
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = original_stdout

def run_hbt_analysis(script_path, state, reserved_shot=None):
    """
    Run the HBT analysis script with the specified state and reserved shot.
    Returns the mean absolute percentage error for the reserved shot.
    """
    # Load the script as a module
    spec = importlib.util.spec_from_file_location("hbt_analysis", script_path)
    hbt_analysis = importlib.util.module_from_spec(spec)
    sys.modules["hbt_analysis"] = hbt_analysis

    # Override state and RESERVED_SHOT in the module
    hbt_analysis.state = state
    if reserved_shot is not None:
        hbt_analysis.RESERVED_SHOT = reserved_shot
    else:
        # Use a default reserved shot for consistency
        if state == 1:
            hbt_analysis.RESERVED_SHOT = 119671
        elif state == 2:
            hbt_analysis.RESERVED_SHOT = 114412
        elif state == 3:
            hbt_analysis.RESERVED_SHOT = 119671

    # Execute the script with suppressed output
    with suppress_stdout():
        spec.loader.exec_module(hbt_analysis)

    # Extract the MAPE from the reserved shot predictions
    # Compute MAPE manually since plot_reserved_shot_predictions prints it
    shot_idx = hbt_analysis.shot_list.index(hbt_analysis.RESERVED_SHOT)
    camera_data = hbt_analysis.reserved_shot_cut_2d
    hbt_data = hbt_analysis.hbt_ma2_data[hbt_analysis.valid_shots.index(hbt_analysis.RESERVED_SHOT)][:, 0]
    input_data = np.array(camera_data).reshape(-1, 32, 32, 1)
    predictions = hbt_analysis.william_model.predict(input_data, verbose=0)[:, 0] * hbt_analysis.ma_norm
    prediction_errors = np.abs(hbt_data - predictions) / hbt_analysis.ma_norm * 100
    mape = np.mean(prediction_errors)

    return mape

def compare_configurations():
    """Run all configurations and generate a comparison plot."""
    configurations = [
        ('trimmed', 1), ('trimmed', 2), ('trimmed', 3),
        ('untrimmed', 1), ('untrimmed', 2), ('untrimmed', 3)
    ]
    mapes = []
    labels = []

    for trim_type, state in configurations:
        script_path = f"{trim_type}_HBT_analysis.py"
        if not os.path.exists(script_path):
            print(f"Script {script_path} not found. Skipping.")
            continue
        print(f"Running {trim_type} HBT analysis for state {state}...")
        mape = run_hbt_analysis(script_path, state)
        mapes.append(mape)
        labels.append(f"{trim_type.capitalize()} State {state}")
        print(f"MAPE for {trim_type} State {state}: {mape:.2f}%")

    # Generate bar plot
    plt.figure(figsize=(10, 6))
    x = np.arange(len(mapes))
    plt.bar(x, mapes, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    plt.xticks(x, labels, rotation=45)
    plt.ylabel('Mean Absolute Percentage Error (%)')
    plt.title('HBT Model Performance Across Configurations')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_configurations()