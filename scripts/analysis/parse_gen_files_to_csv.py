"""
Optimization Results Parser Script

This script parses genetic algorithm optimization results from individual generation
files and consolidates them into a single CSV file for analysis. It extracts
parameter configurations, performance metrics, and other relevant data from
the optimization run directories.

Features:
- Scans optimization directories for individual result files
- Extracts parameters and performance metrics from each generation
- Consolidates data into a single CSV file for analysis
- Handles missing or corrupted result files gracefully
- Provides data restoration capabilities for incomplete runs
"""

import pandas as pd
import re
import ast
import os
import glob
import numpy as np
import shutil
from pathlib import Path

# Centralized project root and path utilities
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)

# Configuration
RUN_DIR = project_path('data', 'optimization_results', 'run_20250804_150431')  # Specify your run directory here
OUTPUT_CSV = 'hbt_optimization_results_restored.csv'

def compute_mape(true_data, pred_data):
    """Calculate MAPE between true and predicted arrays."""
    if true_data is None or pred_data is None or len(true_data) == 0 or len(pred_data) == 0:
        return None
    true = true_data.flatten()
    pred = pred_data.flatten()
    min_length = min(len(true), len(pred))
    errors = np.abs((true[:min_length] - pred[:min_length]) / (np.max(np.abs(true[:min_length])) + 1e-8))
    return np.mean(errors) * 100

def load_array(file_path):
    """Load a NumPy array with detailed error handling."""
    if not os.path.exists(file_path):
        return None
    try:
        return np.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def parse_executing_line(line):
    """Parse an 'Executing notebook with parameters' line from gen*.txt and extract parameters."""
    # Regular expression to match the Executing line and capture parameters
    pattern = (
        r"Executing notebook with parameters: \{"
        r"'state': (?P<state>\d+), "
        r"'selected_data_type': 'ma2', "
        r"'RESERVED_SHOT': (?P<reserved_shot>\d+), "
        r"'EPOCH_NUM': (?P<epochs>\d+), "
        r"'VALIDATION_SPLIT': (?P<validation_split>[\d.]+), "
        r"'ACTIVATION_FUNC': '(?P<activation_func>[^']+)', "
        r"'LOSS_FUNC': '(?P<loss_func>[^']+)', "
        r"'OPTIMIZER_FUNC': '(?P<optimizer_func>[^']+)', "
        r"'OUTLIER_CUTOFF': (?P<outlier_cutoff>[\d.]+), "
        r"'NUM_CONV2D_LAYERS': (?P<num_conv2d_layers>\d+), "
        r"'NUM_DENSE_LAYERS': (?P<num_dense_layers>\d+), "
        r"'CONV2D_NEURONS': (?P<conv2d_neurons>\[.*?\]), "
        r"'CONV2D_SIZE': (?P<conv2d_size>\[.*?\]), "
        r"'DENSE_LAYER_NEURONS': (?P<dense_layer_neurons>\[.*?\]), "
        r"'MAX_POOLING_SIZE': (?P<max_pooling_size>\([\d, ]+\))"
        r"\}"
    )
    match = re.match(pattern, line.strip())
    if not match:
        return None
    
    params = match.groupdict()
    
    # Convert string representations to appropriate types
    params['state'] = int(params['state'])
    params['reserved_shot'] = int(params['reserved_shot'])
    params['epochs'] = int(params['epochs'])
    params['validation_split'] = float(params['validation_split'])
    params['outlier_cutoff'] = float(params['outlier_cutoff'])
    params['num_conv2d_layers'] = int(params['num_conv2d_layers'])
    params['num_dense_layers'] = int(params['num_dense_layers'])
    
    # Safely evaluate list and tuple strings
    try:
        params['conv2d_neurons'] = ast.literal_eval(params['conv2d_neurons'])
        params['conv2d_size'] = ast.literal_eval(params['conv2d_size'])
        params['dense_layer_neurons'] = ast.literal_eval(params['dense_layer_neurons'])
        params['max_pooling_size'] = ast.literal_eval(params['max_pooling_size'])
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing lists/tuples for parameters: {e}")
        return None
    
    return params

def get_individual_id_and_notebook_type(line_number, lines):
    """Find the individual ID and notebook type by searching backward from the Executing line."""
    for i in range(line_number - 1, -1, -1):
        if lines[i].startswith('Evaluating'):
            # Extract ID from "Evaluating X/40 (ID: ...)"
            match = re.search(r"Evaluating \d+/\d+ \(ID: ([^\)]+)\)", lines[i])
            if match:
                individual_id = match.group(1)
                # Look for the corresponding Success or failure line to infer notebook type
                for j in range(line_number, len(lines)):
                    if lines[j].startswith('Success for') and individual_id in lines[j]:
                        # Extract notebook_type from Success line
                        success_match = re.search(r"type=([^\s]+)", lines[j])
                        if success_match:
                            return individual_id, success_match.group(1)
                    elif individual_id in lines[j] and ('Missing result files' in lines[j] or 'Invalid values' in lines[j] or 'Mismatched state' in lines[j]):
                        # Infer notebook_type from true/pred file if exists
                        dir_path = os.path.join(RUN_DIR, f"individual_{individual_id}")
                        npy_files = glob.glob(os.path.join(dir_path, "*.npy"))
                        for f in npy_files:
                            if 'true.npy' in f:
                                base_name = os.path.basename(f).replace('_true.npy', '')
                                parts = base_name.split('_')
                                if len(parts) > 1:
                                    return individual_id, parts[1]  # e.g., 'untrimmed' or 'trimmed'
                        return individual_id, None
                return individual_id, None
    return None, None

def delete_non_gen1_individuals(run_dir):
    """Delete individual directories not referenced in gen1.txt."""
    gen1_file = os.path.join(run_dir, 'gen1.txt')
    if not os.path.exists(gen1_file):
        print(f"gen1.txt not found in {run_dir}. Skipping deletion.")
        return
    
    # Extract individual IDs from gen1.txt
    gen1_ids = set()
    with open(gen1_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('Evaluating'):
                match = re.search(r"Evaluating \d+/\d+ \(ID: ([^\)]+)\)", line)
                if match:
                    gen1_ids.add(match.group(1))
    
    print(f"Found {len(gen1_ids)} individual IDs in gen1.txt: {gen1_ids}")
    
    # Find all individual directories
    individual_dirs = glob.glob(os.path.join(run_dir, "individual_*"))
    deleted_count = 0
    
    for dir_path in individual_dirs:
        # Extract individual ID from directory name
        dir_name = os.path.basename(dir_path)
        individual_id = dir_name[len("individual_"):] if dir_name.startswith("individual_") else None
        
        if individual_id and individual_id not in gen1_ids:
            try:
                shutil.rmtree(dir_path)
                print(f"Deleted directory: {dir_path}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting directory {dir_path}: {e}")
    
    print(f"Deleted {deleted_count} individual directories not in gen1.txt.")

def create_csv_from_gen_files(run_dir, output_csv):
    """Read all gen*.txt files in the run directory, compute MAPE, and create a single CSV."""
    # Delete individuals not in gen1.txt before processing
    delete_non_gen1_individuals(run_dir)
    
    # Define the expected columns for the CSV
    columns = [
        'generation', 'individual_id', 'notebook_type', 'state', 'reserved_shot',
        'epochs', 'validation_split', 'activation_func', 'loss_func',
        'optimizer_func', 'outlier_cutoff', 'num_conv2d_layers',
        'num_dense_layers', 'conv2d_neurons', 'conv2d_size',
        'dense_layer_neurons', 'max_pooling_size', 'mape'
    ]
    
    # Initialize an empty list to store parsed data
    data = []
    
    # Find all gen*.txt files in the run directory
    gen_files = sorted(glob.glob(os.path.join(run_dir, 'gen*.txt')))
    if not gen_files:
        print(f"No gen*.txt files found in {run_dir}.")
        return
    
    # Process each gen file
    for gen_file in gen_files:
        # Extract generation number from filename (e.g., gen1.txt → 1)
        gen_match = re.match(r'gen(\d+)\.txt', os.path.basename(gen_file))
        if not gen_match:
            print(f"Skipping file with invalid name format: {gen_file}")
            continue
        generation = int(gen_match.group(1))
        
        print(f"Processing file: {gen_file} (Generation {generation})")
        
        # Read the file
        with open(gen_file, 'r') as f:
            lines = f.readlines()
        
        # Process each line
        for i, line in enumerate(lines):
            if line.startswith('Executing notebook with parameters'):
                params = parse_executing_line(line)
                if not params:
                    print(f"Skipping invalid executing line in {gen_file} at line {i+1}")
                    continue
                
                # Get individual_id and notebook_type
                individual_id, notebook_type = get_individual_id_and_notebook_type(i, lines)
                if not individual_id:
                    print(f"Could not find individual ID for line {i+1} in {gen_file}")
                    continue
                if not notebook_type:
                    print(f"Could not determine notebook_type for ID {individual_id}")
                    notebook_type = 'unknown'
                
                # Compute MAPE by loading true and pred files
                dir_path = os.path.join(run_dir, f"individual_{individual_id}")
                npy_files = glob.glob(os.path.join(dir_path, "*.npy"))
                true_file = None
                pred_file = None
                for f in npy_files:
                    if 'true.npy' in f:
                        true_file = f
                    elif 'pred.npy' in f:
                        pred_file = f
                
                mape = None
                if true_file and pred_file:
                    true_data = load_array(true_file)
                    pred_data = load_array(pred_file)
                    if true_data is not None and pred_data is not None:
                        if not (np.any(np.isnan(true_data)) or np.any(np.isnan(pred_data)) or 
                                np.any(np.isinf(true_data)) or np.any(np.isinf(pred_data))):
                            mape = compute_mape(true_data, pred_data)
                        else:
                            print(f"Invalid data (NaN/Inf) for ID {individual_id}")
                    else:
                        print(f"Failed to load data for ID {individual_id}: true={true_file}, pred={pred_file}")
                else:
                    print(f"Missing .npy files for ID {individual_id}: true={true_file}, pred={pred_file}")
                
                # Create result dictionary
                result = {
                    'generation': generation,
                    'individual_id': individual_id,
                    'notebook_type': notebook_type,
                    'state': params['state'],
                    'reserved_shot': params['reserved_shot'],
                    'epochs': params['epochs'],
                    'validation_split': params['validation_split'],
                    'activation_func': params['activation_func'],
                    'loss_func': params['loss_func'],
                    'optimizer_func': params['optimizer_func'],
                    'outlier_cutoff': params['outlier_cutoff'],
                    'num_conv2d_layers': params['num_conv2d_layers'],
                    'num_dense_layers': params['num_dense_layers'],
                    'conv2d_neurons': params['conv2d_neurons'],
                    'conv2d_size': params['conv2d_size'],
                    'dense_layer_neurons': params['dense_layer_neurons'],
                    'max_pooling_size': params['max_pooling_size'],
                    'mape': mape
                }
                data.append(result)
                print(f"Processed ID {individual_id}: MAPE={mape if mape is not None else 'None'}")
    
    if not data:
        print(f"No valid data processed from gen*.txt files in {run_dir}.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV in the run directory
    output_path = os.path.join(run_dir, output_csv)
    df.to_csv(output_path, index=False)
    print(f"CSV file created successfully: {output_path}")
    print(f"Total number of records: {len(df)}")
    print(f"Processed files: {', '.join(os.path.basename(f) for f in gen_files)}")

if __name__ == "__main__":
    # Ensure the run directory exists
    if not os.path.exists(RUN_DIR):
        print(f"Run directory {RUN_DIR} does not exist. Please update RUN_DIR in the script.")
    else:
        create_csv_from_gen_files(RUN_DIR, OUTPUT_CSV)