"""
Missing Results Analysis Script

This script analyzes optimization runs to identify missing or incomplete results.
It scans through optimization directories to find individuals that failed to complete
or produced incomplete output files.

Features:
- Scans optimization result directories for missing files
- Identifies failed or incomplete optimization runs
- Provides statistics on completion rates
- Helps debug optimization workflow issues
"""

import os
import numpy as np
from pathlib import Path

# Centralized project root and path utilities
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)

# Configuration
RUN_DIR = project_path('data', 'optimization_results', 'run_20250804_150431')

def print_npy_contents(file_path):
    """Print contents of a .npy file."""
    try:
        data = np.load(file_path)
        print(f"  Contents of {os.path.basename(file_path)}:")
        print(f"    Shape: {data.shape}")
        print(f"    Data type: {data.dtype}")
        print(f"    Sample (first 5 elements or less): {data.flatten()[:5]}")
        print(f"    Min: {np.min(data):.4f}, Max: {np.max(data):.4f}, Mean: {np.mean(data):.4f}")
    except Exception as e:
        print(f"    Error loading {os.path.basename(file_path)}: {e}")

def print_npz_contents(file_path):
    """Print contents of a .npz file."""
    try:
        with np.load(file_path) as data:
            print(f"  Contents of {os.path.basename(file_path)}:")
            for key in data.files:
                array = data[key]
                print(f"    Array '{key}':")
                print(f"      Shape: {array.shape}")
                print(f"      Data type: {array.dtype}")
                print(f"      Sample (first 5 elements or less): {array.flatten()[:5]}")
                print(f"      Min: {np.min(array):.4f}, Max: {np.max(array):.4f}, Mean: {np.mean(array):.4f}")
    except Exception as e:
        print(f"    Error loading {os.path.basename(file_path)}: {e}")

def list_result_files_contents(run_dir, individual_id):
    """List all files in the individual's directory and print contents of .npy/.npz files."""
    individual_dir = os.path.join(run_dir, f"individual_{individual_id}")
    
    # Check if directory exists
    if not os.path.exists(individual_dir):
        print(f"Directory not found for individual {individual_id}: {individual_dir}")
        return
    if not os.path.isdir(individual_dir):
        print(f"Path exists but is not a directory for individual {individual_id}: {individual_dir}")
        return
    
    # Check directory permissions
    try:
        files = sorted(os.listdir(individual_dir))
    except PermissionError:
        print(f"Permission denied accessing directory for individual {individual_id}: {individual_dir}")
        return
    except Exception as e:
        print(f"Error accessing directory for individual {individual_id}: {e}")
        return

    if not files:
        print(f"No files found in {individual_dir}")
        return

    print(f"\nFiles for individual {individual_id}:")
    for file in files:
        print(f"  {file}")
        file_path = os.path.join(individual_dir, file)
        if file.endswith('.npy'):
            print_npy_contents(file_path)
        elif file.endswith('.npz'):
            print_npz_contents(file_path)
        else:
            print(f"    Skipping non-.npy/.npz file")

def main():
    # Verify run directory
    if not os.path.exists(RUN_DIR):
        print(f"Run directory not found: {RUN_DIR}")
        return

    # List files and contents for the specified individuals
    individual_ids = [
        '38d893a4-3239-45fe-bebc-7c63d2b35965',
        '15effe14-53e2-439d-80b1-f8b1031d2071'
    ]
    for individual_id in individual_ids:
        print(f"\nChecking individual {individual_id}")
        list_result_files_contents(RUN_DIR, individual_id)

if __name__ == "__main__":
    main()