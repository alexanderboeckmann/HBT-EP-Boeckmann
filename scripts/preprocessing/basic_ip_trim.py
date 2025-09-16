#!/usr/bin/env python
# coding: utf-8

"""
Basic IP Trim Script

This script finds two indices per shot based on plasma current data,
where there is extraneous noise unhelpful for modeling. It identifies
initial spikes and end jumps in the IP data to determine optimal
cutoff points for data preprocessing.

Key features:
- Loads IP data for specified shot lists
- Finds initial cutoff points to remove startup spikes
- Finds end cutoff points to remove shutdown jumps
- Supports multiple plasma states (1, 2, 3)
- Generates visualization plots showing the trimming process
- Saves plots to outputs directory

Usage:
    python basic_ip_trim.py --state 2 --output_dir outputs
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path


# Centralized project root and path utilities
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def load_ip_data(shot_list, file_paths):
    """
    Load IP data for a list of shots.
    
    Parameters:
    - shot_list: List of shot numbers
    - file_paths: Directory or list of directories containing IP data files
    
    Returns:
    - ip_data: Array of IP data for all shots
    """
    ip_data = []
    
    # Handle both single path and multiple paths
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    for shot in shot_list:
        shot_data = None
        for file_path in file_paths:
            try:
                shot_data = np.load(os.path.join(file_path, f'{shot}ip.npy'))
                break  # Found the data, no need to check other paths
            except FileNotFoundError:
                continue
        
        if shot_data is not None:
            ip_data.append(shot_data)
        else:
            print(f"Warning: Could not find IP data for shot {shot} in any of the provided paths")
    
    return np.array(ip_data)


def format_ip_data(data, target_length=800):
    """
    Format plasma current data to match target length through downsampling.
    
    Parameters:
    - data: Array of IP data for multiple shots
    - target_length: Target number of frames (default: 800)
    
    Returns:
    - formatted_data: Downsampled data with shape (n_shots, target_length, 1)
    """
    # Convert to array and get frame ratio
    data = np.asarray(data, dtype=float)
    frame_ratio = data[0].shape[0] // target_length
    
    # Reshape and downsample in one go
    data = np.reshape(data, (len(data), -1, 1))
    data = data[:,::frame_ratio,:]
    data = data[:,:target_length,:]
    return data


def find_initial_cutoff_index(ip_data, window_size=5, start_index=50):
    """
    Find initial cutoff index to remove startup spikes.
    
    Parameters:
    - ip_data: 1D array of IP values for a single shot
    - window_size: Size of the moving average window for smoothing
    - start_index: Starting index for peak detection
    
    Returns:
    - cutoff_index: Index after which data should be included
    """
    # Smooth the data with a moving average
    smoothed_ip = np.convolve(ip_data, np.ones(window_size)/window_size, mode='valid')
    # If points were removed from the smoothing, pad adds extra
    pad = window_size // 2
    smoothed_ip = np.pad(smoothed_ip, (pad, pad), mode='edge')
    # If padding adds too many, cut off to maintain size
    if len(smoothed_ip) > len(ip_data):
        smoothed_ip = smoothed_ip[:len(ip_data)]
    
    # Compute the difference (first derivative)
    diff = np.diff(smoothed_ip)
    
    # Find the first peak: where diff changes from positive to negative
    peak_index = None
    for i in range(start_index, len(diff)):
        if diff[i-1] > 0 and diff[i] < 0 and ip_data[i] > 0:
            peak_index = i
            break
    if peak_index is None:
        return 0  # No peak found, assume no spike or include all data
    
    # Find the next valley: where diff changes from negative to positive after peak
    valley_index = None
    for i in range(peak_index + 1, len(diff)):
        if diff[i-1] < 0 and diff[i] > 0 and ip_data[i] > 0:
            valley_index = i
            break
    if valley_index is None:
        return peak_index  # No valley found, cut after peak
    
    return valley_index


def find_end_cutoff_index(ip_data, window_size=5, jump_ratio=2.5, lookback_window=10, stability_window=30):
    """
    Detect the cutoff index before a sudden jump at the end of the IP data.
    
    Parameters:
    - ip_data: 1D array of IP values for a single shot
    - window_size: Size of the moving average window for smoothing
    - jump_ratio: Multiplier of the median derivative to detect a jump
    - lookback_window: Number of frames to look back for jump detection
    - stability_window: Number of frames to establish baseline median derivative
    
    Returns:
    - end_cutoff_index: Index before the sudden jump
    """
    # Smooth the data with a moving average
    smoothed_ip = np.convolve(ip_data, np.ones(window_size)/window_size, mode='valid')
    pad = window_size // 2
    smoothed_ip = np.pad(smoothed_ip, (pad, pad), mode='edge')
    if len(smoothed_ip) > len(ip_data):
        smoothed_ip = smoothed_ip[:len(ip_data)]
    
    # Compute the first derivative
    diff = np.diff(smoothed_ip)
    
    # Calculate the median derivative over the initial stability window as baseline
    if len(diff) > stability_window:
        baseline_median = np.median(diff[:stability_window])
    else:
        baseline_median = np.median(diff) if len(diff) > 0 else 0.0
    
    # Move forward to find the sudden jump
    for i in range(stability_window, len(diff) - lookback_window):
        # Look at the maximum derivative in the next lookback_window
        max_deriv = np.max(diff[i:i + lookback_window])
        if max_deriv > jump_ratio * abs(baseline_median) and diff[i] < max_deriv * 0.2:
            # Found the onset of a sudden jump; set cutoff before this point
            return max(0, i - window_size)  # Buffer to include some context
    
    # If no jump detected, check for rapid decrease
    for i in range(stability_window, len(diff)):
        if diff[i] < -jump_ratio * abs(baseline_median):
            # Found a rapid decrease; set cutoff before this point
            return max(0, i - window_size)  # Buffer to include some context
    
    return len(ip_data)  # No jump detected, use full length


def get_shot_list_and_path(state):
    """
    Get shot list and file path based on state.
    
    Parameters:
    - state: Plasma state (1, 2, or 3)
    
    Returns:
    - shot_list: List of shot numbers
    - file_path: Path to IP data files
    """
    if state == 1:
        shot_list = [119591, 119599, 119601, 119646, 119648, 119653, 119654, 119658, 119659,
                     119661, 119662, 119663, 119665, 119666, 119667, 119669, 119670, 119671,
                     119673, 119675, 119748, 119750, 119751, 119752, 119754, 119755, 119756,
                     119757, 119760, 119761, 119762, 119763, 119764, 119766, 119767, 119768,
                     119769]
        file_path = project_path('data', 'shots', 'new', 'ip_data')
    elif state == 2:
        shot_list = [114407,114408,114411,114412,114413,114415,114416,114417,114418,114419,114420,114422,114424,114425,114428,114429,114431,114432,114433,
                     114434,114435,114436,114438,114439,114441,114443,114444,114445,114448,114450,114451,114453,114454,114455,114456,114457,114458,114460,
                     114462,114464,114467,114468,114472,114473]
        file_path = project_path('data', 'shots', 'old')
    elif state == 3:
        # State 3 combines both state 1 and state 2 shots
        shot_list_1 = [119591, 119599, 119601, 119646, 119648, 119653, 119654, 119658, 119659,
                       119661, 119662, 119663, 119665, 119666, 119667, 119669, 119670, 119671,
                       119673, 119675, 119748, 119750, 119751, 119752, 119754, 119755, 119756,
                       119757, 119760, 119761, 119762, 119763, 119764, 119766, 119767, 119768,
                       119769]
        shot_list_2 = [114407,114408,114411,114412,114413,114415,114416,114417,114418,114419,114420,114422,114424,114425,114428,114429,114431,114432,114433,
                       114434,114435,114436,114438,114439,114441,114443,114444,114445,114448,114450,114451,114453,114454,114455,114456,114457,114458,114460,
                       114462,114464,114467,114468,114472,114473]
        shot_list = shot_list_1 + shot_list_2
        # For state 3, we need to check both new and old shot directories
        file_path = [project_path('data', 'shots', 'new'), project_path('data', 'shots', 'old')]
    else:
        raise ValueError(f"Invalid state: {state}. Must be 1, 2, or 3.")
    
    return shot_list, file_path


def create_visualizations(formatted_ip_data, shot_list, initial_cutoff_indices, end_cutoff_indices, output_dir):
    """
    Create visualization plots for validation.
    
    Parameters:
    - formatted_ip_data: Formatted IP data array
    - shot_list: List of shot numbers
    - initial_cutoff_indices: List of initial cutoff indices
    - end_cutoff_indices: List of end cutoff indices
    - output_dir: Directory to save plots
    """
    # Plot all shots overlaid with cutoff points for validation
    plt.figure(figsize=(15, 6))
    for i, shot_num in enumerate(shot_list):
        plt.plot(formatted_ip_data[i, :, 0], alpha=0.5)
        plt.axvline(initial_cutoff_indices[i], color='r', linestyle='--', alpha=0.3, label='Initial Cutoff' if i == 0 else "")
        plt.axvline(end_cutoff_indices[i], color='g', linestyle='--', alpha=0.3, label='End Cutoff' if i == 0 else "")
    plt.title('IP Data - All Shots Overlay with Cutoff Points')
    plt.xlabel('Time Index')
    plt.ylabel('IP Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ip_data_with_cutoffs.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Plot truncated IP data (post-initial-cutoff, pre-end-cutoff) overlaid
    plt.figure(figsize=(15, 6))
    for i, shot_num in enumerate(shot_list):
        start_cutoff = initial_cutoff_indices[i]
        end_cutoff = end_cutoff_indices[i]
        truncated_ip = formatted_ip_data[i, start_cutoff:end_cutoff, 0]
        plt.plot(truncated_ip, alpha=0.5)
    plt.title('Doubly Truncated IP Data - All Shots Overlay')
    plt.xlabel('Time Index (Post-Initial-Cutoff)')
    plt.ylabel('IP Value')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'truncated_ip_data.png'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to process IP data and find cutoff indices."""
    parser = argparse.ArgumentParser(description='Basic IP Trim Script')
    parser.add_argument('--state', type=int, default=2, choices=[1, 2, 3], 
                       help='Plasma state (1, 2, or 3)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for plots')
    parser.add_argument('--target_frames', type=int, default=800,
                       help='Target number of frames for downsampling')
    parser.add_argument('--window_size', type=int, default=5,
                       help='Window size for moving average smoothing')
    parser.add_argument('--jump_ratio', type=float, default=2.5,
                       help='Jump ratio for end cutoff detection')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get shot list and file path
    shot_list, file_path = get_shot_list_and_path(args.state)
    print(f"Processing state {args.state} with {len(shot_list)} shots")
    if isinstance(file_path, list):
        print(f"IP data paths: {file_path}")
    else:
        print(f"IP data path: {file_path}")
    
    # Load and format IP data
    print("Loading IP data...")
    ip_data = load_ip_data(shot_list, file_path)
    formatted_ip_data = format_ip_data(ip_data, args.target_frames)
    print(f"IP data shape: {formatted_ip_data.shape}")
    
    # Calculate initial cutoff indices
    print("Finding initial cutoff indices...")
    initial_cutoff_indices = []
    for i in range(len(shot_list)):
        ip = formatted_ip_data[i, :, 0]
        initial_cutoff_index = find_initial_cutoff_index(ip, window_size=args.window_size, start_index=50)
        initial_cutoff_indices.append(initial_cutoff_index)
    
    # Calculate end cutoff indices for truncated data
    print("Finding end cutoff indices...")
    end_cutoff_indices = []
    for i in range(len(shot_list)):
        start_cutoff = initial_cutoff_indices[i]
        truncated_ip = formatted_ip_data[i, start_cutoff:, 0]
        end_cutoff = find_end_cutoff_index(truncated_ip, window_size=args.window_size, jump_ratio=args.jump_ratio)
        end_cutoff_indices.append(end_cutoff + start_cutoff if end_cutoff < len(truncated_ip) else args.target_frames)
    
    # Print summary statistics
    print(f"\nCutoff Statistics:")
    print(f"Initial cutoff range: {min(initial_cutoff_indices)} - {max(initial_cutoff_indices)}")
    print(f"End cutoff range: {min(end_cutoff_indices)} - {max(end_cutoff_indices)}")
    print(f"Average initial cutoff: {np.mean(initial_cutoff_indices):.1f}")
    print(f"Average end cutoff: {np.mean(end_cutoff_indices):.1f}")
    
    # Create visualization plots
    print("Creating visualization plots...")
    create_visualizations(formatted_ip_data, shot_list, initial_cutoff_indices, end_cutoff_indices, args.output_dir)
    print(f"Plots saved to {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
