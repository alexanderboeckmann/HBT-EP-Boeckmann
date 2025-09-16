#!/usr/bin/env python
# coding: utf-8

"""
Camera Frame Finder Script

This script helps locate and visualize specific camera frames from plasma shot data.
It searches through shot directories to find and display camera images, useful for
data exploration and debugging HBT analysis workflows.

Features:
- Searches through shot data directories for camera frames
- Displays found frames with metadata
- Supports both old and new shot data formats
- Generates visualization outputs for frame inspection
- Helps identify data quality and availability issues
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
import argparse
from pathlib import Path
import logging

# Set matplotlib global font size and DPI
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 28,
    'figure.dpi': 600,
    'axes.linewidth': 2.5,
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0,
    'xtick.minor.width': 1.5,
    'ytick.minor.width': 1.5,
    'lines.linewidth': 3.0,
    'patch.linewidth': 2.0
})

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Centralized project root and path utilities
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)

# Parameters
CAMERA_DEPTH = 65535.0  # 2^16
DEFAULT_FRAME_COUNT = 800
DPI = 600  # Resolution for saving the image

def downsample_time_data(time_data, target_length=DEFAULT_FRAME_COUNT):
    """Downsample time data to target length using the same logic as basic_ip_trim.py."""
    time_data = np.asarray(time_data, dtype=float)
    logger.info(f"Raw time data length: {len(time_data)}")
    # Use the same frame ratio calculation as basic_ip_trim.py
    frame_ratio = len(time_data) // target_length
    if frame_ratio == 0:
        frame_ratio = 1  # Avoid division by zero
    downsampled_data = time_data[::frame_ratio][:target_length]
    logger.info(f"Downsampled time data length: {len(downsampled_data)}")
    return downsampled_data

def load_m2_data(shot_number, data_dir):
    """Load m=2 mode amplitude data for a specific shot."""
    hbt_path = Path(data_dir) / 'new'
    m2_file = hbt_path / f'{shot_number}m2Amp.npy'
    
    if not m2_file.exists():
        raise FileNotFoundError(f"m=2 amplitude data not found: {m2_file}")
    
    m2_data = np.load(m2_file)
    logger.info(f"Loaded m=2 data for shot {shot_number}, shape: {m2_data.shape}")
    return m2_data

def load_ip_data(shot_number, data_dir):
    """Load IP (plasma current) data for a specific shot."""
    # Try both new and old directories
    for subdir in ['new', 'old']:
        ip_path = Path(data_dir) / subdir
        ip_file = ip_path / f'{shot_number}ip.npy'
        
        if ip_file.exists():
            ip_data = np.load(ip_file)
            logger.info(f"Loaded IP data for shot {shot_number} from {subdir}, shape: {ip_data.shape}")
            return ip_data
    
    raise FileNotFoundError(f"IP data not found for shot {shot_number} in {data_dir}/new or {data_dir}/old")

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

def find_end_cutoff_index(ip_data, window_size=5, jump_ratio=5, lookback_window=5, stability_window=30):
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
            print(f"Found a sudden jump at index {i}")
            return max(0, i)  # Buffer to include some context
    
    # If no jump detected, check for rapid decrease
    for i in range(stability_window, len(diff)):
        if diff[i] < -jump_ratio * abs(baseline_median):
            # Found a rapid decrease; set cutoff before this point
            print(f"Found a rapid decrease at index {i}")
            return max(0, i)  # Buffer to include some context
    
    return len(ip_data)  # No jump detected, use full length

def create_amplitude_visualization(shot_number, time_data, m2_data, ip_data, frame_times, frame_indices, output_dir):
    """Create the amplitude visualization with m=2 data, IP data, and frame markers."""
    # Downsample data to match time_data length using the same logic as basic_ip_trim.py
    frame_ratio = len(m2_data) // len(time_data)
    if frame_ratio == 0:
        frame_ratio = 1  # Avoid division by zero
    m2_downsampled = m2_data[::frame_ratio][:len(time_data)]
    ip_downsampled = ip_data[::frame_ratio][:len(time_data)]
    
    # Find cutoff points in the downsampled data
    initial_cutoff = find_initial_cutoff_index(ip_downsampled)
    # For end cutoff, only check data after the initial cutoff
    truncated_ip = ip_downsampled[initial_cutoff:]
    end_cutoff_relative = find_end_cutoff_index(truncated_ip)
    # Use the same fallback logic as basic_ip_trim.py
    end_cutoff = initial_cutoff + end_cutoff_relative if end_cutoff_relative < len(truncated_ip) else len(ip_downsampled)
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(18, 12))
    
    # Plot m=2 data on left y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Time (ms)', color='black', fontsize=24)
    ax1.set_ylabel('Mode Amplitude (G)', color='black', fontsize=24)
    line1 = ax1.plot(time_data, m2_downsampled, color=color1, linewidth=4, label='m=2 Mode Amplitude')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.tick_params(axis='x', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for IP data
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Plasma Current (kA)', color='black', fontsize=24)
    line2 = ax2.plot(time_data, ip_downsampled / 1000, color=color2, linewidth=4, label='Plasma Current')
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Add vertical lines for initialization and quenching points (no labels)
    init_time = time_data[initial_cutoff] if initial_cutoff < len(time_data) else time_data[0]
    end_time = time_data[end_cutoff] if end_cutoff < len(time_data) else time_data[-1]
    
    ax1.axvline(init_time, color='black', linestyle='--', linewidth=3, alpha=0.8)
    ax1.axvline(end_time, color='black', linestyle='--', linewidth=3, alpha=0.8)
    
    # Add arrows pointing to frame locations with labels just above m=2 data points
    arrow_props = dict(arrowstyle='->', lw=4, color='green', alpha=0.8)
    for i, (frame_time, frame_idx) in enumerate(zip(frame_times, frame_indices)):
        if frame_idx < len(m2_downsampled):
            y_pos = m2_downsampled[frame_idx]
            # Determine horizontal alignment based on position to avoid overlap with y-axis
            if frame_time > 6.5:  # 4th frame - smallest shift
                ha = 'right'
                x_offset = -0.05  # Smallest shift for 4th frame
            elif frame_time > 5.0:  # 3rd frame - medium shift
                ha = 'right'
                x_offset = -0.1  # Medium shift for 3rd frame
            else:
                ha = 'center'
                x_offset = 0
            
            # Place label just above the m=2 data point with a small offset
            ax1.annotate(f'Frame {frame_idx}', 
                        xy=(frame_time, y_pos),  # Point to data
                        xytext=(frame_time + x_offset, y_pos + 0.1 * (max(m2_downsampled) - min(m2_downsampled))),  # Small offset above
                        arrowprops=arrow_props,
                        fontsize=20, 
                        ha=ha,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.9, linewidth=2))
    
    # Set x-axis limits to cut off wasted space after 8ms
    ax1.set_xlim(left=time_data[0], right=8.0)
    
    # Set title and legends
    plt.title(f'Testing Mode Amplitude Shot: {shot_number}', fontsize=28, color='black')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Adjust layout and save
    plt.tight_layout()
    output_file = output_dir / f'shot_{shot_number}_amplitude_with_frames.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved amplitude visualization for shot {shot_number} to {output_file}")
    return output_file

def process_shot_data(folder_path, initial_cutoff, end_cutoff, max_pixel_value=CAMERA_DEPTH):
    """Process TIFF images in a folder, cropping to 32x32 and normalizing."""
    folder_path = Path(folder_path)
    tiff_files = sorted(glob.glob(str(folder_path / "*.tiff")))
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {folder_path}")
    
    logger.info(f"Found {len(tiff_files)} TIFF files in {folder_path}")
    end_cutoff = min(end_cutoff, len(tiff_files))
    if initial_cutoff >= end_cutoff:
        raise ValueError(f"Invalid cutoff indices: initial={initial_cutoff}, end={end_cutoff}")
    
    shot_2d = []
    for tiff_file in tiff_files[initial_cutoff:end_cutoff]:
        try:
            with Image.open(tiff_file) as im:
                img = np.array(im, dtype=np.float32) / max_pixel_value
                h, w = img.shape
                if h < 32 or w < 32:
                    raise ValueError(f"Image too small to crop to 32x32: got {h}x{w}")
                
                start_h, start_w = (h - 32) // 2, (w - 32) // 2
                cropped = img[start_h:start_h + 32, start_w:start_w + 32]
                shot_2d.append(cropped)
        except Exception as e:
            logger.warning(f"Error loading {tiff_file}: {e}")
            continue
    
    if not shot_2d:
        raise ValueError(f"No valid frames processed for {folder_path}")
    
    return np.array(shot_2d)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process camera frames for a shot.")
    parser.add_argument('--data-dir', default=project_path('data', 'shots'), help='Base directory for shot data (default: data/shots)')
    parser.add_argument('--shot', type=int, default=119671, help='Shot number (default: 119671)')
    parser.add_argument('--shot-type', choices=['old', 'new'], default='new', help='Shot type: old or new (default: new)')
    parser.add_argument('--output-dir', default=project_path('outputs'), help='Directory for output PNG (default: outputs)')
    args = parser.parse_args()

    # Construct paths
    data_dir = Path(args.data_dir) / args.shot_type / str(args.shot) / 'CAM-26731'
    tiff_dir = data_dir / 'tiff'
    time_file = Path(args.data_dir) / args.shot_type / f'{args.shot}time.npy'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate paths
    if not tiff_dir.exists():
        raise FileNotFoundError(f"TIFF directory not found: {tiff_dir}")
    if not time_file.exists():
        raise FileNotFoundError(f"Time data file not found: {time_file}")

    # Process camera data
    initial_cutoff, end_cutoff = 0, DEFAULT_FRAME_COUNT
    try:
        logger.info(f"Processing shot {args.shot} ({args.shot_type})")
        shot_2d = process_shot_data(tiff_dir, initial_cutoff, end_cutoff)
        logger.info(f"Successfully processed shot {args.shot} with {len(shot_2d)} frames")

        # Load and downsample time data (convert to milliseconds)
        time_data = np.load(time_file, allow_pickle=True) * 1000  # Convert to ms
        time_data = downsample_time_data(time_data)
        logger.info(f"Loaded and downsampled {len(time_data)} time points for shot {args.shot}")

        # Select four frames at different stages
        num_frames = len(shot_2d)
        if num_frames == 0:
            raise ValueError(f"No frames available for shot {args.shot}")
        frame_indices = [
            initial_cutoff + int(0.25 * (end_cutoff - initial_cutoff)),
            initial_cutoff + int(0.4 * (end_cutoff - initial_cutoff)),
            initial_cutoff + int(0.6 * (end_cutoff - initial_cutoff)),
            initial_cutoff + int(0.8 * (end_cutoff - initial_cutoff))
        ]
        frame_indices = [min(max(0, idx), len(time_data) - 1) for idx in frame_indices]
        frame_times = [time_data[i] if i < len(time_data) else time_data[-1] for i in frame_indices]

        # Plot four frames in a 2x2 grid with a shared colorbar to the right
        fig, axes = plt.subplots(2, 2, figsize=(16, 16), sharex=True, sharey=True)
        plt.suptitle(f"Shot {args.shot}", fontsize=32)
        images = []
        for ax, idx, time in zip(axes.flat, frame_indices, frame_times):
            if idx < len(shot_2d):
                im = ax.imshow(shot_2d[idx], cmap='gray', vmin=0, vmax=1)
                images.append(im)
                ax.set_title(f'Frame {idx} at {time:.3f}ms', fontsize=22)
                ax.set_xlabel('Pixel X', fontsize=20)
                ax.set_ylabel('Pixel Y', fontsize=20)
        
        # Add a separate colorbar to the right
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        plt.colorbar(images[0], cax=cbar_ax, orientation='vertical')
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout for colorbar
        output_file = output_dir / f'shot_{args.shot}_four_frames.png'
        plt.savefig(output_file, dpi=DPI)
        plt.close()

        logger.info(f"Saved four frames plot for shot {args.shot} to {output_file}")
        
        # Create amplitude visualization for any shot
        try:
            logger.info(f"Creating amplitude visualization for shot {args.shot}")
            
            # Load m=2 and IP data
            m2_data = load_m2_data(args.shot, args.data_dir)
            ip_data = load_ip_data(args.shot, args.data_dir)
            
            # Create the amplitude visualization
            amplitude_output = create_amplitude_visualization(
                args.shot, time_data, m2_data, ip_data, frame_times, frame_indices, output_dir
            )
            logger.info(f"Successfully created amplitude visualization: {amplitude_output}")
            
        except Exception as e:
            logger.error(f"Error creating amplitude visualization for shot {args.shot}: {e}")
            # Don't raise here, just log the error and continue
    except Exception as e:
        logger.error(f"Error processing shot {args.shot}: {e}")
        raise

if __name__ == "__main__":
    main()