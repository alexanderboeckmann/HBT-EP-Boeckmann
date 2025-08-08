#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image

# Parameters (updated for shot 119671)
RESERVED_SHOT = 114451
CAMERA_DEPTH = 65535.0  # 2^16
DEFAULT_FRAME_COUNT = 800
DPI = 300  # Resolution for saving the image

# Paths
data_path = '/Users/aboeckmann/Documents/Columbia/PlasmaLab/HBT-EP-Boeckmann/HighFreqMLModeling/Training/Input Data/Old Shots/'
hbt_path = '/Users/aboeckmann/Documents/Columbia/PlasmaLab/HBT-EP-Boeckmann/HighFreqMLModeling/Training/oldshot_python_hbteplib_data/'

def downsample_time_data(time_data, target_length=DEFAULT_FRAME_COUNT):
    """Downsample time data to target length."""
    time_data = np.asarray(time_data, dtype=float)
    print(f"Raw time data length: {len(time_data)}")
    frame_ratio = max(1, len(time_data) // target_length)  # Avoid division by zero
    downsampled_data = time_data[::frame_ratio][:target_length]
    print(f"Downsampled time data length: {len(downsampled_data)}")
    return downsampled_data

def process_shot_data(folder_path, initial_cutoff, end_cutoff, max_pixel_value=CAMERA_DEPTH):
    """Process TIFF images in a folder, cropping to 32x32 and normalizing."""
    tiff_files = sorted(glob.glob(os.path.join(folder_path, "*.tiff")))
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {folder_path}")
    
    print(f"Found {len(tiff_files)} TIFF files in {folder_path}")
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
            print(f"Error loading {tiff_file}: {e}")
            continue
    
    if not shot_2d:
        raise ValueError(f"No valid frames processed for {folder_path}")
    
    return np.array(shot_2d)

# Process camera data for the reserved shot
folder_path = os.path.join(data_path, str(RESERVED_SHOT), 'CAM-26731/tiff/')
initial_cutoff, end_cutoff = 0, DEFAULT_FRAME_COUNT
try:
    print(f"Processing shot {RESERVED_SHOT}")
    shot_2d = process_shot_data(folder_path, initial_cutoff, end_cutoff)
    print(f"Successfully processed shot {RESERVED_SHOT} with {len(shot_2d)} frames")

    # Load and downsample time data (convert to milliseconds)
    time_data = np.load(os.path.join(hbt_path, f'{RESERVED_SHOT}time.npy'), allow_pickle=True) * 1000  # Convert to ms
    time_data = downsample_time_data(time_data)
    print(f"Loaded and downsampled {len(time_data)} time points for shot {RESERVED_SHOT}")

    # Select four frames at different stages
    num_frames = len(shot_2d)
    if num_frames == 0:
        raise ValueError(f"No frames available for shot {RESERVED_SHOT}")
    frame_indices = [
        initial_cutoff + int(0.25 * (end_cutoff - initial_cutoff)),
        initial_cutoff + int(0.4 * (end_cutoff - initial_cutoff)),
        initial_cutoff + int(0.6 * (end_cutoff - initial_cutoff)),
        initial_cutoff + int(0.8 * (end_cutoff - initial_cutoff))
    ]
    frame_indices = [min(max(0, idx), len(time_data) - 1) for idx in frame_indices]  # Ensure indices are valid
    frame_times = [time_data[i] if i < len(time_data) else time_data[-1] for i in frame_indices]

    # Plot four frames in a 2x2 grid with a shared colorbar to the right
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    plt.suptitle(f"Shot {RESERVED_SHOT}")
    images = []
    for ax, idx, time in zip(axes.flat, frame_indices, frame_times):
        if idx < len(shot_2d):
            im = ax.imshow(shot_2d[idx], cmap='gray', vmin=0, vmax=1)
            images.append(im)
            ax.set_title(f'Frame {idx} at {time:.3f}ms')
            ax.set_xlabel('Pixel X')
            ax.set_ylabel('Pixel Y')
    
    # Add a separate colorbar to the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    plt.colorbar(images[0], cax=cbar_ax, orientation='vertical')
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to leave space for colorbar
    plt.savefig(f'shot_{RESERVED_SHOT}_four_frames.png', dpi=DPI)
    plt.close()

    print(f"Saved four frames plot for shot {RESERVED_SHOT}")
except Exception as e:
    print(f"Error processing shot {RESERVED_SHOT}: {e}")