#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image

# Parameters (updated for shot 119671)
RESERVED_SHOT = 119671
CAMERA_DEPTH = 65535.0  # 2^16
DEFAULT_FRAME_COUNT = 800
DPI = 300  # Resolution for saving the image
DEFAULT_FRAME_COUNT = 800

# Paths (for new_shots, based on original script)
data_path = '/Users/aboeckmann/Documents/Columbia/PlasmaLab/HBT-EP-Boeckmann/HighFreqMLModeling/Training/Input Data/Shots/'
hbt_path = '/Users/aboeckmann/Documents/Columbia/PlasmaLab/HBT-EP-Boeckmann/HighFreqMLModeling/Training/python_hbteplib_data/'
ip_path = '/Users/aboeckmann/Documents/Columbia/PlasmaLab/HBT-EP-Boeckmann/HighFreqMLModeling/ip_Data/'

def smooth_data(data, window_size=5):
    """Smooth data using a moving average with edge padding."""
    smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    pad = window_size // 2
    smoothed = np.pad(smoothed, (pad, pad), mode='edge')
    return smoothed[:len(data)]

def load_ip_data(shot):
    """Load IP data for a given shot."""
    try:
        ip_data = np.load(os.path.join(ip_path, f'{shot}ip.npy'))
        print(f"Loaded IP data for shot {shot}: {len(ip_data)} frames")
        return ip_data
    except FileNotFoundError:
        print(f"IP data file for shot {shot} not found.")
        return None

def format_ip_data(data, target_length=DEFAULT_FRAME_COUNT):
    """Format IP data to target length and shape."""
    data = np.asarray(data, dtype=float)
    print(f"Raw IP data length: {len(data)}")
    frame_ratio = max(1, data.shape[0] // target_length)  # Avoid division by zero
    formatted_data = data[::frame_ratio][:target_length]
    print(f"Formatted IP data length: {len(formatted_data)}")
    return formatted_data

def find_initial_cutoff_index(ip_data, window_size=5, start_index=50):
    """Find initial cutoff index based on peak and valley detection."""
    if len(ip_data) < start_index + 1:
        print(f"IP data too short ({len(ip_data)} frames) for initial cutoff detection. Returning 0.")
        return 0
    smoothed_ip = smooth_data(ip_data, window_size)
    diff = np.diff(smoothed_ip)
    
    for i in range(start_index, len(diff)):
        if diff[i-1] > 0 and diff[i] < 0 and ip_data[i] > 0:
            peak_index = i
            for j in range(peak_index + 1, len(diff)):
                if diff[j-1] < 0 and diff[j] > 0 and ip_data[j] > 0:
                    print(f"Initial cutoff index: {j}")
                    return j
            print(f"Initial cutoff index (peak): {peak_index}")
            return peak_index
    print("No valid initial cutoff found. Returning 0.")
    return 0

def find_end_cutoff_index(ip_data, window_size=5, jump_ratio=2.5, lookback_window=10, stability_window=30):
    """Find end cutoff index based on derivative jumps."""
    if len(ip_data) < stability_window + 1:
        print(f"IP data too short ({len(ip_data)} frames) for end cutoff detection. Returning {len(ip_data)}.")
        return len(ip_data)
    smoothed_ip = smooth_data(ip_data, window_size)
    diff = np.diff(smoothed_ip)
    
    baseline_median = np.median(diff[:stability_window]) if len(diff) > stability_window else np.median(diff)
    
    for i in range(stability_window, len(diff) - lookback_window):
        if np.max(diff[i:i + lookback_window]) > jump_ratio * abs(baseline_median) and diff[i] < 0.2 * np.max(diff[i:i + lookback_window]):
            print(f"End cutoff index (jump): {max(0, i - window_size)}")
            return max(0, i - window_size)
    for i in range(stability_window, len(diff)):
        if diff[i] < -jump_ratio * abs(baseline_median):
            print(f"End cutoff index (negative jump): {max(0, i - window_size)}")
            return max(0, i - window_size)
    print(f"No valid end cutoff found. Returning {len(ip_data)}.")
    return len(ip_data)

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

# Load and process IP data to get cutoff indices
ip_data = load_ip_data(RESERVED_SHOT)
if ip_data is not None:
    formatted_ip_data = format_ip_data(ip_data)
    initial_cutoff = find_initial_cutoff_index(formatted_ip_data)
    end_cutoff = find_end_cutoff_index(formatted_ip_data)
    # Ensure valid cutoff indices
    if initial_cutoff >= end_cutoff or end_cutoff == 0:
        print(f"Invalid cutoff indices (initial={initial_cutoff}, end={end_cutoff}). Using defaults.")
        initial_cutoff, end_cutoff = 0, DEFAULT_FRAME_COUNT
else:
    print("IP data missing. Using default cutoff indices.")
    initial_cutoff, end_cutoff = 0, DEFAULT_FRAME_COUNT

# Process camera data for the reserved shot
folder_path = os.path.join(data_path, str(RESERVED_SHOT), 'CAM-26731/tiff/')
try:
    shot_2d = process_shot_data(folder_path, initial_cutoff, end_cutoff)
    print(f"Successfully processed shot {RESERVED_SHOT} with {len(shot_2d)} frames")
    
    # Select a single frame from the middle
    middle_index = len(shot_2d) // 2
    frame_to_display = shot_2d[middle_index] if len(shot_2d) > 0 else None
    
    if frame_to_display is not None:
        # Create and save the figure
        plt.figure(figsize=(6, 6))
        plt.imshow(frame_to_display, cmap='gray', vmin=0, vmax=1)
        plt.colorbar()
        plt.title(f'Shot {RESERVED_SHOT}: Trimmed Camera Frame (Frame {middle_index})')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        
        # Save as PNG with 300 DPI
        output_filename = f'shot_{RESERVED_SHOT}_middle_frame.png'
        plt.savefig(output_filename, dpi=DPI, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print(f"Saved middle frame (index {middle_index}) of shot {RESERVED_SHOT} as {output_filename} with shape {frame_to_display.shape}")
    else:
        print(f"No valid frames available for shot {RESERVED_SHOT}")
except Exception as e:
    print(f"Error processing shot {RESERVED_SHOT}: {e}")

# Load time data
time_data = np.load(os.path.join(hbt_path, f'{RESERVED_SHOT}time.npy'), allow_pickle=True)

# Calculate number of processed frames
tiff_files = sorted(glob.glob(os.path.join(f'/Users/aboeckmann/Documents/Columbia/PlasmaLab/HBT-EP-Boeckmann/HighFreqMLModeling/Training/Input Data/Shots/{RESERVED_SHOT}/CAM-26731/tiff/', "*.tiff")))
end_cutoff = min(end_cutoff, len(tiff_files))
num_frames = end_cutoff - initial_cutoff

# Calculate middle frame index
middle_index = num_frames // 2

# Map to original time data index
time_index = initial_cutoff + 2500

# Get the time of the middle frame
if time_index < len(time_data):
    middle_frame_time = time_data[time_index]
    print(f"Time of the middle frame (index {middle_index}, original index {time_index}) for shot {RESERVED_SHOT}: {middle_frame_time} seconds")
else:
    print(f"Error: Time index {time_index} out of bounds for time_data (length {len(time_data)})")