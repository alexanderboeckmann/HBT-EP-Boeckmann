#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
import argparse
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parameters
CAMERA_DEPTH = 65535.0  # 2^16
DEFAULT_FRAME_COUNT = 800
DPI = 300  # Resolution for saving the image

def downsample_time_data(time_data, target_length=DEFAULT_FRAME_COUNT):
    """Downsample time data to target length."""
    time_data = np.asarray(time_data, dtype=float)
    logger.info(f"Raw time data length: {len(time_data)}")
    frame_ratio = max(1, len(time_data) // target_length)  # Avoid division by zero
    downsampled_data = time_data[::frame_ratio][:target_length]
    logger.info(f"Downsampled time data length: {len(downsampled_data)}")
    return downsampled_data

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
    parser.add_argument('--data-dir', default='data/shots', help='Base directory for shot data (default: data/shots)')
    parser.add_argument('--shot', type=int, default=114451, help='Shot number (default: 114451)')
    parser.add_argument('--shot-type', choices=['old', 'new'], default='old', help='Shot type: old or new (default: old)')
    parser.add_argument('--output-dir', default='outputs', help='Directory for output PNG (default: outputs)')
    args = parser.parse_args()

    # Construct paths
    data_dir = Path(args.data_dir) / args.shot_type / str(args.shot) / 'CAM-26731'
    tiff_dir = data_dir / 'tiff'
    time_file = data_dir.parent / 'time.npy'
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
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
        plt.suptitle(f"Shot {args.shot}")
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
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout for colorbar
        output_file = output_dir / f'shot_{args.shot}_four_frames.png'
        plt.savefig(output_file, dpi=DPI)
        plt.close()

        logger.info(f"Saved four frames plot for shot {args.shot} to {output_file}")
    except Exception as e:
        logger.error(f"Error processing shot {args.shot}: {e}")
        raise

if __name__ == "__main__":
    main()