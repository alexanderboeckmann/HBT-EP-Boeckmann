#!/usr/bin/env python
# coding: utf-8

"""
Trimmed HBT Analysis Class

This module provides the trimmed (preprocessed) variant of HBT analysis.
It extends the base class with trimmed-specific data processing that includes
outlier removal and IP-based cutoff detection.

Key features:
- IP data-based cutoff detection for plasma shots
- Outlier removal and data cleaning
- Trimmed data processing pipeline
- Support for both standard and crossover validation
"""

import numpy as np
import os
import glob
from PIL import Image
from typing import Dict, List, Tuple, Optional
import tensorflow as tf

from .base import HBTAnalysisBase


class HBTAnalysisTrimmed(HBTAnalysisBase):
    """
    Trimmed HBT Analysis implementation.
    
    This class handles trimmed plasma data analysis with IP-based cutoff detection
    and outlier removal preprocessing.
    """
    
    def __init__(self, config: Dict):
        """Initialize trimmed analysis with configuration."""
        config['notebook_type'] = 'trimmed'
        config['default_frame_count'] = 800
        super().__init__(config)
    
    def load_ip_data(self, shot_list: List[int]) -> List[np.ndarray]:
        """Load IP data for given shots, handling missing files."""
        ip_data = []
        for shot in shot_list:
            _, _, ip_path = self.get_paths_for_shot(shot)
            try:
                ip_data.append(np.load(os.path.join(ip_path, f'{shot}ip.npy')))
            except FileNotFoundError:
                print(f"IP data file for shot {shot} not found. Skipping.")
        return ip_data
    
    def format_ip_data(self, data: List[np.ndarray], target_length: int = None) -> np.ndarray:
        """Format IP data to target length and shape."""
        if target_length is None:
            target_length = self.default_frame_count
            
        data = np.asarray(data, dtype=float)
        frame_ratio = data[0].shape[0] // target_length
        return data[:, ::frame_ratio, np.newaxis][:, :target_length, :]
    
    def find_initial_cutoff_index(self, ip_data: np.ndarray, window_size: int = 5, start_index: int = 50) -> int:
        """Find initial cutoff index based on peak and valley detection."""
        smoothed_ip = self.smooth_data(ip_data, window_size)
        diff = np.diff(smoothed_ip)
        
        for i in range(start_index, len(diff)):
            if diff[i-1] > 0 and diff[i] < 0 and ip_data[i] > 0:
                peak_index = i
                for j in range(peak_index + 1, len(diff)):
                    if diff[j-1] < 0 and diff[j] > 0 and ip_data[j] > 0:
                        return j
                return peak_index
        return 0
    
    def find_end_cutoff_index(self, ip_data: np.ndarray, window_size: int = 5, 
                            jump_ratio: float = 2.5, lookback_window: int = 10, 
                            stability_window: int = 30) -> int:
        """Find end cutoff index based on derivative jumps."""
        smoothed_ip = self.smooth_data(ip_data, window_size)
        diff = np.diff(smoothed_ip)
        
        baseline_median = np.median(diff[:stability_window]) if len(diff) > stability_window else np.median(diff)
        
        for i in range(stability_window, len(diff) - lookback_window):
            if np.max(diff[i:i + lookback_window]) > jump_ratio * abs(baseline_median) and diff[i] < 0.2 * np.max(diff[i:i + lookback_window]):
                return max(0, i - window_size)
        for i in range(stability_window, len(diff)):
            if diff[i] < -jump_ratio * abs(baseline_median):
                return max(0, i - window_size)
        return len(ip_data)
    
    def process_shot_data(self, folder_path: str, initial_cutoff: int, end_cutoff: int, 
                         max_pixel_value: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Process TIFF images in a folder, cropping to 32x32 and normalizing."""
        if max_pixel_value is None:
            max_pixel_value = self.camera_depth
            
        tiff_files = sorted(glob.glob(os.path.join(folder_path, "*.tiff")))
        if not tiff_files:
            raise ValueError(f"No TIFF files found in {folder_path}")
        
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
        
        shot_2d = np.array(shot_2d)
        return shot_2d, shot_2d, shot_2d.reshape(len(shot_2d), -1), len(shot_2d)
    
    def process_all_shots(self, shot_list: List[int], initial_cutoff_indices: List[int], 
                         end_cutoff_indices: List[int], frame_counts: List[int]) -> Tuple:
        """Process all shots, returning 2D, cut, flat data, valid shots, and frame counts."""
        training_data_2D, cut_training_data_2D, flat_training_data = [], [], []
        valid_shots, actual_frame_counts = [], []
        
        for i, shot in enumerate(shot_list):
            if shot == self.config['reserved_shot']:
                continue
            data_path, _, _ = self.get_paths_for_shot(shot)
            folder_path = os.path.join(data_path, str(shot), 'CAM-26731', 'tiff')
            try:
                shot_2d, cut_2d, flat_data, actual_frames = self.process_shot_data(
                    folder_path, initial_cutoff_indices[i], end_cutoff_indices[i]
                )
                if actual_frames > 0:
                    training_data_2D.append(shot_2d)
                    cut_training_data_2D.append(cut_2d)
                    flat_training_data.append(flat_data)
                    valid_shots.append(shot)
                    actual_frame_counts.append(actual_frames)
                else:
                    print(f"Shot {shot} produced 0 frames. Skipping.")
            except Exception as e:
                print(f"Error processing shot {shot}: {e}")
        
        return training_data_2D, cut_training_data_2D, flat_training_data, valid_shots, actual_frame_counts
    
    def format_hbt_data(self, data: List, mode_num: int, initial_cutoffs: List[int], 
                       end_cutoffs: List[int], frame_ratio: int) -> List[np.ndarray]:
        """Format HBT data for a given mode, trimming to scaled cutoff indices."""
        formatted = []
        for shot in self.valid_shots:
            idx = self.shot_list.index(shot)
            initial = initial_cutoffs[idx] * frame_ratio
            end = min(end_cutoffs[idx] * frame_ratio, len(data[idx][mode_num-1]))
            hbt_slice = data[idx][mode_num-1][initial:end].reshape(-1, 1)
            formatted.append(hbt_slice)
        return formatted
    
    def load_hbt_data(self, shot_list: List[int], valid_shots: List[int], 
                     initial_cutoff_indices: List[int], end_cutoff_indices: List[int], 
                     reserved_shot: int = None, reserved_frame_count: int = None, 
                     ma_norm: float = None) -> Dict:
        """Load and format HBT amplitude, phase, and time data, with proper scaling of cutoff indices."""
        hbt_data = {'amplitudes': [], 'phases': [], 'times': []}
        
        frame_ratio = self.ip_data[0].shape[0] // self.default_frame_count
        
        for shot in shot_list:
            _, hbt_path, _ = self.get_paths_for_shot(shot)
            modes = {
                f'mode_{m}': {
                    'amp': np.load(os.path.join(hbt_path, f'{shot}m{m}Amp.npy')),
                    'phase': np.load(os.path.join(hbt_path, f'{shot}m{m}Phase.npy'))
                } for m in range(1, 5)
            }
            hbt_data['amplitudes'].append([modes[f'mode_{m}']['amp'] for m in range(1, 5)])
            hbt_data['phases'].append([modes[f'mode_{m}']['phase'] for m in range(1, 5)])
            hbt_data['times'].append(np.load(os.path.join(hbt_path, f'{shot}time.npy')))
        
        formatted_data = {
            'amplitudes': [self.format_hbt_data(hbt_data['amplitudes'], m, initial_cutoff_indices, end_cutoff_indices, frame_ratio) for m in range(1, 5)],
            'phases': [self.format_hbt_data(hbt_data['phases'], m, initial_cutoff_indices, end_cutoff_indices, frame_ratio) for m in range(1, 5)],
            'times': [
                hbt_data['times'][shot_list.index(shot)][initial_cutoff_indices[shot_list.index(shot)]*frame_ratio:end_cutoff_indices[shot_list.index(shot)]*frame_ratio]
                for shot in valid_shots
            ]
        }
        
        reserved_shot_hbt = None
        reserved_shot_time = None
        if reserved_shot is not None and reserved_frame_count is not None:
            idx = shot_list.index(reserved_shot)
            initial = initial_cutoff_indices[idx] * frame_ratio
            end = min(end_cutoff_indices[idx] * frame_ratio, len(hbt_data['times'][idx]))
            mode_index = {'ma1': 0, 'ma2': 1, 'ma3': 2, 'ma4': 3, 'mp1': 0, 'mp2': 1, 'mp3': 2, 'mp4': 3}
            data_type = 'amplitudes' if self.config['selected_data_type'].startswith('ma') else 'phases'
            mode_num = mode_index[self.config['selected_data_type']]
            hbt_data_selected = hbt_data[data_type][idx][mode_num]
            reserved_shot_hbt = hbt_data_selected[initial:end]
            hbt_frame_ratio = (end - initial) // reserved_frame_count if reserved_frame_count > 0 else 1
            reserved_shot_hbt = reserved_shot_hbt[::hbt_frame_ratio][:reserved_frame_count]
            if ma_norm is not None:
                clipped_count = np.sum(reserved_shot_hbt > 3 * ma_norm) + np.sum(reserved_shot_hbt < -3 * ma_norm)
                print(f"RESERVED_SHOT {reserved_shot}: {clipped_count} values clipped out of {len(reserved_shot_hbt)}")
                reserved_shot_hbt = np.clip(reserved_shot_hbt, -3 * ma_norm, 3 * ma_norm) / ma_norm
            reserved_shot_hbt = reserved_shot_hbt.reshape(-1, 1)
            reserved_shot_time = hbt_data['times'][idx][initial:end][::hbt_frame_ratio][:reserved_frame_count]
            print(f"RESERVED_SHOT {reserved_shot}: HBT {self.config['selected_data_type']} frames={len(reserved_shot_hbt)}")
            print(f"RESERVED_SHOT {reserved_shot}: Time data frames={len(reserved_shot_time)}")
        
        return formatted_data, reserved_shot_hbt, reserved_shot_time
    
    def compute_reserved_shot_predictions(self, model: tf.keras.Model, reserved_shot: int, 
                                        cut_training_data_2d: np.ndarray, reserved_shot_cut_2d: np.ndarray,
                                        target_data: List, ma_norm: float, reserved_shot_hbt: np.ndarray, 
                                        reserved_shot_time: np.ndarray) -> Optional[np.ndarray]:
        """Compute predictions for the reserved shot and print metrics."""
        if reserved_shot_cut_2d is None or reserved_shot_hbt is None or reserved_shot_time is None:
            print(f"No data available for reserved shot {reserved_shot}. No predictions computed.")
            return None
        
        camera_data = reserved_shot_cut_2d
        hbt_data = reserved_shot_hbt[:, 0]
        
        print(f"Shot {reserved_shot}: Camera frames={len(camera_data)}, HBT frames={len(hbt_data)}, Time frames={len(reserved_shot_time)}")
        
        if len(camera_data) == 0:
            print(f"No camera data for shot {reserved_shot}. No predictions computed.")
            return None
        
        input_data = np.array(camera_data).reshape(-1, 32, 32, 1)
        predictions = model.predict(input_data, verbose=0)[:, 0]
        
        hbt_data_original = hbt_data * ma_norm
        predictions_original = predictions * ma_norm
        
        print(f"Shot {reserved_shot}: HBT {self.config['selected_data_type']} min/max (normalized): {np.min(hbt_data):.2f}, {np.max(hbt_data):.2f}")
        print(f"Shot {reserved_shot}: HBT {self.config['selected_data_type']} min/max (original): {np.min(hbt_data_original):.2f}, {np.max(hbt_data_original):.2f}")
        print(f"Shot {reserved_shot}: Predicted {self.config['selected_data_type']} min/max (normalized): {np.min(predictions):.2f}, {np.max(predictions):.2f}")
        print(f"Shot {reserved_shot}: Predicted {self.config['selected_data_type']} min/max (original): {np.min(predictions_original):.2f}, {np.max(predictions_original):.2f}")
        prediction_errors = np.abs(hbt_data_original - predictions_original) / np.max(np.abs(hbt_data_original)) * 100
        print(f"Shot {reserved_shot} - Mean absolute percentage error: {np.mean(prediction_errors):.2f}%")
        
        return predictions
    
    def run_analysis(self, output_dir: Optional[str] = None) -> Dict:
        """Run the complete trimmed analysis pipeline."""
        print(f"Starting trimmed HBT analysis for state {self.config['state']}")
        
        # Load and process IP data
        print("Loading IP data...")
        self.ip_data = self.load_ip_data(self.shot_list)
        formatted_ip_data = self.format_ip_data(self.ip_data)
        
        # Find cutoff indices
        print("Finding cutoff indices...")
        initial_cutoff_indices = [self.find_initial_cutoff_index(formatted_ip_data[i, :, 0]) for i in range(len(self.shot_list))]
        end_cutoff_indices = [
            self.find_end_cutoff_index(formatted_ip_data[i, initial_cutoff_indices[i]:, 0]) + initial_cutoff_indices[i]
            if self.find_end_cutoff_index(formatted_ip_data[i, initial_cutoff_indices[i]:, 0]) < len(formatted_ip_data[i, initial_cutoff_indices[i]:, 0])
            else self.default_frame_count
            for i in range(len(self.shot_list))
        ]
        frame_counts = [end - start for end, start in zip(end_cutoff_indices, initial_cutoff_indices)]
        
        # Process all shots
        print("Processing shot data...")
        training_data_2D, cut_training_data_2D, flat_training_data, self.valid_shots, actual_frame_counts = self.process_all_shots(
            self.shot_list, initial_cutoff_indices, end_cutoff_indices, frame_counts
        )
        
        # Process reserved shot
        print(f"Processing reserved shot {self.config['reserved_shot']}...")
        reserved_shot_data_2d = None
        reserved_shot_cut_2d = None
        reserved_shot_flat = None
        reserved_shot_frame_count = None
        
        if self.config['reserved_shot'] is not None:
            shot_idx = self.shot_list.index(self.config['reserved_shot'])
            data_path, _, _ = self.get_paths_for_shot(self.config['reserved_shot'])
            folder_path = os.path.join(data_path, str(self.config['reserved_shot']), 'CAM-26731', 'tiff')
            
            print(f"RESERVED_SHOT {self.config['reserved_shot']}: initial_cutoff={initial_cutoff_indices[shot_idx]}, end_cutoff={end_cutoff_indices[shot_idx]}")
            
            try:
                shot_2d, cut_2d, flat_data, actual_frames = self.process_shot_data(
                    folder_path, initial_cutoff_indices[shot_idx], end_cutoff_indices[shot_idx]
                )
                print(f"RESERVED_SHOT {self.config['reserved_shot']}: actual_frames={actual_frames}")
                reserved_shot_data_2d = shot_2d
                reserved_shot_cut_2d = cut_2d
                reserved_shot_flat = flat_data
                reserved_shot_frame_count = actual_frames
                print(f"Successfully processed RESERVED_SHOT {self.config['reserved_shot']} with {actual_frames} frames")
            except Exception as e:
                print(f"Error processing RESERVED_SHOT {self.config['reserved_shot']}: {e}")
        
        # Load HBT data
        print("Loading HBT data...")
        hbt_data, reserved_shot_hbt, reserved_shot_time = self.load_hbt_data(
            self.shot_list, self.valid_shots, initial_cutoff_indices, end_cutoff_indices, 
            reserved_shot=self.config['reserved_shot'], reserved_frame_count=reserved_shot_frame_count
        )
        
        # Prepare target data
        data_type_mapping = {
            'ma1': hbt_data['amplitudes'][0],
            'ma2': hbt_data['amplitudes'][1],
            'ma3': hbt_data['amplitudes'][2],
            'ma4': hbt_data['amplitudes'][3],
            'mp1': hbt_data['phases'][0],
            'mp2': hbt_data['phases'][1],
            'mp3': hbt_data['phases'][2],
            'mp4': hbt_data['phases'][3]
        }
        
        if self.config['selected_data_type'] not in data_type_mapping:
            raise ValueError(f"Invalid selected_data_type: {self.config['selected_data_type']}. Choose from {list(data_type_mapping.keys())}")
        
        target_data = data_type_mapping[self.config['selected_data_type']]
        training_data = cut_training_data_2D
        
        # Calculate normalization
        print("Calculating normalization...")
        raw_target_vector = []
        for i, shot in enumerate(self.valid_shots):
            if shot == self.config['reserved_shot']:
                continue
            for j in range(actual_frame_counts[i]):
                raw_target_vector.append(target_data[i][j])
        
        raw_target_vector = np.asarray(raw_target_vector, dtype=np.float32)[:, 0]
        percentile_cutoff = np.percentile(np.abs(raw_target_vector), self.config['outlier_cutoff'])
        ma_norm = percentile_cutoff if percentile_cutoff > 0 else 1.0
        outlier_threshold = 3 * ma_norm
        print(f"Normalization factor ({self.config['outlier_cutoff']} percentile): {ma_norm:.2f}")
        print(f"Number of outliers (|value| > {outlier_threshold:.2f}): {np.sum(np.abs(raw_target_vector) > outlier_threshold)}")
        
        raw_target_vector = np.clip(raw_target_vector, -outlier_threshold, outlier_threshold)
        
        # Prepare training data
        print("Preparing training data...")
        target_vector = []
        training_vector = []
        for i, shot in enumerate(self.valid_shots):
            if shot == self.config['reserved_shot']:
                continue
            for j in range(actual_frame_counts[i]):
                target_vector.append(target_data[i][j])
                training_vector.append(training_data[i][j])
        
        import random
        random.seed(123)
        zip_list = list(zip(target_vector, training_vector))
        random.shuffle(zip_list)
        target_vector, training_vector = zip(*zip_list)
        
        target_vector = np.asarray(target_vector, dtype=np.float32)[:, 0]
        target_vector = np.clip(target_vector, -outlier_threshold, outlier_threshold) / ma_norm
        training_vector = np.asarray(training_vector, dtype=np.float32).reshape(-1, 32, 32, 1)
        
        # Split data
        test_size = 400
        testing_inputs = training_vector[-test_size:]
        testing_labels = target_vector[-test_size:]
        training_vector = training_vector[:-test_size]
        target_vector = target_vector[:-test_size]
        
        # Save normalization info
        self.save_normalization_info(ma_norm, outlier_threshold)
        
        print('Training shape:', training_vector.shape, 'Target shape:', target_vector.shape)
        print('Testing shape:', testing_inputs.shape, 'Testing label shape:', testing_labels.shape)
        
        # Create and train model
        print("Creating model...")
        model = self.create_model()
        model.summary()
        
        print("Training model...")
        history = self.train_model(model, training_vector, target_vector)
        
        # Evaluate model
        print("Evaluating model...")
        predictions = self.evaluate_model(model, testing_inputs, testing_labels)
        
        # Compute reserved shot predictions
        print("Computing reserved shot predictions...")
        reserved_predictions = self.compute_reserved_shot_predictions(
            model, self.config['reserved_shot'], cut_training_data_2D, reserved_shot_cut_2d,
            target_data, ma_norm, reserved_shot_hbt, reserved_shot_time
        )
        
        # Save results
        if reserved_predictions is not None:
            self.save_results(
                self.config['reserved_shot'], 
                reserved_shot_hbt[:, 0], 
                reserved_predictions, 
                reserved_shot_time,
                output_dir
            )
        
        print("Trimmed analysis complete!")
        
        return {
            'model': model,
            'history': history,
            'test_predictions': predictions,
            'reserved_predictions': reserved_predictions,
            'ma_norm': ma_norm,
            'outlier_threshold': outlier_threshold
        }
