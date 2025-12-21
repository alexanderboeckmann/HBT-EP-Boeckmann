#!/usr/bin/env python
# coding: utf-8

"""
Untrimmed HBT Analysis Class

This module provides the untrimmed (raw) variant of HBT analysis.
It extends the base class with untrimmed-specific data processing that uses
raw data without extensive preprocessing or outlier removal.

Key features:
- Raw data processing without IP-based cutoff detection
- Minimal preprocessing to preserve full data range
- Support for both standard and crossover validation
- Simpler data pipeline for faster processing
"""

import numpy as np
import os
import glob
import json
import time
from PIL import Image
from typing import Dict, List, Tuple, Optional
import tensorflow as tf

from .base import HBTAnalysisBase
from ..utils.camera_cache import load_center32_frames_sampled, CameraCacheStats


class HBTAnalysisUntrimmed(HBTAnalysisBase):
    """
    Untrimmed HBT Analysis implementation.
    
    This class handles raw, untrimmed plasma data analysis with minimal preprocessing.
    """
    
    def __init__(self, config: Dict):
        """Initialize untrimmed analysis with configuration."""
        config['notebook_type'] = 'untrimmed'
        config['default_frame_count'] = 800
        super().__init__(config)
    
    def determine_frame_ratio(self, num_frames: int, target_frames: int = None) -> Tuple[int, int]:
        """
        Determines the frame ratio needed to downsample the data to target_frames.
        Returns the ratio and the actual number of frames after downsampling.
        """
        if target_frames is None:
            target_frames = self.default_frame_count
        ratio = max(1, num_frames // target_frames)
        actual_frames = num_frames // ratio
        return ratio, actual_frames
    
    def process_shot_data(self, folder_path: str, target_frame_count: int = None, 
                         max_pixel_value: float = None, shot_num: int = None,
                         cache_stats: Optional[CameraCacheStats] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a single shot's data with automatic frame rate handling.
        Returns: 2D data (32x32), cut 2D data (32x32), and flat data for the shot
        """
        if target_frame_count is None:
            target_frame_count = self.default_frame_count
        if max_pixel_value is None:
            max_pixel_value = self.camera_depth
            
        frame_files: list[str] = []
        for pat in ("*.tif", "*.tiff", "*.png"):
            frame_files.extend(glob.glob(os.path.join(folder_path, pat)))
        frame_files = sorted(frame_files)
        num_frames = len(frame_files)

        if num_frames == 0:
            raise ValueError(f"No frame files found in {folder_path} (expected tif/tiff/png)")

        frame_ratio, _actual_frames = self.determine_frame_ratio(num_frames, target_frame_count)
        sample_indices = list(range(0, num_frames, frame_ratio))[:target_frame_count]

        cache_file = None
        if shot_num is not None:
            cache_file = self.project_path(
                'data', 'cache', 'camera_center32_downsampled', str(target_frame_count), str(shot_num), 'frames.npy'
            )

        shot_2d = load_center32_frames_sampled(
            folder_path=folder_path,
            cache_file=cache_file,
            max_pixel_value=max_pixel_value,
            sample_indices=sample_indices,
            stats=cache_stats,
        )

        cut_shot = shot_2d
        flat_shot = shot_2d.reshape(len(shot_2d), -1)
        return shot_2d, cut_shot, flat_shot
    
    def process_all_shots(self, shot_list: List[int], target_frame_count: int = None, 
                         max_pixel_value: float = None,
                         cache_stats: Optional[CameraCacheStats] = None) -> Tuple:
        """
        Process multiple shots with automatic frame rate handling
        """
        if target_frame_count is None:
            target_frame_count = self.default_frame_count
        if max_pixel_value is None:
            max_pixel_value = self.camera_depth
            
        training_data_2D = []
        cut_training_data_2D = []
        flat_training_data = []
        valid_shots = []
        
        for shot in shot_list:
            if shot == self.config['reserved_shot']:
                continue
            try:
                folder_path = self.resolve_camera_frames_dir(shot)
            except Exception as e:
                print(f"Error resolving camera frames for shot {shot}: {e}")
                continue
            
            try:
                shot_2d, cut_2d, flat_data = self.process_shot_data(
                    folder_path,
                    target_frame_count,
                    max_pixel_value,
                    shot_num=shot,
                    cache_stats=cache_stats,
                )
                
                if len(shot_2d) == target_frame_count:
                    training_data_2D.append(shot_2d)
                    cut_training_data_2D.append(cut_2d)
                    flat_training_data.append(flat_data)
                    valid_shots.append(shot)
                else:
                    print(f"Shot {shot} produced {len(shot_2d)} frames, expected {target_frame_count}. Skipping.")
                    
            except Exception as e:
                print(f"Error processing shot {shot}: {e}")
                continue
        
        return (np.array(training_data_2D), 
                np.array(cut_training_data_2D), 
                np.array(flat_training_data), valid_shots)
    
    def format_hbt_data(self, data: List, mode_num: int) -> np.ndarray:
        """Format HBT data for a given mode."""
        # Determine frame ratio
        original_length = data[0][0].shape[0]  # 5000
        target_length = self.default_frame_count
        frame_ratio = original_length // target_length  # generally 5 (5000/target_frame_count)
        
        data = np.asarray(data, dtype=float)
        data = np.reshape(data[:, mode_num-1, :], (len(data), original_length, 1))
        data = data[:, ::frame_ratio, :]
        data = data[:, :target_length, :]
        return data
    
    def load_hbt_data(self, shot_list: List[int], valid_shots: List[int], **kwargs) -> Dict:
        """Load and format HBT data for untrimmed analysis."""
        hbt_ma_data = []
        hbt_mp_data = []
        hbt_time_data = []
        valid_shots_hbt = []
        
        for i in range(len(shot_list)):
            shot = shot_list[i]
            _, hbt_path, _ = self.get_paths_for_shot(shot)
            
            try:
                # Load flat per-mode files and stack to shape (4, L)
                mode_amplitude = np.vstack([np.load(os.path.join(hbt_path, f'{shot}m{j}Amp.npy')) for j in range(1, 5)])
                mode_phase = np.vstack([np.load(os.path.join(hbt_path, f'{shot}m{j}Phase.npy')) for j in range(1, 5)])
                ma_list = [mode_amplitude[j] for j in range(4)]
                mp_list = [mode_phase[j] for j in range(4)]
                
                hbt_ma_data.append(ma_list)
                hbt_mp_data.append(mp_list)
                time_data = np.load(os.path.join(hbt_path, f'{shot}time.npy'))
                hbt_time_data.append(time_data)
                valid_shots_hbt.append(shot)
            except Exception as e:
                print(f"Error loading HBT data for shot {shot}: {e}")
                continue
        
        # Format HBT data
        hbt_ma1_data = self.format_hbt_data(hbt_ma_data, 1)
        hbt_ma2_data = self.format_hbt_data(hbt_ma_data, 2)
        hbt_ma3_data = self.format_hbt_data(hbt_ma_data, 3)
        hbt_ma4_data = self.format_hbt_data(hbt_ma_data, 4)

        hbt_mp1_data = self.format_hbt_data(hbt_mp_data, 1)
        hbt_mp2_data = self.format_hbt_data(hbt_mp_data, 2)
        hbt_mp3_data = self.format_hbt_data(hbt_mp_data, 3)
        hbt_mp4_data = self.format_hbt_data(hbt_mp_data, 4)
        
        return {
            'amplitudes': [hbt_ma1_data, hbt_ma2_data, hbt_ma3_data, hbt_ma4_data],
            'phases': [hbt_mp1_data, hbt_mp2_data, hbt_mp3_data, hbt_mp4_data],
            'times': hbt_time_data,
            'valid_shots_hbt': valid_shots_hbt
        }
    
    def compute_reserved_shot_predictions(self, model: tf.keras.Model, reserved_shot: int, 
                                        cut_training_data_2d: np.ndarray, reserved_shot_cut_2d: np.ndarray,
                                        hbt_data_type: List, hbt_time_data: List, ma_norm: float) -> Optional[np.ndarray]:
        """Compute predictions for the reserved shot."""
        if reserved_shot_cut_2d is None:
            print(f"No data available for reserved shot {reserved_shot}. No predictions computed.")
            return None
        
        shot_idx = self.shot_list.index(reserved_shot)
        camera_data = reserved_shot_cut_2d
        hbt_data = hbt_data_type[shot_idx][:, 0]
        time_data = hbt_time_data[shot_idx]

        print(f"Shot {reserved_shot}: Camera frames={len(camera_data)}, HBT frames={len(hbt_data)}, Time frames={len(time_data)}")
        print(f"Shape of reserved_shot_cut_2d: {reserved_shot_cut_2d.shape}")
        print(f"Shape of camera_data: {np.array(camera_data).shape}")

        if len(camera_data) == 0:
            print(f"No camera data for shot {reserved_shot}. No predictions computed.")
            return None
        
        if len(camera_data) != len(hbt_data):
            print(f"Warning: Camera data ({len(camera_data)} frames) does not match HBT data ({len(hbt_data)} frames)")

        input_data = np.array(camera_data).reshape(-1, 32, 32, 1)
        print(f"Input data shape: {input_data.shape}")
        print(f"Model output shape: {model.output_shape}")

        predictions = []
        batch_size = 100
        for i in range(0, len(input_data), batch_size):
            batch = input_data[i:i+batch_size]
            batch_pred = model.predict(batch, verbose=0)[:, 0] * ma_norm
            predictions.extend(batch_pred)
        predictions = np.array(predictions)
        print(f"Predictions shape: {predictions.shape}")
        
        # Print metrics
        prediction_errors = np.abs(hbt_data - predictions) / ma_norm * 100
        print(f"Shot {reserved_shot} - Mean absolute percentage error: {np.mean(prediction_errors):.2f}%")
        print(f"Shot {reserved_shot} - Max actual {self.config['selected_data_type']}: {np.max(np.abs(hbt_data)):.2f}")
        print(f"Shot {reserved_shot} - Max predicted {self.config['selected_data_type']}: {np.max(np.abs(predictions)):.2f}")

        return predictions
    
    def run_analysis(self, output_dir: Optional[str] = None) -> Dict:
        """Run the complete untrimmed analysis pipeline."""
        print(f"Starting untrimmed HBT analysis for state {self.config['state']}")
        overall_start = time.perf_counter()
        cache_stats = CameraCacheStats()
        
        # Process all shots
        print("Processing shot data...")
        t0 = time.perf_counter()
        training_data_2D, cut_training_data_2D, flat_training_data, self.valid_shots = self.process_all_shots(
            self.shot_list,
            self.default_frame_count,
            self.camera_depth,
            cache_stats=cache_stats,
        )
        camera_processing_seconds = time.perf_counter() - t0

        if not self.valid_shots:
            raise ValueError(
                f"No usable camera frames found for state {self.config['state']} shots. "
                f"Checked for tif/tiff/png under each shot's CAM-* folders."
            )
        
        # Process reserved shot
        print(f"Processing reserved shot {self.config['reserved_shot']}...")
        reserved_shot_cut_2d = None
        if self.config['reserved_shot'] is not None:
            try:
                folder_path = self.resolve_camera_frames_dir(self.config['reserved_shot'])
            except Exception as e:
                print(f"Error resolving camera frames for RESERVED_SHOT {self.config['reserved_shot']}: {e}")
                folder_path = None
            
            try:
                if folder_path is None:
                    raise ValueError("No camera frames directory found for reserved shot.")
                _, reserved_shot_cut_2d, _ = self.process_shot_data(
                    folder_path,
                    self.default_frame_count,
                    self.camera_depth,
                    shot_num=self.config['reserved_shot'],
                    cache_stats=cache_stats,
                )
                print(f"Successfully processed RESERVED_SHOT {self.config['reserved_shot']} with {len(reserved_shot_cut_2d)} frames")
            except Exception as e:
                print(f"Error processing RESERVED_SHOT {self.config['reserved_shot']}: {e}")
        
        # Load HBT data
        print("Loading HBT data...")
        hbt_data = self.load_hbt_data(self.shot_list, self.valid_shots)
        
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
        for i in range(len(self.valid_shots)):
            shot = self.valid_shots[i]
            if shot == self.config['reserved_shot']:
                continue
            for j in range(len(target_data[i])):
                raw_target_vector.append(target_data[i][j])
        
        if not raw_target_vector:
            raise ValueError("No training samples available after filtering; cannot compute normalization.")

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
        for i in range(len(self.valid_shots)):
            shot = self.valid_shots[i]
            if shot == self.config['reserved_shot']:
                continue
            for j in range(len(target_data[i])):
                target_vector.append(target_data[i][j])
                training_vector.append(training_data[i][j])
        
        # Shuffle the data
        import random
        random.seed(123)
        zip_list = list(zip(target_vector, training_vector))
        random.shuffle(zip_list)
        target_vector, training_vector = zip(*zip_list)
        
        # Convert to numpy arrays and normalize with clipping
        target_vector = np.asarray(target_vector, dtype=np.float32)[:, 0]
        target_vector = np.clip(target_vector, -outlier_threshold, outlier_threshold) / ma_norm
        training_vector = np.asarray(training_vector, dtype=np.float32).reshape(-1, 32, 32, 1)
        
        # Split into training and testing sets
        test_size = 200
        if len(training_vector) <= test_size:
            raise ValueError(
                f"Not enough samples to split: have {len(training_vector)} frames but test_size={test_size}. "
                f"Add more camera frames or reduce test_size."
            )
        testing_inputs = training_vector[-test_size:]
        testing_labels = target_vector[-test_size:]
        training_vector = training_vector[:-test_size]
        target_vector = target_vector[:-test_size]
        
        # Save normalization info (write into output_dir when provided to avoid parallel workers clobbering a shared file)
        self.save_normalization_info(ma_norm, outlier_threshold, output_dir=output_dir)
        
        print('Training shape:', training_vector.shape, 'Target shape:', target_vector.shape)
        print('Testing shape:', testing_inputs.shape, 'Testing label shape:', testing_labels.shape)
        
        # Create and train model
        print("Creating model...")
        model = self.create_model()
        if bool(self.config.get('model_summary', True)):
            model.summary()
        
        print("Training model...")
        train_start = time.perf_counter()
        history = self.train_model(model, training_vector, target_vector)
        train_seconds = time.perf_counter() - train_start
        
        # Evaluate model
        print("Evaluating model...")
        predictions = self.evaluate_model(model, testing_inputs, testing_labels)
        
        # Calculate prediction errors
        prediction_errors = np.abs(testing_labels - predictions[:, 0]) * 100
        print(f"Maximum actual mode amplitude (normalized): {np.max(np.abs(testing_labels)):.2f}")
        print(f"Maximum predicted mode amplitude (normalized): {np.max(np.abs(predictions[:, 0])):.2f}")
        print(f"Mean absolute percentage error: {np.mean(prediction_errors):.2f}%")
        
        # Compute reserved shot predictions
        print("Computing reserved shot predictions...")
        reserved_predictions = self.compute_reserved_shot_predictions(
            model, self.config['reserved_shot'], cut_training_data_2D, reserved_shot_cut_2d,
            target_data, hbt_data['times'], ma_norm
        )
        
        # Save results
        if reserved_predictions is not None:
            time_data = hbt_data['times'][self.shot_list.index(self.config['reserved_shot'])]
            self.save_results(
                self.config['reserved_shot'], 
                target_data[self.shot_list.index(self.config['reserved_shot'])][:, 0], 
                reserved_predictions, 
                time_data,
                output_dir
            )
        
        print("Untrimmed analysis complete!")

        # Persist cache stats + coarse timing so we can verify cache effectiveness.
        stats_out_dir = output_dir or self.config.get('output_dir')
        if stats_out_dir:
            try:
                os.makedirs(stats_out_dir, exist_ok=True)
                payload = {
                    "notebook_type": "untrimmed",
                    "state": int(self.config["state"]),
                    "selected_data_type": str(self.config["selected_data_type"]),
                    "target_frame_count": int(self.default_frame_count),
                    "camera_cache": {
                        "hits": int(cache_stats.hits),
                        "misses": int(cache_stats.misses),
                        "saves": int(cache_stats.saves),
                        "load_seconds": float(cache_stats.load_seconds),
                        "build_seconds": float(cache_stats.build_seconds),
                        "cache_root": self.project_path("data", "cache", "camera_center32_downsampled", str(self.default_frame_count)),
                    },
                    "timing_seconds": {
                        "camera_processing_seconds": float(camera_processing_seconds),
                        "train_seconds": float(train_seconds),
                        "overall_seconds": float(time.perf_counter() - overall_start),
                    },
                }
                with open(os.path.join(stats_out_dir, "camera_cache_stats.json"), "w") as f:
                    json.dump(payload, f, indent=2)
                print(f"Wrote camera cache stats to {self.pretty_path(os.path.join(stats_out_dir, 'camera_cache_stats.json'))}")
            except Exception as e:
                print(f"Warning: failed to write camera cache stats: {e}")
        
        return {
            'model': model,
            'history': history,
            'test_predictions': predictions,
            'reserved_predictions': reserved_predictions,
            'ma_norm': ma_norm,
            'outlier_threshold': outlier_threshold
        }
