#!/usr/bin/env python
# coding: utf-8

"""
Base HBT Analysis Class

This module provides the core functionality for analysis of plasma data.
It contains the shared logic used across all analysis variants.

Key features:
- Common data loading and preprocessing
- Model architecture definition
- Training and evaluation logic
- Result saving and metrics calculation
- GPU support (optional)

This base class is extended by specialized classes for different analysis types.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import glob
import random
import ast
from PIL import Image
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional, Union
import sys

from ..utils.gpu import (
    configure_gpu_memory, 
    get_optimal_batch_size, 
    create_gpu_optimized_model,
    setup_gpu_environment
)


class HBTAnalysisBase:
    """
    Base class for HBT analysis providing common functionality.
    
    This class contains all the shared logic used across different analysis variants:
    - Data loading and preprocessing
    - Model creation and training
    - Evaluation and metrics
    - Result saving
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the HBT analysis with configuration.
        
        Args:
            config: Dictionary containing analysis configuration parameters
        """
        self.config = config
        self.project_root = str(Path(__file__).resolve().parents[3])
        
        # Initialize shot paths
        self.shot_paths = {
            'new_shots': {
                'range': (119591, 119769),
                'data_path': None,
                'hbt_path': None,
                'ip_path': None
            },
            'old_shots': {
                'range': (114407, 114473),
                'data_path': None,
                'hbt_path': None,
                'ip_path': None
            }
        }
        
        # Initialize paths
        self._initialize_paths()
        
        # GPU configuration
        self.use_gpu = config.get('use_gpu', False)
        self.gpu_memory_limit = config.get('gpu_memory_limit', None)
        self._setup_gpu()
        
        # Initialize shot lists
        self.shot_list = self._get_shot_list(config['state'])
        self.notebook_type = config.get('notebook_type', 'base')
        
        # Constants
        self.camera_depth = 65535.0
        self.default_frame_count = config.get('default_frame_count', 800)
        
    def _initialize_paths(self):
        """Initialize absolute paths for shot data."""
        for shot_type in self.shot_paths:
            self.shot_paths[shot_type]['data_path'] = self.project_path('data', 'shots', 'new' if 'new' in shot_type else 'old')
            self.shot_paths[shot_type]['hbt_path'] = self.project_path('data', 'shots', 'new' if 'new' in shot_type else 'old')
            self.shot_paths[shot_type]['ip_path'] = self.project_path('data', 'shots', 'new' if 'new' in shot_type else 'old')
    
    def _setup_gpu(self):
        """Setup GPU configuration if requested."""
        if self.use_gpu:
            print("Configuring GPU environment...")
            gpu_config = configure_gpu_memory(
                gpu_memory_limit=self.gpu_memory_limit,
                allow_memory_growth=True
            )
            
            if gpu_config.get('gpu_available', False):
                print(f"GPU setup successful: {gpu_config['device_count']} GPU(s) available")
                setup_gpu_environment()
            else:
                print("GPU not available, falling back to CPU")
                self.use_gpu = False
        else:
            print("Using CPU execution")
    
    def project_path(self, *parts):
        """Create absolute path from project root."""
        return os.path.join(self.project_root, *parts)
    
    def _get_shot_list(self, state: int) -> List[int]:
        """Get shot list based on state."""
        if state == 1:
            return [119591, 119599, 119601, 119646, 119648, 119653, 119654, 119658, 119659,
                    119661, 119662, 119663, 119665, 119666, 119667, 119669, 119670, 119671,
                    119673, 119675, 119748, 119750, 119751, 119752, 119754, 119755, 119756,
                    119757, 119760, 119761, 119762, 119763, 119764, 119766, 119767, 119768, 119769]
        elif state == 2:
            return [114407, 114408, 114411, 114412, 114413, 114415, 114416, 114417, 114418, 114419,
                    114420, 114422, 114424, 114425, 114428, 114429, 114431, 114432, 114433, 114434,
                    114435, 114436, 114438, 114439, 114441, 114443, 114444, 114445, 114448, 114450,
                    114451, 114453, 114454, 114455, 114456, 114457, 114458, 114460, 114462, 114464,
                    114467, 114468, 114472, 114473]
        elif state == 3:
            return [119591, 119599, 119601, 119646, 119648, 119653, 119654, 119658, 119659,
                    119661, 119662, 119663, 119665, 119666, 119667, 119669, 119670, 119671,
                    119673, 119675, 119748, 119750, 119751, 119752, 119754, 119755, 119756,
                    119757, 119760, 119761, 119762, 119763, 119764, 119766, 119767, 119768, 119769,
                    114407, 114408, 114411, 114412, 114413, 114415, 114416, 114417, 114418, 114419,
                    114420, 114422, 114424, 114425, 114428, 114429, 114431, 114432, 114433, 114434,
                    114435, 114436, 114438, 114439, 114441, 114443, 114444, 114445, 114448, 114450,
                    114451, 114453, 114454, 114455, 114456, 114457, 114458, 114460, 114462, 114464,
                    114467, 114468, 114472, 114473]
        else:
            raise ValueError(f"Invalid state: {state}. Must be 1, 2, or 3.")
    
    def get_paths_for_shot(self, shot_num: int) -> Tuple[str, str, str]:
        """Return data, HBT, and IP paths for a given shot number."""
        for paths in self.shot_paths.values():
            if paths['range'][0] <= shot_num <= paths['range'][1]:
                return paths['data_path'], paths['hbt_path'], paths['ip_path']
        raise ValueError(f"Shot number {shot_num} is not in the defined ranges.")
    
    def smooth_data(self, data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Smooth data using a moving average with edge padding."""
        smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        pad = window_size // 2
        smoothed = np.pad(smoothed, (pad, pad), mode='edge')
        return smoothed[:len(data)]
    
    def process_shot_data(self, folder_path: str, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Process TIFF images in a folder. To be implemented by subclasses.
        
        Args:
            folder_path: Path to folder containing TIFF files
            **kwargs: Additional processing parameters
            
        Returns:
            Tuple of (shot_2d, cut_2d, flat_data, frame_count)
        """
        raise NotImplementedError("Subclasses must implement process_shot_data")
    
    def load_hbt_data(self, shot_list: List[int], valid_shots: List[int], **kwargs) -> Dict:
        """
        Load and format HBT data. To be implemented by subclasses.
        
        Args:
            shot_list: List of all shot numbers
            valid_shots: List of valid shot numbers (excluding reserved shot)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing formatted HBT data
        """
        raise NotImplementedError("Subclasses must implement load_hbt_data")
    
    def create_model(self) -> tf.keras.Model:
        """Create the neural network model."""
        # Parse configuration
        conv2d_size = [
            tuple(int(x) for x in ast.literal_eval(size) if isinstance(x, (int, float)))
            if isinstance(size, str) else tuple(int(x) for x in size)
            for size in self.config['conv2d_size']
        ]
        max_pooling_size = (
            tuple(int(x) for x in ast.literal_eval(self.config['max_pooling_size']) if isinstance(x, (int, float)))
            if isinstance(self.config['max_pooling_size'], str) else tuple(int(x) for x in self.config['max_pooling_size'])
        )
        
        if self.use_gpu:
            # Create GPU-optimized model
            model = create_gpu_optimized_model(
                input_shape=(32, 32, 1),
                conv2d_neurons=self.config['conv2d_neurons'],
                conv2d_size=conv2d_size,
                dense_layer_neurons=self.config['dense_layer_neurons'],
                num_conv2d_layers=self.config['num_conv2d_layers'],
                num_dense_layers=self.config['num_dense_layers'],
                max_pooling_size=max_pooling_size,
                activation_func=self.config['activation_func'],
                loss_func=self.config['loss_func'],
                optimizer_func=self.config['optimizer_func']
            )
            
            # Optimize batch size for GPU
            if self.config.get('batch_size', 32) == 32:
                optimal_batch = get_optimal_batch_size(model, (32, 32, 1))
                self.config['batch_size'] = min(optimal_batch, 256)
                print(f"Optimized batch size for GPU: {self.config['batch_size']}")
        else:
            # Create standard CPU model
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.InputLayer(shape=(32, 32, 1)))

            for i in range(self.config['num_conv2d_layers']):
                model.add(tf.keras.layers.Conv2D(
                    self.config['conv2d_neurons'][i], 
                    conv2d_size[i], 
                    padding='same', 
                    activation=self.config['activation_func']
                ))
                model.add(tf.keras.layers.MaxPooling2D(max_pooling_size, padding='same'))

            model.add(tf.keras.layers.Flatten())
            for i in range(self.config['num_dense_layers']):
                model.add(tf.keras.layers.Dense(
                    self.config['dense_layer_neurons'][i], 
                    activation=self.config['activation_func']
                ))
                model.add(tf.keras.layers.Dropout(0.2))

            model.add(tf.keras.layers.Dense(1))
            model.compile(
                optimizer=self.config['optimizer_func'], 
                loss=self.config['loss_func']
            )
        
        return model
    
    def train_model(self, model: tf.keras.Model, training_data: np.ndarray, 
                   target_data: np.ndarray) -> tf.keras.callbacks.History:
        """Train the model with early stopping."""
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=self.config.get('early_stopping_patience', 20),
            min_delta=self.config.get('early_stopping_min_delta', 0.01)
        )
        
        history = model.fit(
            training_data, target_data,
            epochs=self.config['epoch_num'],
            validation_split=self.config['validation_split'],
            batch_size=self.config.get('batch_size', 32),
            verbose=1,
            callbacks=[early_stop]
        )
        
        return history
    
    def evaluate_model(self, model: tf.keras.Model, test_data: np.ndarray, 
                      test_labels: np.ndarray) -> np.ndarray:
        """Evaluate model on test data."""
        predictions = model.predict(test_data)
        return predictions
    
    def compute_reserved_shot_predictions(self, model: tf.keras.Model, 
                                        reserved_shot: int, **kwargs) -> Optional[np.ndarray]:
        """
        Compute predictions for the reserved shot. To be implemented by subclasses.
        
        Args:
            model: Trained model
            reserved_shot: Shot number to predict
            **kwargs: Additional parameters
            
        Returns:
            Predictions array or None if no data available
        """
        raise NotImplementedError("Subclasses must implement compute_reserved_shot_predictions")
    
    def save_results(self, reserved_shot: int, true_data: np.ndarray, 
                    predictions: np.ndarray, time_data: np.ndarray, 
                    output_dir: Optional[str] = None):
        """Save analysis results to files."""
        if output_dir is None:
            output_dir = self.project_path('data', 'predictions')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        np.save(os.path.join(output_dir, f'results_{self.notebook_type}_state_{self.config["state"]}_{self.config["selected_data_type"]}_true.npy'), true_data)
        np.save(os.path.join(output_dir, f'results_{self.notebook_type}_state_{self.config["state"]}_{self.config["selected_data_type"]}_pred.npy'), predictions)
        np.save(os.path.join(output_dir, f'results_{self.notebook_type}_state_{self.config["state"]}_{self.config["selected_data_type"]}_time.npy'), time_data)
        
        print(f"Results saved to {output_dir}")
    
    def save_normalization_info(self, ma_norm: float, outlier_threshold: float):
        """Save normalization information."""
        pred_dir = self.project_path('data', 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        
        normalization_filename = os.path.join(
            pred_dir, 
            f"normalization_{self.notebook_type}_state_{self.config['state']}.npz"
        )
        
        np.savez(normalization_filename,
                 ma_norm=ma_norm,
                 outlier_threshold=outlier_threshold,
                 selected_data_type=self.config['selected_data_type'])
        
        print(f"Saved normalization info to {normalization_filename}")
    
    def run_analysis(self, output_dir: Optional[str] = None) -> Dict:
        """
        Run the complete analysis pipeline. To be implemented by subclasses.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing analysis results
        """
        raise NotImplementedError("Subclasses must implement run_analysis")
    
    @staticmethod
    def create_argument_parser() -> argparse.ArgumentParser:
        """Create command-line argument parser with common arguments."""
        parser = argparse.ArgumentParser(description='HBT Analysis Script')
        
        # Core parameters
        parser.add_argument('--state', type=int, required=True, help='State number (1, 2, or 3)')
        parser.add_argument('--selected_data_type', type=str, default='ma2', 
                          help='Data type: ma1-ma4 (mode amplitude 1-4) or mp1-mp4 (mode phase 1-4) (default: ma2)')
        parser.add_argument('--RESERVED_SHOT', type=int, help='Reserved shot number')
        parser.add_argument('--output_dir', type=str, help='Output directory for saving results')
        
        # Training parameters
        parser.add_argument('--EPOCH_NUM', type=int, default=15, help='Number of epochs (default: 15)')
        parser.add_argument('--VALIDATION_SPLIT', type=float, default=0.2, help='Validation split (default: 0.2)')
        parser.add_argument('--BATCH_SIZE', type=int, default=32, help='Batch size (default: 32)')
        
        # Model architecture
        parser.add_argument('--ACTIVATION_FUNC', type=str, default='relu', help='Activation function (default: relu)')
        parser.add_argument('--LOSS_FUNC', type=str, default='mae', help='Loss function (default: mae)')
        parser.add_argument('--OPTIMIZER_FUNC', type=str, default='adam', help='Optimizer function (default: adam)')
        parser.add_argument('--NUM_CONV2D_LAYERS', type=int, default=3, help='Number of Conv2D layers (default: 3)')
        parser.add_argument('--NUM_DENSE_LAYERS', type=int, default=2, help='Number of dense layers (default: 2)')
        parser.add_argument('--CONV2D_NEURONS', type=str, default='[32, 32, 16]', help='Conv2D neurons as JSON list (default: [32, 32, 16])')
        parser.add_argument('--CONV2D_SIZE', type=str, default='[(8, 8), (8, 8), (4, 4)]', help='Conv2D sizes as JSON list (default: [(8, 8), (8, 8), (4, 4)])')
        parser.add_argument('--DENSE_LAYER_NEURONS', type=str, default='[64, 32]', help='Dense layer neurons as JSON list (default: [64, 32])')
        parser.add_argument('--MAX_POOLING_SIZE', type=str, default='(2, 2)', help='Max pooling size as JSON tuple (default: (2, 2))')
        
        # Early stopping
        parser.add_argument('--EARLY_STOPPING_PATIENCE', type=int, default=20, help='Early stopping patience (default: 20)')
        parser.add_argument('--EARLY_STOPPING_MIN_DELTA', type=float, default=0.01, help='Early stopping minimum delta (default: 0.01)')
        
        # Data processing
        parser.add_argument('--OUTLIER_CUTOFF', type=float, default=99, help='Outlier cutoff percentile (default: 99)')
        
        # GPU options
        parser.add_argument('--USE_GPU', action='store_true', help='Use GPU acceleration')
        parser.add_argument('--GPU_MEMORY_LIMIT', type=int, help='GPU memory limit in MB')
        
        return parser
    
    @staticmethod
    def parse_arguments(args: argparse.Namespace) -> Dict:
        """Parse command-line arguments into configuration dictionary."""
        config = {
            'state': args.state,
            'selected_data_type': args.selected_data_type,
            'reserved_shot': args.RESERVED_SHOT,
            'epoch_num': args.EPOCH_NUM,
            'validation_split': args.VALIDATION_SPLIT,
            'batch_size': args.BATCH_SIZE,
            'activation_func': args.ACTIVATION_FUNC,
            'loss_func': args.LOSS_FUNC,
            'optimizer_func': args.OPTIMIZER_FUNC,
            'outlier_cutoff': args.OUTLIER_CUTOFF,
            'num_conv2d_layers': args.NUM_CONV2D_LAYERS,
            'num_dense_layers': args.NUM_DENSE_LAYERS,
            'conv2d_neurons': ast.literal_eval(args.CONV2D_NEURONS),
            'conv2d_size': ast.literal_eval(args.CONV2D_SIZE),
            'dense_layer_neurons': ast.literal_eval(args.DENSE_LAYER_NEURONS),
            'max_pooling_size': ast.literal_eval(args.MAX_POOLING_SIZE),
            'early_stopping_patience': args.EARLY_STOPPING_PATIENCE,
            'early_stopping_min_delta': args.EARLY_STOPPING_MIN_DELTA,
            'use_gpu': args.USE_GPU,
            'gpu_memory_limit': args.GPU_MEMORY_LIMIT,
            'output_dir': args.output_dir
        }
        
        # Set default reserved shot based on state
        if config['reserved_shot'] is None:
            if config['state'] == 1:
                config['reserved_shot'] = 119671
            elif config['state'] == 2:
                config['reserved_shot'] = 114458
            else:
                config['reserved_shot'] = 119671
        
        return config
