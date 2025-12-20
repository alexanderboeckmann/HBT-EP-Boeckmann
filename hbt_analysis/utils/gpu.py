"""
GPU Utilities for HBT Analysis

This module provides GPU configuration and utilities for running analysis
on GPU-enabled servers. It includes memory management, device detection,
and performance monitoring.
"""

import os
import tensorflow as tf
import numpy as np
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def configure_gpu_memory(gpu_memory_limit: Optional[int] = None, 
                        allow_memory_growth: bool = True) -> Dict[str, Any]:
    """
    Configure GPU memory settings for optimal performance.
    
    Args:
        gpu_memory_limit: Maximum GPU memory to use in MB (None for no limit)
        allow_memory_growth: Allow TensorFlow to allocate memory as needed
        
    Returns:
        Dictionary with GPU configuration info
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if not gpus:
        logger.warning("No GPU devices found. Falling back to CPU.")
        return {"gpu_available": False, "device_count": 0}
    
    try:
        # Configure each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, allow_memory_growth)
            
            if gpu_memory_limit is not None:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=gpu_memory_limit
                    )]
                )
        
        # Enable mixed precision for better performance (TensorFlow 2.10+)
        try:
            tf.config.experimental.enable_mixed_precision_policy('mixed_float16')
        except AttributeError:
            # Fallback for older TensorFlow versions
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        logger.info(f"GPU configuration successful. Found {len(gpus)} GPU(s)")
        return {
            "gpu_available": True,
            "device_count": len(gpus),
            "memory_growth": allow_memory_growth,
            "memory_limit": gpu_memory_limit
        }
        
    except RuntimeError as e:
        logger.error(f"GPU configuration failed: {e}")
        return {"gpu_available": False, "error": str(e)}

def get_optimal_batch_size(model: tf.keras.Model, 
                          input_shape: tuple, 
                          max_batch_size: int = 256) -> int:
    """
    Find optimal batch size for GPU memory.
    
    Args:
        model: Keras model
        input_shape: Input data shape
        max_batch_size: Maximum batch size to test
        
    Returns:
        Optimal batch size
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        return 32  # Default CPU batch size
    
    # Test different batch sizes
    for batch_size in [64, 128, 256, 512]:
        if batch_size > max_batch_size:
            break
            
        try:
            # Create dummy data
            dummy_data = tf.random.normal((batch_size,) + input_shape)
            
            # Test forward pass
            with tf.device('/GPU:0'):
                _ = model(dummy_data, training=False)
            
            logger.info(f"Batch size {batch_size} works on GPU")
            
        except tf.errors.ResourceExhaustedError:
            logger.info(f"Batch size {batch_size} exceeds GPU memory")
            return max(32, batch_size // 2)
        except Exception as e:
            logger.warning(f"Error testing batch size {batch_size}: {e}")
            return max(32, batch_size // 2)
    
    return min(256, max_batch_size)

def monitor_gpu_usage() -> Dict[str, Any]:
    """
    Monitor current GPU usage.
    
    Returns:
        Dictionary with GPU usage statistics
    """
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if not gpus:
            return {"gpu_available": False}
        
        # Get GPU memory info
        gpu_details = []
        for i, gpu in enumerate(gpus):
            try:
                # This is a simplified version - in practice you'd use nvidia-ml-py
                # or similar for detailed memory monitoring
                gpu_details.append({
                    "device_id": i,
                    "name": gpu.name,
                    "memory_limit": "Unknown"  # Would need nvidia-ml-py for actual values
                })
            except Exception as e:
                logger.warning(f"Could not get details for GPU {i}: {e}")
        
        return {
            "gpu_available": True,
            "device_count": len(gpus),
            "devices": gpu_details
        }
        
    except Exception as e:
        logger.error(f"GPU monitoring failed: {e}")
        return {"gpu_available": False, "error": str(e)}

def create_gpu_optimized_model(input_shape: tuple, 
                              conv2d_neurons: list,
                              conv2d_size: list,
                              dense_layer_neurons: list,
                              num_conv2d_layers: int,
                              num_dense_layers: int,
                              max_pooling_size: tuple,
                              activation_func: str = 'relu',
                              loss_func: str = 'mae',
                              optimizer_func: str = 'adam') -> tf.keras.Model:
    """
    Create a GPU-optimized model with mixed precision support.
    
    Args:
        input_shape: Input data shape
        conv2d_neurons: List of neurons for Conv2D layers
        conv2d_size: List of kernel sizes for Conv2D layers
        dense_layer_neurons: List of neurons for dense layers
        num_conv2d_layers: Number of Conv2D layers
        num_dense_layers: Number of dense layers
        max_pooling_size: Max pooling size
        activation_func: Activation function
        loss_func: Loss function
        optimizer_func: Optimizer function
        
    Returns:
        Compiled Keras model optimized for GPU
    """
    # Use mixed precision policy
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(shape=input_shape))
    
    # Add Conv2D layers
    for i in range(num_conv2d_layers):
        model.add(tf.keras.layers.Conv2D(
            conv2d_neurons[i], 
            conv2d_size[i], 
            padding='same', 
            activation=activation_func
        ))
        model.add(tf.keras.layers.MaxPooling2D(max_pooling_size, padding='same'))
    
    # Flatten and add dense layers
    model.add(tf.keras.layers.Flatten())
    for i in range(num_dense_layers):
        model.add(tf.keras.layers.Dense(
            dense_layer_neurons[i], 
            activation=activation_func
        ))
        model.add(tf.keras.layers.Dropout(0.2))
    
    # Output layer
    model.add(tf.keras.layers.Dense(1, dtype='float32'))  # Ensure output is float32
    
    # Compile with mixed precision optimizer
    if optimizer_func == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    elif optimizer_func == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    elif optimizer_func == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    else:
        optimizer = optimizer_func
    
    # Wrap optimizer for mixed precision
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(optimizer=optimizer, loss=loss_func)
    
    return model

def setup_gpu_environment():
    """
    Set up the complete GPU environment for HBT analysis.
    
    Returns:
        Dictionary with environment configuration
    """
    # Configure GPU memory
    gpu_config = configure_gpu_memory(
        gpu_memory_limit=None,  # Use all available memory
        allow_memory_growth=True
    )
    
    # Set environment variables for optimal performance
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = '2'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
    
    # Enable XLA compilation for better performance
    tf.config.optimizer.set_jit(True)
    
    logger.info("GPU environment setup complete")
    return gpu_config

if __name__ == "__main__":
    # Test GPU configuration
    logging.basicConfig(level=logging.INFO)
    config = setup_gpu_environment()
    print(f"GPU Configuration: {config}")
    
    # Test model creation
    model = create_gpu_optimized_model(
        input_shape=(32, 32, 1),
        conv2d_neurons=[32, 32, 16],
        conv2d_size=[(8, 8), (8, 8), (4, 4)],
        dense_layer_neurons=[64, 32],
        num_conv2d_layers=3,
        num_dense_layers=2,
        max_pooling_size=(2, 2)
    )
    print(f"Model created successfully: {model.summary()}")
