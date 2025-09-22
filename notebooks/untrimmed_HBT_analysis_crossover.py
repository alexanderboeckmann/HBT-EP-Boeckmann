#!/usr/bin/env python
# coding: utf-8

"""
Untrimmed HBT Analysis Crossover Script

This script performs analysis on raw, untrimmed plasma data
with crossover validation methodology. It trains CNN models using a specific data splitting
approach where different states are used for training and testing.

Key features:
- Uses raw (untrimmed) plasma data without extensive preprocessing
- Implements crossover validation between different plasma states
- Trains models on one state and tests on another
- Supports configurable model architecture and training parameters
- Saves results for genetic algorithm optimization

This version is specifically designed for parameter optimization workflows
where different state combinations are tested systematically using raw data.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import PIL
from PIL import Image
import glob
import random
import ast
from pathlib import Path
import argparse

# Parameters
state = None
selected_data_type = None
RESERVED_SHOT = None
EPOCH_NUM = None
VALIDATION_SPLIT = None
ACTIVATION_FUNC = None
LOSS_FUNC = None
OPTIMIZER_FUNC = None
OUTLIER_CUTOFF = None
NUM_CONV2D_LAYERS = None
NUM_DENSE_LAYERS = None
CONV2D_NEURONS = None
CONV2D_SIZE = None
DENSE_LAYER_NEURONS = None
MAX_POOLING_SIZE = None

# Command-line argument parsing
parser = argparse.ArgumentParser(description='HBT Analysis Crossover Script')
parser.add_argument('--state', type=int, help='State number (1, 2, or 3)')
parser.add_argument('--selected_data_type', type=str, default='ma2', help='Data type (default: ma2)')
parser.add_argument('--RESERVED_SHOT', type=int, help='Reserved shot number')
parser.add_argument('--EPOCH_NUM', type=int, default=15, help='Number of epochs (default: 15)')
parser.add_argument('--VALIDATION_SPLIT', type=float, default=0.2, help='Validation split (default: 0.2)')
parser.add_argument('--ACTIVATION_FUNC', type=str, default='relu', help='Activation function (default: relu)')
parser.add_argument('--LOSS_FUNC', type=str, default='mae', help='Loss function (default: mae)')
parser.add_argument('--OPTIMIZER_FUNC', type=str, default='adam', help='Optimizer function (default: adam)')
parser.add_argument('--OUTLIER_CUTOFF', type=float, default=99, help='Outlier cutoff percentile (default: 99)')
parser.add_argument('--NUM_CONV2D_LAYERS', type=int, default=2, help='Number of Conv2D layers (default: 2)')
parser.add_argument('--NUM_DENSE_LAYERS', type=int, default=1, help='Number of dense layers (default: 1)')
parser.add_argument('--CONV2D_NEURONS', type=str, default='[16, 16]', help='Conv2D neurons as JSON list (default: [16, 16])')
parser.add_argument('--CONV2D_SIZE', type=str, default='[(8, 8), (8, 8)]', help='Conv2D sizes as JSON list (default: [(8, 8), (8, 8)])')
parser.add_argument('--DENSE_LAYER_NEURONS', type=str, default='[10]', help='Dense layer neurons as JSON list (default: [10])')
parser.add_argument('--MAX_POOLING_SIZE', type=str, default='(4, 4)', help='Max pooling size as JSON tuple (default: (4, 4))')
parser.add_argument('--output_dir', type=str, help='Output directory for saving results (default: data/predictions)')

args = parser.parse_args()

# Use command-line arguments if provided, otherwise use manual override
if args.state is not None:
    # Use command-line arguments
    state = args.state
    selected_data_type = args.selected_data_type
    RESERVED_SHOT = args.RESERVED_SHOT
    EPOCH_NUM = args.EPOCH_NUM
    VALIDATION_SPLIT = args.VALIDATION_SPLIT
    ACTIVATION_FUNC = args.ACTIVATION_FUNC
    LOSS_FUNC = args.LOSS_FUNC
    OPTIMIZER_FUNC = args.OPTIMIZER_FUNC
    OUTLIER_CUTOFF = args.OUTLIER_CUTOFF
    NUM_CONV2D_LAYERS = args.NUM_CONV2D_LAYERS
    NUM_DENSE_LAYERS = args.NUM_DENSE_LAYERS
    CONV2D_NEURONS = ast.literal_eval(args.CONV2D_NEURONS)
    CONV2D_SIZE = ast.literal_eval(args.CONV2D_SIZE)
    DENSE_LAYER_NEURONS = ast.literal_eval(args.DENSE_LAYER_NEURONS)
    MAX_POOLING_SIZE = ast.literal_eval(args.MAX_POOLING_SIZE)
    
    # Set RESERVED_SHOT based on state if not provided
    if RESERVED_SHOT is None:
        if state == 1:
            RESERVED_SHOT = 119671
        elif state == 2:
            RESERVED_SHOT = 114458
        else:
            RESERVED_SHOT = 119671
else:
    # Manual override logic - toggle this to True to use manual values
    manual = False
    
    if manual:
        state = 1  # state 1, 2, or 3
        selected_data_type = 'ma2'  # ma2 basically always
        if state == 1:
            RESERVED_SHOT = 119671  # for state 1
        elif state == 2:
            RESERVED_SHOT = 114458  # for state 2
        else:
            RESERVED_SHOT = 119671  # Default for state 3
        EPOCH_NUM = 15
        VALIDATION_SPLIT = 0.2
        ACTIVATION_FUNC = 'relu'
        LOSS_FUNC = 'mae'
        OPTIMIZER_FUNC = 'adam'
        OUTLIER_CUTOFF = 99
        NUM_CONV2D_LAYERS = 2
        NUM_DENSE_LAYERS = 1
        CONV2D_NEURONS = [16, 16]
        CONV2D_SIZE = [(8, 8), (8, 8)]
        DENSE_LAYER_NEURONS = [10]
        MAX_POOLING_SIZE = (4, 4)


# Define shot lists and paths
SHOT_PATHS = {
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

# Centralized project root and path utilities
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])

def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)

# Initialize absolute paths in SHOT_PATHS
SHOT_PATHS['new_shots']['data_path'] = project_path('data', 'shots', 'new')
SHOT_PATHS['new_shots']['hbt_path'] = project_path('data', 'shots', 'new')
SHOT_PATHS['new_shots']['ip_path'] = project_path('data', 'shots', 'new')
SHOT_PATHS['old_shots']['data_path'] = project_path('data', 'shots', 'old')
SHOT_PATHS['old_shots']['hbt_path'] = project_path('data', 'shots', 'old')
SHOT_PATHS['old_shots']['ip_path'] = project_path('data', 'shots', 'old')

if state == 1:
    shot_list = [119591, 119599, 119601, 119646, 119648, 119653, 119654, 119658, 119659,
                 119661, 119662, 119663, 119665, 119666, 119667, 119669, 119670, 119671,
                 119673, 119675, 119748, 119750, 119751, 119752, 119754, 119755, 119756,
                 119757, 119760, 119761, 119762, 119763, 119764, 119766, 119767, 119768, 119769]
elif state == 2:
    shot_list = [114407, 114408, 114411, 114412, 114413, 114415, 114416, 114417, 114418, 114419,
                 114420, 114422, 114424, 114425, 114428, 114429, 114431, 114432, 114433, 114434,
                 114435, 114436, 114438, 114439, 114441, 114443, 114444, 114445, 114448, 114450,
                 114451, 114453, 114454, 114455, 114456, 114457, 114458, 114460, 114462, 114464,
                 114467, 114468, 114472, 114473]
elif state == 3:
    shot_list = [119591, 119599, 119601, 119646, 119648, 119653, 119654, 119658, 119659,
                 119661, 119662, 119663, 119665, 119666, 119667, 119669, 119670, 119671,
                 119673, 119675, 119748, 119750, 119751, 119752, 119754, 119755, 119756,
                 119757, 119760, 119761, 119762, 119763, 119764, 119766, 119767, 119768, 119769,
                 114407, 114408, 114411, 114412, 114413, 114415, 114416, 114417, 114418, 114419,
                 114420, 114422, 114424, 114425, 114428, 114429, 114431, 114432, 114433, 114434,
                 114435, 114436, 114438, 114439, 114441, 114443, 114444, 114445, 114448, 114450,
                 114451, 114453, 114454, 114455, 114456, 114457, 114458, 114460, 114462, 114464,
                 114467, 114468, 114472, 114473]

notebook_type = 'untrimmed'
CAMERA_DEPTH = 65535.0
DEFAULT_FRAME_COUNT = 800

def get_paths_for_shot(shot_num):
    """Return data and HBT paths for a given shot number."""
    for paths in SHOT_PATHS.values():
        if paths['range'][0] <= shot_num <= paths['range'][1]:
            return paths['data_path'], paths['hbt_path']
    raise ValueError(f"Shot number {shot_num} is not in the defined ranges.")

def process_shot_data(folder_path, max_frames=DEFAULT_FRAME_COUNT, max_pixel_value=CAMERA_DEPTH):
    """Process TIFF images in a folder, cropping to 32x32 and normalizing, up to max_frames."""
    tiff_files = sorted(glob.glob(os.path.join(folder_path, "*.tiff")))
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {folder_path}")
    
    # Limit to max_frames
    tiff_files = tiff_files[:max_frames]
    
    shot_2d = []
    for tiff_file in tiff_files:
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
    actual_frames = len(shot_2d)
    return shot_2d, shot_2d, shot_2d.reshape(actual_frames, -1), actual_frames

def process_all_shots(shot_list):
    """Process all shots for untrimmed data, returning 2D, cut, flat data, valid shots, and frame counts."""
    training_data_2D, cut_training_data_2D, flat_training_data = [], [], []
    valid_shots, actual_frame_counts = [], []
    
    for shot in shot_list:
        if shot == RESERVED_SHOT:
            continue
        data_path, _ = get_paths_for_shot(shot)
        folder_path = os.path.join(data_path, str(shot), 'CAM-26731/tiff/')
        try:
            shot_2d, cut_2d, flat_data, actual_frames = process_shot_data(folder_path)
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

# Process all shots
training_data_2D, cut_training_data_2D, flat_training_data, valid_shots, actual_frame_counts = process_all_shots(shot_list)

# Process RESERVED_SHOT separately for testing
reserved_shot_data_2d = None
reserved_shot_cut_2d = None
reserved_shot_flat = None
reserved_shot_frame_count = None
if RESERVED_SHOT is not None:
    data_path, _ = get_paths_for_shot(RESERVED_SHOT)
    folder_path = os.path.join(data_path, str(RESERVED_SHOT), 'CAM-26731/tiff/')
    
    try:
        shot_2d, cut_2d, flat_data, actual_frames = process_shot_data(folder_path)
        reserved_shot_data_2d = shot_2d
        reserved_shot_cut_2d = cut_2d
        reserved_shot_flat = flat_data
        reserved_shot_frame_count = actual_frames
        print(f"Successfully processed RESERVED_SHOT {RESERVED_SHOT} with {actual_frames} frames")
    except Exception as e:
        print(f"Error processing RESERVED_SHOT {RESERVED_SHOT}: {e}")

# Set RESERVED_SHOT if None
if RESERVED_SHOT is None:
    RESERVED_SHOT = random.choice(valid_shots)
    print(f"Randomly selected reserved shot: {RESERVED_SHOT}")
    data_path, _ = get_paths_for_shot(RESERVED_SHOT)
    folder_path = os.path.join(data_path, str(RESERVED_SHOT), 'CAM-26731/tiff/')
    shot_2d, cut_2d, flat_data, actual_frames = process_shot_data(folder_path)
    reserved_shot_data_2d = shot_2d
    reserved_shot_cut_2d = cut_2d
    reserved_shot_flat = flat_data
    reserved_shot_frame_count = actual_frames

def load_hbt_data_untrimmed(shot_list, valid_shots, selected_data_type):
    """Load and format HBT data for untrimmed analysis."""
    hbt_data = {'amplitudes': [], 'phases': [], 'times': []}
    
    mode_index = {'ma1': 0, 'ma2': 1, 'ma3': 2, 'ma4': 3, 'mp1': 0, 'mp2': 1, 'mp3': 2, 'mp4': 3}
    data_type = 'amplitudes' if selected_data_type.startswith('ma') else 'phases'
    mode_num = mode_index[selected_data_type]
    
    for shot in shot_list:
        _, hbt_path = get_paths_for_shot(shot)
        mode_amp_data = np.load(os.path.join(hbt_path, f'{shot}m2Amp.npy'))
        mode_phase_data = np.load(os.path.join(hbt_path, f'{shot}m2Phase.npy'))
        time_data = np.load(os.path.join(hbt_path, f'{shot}time.npy'))
        
        # Downsample to DEFAULT_FRAME_COUNT if necessary
        if len(time_data) > DEFAULT_FRAME_COUNT:
            frame_ratio = len(time_data) // DEFAULT_FRAME_COUNT
            time_data = time_data[::frame_ratio][:DEFAULT_FRAME_COUNT]
            amp_data = mode_amp_data[::frame_ratio][:DEFAULT_FRAME_COUNT]
            phase_data = mode_phase_data[::frame_ratio][:DEFAULT_FRAME_COUNT]
        else:
            amp_data = mode_amp_data
            phase_data = mode_phase_data
        # Ensure shape (-1, 1) for downstream indexing
        amp_data = np.asarray(amp_data, dtype=np.float32).reshape(-1, 1)
        phase_data = np.asarray(phase_data, dtype=np.float32).reshape(-1, 1)

        hbt_data['amplitudes'].append(amp_data)
        hbt_data['phases'].append(phase_data)
        hbt_data['times'].append(time_data)
    
    # Select relevant data
    selected_hbt_data = [hbt_data[data_type][shot_list.index(shot)] for shot in valid_shots]
    hbt_time_data = [hbt_data['times'][shot_list.index(shot)] for shot in valid_shots]
    
    # For reserved shot
    reserved_shot_hbt = None
    reserved_shot_time = None
    if RESERVED_SHOT is not None:
        res_idx = shot_list.index(RESERVED_SHOT)
        reserved_shot_hbt = hbt_data[data_type][res_idx].reshape(-1, 1)
        reserved_shot_time = hbt_data['times'][res_idx]
        print(f"RESERVED_SHOT {RESERVED_SHOT}: HBT {selected_data_type} frames={len(reserved_shot_hbt)}")
        print(f"RESERVED_SHOT {RESERVED_SHOT}: Time data frames={len(reserved_shot_time)}")
    
    return selected_hbt_data, hbt_time_data, reserved_shot_hbt, reserved_shot_time

# Load HBT data
hbt_dataType, hbt_time_data, reserved_shot_hbt, reserved_shot_time = load_hbt_data_untrimmed(
    shot_list, valid_shots, selected_data_type
)

# Prepare data for model
target_data = hbt_dataType
training_data = cut_training_data_2D

# Normalization
raw_target_vector = []
for i, shot in enumerate(valid_shots):
    if shot == RESERVED_SHOT:
        continue
    for j in range(actual_frame_counts[i]):
        raw_target_vector.append(target_data[i][j])

raw_target_vector = np.asarray(raw_target_vector, dtype=np.float32)[:, 0]
percentile_cutoff = np.percentile(np.abs(raw_target_vector), OUTLIER_CUTOFF)
ma_norm = percentile_cutoff if percentile_cutoff > 0 else 1.0
outlier_threshold = 3 * ma_norm
outliers = np.abs(raw_target_vector) > outlier_threshold
print(f"Normalization factor ({OUTLIER_CUTOFF} percentile): {ma_norm:.2f}")
print(f"Number of outliers (|value| > {outlier_threshold:.2f}): {np.sum(outliers)}")

raw_target_vector = np.clip(raw_target_vector, -outlier_threshold, outlier_threshold)

# Reshape training data and labels
target_vector = []
training_vector = []
for i, shot in enumerate(valid_shots):
    if shot == RESERVED_SHOT:
        continue
    for j in range(actual_frame_counts[i]):
        target_vector.append(target_data[i][j])
        training_vector.append(training_data[i][j])

random.seed(123)
zip_list = list(zip(target_vector, training_vector))
random.shuffle(zip_list)
target_vector, training_vector = zip(*zip_list)

target_vector = np.asarray(target_vector, dtype=np.float32)[:, 0]
target_vector = np.clip(target_vector, -outlier_threshold, outlier_threshold) / ma_norm
training_vector = np.asarray(training_vector, dtype=np.float32).reshape(-1, 32, 32, 1)

test_size = 400
testing_inputs = training_vector[-test_size:]
testing_labels = target_vector[-test_size:]
training_vector = training_vector[:-test_size]
target_vector = target_vector[:-test_size]

# Save normalization in data/predictions
pred_dir_npz = project_path('data', 'predictions')
os.makedirs(pred_dir_npz, exist_ok=True)
normalization_filename = os.path.join(pred_dir_npz, f"normalization_{notebook_type}_state_{state}.npz")
np.savez(normalization_filename,
         ma_norm=ma_norm,
         outlier_threshold=outlier_threshold,
         selected_data_type=selected_data_type)
print(f"Saved normalization info to {normalization_filename}")

print('Training shape:', training_vector.shape, 'Target shape:', target_vector.shape)
print('Testing shape:', testing_inputs.shape, 'Testing label shape:', testing_labels.shape)

print(f"ma_norm: {ma_norm}")
print(f"Raw target min/max: {np.min(raw_target_vector)}, {np.max(raw_target_vector)}")
print(f"Clipped target min/max: {np.min(target_vector * ma_norm)}, {np.max(target_vector * ma_norm)}")
print(f"Reserved shot ma2 min/max: {np.min(reserved_shot_hbt[:, 0]) if reserved_shot_hbt is not None else 'N/A'}, {np.max(reserved_shot_hbt[:, 0]) if reserved_shot_hbt is not None else 'N/A'}")

# Print frame counts for reserved shot
if reserved_shot_hbt is not None and reserved_shot_time is not None:
    print(f"Post-processed ma2 data: {len(reserved_shot_hbt)} frames")
    print(f"Post-processed time data: {len(reserved_shot_time)} frames")
else:
    print(f"No post-processed data available for shot {RESERVED_SHOT}")

# Define model architecture
num_conv2d_layers = NUM_CONV2D_LAYERS
num_dense_layers = NUM_DENSE_LAYERS
conv2d_neurons = CONV2D_NEURONS
conv2d_size = CONV2D_SIZE 
dense_layer_neurons = DENSE_LAYER_NEURONS 
max_pooling_size = MAX_POOLING_SIZE 
activation_func = ACTIVATION_FUNC 
loss_func = LOSS_FUNC 
optimizer_func = OPTIMIZER_FUNC 

conv2d_size = [
    tuple(int(x) for x in ast.literal_eval(size) if isinstance(x, (int, float)))
    if isinstance(size, str) else tuple(int(x) for x in size)
    for size in conv2d_size
]
max_pooling_size = (
    tuple(int(x) for x in ast.literal_eval(max_pooling_size) if isinstance(x, (int, float)))
    if isinstance(max_pooling_size, str) else tuple(int(x) for x in max_pooling_size)
)

william_model = tf.keras.models.Sequential()
william_model.add(tf.keras.layers.InputLayer(shape=(32, 32, 1)))

for i in range(num_conv2d_layers):
    william_model.add(tf.keras.layers.Conv2D(conv2d_neurons[i], conv2d_size[i], padding='same', activation=activation_func))
    william_model.add(tf.keras.layers.MaxPooling2D(max_pooling_size, padding='same'))

william_model.add(tf.keras.layers.Flatten())
for i in range(num_dense_layers):
    william_model.add(tf.keras.layers.Dense(dense_layer_neurons[i], activation=activation_func))
    william_model.add(tf.keras.layers.Dropout(0.2))

william_model.add(tf.keras.layers.Dense(1))
william_model.compile(optimizer=optimizer_func, loss=loss_func)
william_model.summary()

# Train the model
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
Model = william_model
history = Model.fit(training_vector, target_vector,
                    epochs=EPOCH_NUM,
                    validation_split=VALIDATION_SPLIT,
                    batch_size=32,
                    verbose=1,
                    callbacks=[early_stop])

# Evaluate model
predictions = Model.predict(testing_inputs)

# Load state2 model for crossover analysis
model_path_new = project_path('data', 'models', 'untrimmed_model_good_state2.keras')
if os.path.exists(model_path_new):
    state2_model = keras.models.load_model(model_path_new)
    print(f"Loaded state 2 model from {model_path_new} for predictions.")
else:
    raise FileNotFoundError("State 2 model not found. Expected at data/models/untrimmed_model_good_state2.keras")
def compute_RESERVED_SHOT_predictions(shot, shot_list, cut_training_data_2d, reserved_shot_cut_2d, hbt_dataType, hbt_time_data, model, ma_norm):
    """Compute predictions for the reserved shot using state2 model and print metrics."""
    if reserved_shot_cut_2d is None:
        print(f"No data available for reserved shot {shot}. No predictions computed.")
        return None
    
    shot_idx = shot_list.index(shot)
    camera_data = reserved_shot_cut_2d
    hbt_data = hbt_dataType[shot_idx][:, 0]
    time_data = hbt_time_data[shot_idx]

    print(f"Shot {shot}: Camera frames={len(camera_data)}, HBT frames={len(hbt_data)}, Time frames={len(time_data)}")
    print(f"Shape of reserved_shot_cut_2d: {reserved_shot_cut_2d.shape}")
    print(f"Shape of camera_data: {np.array(camera_data).shape}")

    if len(camera_data) == 0:
        print(f"No camera data for shot {shot}. No predictions computed.")
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
    print(f"Shot {shot} (State 2 Model) - Mean absolute percentage error: {np.mean(prediction_errors):.2f}%")
    print(f"Shot {shot} (State 2 Model) - Max actual {selected_data_type}: {np.max(np.abs(hbt_data)):.2f}")
    print(f"Shot {shot} (State 2 Model) - Max predicted {selected_data_type}: {np.max(np.abs(predictions)):.2f}")

    return predictions

reserved_predictions_state1 = compute_RESERVED_SHOT_predictions(
    RESERVED_SHOT, shot_list, cut_training_data_2D, reserved_shot_cut_2d, hbt_dataType, hbt_time_data, state2_model, ma_norm
)

# Save true ma2 data, predictions, and time data for reserved shot
time_data = hbt_time_data[shot_list.index(RESERVED_SHOT)]

print(f"Untrimmed: Saving results for state 2, shot {RESERVED_SHOT} from state 1 using state 2 model")
print(f"True data defined: {'hbt_dataType' in locals()}, shape: {hbt_dataType[shot_list.index(RESERVED_SHOT)][:, 0].shape if 'hbt_dataType' in locals() else 'N/A'}")
print(f"Predictions (state 2) defined: {'reserved_predictions_state1' in locals()}, shape: {reserved_predictions_state1.shape if 'reserved_predictions_state1' in locals() else 'N/A'}")
print(f"Time data defined: {'hbt_time_data' in locals()}, shape: {time_data.shape if 'time_data' in locals() else 'N/A'}")
# Ensure predictions directory exists and save results
if args.output_dir:
    pred_dir = args.output_dir
else:
    pred_dir = project_path('data', 'predictions')
os.makedirs(pred_dir, exist_ok=True)
np.save(os.path.join(pred_dir, f'results_{notebook_type}_state_2_{selected_data_type}_true.npy'), hbt_dataType[shot_list.index(RESERVED_SHOT)][:, 0])
np.save(os.path.join(pred_dir, f'results_{notebook_type}_state_2_{selected_data_type}_pred.npy'), reserved_predictions_state1)
np.save(os.path.join(pred_dir, f'results_{notebook_type}_state_2_{selected_data_type}_time.npy'), time_data)
print("Done")