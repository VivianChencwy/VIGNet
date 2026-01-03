"""
EEG Data Processing and Modeling Pipeline
==========================================

Complete pipeline for processing EEG + Eye state data and training VIGNet model.
Matches SEED-VIG preprocessing pipeline for consistent comparison.

Pipeline Steps:
1. Load merged_data.csv
2. Trim first 5 minutes (300 seconds)
3. Resample from ~500Hz to 200Hz
4. Segment into 8-second windows with 4-second stride
5. Extract DE (Differential Entropy) features for 25 frequency bands
6. Apply LDS (Kalman filter) smoothing
7. Compute PERCLOS labels (60-second window)
8. Save preprocessed data
9. Train VIGNet model with 70/15/15 split
10. Evaluate and save results
11. Plot prediction results (scatter plots, time series, residuals)

Usage:
    conda activate eeg
    cd /home/vivian/eeg/SEED_VIG/VIGNet
    python pipeline_weiyu.py --data_dir Weiyu_01
    
    Or specify a different data directory:
    python pipeline_weiyu.py --data_dir Subject_02
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import uniform_filter1d

# Disable XLA before importing TensorFlow
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"
os.environ["TF_DISABLE_XLA"] = "1"

import tensorflow as tf

# Disable XLA compilation
try:
    tf.config.optimizer.set_jit(False)
except:
    pass
try:
    tf.config.experimental.enable_op_determinism()
except:
    pass

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Matplotlib for plotting
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Optional: Kalman filter for LDS smoothing
try:
    from pykalman import KalmanFilter
    HAS_PYKALMAN = True
except ImportError:
    HAS_PYKALMAN = False
    print("Warning: pykalman not installed. Using moving average instead of LDS.")
    print("Install with: pip install pykalman")


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Pipeline configuration matching SEED-VIG parameters"""
    
    # Input/Output paths
    INPUT_CSV = "merged_data.csv"
    OUTPUT_DIR = "processed"
    LOG_DIR = "logs"
    
    # Preprocessing parameters (matching SEED-VIG)
    TARGET_FS = 200              # Target sampling rate (Hz)
    WINDOW_SEC = 8.0             # Window size (seconds)
    STRIDE_SEC = 4.0             # Stride size (seconds) - 50% overlap
    TRIM_START_SEC = 300.0       # Trim first 5 minutes
    TRIM_END_SEC = 0.0           # Trim from end
    
    # PERCLOS calculation
    PERCLOS_WINDOW_SEC = 60.0    # 60-second window for PERCLOS
    
    # Feature extraction
    N_FREQ_BANDS = 25            # 25 frequency bands (2Hz resolution, 1-50Hz)
    FREQ_BANDS = [(i, i+2) for i in range(1, 50, 2)]  # [(1,3), (3,5), ..., (49,51)]
    
    # Data columns
    EEG_COLUMNS = ['fp1_clean', 'fp2_clean']
    EYE_STATE_COLUMN = 'eye_state'
    LSL_TIMESTAMP_COLUMN = 'lsl_timestamp'
    UNIX_TIMESTAMP_COLUMN = 'unix_timestamp'
    
    # Training parameters
    LEARNING_RATE = 0.005
    NUM_EPOCHS = 500
    BATCH_SIZE = 8
    EARLY_STOPPING_PATIENCE = 50
    
    # Data split
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    BLOCK_SIZE = 8               # Windows per block for split
    GAP_SIZE = 2                 # Gap between blocks to prevent leakage
    RANDOM_SEED = 42
    
    # GPU
    GPU_IDX = 0


# =============================================================================
# Step 1: Data Loading and Trimming
# =============================================================================

def load_and_trim_data(data_dir, config, logger):
    """Load merged CSV and trim first 5 minutes"""
    logger.info("=" * 60)
    logger.info("STEP 1: Loading and Trimming Data")
    logger.info("=" * 60)
    
    input_path = os.path.join(data_dir, config.INPUT_CSV)
    logger.info(f"Loading data from: {input_path}")
    
    # Load CSV
    df = pd.read_csv(input_path, low_memory=False)
    logger.info(f"Loaded {len(df)} samples")
    
    # Get timestamps
    lsl_timestamps = df[config.LSL_TIMESTAMP_COLUMN].values
    total_duration = lsl_timestamps[-1] - lsl_timestamps[0]
    logger.info(f"Original duration: {total_duration:.2f}s ({total_duration/60:.2f} min)")
    
    # Estimate original sampling rate
    time_diffs = np.diff(lsl_timestamps)
    original_fs = 1.0 / np.median(time_diffs)
    logger.info(f"Original sampling rate: {original_fs:.2f} Hz")
    
    # Trim start and end
    if config.TRIM_START_SEC > 0 or config.TRIM_END_SEC > 0:
        logger.info(f"Trimming: first {config.TRIM_START_SEC}s, last {config.TRIM_END_SEC}s")
        
        start_time = lsl_timestamps[0] + config.TRIM_START_SEC
        end_time = lsl_timestamps[-1] - config.TRIM_END_SEC
        
        mask = (df[config.LSL_TIMESTAMP_COLUMN] >= start_time) & \
               (df[config.LSL_TIMESTAMP_COLUMN] <= end_time)
        df = df[mask].reset_index(drop=True)
        
        new_duration = df[config.LSL_TIMESTAMP_COLUMN].iloc[-1] - df[config.LSL_TIMESTAMP_COLUMN].iloc[0]
        logger.info(f"After trimming: {new_duration:.2f}s ({new_duration/60:.2f} min)")
        logger.info(f"Samples after trimming: {len(df)}")
    
    return df, original_fs


# =============================================================================
# Step 2: Resampling
# =============================================================================

def resample_data(eeg_data, original_fs, target_fs, logger):
    """Resample EEG data to target sampling rate"""
    logger.info("=" * 60)
    logger.info("STEP 2: Resampling")
    logger.info("=" * 60)
    
    if abs(original_fs - target_fs) < 1:
        logger.info("Sampling rates are similar, no resampling needed")
        return eeg_data
    
    n_samples = eeg_data.shape[0]
    duration = n_samples / original_fs
    target_samples = int(duration * target_fs)
    
    logger.info(f"Resampling from {original_fs:.2f} Hz to {target_fs} Hz")
    logger.info(f"Samples: {n_samples} -> {target_samples}")
    
    resampled = np.zeros((target_samples, eeg_data.shape[1]))
    for ch in range(eeg_data.shape[1]):
        resampled[:, ch] = signal.resample(eeg_data[:, ch], target_samples)
    
    return resampled


# =============================================================================
# Step 3: Segmentation
# =============================================================================

def segment_signal(data, window_samples, stride_samples, logger):
    """Segment signal into overlapping windows"""
    logger.info("=" * 60)
    logger.info("STEP 3: Segmentation")
    logger.info("=" * 60)
    
    n_samples = data.shape[0]
    segments = []
    
    for start in range(0, n_samples - window_samples + 1, stride_samples):
        end = start + window_samples
        segments.append(data[start:end, :])
    
    segments = np.array(segments)
    logger.info(f"Created {segments.shape[0]} windows")
    logger.info(f"Window shape: {segments.shape[1]} samples x {segments.shape[2]} channels")
    
    return segments


# =============================================================================
# Step 4: DE Feature Extraction
# =============================================================================

def extract_de_features(segments, config, logger):
    """Extract Differential Entropy features for each frequency band"""
    logger.info("=" * 60)
    logger.info("STEP 4: Extracting DE Features")
    logger.info("=" * 60)
    
    n_windows = segments.shape[0]
    n_channels = segments.shape[2]
    n_bands = config.N_FREQ_BANDS
    
    de_features = np.zeros((n_channels, n_windows, n_bands))
    
    logger.info(f"Extracting DE for {n_windows} windows, {n_channels} channels, {n_bands} bands")
    
    for w in range(n_windows):
        if (w + 1) % 100 == 0 or w == n_windows - 1:
            logger.info(f"  Processing window {w + 1}/{n_windows}")
        
        for ch in range(n_channels):
            segment = segments[w, :, ch]
            
            for i, (low_freq, high_freq) in enumerate(config.FREQ_BANDS):
                # Design bandpass filter
                nyq = config.TARGET_FS / 2.0
                low = max(0.01, low_freq / nyq)
                high = min(0.99, high_freq / nyq)
                
                if low < high:
                    # Apply bandpass filter (4th order Butterworth)
                    b, a = signal.butter(4, [low, high], btype='band')
                    filtered = signal.filtfilt(b, a, segment)
                    
                    # Compute DE: 0.5 * log(2 * pi * e * variance)
                    variance = np.var(filtered)
                    de = 0.5 * np.log(2 * np.pi * np.e * (variance + 1e-10))
                    de_features[ch, w, i] = de
    
    logger.info(f"DE features shape: {de_features.shape}")
    return de_features


# =============================================================================
# Step 5: Smoothing (Moving Average + LDS)
# =============================================================================

def apply_smoothing(features, logger):
    """Apply Moving Average and LDS smoothing"""
    logger.info("=" * 60)
    logger.info("STEP 5: Applying Smoothing")
    logger.info("=" * 60)
    
    n_channels, n_windows, n_freqs = features.shape
    
    # Moving Average smoothing
    logger.info("Applying moving average (window=5)...")
    smoothed_ma = np.zeros_like(features)
    for ch in range(n_channels):
        for freq in range(n_freqs):
            smoothed_ma[ch, :, freq] = uniform_filter1d(features[ch, :, freq], 
                                                         size=5, mode='nearest')
    
    # LDS (Kalman filter) smoothing
    if HAS_PYKALMAN:
        logger.info("Applying LDS (Kalman filter) smoothing...")
        smoothed_lds = np.zeros_like(features)
        
        for ch in range(n_channels):
            for freq in range(n_freqs):
                ts = features[ch, :, freq]
                
                kf = KalmanFilter(
                    initial_state_mean=ts[0],
                    n_dim_obs=1,
                    n_dim_state=1,
                    transition_matrices=[1],
                    observation_matrices=[1],
                    transition_covariance=0.1 * np.eye(1),
                    observation_covariance=1.0 * np.eye(1)
                )
                
                smoothed_ts, _ = kf.smooth(ts.reshape(-1, 1))
                smoothed_lds[ch, :, freq] = smoothed_ts.flatten()
    else:
        logger.warning("pykalman not available, using moving average as fallback")
        smoothed_lds = smoothed_ma
    
    logger.info("Smoothing completed")
    return smoothed_lds


# =============================================================================
# Step 6: PERCLOS Calculation
# =============================================================================

def compute_perclos(df, window_timestamps, config, logger):
    """Compute PERCLOS for each window"""
    logger.info("=" * 60)
    logger.info("STEP 6: Computing PERCLOS Labels")
    logger.info("=" * 60)
    
    # Get eye state data
    eye_states = df[config.EYE_STATE_COLUMN].values
    eye_timestamps = df[config.UNIX_TIMESTAMP_COLUMN].values
    
    # Check actual eye state values in data
    unique_states = np.unique(eye_states)
    logger.info(f"Unique eye states in data: {unique_states}")
    
    # Convert eye states to binary (1 = closed/close, 0 = open)
    # Handle both 'closed' and 'close' variants
    eye_closed = np.isin(eye_states, ['closed', 'close']).astype(float)
    
    closed_count = np.sum(eye_closed)
    total_count = len(eye_closed)
    logger.info(f"Eye closed samples: {closed_count} / {total_count} ({100*closed_count/total_count:.2f}%)")
    
    perclos = np.zeros(len(window_timestamps))
    
    logger.info(f"Computing PERCLOS with {config.PERCLOS_WINDOW_SEC}s window...")
    
    for i, win_ts in enumerate(window_timestamps):
        # Find eye states in the preceding window
        window_start = win_ts - config.PERCLOS_WINDOW_SEC
        window_mask = (eye_timestamps >= window_start) & (eye_timestamps <= win_ts)
        
        if np.sum(window_mask) > 0:
            perclos[i] = np.mean(eye_closed[window_mask])
        else:
            perclos[i] = 0.0
    
    logger.info(f"PERCLOS range: [{perclos.min():.4f}, {perclos.max():.4f}]")
    logger.info(f"PERCLOS mean: {perclos.mean():.4f}, std: {perclos.std():.4f}")
    
    # Print distribution
    awake = np.sum(perclos < 0.35)
    tired = np.sum((perclos >= 0.35) & (perclos < 0.7))
    drowsy = np.sum(perclos >= 0.7)
    total = len(perclos)
    
    logger.info(f"Distribution: Awake={awake} ({100*awake/total:.1f}%), "
                f"Tired={tired} ({100*tired/total:.1f}%), "
                f"Drowsy={drowsy} ({100*drowsy/total:.1f}%)")
    
    return perclos


# =============================================================================
# Step 7: Save Preprocessed Data
# =============================================================================

def save_preprocessed_data(data_dir, de_features, perclos, timestamps, config, logger):
    """Save preprocessed data to files"""
    logger.info("=" * 60)
    logger.info("STEP 7: Saving Preprocessed Data")
    logger.info("=" * 60)
    
    output_dir = os.path.join(data_dir, config.OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Transpose features: (n_channels, n_windows, 25) -> (n_windows, n_channels, 25)
    de_transposed = np.moveaxis(de_features, 0, 1)
    
    # Save features
    features_path = os.path.join(output_dir, 'de_features.npy')
    np.save(features_path, de_transposed)
    logger.info(f"Saved DE features: {features_path}")
    logger.info(f"  Shape: {de_transposed.shape}")
    
    # Save PERCLOS labels
    perclos_path = os.path.join(output_dir, 'perclos_labels.npy')
    np.save(perclos_path, perclos)
    logger.info(f"Saved PERCLOS labels: {perclos_path}")
    
    # Save timestamps
    ts_path = os.path.join(output_dir, 'timestamps.npy')
    np.save(ts_path, timestamps)
    logger.info(f"Saved timestamps: {ts_path}")
    
    # Save metadata
    metadata = {
        'input_file': config.INPUT_CSV,
        'eeg_channels': config.EEG_COLUMNS,
        'target_sampling_rate': config.TARGET_FS,
        'window_sec': config.WINDOW_SEC,
        'stride_sec': config.STRIDE_SEC,
        'perclos_window_sec': config.PERCLOS_WINDOW_SEC,
        'n_windows': de_transposed.shape[0],
        'n_channels': de_transposed.shape[1],
        'n_bands': config.N_FREQ_BANDS,
        'trim_start_sec': config.TRIM_START_SEC,
        'processing_date': datetime.now().isoformat()
    }
    metadata_path = os.path.join(output_dir, 'metadata.npy')
    np.save(metadata_path, metadata)
    logger.info(f"Saved metadata: {metadata_path}")
    
    return de_transposed


# =============================================================================
# Step 8: Data Splitting (Block-wise Random)
# =============================================================================

def blockwise_random_split(n_samples, config, logger):
    """Block-wise random split to avoid data leakage"""
    logger.info("=" * 60)
    logger.info("STEP 8: Data Splitting (Block-wise Random)")
    logger.info("=" * 60)
    
    np.random.seed(config.RANDOM_SEED)
    
    # Create blocks
    blocks = []
    unit_size = config.BLOCK_SIZE + config.GAP_SIZE
    
    start_idx = 0
    while start_idx + config.BLOCK_SIZE <= n_samples:
        block_indices = list(range(start_idx, start_idx + config.BLOCK_SIZE))
        blocks.append(block_indices)
        start_idx += unit_size
    
    n_blocks = len(blocks)
    logger.info(f"Created {n_blocks} blocks (block_size={config.BLOCK_SIZE}, gap={config.GAP_SIZE})")
    
    # Shuffle blocks
    block_order = np.random.permutation(n_blocks)
    
    # Assign blocks to train/val/test
    n_train_blocks = int(n_blocks * config.TRAIN_RATIO)
    n_val_blocks = int(n_blocks * config.VAL_RATIO)
    
    train_block_ids = block_order[:n_train_blocks]
    val_block_ids = block_order[n_train_blocks:n_train_blocks + n_val_blocks]
    test_block_ids = block_order[n_train_blocks + n_val_blocks:]
    
    # Collect indices
    train_idx = np.array([idx for bid in train_block_ids for idx in blocks[bid]])
    val_idx = np.array([idx for bid in val_block_ids for idx in blocks[bid]])
    test_idx = np.array([idx for bid in test_block_ids for idx in blocks[bid]])
    
    total_used = len(train_idx) + len(val_idx) + len(test_idx)
    discarded = n_samples - total_used
    
    logger.info(f"Split results:")
    logger.info(f"  Train: {len(train_idx)} samples ({100*len(train_idx)/total_used:.1f}%)")
    logger.info(f"  Valid: {len(val_idx)} samples ({100*len(val_idx)/total_used:.1f}%)")
    logger.info(f"  Test:  {len(test_idx)} samples ({100*len(test_idx)/total_used:.1f}%)")
    logger.info(f"  Discarded (gaps): {discarded} samples")
    
    return train_idx, val_idx, test_idx


# =============================================================================
# Step 9: Model Definition (VIGNet-FP)
# =============================================================================

class VIGNetFP(tf.keras.Model):
    """VIGNet model for FP1/FP2 forehead channels (2 channels)"""
    
    def __init__(self, mode='RGS'):
        tf.keras.backend.set_floatx("float64")
        super(VIGNetFP, self).__init__()
        
        self.mode = mode
        self.regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2=0.05)
        self.activation = tf.nn.leaky_relu
        
        # Convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(10, (1, 5), kernel_regularizer=self.regularizer, activation=None)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        
        self.conv2 = tf.keras.layers.Conv2D(10, (1, 5), kernel_regularizer=self.regularizer, activation=None)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        
        self.conv3 = tf.keras.layers.Conv2D(10, (1, 5), kernel_regularizer=self.regularizer, activation=None)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(0.3)
        
        # Spatial fusion (2 channels -> 1)
        self.conv4 = tf.keras.layers.Conv2D(20, (2, 1), kernel_regularizer=self.regularizer, activation=self.activation)
        
        self.flatten = tf.keras.layers.Flatten()
        
        if self.mode == "CLF":
            self.dense = tf.keras.layers.Dense(3)
        else:  # RGS
            self.dense = tf.keras.layers.Dense(1)
    
    def MHRSSA(self, x, out_filter, num_channel=2):
        """Multi-head Residual Spectro-Spatio Attention module"""
        for i in range(out_filter):
            tmp = tf.keras.layers.Conv2D(num_channel, (num_channel, 1), 
                                         kernel_regularizer=self.regularizer, activation=None)(x)
            if i == 0:
                MHRSSA = tmp
            else:
                MHRSSA = tf.concat((MHRSSA, tmp), 1)
        
        MHRSSA = tf.transpose(MHRSSA, perm=[0, 3, 2, 1])
        MHRSSA = tf.keras.layers.DepthwiseConv2D((1, 5), kernel_regularizer=self.regularizer, activation=None)(MHRSSA)
        MHRSSA = tf.keras.activations.softmax(MHRSSA)
        return MHRSSA
    
    def call(self, x, training=False):
        att1 = self.MHRSSA(x, 10)
        hidden = self.conv1(x)
        hidden = self.bn1(hidden, training=training)
        hidden = self.activation(hidden)
        hidden *= att1
        hidden = self.dropout1(hidden, training=training)
        
        att2 = self.MHRSSA(hidden, 10)
        hidden = self.conv2(hidden)
        hidden = self.bn2(hidden, training=training)
        hidden = self.activation(hidden)
        hidden *= att2
        hidden = self.dropout2(hidden, training=training)
        
        att3 = self.MHRSSA(hidden, 10)
        hidden = self.conv3(hidden)
        hidden = self.bn3(hidden, training=training)
        hidden = self.activation(hidden)
        hidden *= att3
        hidden = self.dropout3(hidden, training=training)
        
        hidden = self.conv4(hidden)
        hidden = self.flatten(hidden)
        hidden = self.dense(hidden)
        
        if self.mode == "CLF":
            return tf.keras.activations.softmax(hidden)
        else:
            return hidden


# =============================================================================
# Step 10: Training
# =============================================================================

def train_model(data_dir, features, labels, train_idx, val_idx, test_idx, config, logger):
    """Train VIGNet model"""
    logger.info("=" * 60)
    logger.info("STEP 9-10: Model Training")
    logger.info("=" * 60)
    
    # Configure GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU_IDX)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Using GPU: {gpus[0]}")
        except RuntimeError as e:
            logger.warning(f"GPU configuration error: {e}")
    else:
        logger.warning("No GPU found, using CPU")
    
    tf.config.run_functions_eagerly(True)
    
    # Prepare data
    X_train = features[train_idx]
    X_val = features[val_idx]
    X_test = features[test_idx]
    
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Normalize features
    train_shape = X_train.shape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(train_shape[0], -1)).reshape(train_shape)
    X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
    logger.info("Applied StandardScaler normalization")
    
    # Add channel dimension: (N, 2, 25) -> (N, 2, 25, 1)
    X_train = np.expand_dims(X_train, -1)
    X_val = np.expand_dims(X_val, -1)
    X_test = np.expand_dims(X_test, -1)
    
    # Expand labels
    y_train = np.expand_dims(y_train, -1)
    y_val_orig = y_val.copy()
    y_test_orig = y_test.copy()
    y_val = np.expand_dims(y_val, -1)
    y_test = np.expand_dims(y_test, -1)
    
    # Convert to tensors
    X_train = tf.constant(X_train, dtype=tf.float64)
    y_train = tf.constant(y_train, dtype=tf.float64)
    X_val = tf.constant(X_val, dtype=tf.float64)
    X_test = tf.constant(X_test, dtype=tf.float64)
    
    # Initialize model
    model = VIGNetFP(mode='RGS')
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    
    logger.info(f"Learning rate: {config.LEARNING_RATE}")
    logger.info(f"Max epochs: {config.NUM_EPOCHS}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None
    
    num_batch_iter = int(X_train.shape[0] / config.BATCH_SIZE)
    
    for epoch in range(config.NUM_EPOCHS):
        loss_per_epoch = 0
        
        # Shuffle training data
        tf.random.set_seed(config.RANDOM_SEED + epoch)
        rand_idx = tf.random.shuffle(tf.range(tf.shape(X_train)[0]))
        X_train_shuffled = tf.gather(X_train, rand_idx)
        y_train_shuffled = tf.gather(y_train, rand_idx)
        
        for batch in range(num_batch_iter):
            x = X_train_shuffled[batch * config.BATCH_SIZE:(batch + 1) * config.BATCH_SIZE]
            y = y_train_shuffled[batch * config.BATCH_SIZE:(batch + 1) * config.BATCH_SIZE]
            
            # Forward pass and compute gradients
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss = tf.keras.losses.MSE(y, y_pred)
            
            grads = tape.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            loss_per_epoch += tf.reduce_mean(loss).numpy()
        
        avg_loss = loss_per_epoch / num_batch_iter
        
        # Validation
        y_val_pred = model(X_val, training=False)
        val_loss = tf.reduce_mean((y_val_orig - tf.squeeze(y_val_pred).numpy()) ** 2).numpy()
        
        logger.info(f"Epoch {epoch + 1:3d}: Train Loss = {avg_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = model.get_weights()
            logger.info(f"  -> New best validation loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Restore best weights
    if best_weights is not None:
        model.set_weights(best_weights)
        logger.info("Restored best model weights")
    
    # Final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    
    # Validation metrics
    y_val_pred = model(X_val, training=False).numpy().squeeze()
    val_mse = mean_squared_error(y_val_orig, y_val_pred)
    val_mae = mean_absolute_error(y_val_orig, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_corr, val_p = pearsonr(y_val_orig, y_val_pred)
    
    logger.info(f"Validation: MSE={val_mse:.6f}, MAE={val_mae:.6f}, RMSE={val_rmse:.6f}, Corr={val_corr:.6f}")
    
    # Test metrics
    y_test_pred = model(X_test, training=False).numpy().squeeze()
    test_mse = mean_squared_error(y_test_orig, y_test_pred)
    test_mae = mean_absolute_error(y_test_orig, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_corr, test_p = pearsonr(y_test_orig, y_test_pred)
    
    logger.info(f"Test:       MSE={test_mse:.6f}, MAE={test_mae:.6f}, RMSE={test_rmse:.6f}, Corr={test_corr:.6f}")
    
    # Save model and scaler
    logger.info("\nSaving model and results...")
    log_dir = os.path.join(data_dir, config.LOG_DIR)
    models_dir = os.path.join(log_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model weights
    try:
        model_path = os.path.join(models_dir, "best_model")
        model.save(model_path, save_format='tf')
        logger.info(f"Saved model to: {model_path}")
    except Exception as e:
        weights_path = os.path.join(models_dir, "best_weights.h5")
        model.save_weights(weights_path)
        logger.info(f"Saved weights to: {weights_path}")
    
    # Save scaler
    import pickle
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved scaler to: {scaler_path}")
    
    # Save predictions
    predictions = {
        'val_true': y_val_orig,
        'val_pred': y_val_pred,
        'test_true': y_test_orig,
        'test_pred': y_test_pred,
        'metrics': {
            'val_mse': val_mse, 'val_mae': val_mae, 'val_rmse': val_rmse, 'val_corr': val_corr,
            'test_mse': test_mse, 'test_mae': test_mae, 'test_rmse': test_rmse, 'test_corr': test_corr
        }
    }
    pred_path = os.path.join(models_dir, "predictions.npy")
    np.save(pred_path, predictions)
    logger.info(f"Saved predictions to: {pred_path}")
    
    return predictions


# =============================================================================
# Step 11: Plot Prediction Results
# =============================================================================

def plot_perclos_predictions(predictions, output_path, logger):
    """Plot comprehensive visualization of PERCLOS predictions"""
    logger.info("=" * 60)
    logger.info("STEP 11: Plotting Prediction Results")
    logger.info("=" * 60)
    
    val_true = predictions['val_true']
    val_pred = predictions['val_pred']
    test_true = predictions['test_true']
    test_pred = predictions['test_pred']
    metrics = predictions['metrics']
    
    # Create figure with subplots (2 rows x 3 columns for each set)
    fig = plt.figure(figsize=(18, 12))
    
    # Helper function to plot one set
    def plot_set(ax1, ax2, ax3, y_true, y_pred, set_name, set_metrics):
        # 1. Scatter plot with regression line
        ax1.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Add diagonal line (perfect prediction)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Add regression line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax1.plot(y_true, p(y_true), "b-", lw=2, alpha=0.8, label=f'Regression (slope={z[0]:.3f})')
        
        ax1.set_xlabel('True PERCLOS', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Predicted PERCLOS', fontsize=11, fontweight='bold')
        ax1.set_title(f'{set_name.upper()} Set: Scatter Plot\n'
                      f'r = {set_metrics["corr"]:.4f}, RMSE = {set_metrics["rmse"]:.4f}',
                      fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        # 2. Time series comparison
        time_points = np.arange(len(y_true))
        ax2.plot(time_points, y_true, 'o-', label='True PERCLOS', alpha=0.7, linewidth=2, markersize=3)
        ax2.plot(time_points, y_pred, 's-', label='Predicted PERCLOS', alpha=0.7, linewidth=2, markersize=3)
        ax2.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
        ax2.set_ylabel('PERCLOS', fontsize=11, fontweight='bold')
        ax2.set_title(f'{set_name.upper()} Set: Time Series', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. Residuals plot
        residuals = y_true - y_pred
        ax3.scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('Predicted PERCLOS', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Residuals (True - Predicted)', fontsize=11, fontweight='bold')
        ax3.set_title(f'{set_name.upper()} Set: Residuals\n'
                      f'Mean = {residuals.mean():.4f}, Std = {residuals.std():.4f}',
                      fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # Validation set plots (top row)
    val_metrics = {
        'corr': metrics['val_corr'],
        'rmse': metrics['val_rmse'],
        'mae': metrics['val_mae'],
        'mse': metrics['val_mse']
    }
    plot_set(plt.subplot(2, 3, 1), plt.subplot(2, 3, 2), plt.subplot(2, 3, 3),
             val_true, val_pred, 'Validation', val_metrics)
    
    # Test set plots (bottom row)
    test_metrics = {
        'corr': metrics['test_corr'],
        'rmse': metrics['test_rmse'],
        'mae': metrics['test_mae'],
        'mse': metrics['test_mse']
    }
    plot_set(plt.subplot(2, 3, 4), plt.subplot(2, 3, 5), plt.subplot(2, 3, 6),
             test_true, test_pred, 'Test', test_metrics)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved prediction plots to: {output_path}")


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    """Run complete pipeline"""
    parser = argparse.ArgumentParser(description='EEG Data Processing and Modeling Pipeline')
    parser.add_argument('--data_dir', type=str, default='Weiyu_01',
                        help='Data directory containing merged_data.csv (default: Weiyu_01)')
    args = parser.parse_args()
    
    # Get absolute path to data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, args.data_dir)
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    config = Config()
    
    # Setup logging
    log_dir = os.path.join(data_dir, config.LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    
    # Configure logger
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("DATA PROCESSING AND MODELING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_path}")
    logger.info("")
    
    try:
        # Step 1: Load and trim data
        df, original_fs = load_and_trim_data(data_dir, config, logger)
        
        # Extract EEG data
        eeg_data = df[config.EEG_COLUMNS].values
        lsl_timestamps = df[config.LSL_TIMESTAMP_COLUMN].values
        
        # Step 2: Resample
        eeg_resampled = resample_data(eeg_data, original_fs, config.TARGET_FS, logger)
        
        # Compute resampled timestamps
        resampled_timestamps = np.linspace(lsl_timestamps[0], lsl_timestamps[-1], len(eeg_resampled))
        
        # Step 3: Segment
        window_samples = int(config.WINDOW_SEC * config.TARGET_FS)
        stride_samples = int(config.STRIDE_SEC * config.TARGET_FS)
        segments = segment_signal(eeg_resampled, window_samples, stride_samples, logger)
        
        # Compute window center timestamps
        n_windows = segments.shape[0]
        window_center_indices = [
            int(i * stride_samples + window_samples / 2)
            for i in range(n_windows)
        ]
        window_lsl_timestamps = resampled_timestamps[window_center_indices]
        
        # Map to UNIX timestamps for PERCLOS calculation
        lsl_to_unix_offset = df[config.UNIX_TIMESTAMP_COLUMN].iloc[0] - df[config.LSL_TIMESTAMP_COLUMN].iloc[0]
        window_unix_timestamps = window_lsl_timestamps + lsl_to_unix_offset
        
        # Step 4: Extract DE features
        de_features = extract_de_features(segments, config, logger)
        
        # Step 5: Apply smoothing
        de_smoothed = apply_smoothing(de_features, logger)
        
        # Step 6: Compute PERCLOS
        perclos = compute_perclos(df, window_unix_timestamps, config, logger)
        
        # Step 7: Save preprocessed data
        de_transposed = save_preprocessed_data(data_dir, de_smoothed, perclos, window_unix_timestamps, config, logger)
        
        # Step 8: Split data
        train_idx, val_idx, test_idx = blockwise_random_split(len(perclos), config, logger)
        
        # Step 9-10: Train model
        predictions = train_model(data_dir, de_transposed, perclos, train_idx, val_idx, test_idx, config, logger)
        
        # Step 11: Plot prediction results
        plot_output_path = os.path.join(log_dir, "perclos_prediction_results.png")
        plot_perclos_predictions(predictions, plot_output_path, logger)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Duration: {duration}")
        logger.info("")
        logger.info("Output files:")
        logger.info(f"  - Preprocessed data: {config.OUTPUT_DIR}/")
        logger.info(f"  - Model and results: {config.LOG_DIR}/models/")
        logger.info(f"  - Prediction plots: {plot_output_path}")
        logger.info(f"  - Log file: {log_path}")
        
    except Exception as e:
        logger.error(f"\nPIPELINE FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

