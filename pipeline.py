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
    KSS_COLUMN = 'kss_rating'
    STRESS_COLUMN = 'stress'
    FRUSTRATION_COLUMN = 'frustration'
    ACCURACY_COLUMN = 'rating_label'  # from EDA definition
    
    # Composite label weights (perclos, kss, stress, frustration, accuracy)
    COMPOSITE_WEIGHTS = [3.0, 3.0, 1.0, 1.0, 1.0]
    
    # Training parameters
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 1000
    BATCH_SIZE = 64
    EARLY_STOPPING_PATIENCE = 100
    
    # Data split
    TRAIN_RATIO = 0.60
    VAL_RATIO = 0.20
    TEST_RATIO = 0.20
    BLOCK_SIZE = 4               # Windows per block for split
    GAP_SIZE = 0                 # Gap between blocks
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
    
    # Check if DataFrame is empty
    if len(df) == 0:
        raise ValueError(f"DataFrame is empty after loading from {input_path}")
    
    # Check if required column exists
    if config.LSL_TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(f"Required column '{config.LSL_TIMESTAMP_COLUMN}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Get timestamps and remove NaN values
    lsl_timestamps = df[config.LSL_TIMESTAMP_COLUMN].values
    
    # Check if timestamps are valid
    if len(lsl_timestamps) == 0:
        raise ValueError("No timestamps found in the data")
    
    # Remove rows with NaN timestamps before processing
    nan_mask = pd.isna(df[config.LSL_TIMESTAMP_COLUMN])
    nan_count = nan_mask.sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} rows with NaN timestamps. Removing them.")
        df = df[~nan_mask].reset_index(drop=True)
        lsl_timestamps = df[config.LSL_TIMESTAMP_COLUMN].values
        logger.info(f"After removing NaN timestamps: {len(df)} samples")
    
    if len(lsl_timestamps) == 0:
        raise ValueError("No valid timestamps found in the data after removing NaN values")
    
    total_duration = lsl_timestamps[-1] - lsl_timestamps[0]
    logger.info(f"Original duration: {total_duration:.2f}s ({total_duration/60:.2f} min)")
    
    # Estimate original sampling rate (robust to duplicates/zeros/outliers)
    time_diffs = np.diff(lsl_timestamps)
    if len(time_diffs) == 0:
        raise ValueError("Cannot estimate sampling rate: insufficient data points")

    # Keep positive diffs only to avoid duplicates/zeros
    pos_diffs = time_diffs[time_diffs > 0]
    if len(pos_diffs) == 0:
        raise ValueError("No positive timestamp differences; cannot estimate sampling rate.")

    raw_fs = 1.0 / np.median(pos_diffs)

    # Trim outliers (1st–99th percentile) for a more stable estimate
    lo, hi = np.percentile(pos_diffs, [1, 99])
    trimmed = pos_diffs[(pos_diffs >= lo) & (pos_diffs <= hi)]
    trimmed = trimmed if len(trimmed) > 0 else pos_diffs
    robust_fs = 1.0 / np.median(trimmed)

    # Estimate duration from row count with robust_fs for sanity
    est_duration_rows = len(df) / robust_fs

    logger.info(f"Sampling rate (raw median, pos diffs): {raw_fs:.2f} Hz")
    logger.info(f"Sampling rate (trimmed 1-99%): {robust_fs:.2f} Hz")
    logger.info(f"Duration by timestamps: {total_duration:.2f}s ({total_duration/60:.2f} min)")
    logger.info(f"Duration by rows@robust_fs: {est_duration_rows:.2f}s ({est_duration_rows/60:.2f} min)")
    original_fs = robust_fs
    
    # Trim start and end
    if config.TRIM_START_SEC > 0 or config.TRIM_END_SEC > 0:
        logger.info(f"Trimming: first {config.TRIM_START_SEC}s, last {config.TRIM_END_SEC}s")
        
        start_time = lsl_timestamps[0] + config.TRIM_START_SEC
        end_time = lsl_timestamps[-1] - config.TRIM_END_SEC
        
        mask = (df[config.LSL_TIMESTAMP_COLUMN] >= start_time) & \
               (df[config.LSL_TIMESTAMP_COLUMN] <= end_time)
        df = df[mask].reset_index(drop=True)
        
        # Check if DataFrame is empty after trimming
        if len(df) == 0:
            raise ValueError(f"DataFrame is empty after trimming. Start time: {start_time}, End time: {end_time}, "
                           f"Original range: [{lsl_timestamps[0]}, {lsl_timestamps[-1]}]")
        
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
# Step 6b: Composite Label Calculation
# =============================================================================

def compute_composite_label(df, window_timestamps, perclos, config, logger, train_idx=None):
    """
    Compute composite label using z-scored metrics with weights 3:3:1:1:1
    Components: perclos, KSS, stress, frustration, accuracy (rating_label)
    
    If train_idx is provided, z-score normalization uses only training data to avoid data leakage.
    """
    logger.info("=" * 60)
    logger.info("STEP 6b: Computing Composite Label")
    logger.info("=" * 60)
    
    if train_idx is not None:
        logger.info(f"Using training set only for z-score normalization (train_idx length: {len(train_idx)})")
    else:
        logger.warning("No train_idx provided; using all data for z-score (may cause data leakage)")

    unix_ts = df[config.UNIX_TIMESTAMP_COLUMN].values

    def window_mean(col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found for composite label.")
        series = pd.to_numeric(df[col], errors='coerce').values
        vals = []
        for win_ts in window_timestamps:
            win_start = win_ts - config.PERCLOS_WINDOW_SEC
            mask = (unix_ts >= win_start) & (unix_ts <= win_ts)
            if mask.sum() > 0:
                vals.append(np.nanmean(series[mask]))
            else:
                vals.append(np.nan)
        return np.array(vals, dtype=float)

    kss = window_mean(config.KSS_COLUMN)
    stress = window_mean(config.STRESS_COLUMN)
    frustration = window_mean(config.FRUSTRATION_COLUMN)
    accuracy = window_mean(config.ACCURACY_COLUMN)

    components = {
        "perclos": perclos,
        "kss": kss,
        "stress": stress,
        "frustration": frustration,
        "accuracy": accuracy,
    }

    norm_components = {}
    for name, arr in components.items():
        finite = np.isfinite(arr)
        if finite.sum() < 2:
            logger.warning(f"{name}: insufficient finite values for z-score; filling zeros")
            norm = np.zeros_like(arr)
        else:
            # Use training set statistics only if train_idx is provided
            if train_idx is not None and len(train_idx) > 0:
                train_arr = arr[train_idx]
                train_finite = finite[train_idx]
                if train_finite.sum() < 2:
                    logger.warning(f"{name}: insufficient training data for z-score; using all data")
                    mean = np.nanmean(arr[finite])
                    std = np.nanstd(arr[finite])
                else:
                    mean = np.nanmean(train_arr[train_finite])
                    std = np.nanstd(train_arr[train_finite])
                    logger.info(f"{name}: z-score stats from training set only (mean={mean:.4f}, std={std:.4f})")
            else:
                mean = np.nanmean(arr[finite])
                std = np.nanstd(arr[finite])
            
            if std < 1e-8:
                logger.warning(f"{name}: std nearly zero; z-score set to zero")
                norm = np.zeros_like(arr)
            else:
                norm = (arr - mean) / std
                norm[~finite] = 0.0
        norm_components[name] = norm
        logger.info(f"{name} z-score: mean={np.nanmean(norm):.4f}, std={np.nanstd(norm):.4f}, finite={finite.sum()}/{len(arr)}")

    w_perclos, w_kss, w_stress, w_frustration, w_acc = config.COMPOSITE_WEIGHTS
    weight_sum = w_perclos + w_kss + w_stress + w_frustration + w_acc

    composite = (
        w_perclos * norm_components["perclos"] +
        w_kss * norm_components["kss"] +
        w_stress * norm_components["stress"] +
        w_frustration * norm_components["frustration"] +
        w_acc * norm_components["accuracy"]
    ) / weight_sum

    finite_comp = np.isfinite(composite)
    logger.info(f"Composite label: min={np.nanmin(composite):.4f}, max={np.nanmax(composite):.4f}, mean={np.nanmean(composite):.4f}, std={np.nanstd(composite):.4f}, finite={finite_comp.sum()}/{len(composite)}")

    return composite, norm_components


# =============================================================================
# Step 7: Save Preprocessed Data
# =============================================================================

def save_preprocessed_data(data_dir, de_features, perclos, composite, timestamps, config, logger):
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
    
    # Save labels
    perclos_path = os.path.join(output_dir, 'perclos_labels.npy')
    np.save(perclos_path, perclos)
    logger.info(f"Saved PERCLOS labels: {perclos_path}")

    composite_path = os.path.join(output_dir, 'composite_labels.npy')
    np.save(composite_path, composite)
    logger.info(f"Saved composite labels: {composite_path}")
    
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
        'composite_weights': config.COMPOSITE_WEIGHTS,
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

def blockwise_random_split(perclos, config, logger):
    """Block-wise stratified split on block-level PERCLOS to reduce leakage and class imbalance"""
    logger.info("=" * 60)
    logger.info("STEP 8: Data Splitting (Block-wise Stratified)")
    logger.info("=" * 60)
    
    np.random.seed(config.RANDOM_SEED)
    
    # Create blocks
    n_samples = len(perclos)
    blocks = []
    unit_size = config.BLOCK_SIZE + config.GAP_SIZE
    
    start_idx = 0
    while start_idx + config.BLOCK_SIZE <= n_samples:
        block_indices = list(range(start_idx, start_idx + config.BLOCK_SIZE))
        blocks.append(block_indices)
        start_idx += unit_size
    
    n_blocks = len(blocks)
    logger.info(f"Created {n_blocks} blocks (block_size={config.BLOCK_SIZE}, gap={config.GAP_SIZE})")
    
    if n_blocks == 0:
        raise ValueError("No blocks created for splitting; check BLOCK_SIZE/GAP_SIZE vs data length.")
    
    # Compute block-level scores (mean PERCLOS per block) for stratification
    block_scores = np.array([float(np.mean(perclos[blk])) for blk in blocks])
    logger.info(f"Block PERCLOS range: [{block_scores.min():.4f}, {block_scores.max():.4f}]")
    
    # Bin blocks by quantiles (low / mid / high) to preserve distribution across splits
    quantiles = np.quantile(block_scores, [0, 1/3, 2/3, 1])
    bin_edges = np.unique(quantiles)
    if len(bin_edges) < 4:
        # Fallback to evenly spaced bins if quantiles collapse
        bin_edges = np.linspace(block_scores.min(), block_scores.max(), 4)
    labels = np.digitize(block_scores, bin_edges[1:-1], right=False)
    
    # Stratified assignment at block level
    train_blocks, val_blocks, test_blocks = [], [], []
    for lbl in np.unique(labels):
        lbl_blocks = np.where(labels == lbl)[0]
        np.random.shuffle(lbl_blocks)
        n = len(lbl_blocks)
        n_train = int(n * config.TRAIN_RATIO)
        n_val = int(n * config.VAL_RATIO)
        train_blocks.extend(lbl_blocks[:n_train])
        val_blocks.extend(lbl_blocks[n_train:n_train + n_val])
        test_blocks.extend(lbl_blocks[n_train + n_val:])
        logger.info(f"Label {lbl}: blocks={n}, -> train={len(train_blocks)}, val={len(val_blocks)}, test={len(test_blocks)}")
    
    # Collect indices
    train_idx = np.array([idx for bid in train_blocks for idx in blocks[bid]])
    val_idx = np.array([idx for bid in val_blocks for idx in blocks[bid]])
    test_idx = np.array([idx for bid in test_blocks for idx in blocks[bid]])
    
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
        # Note: DepthwiseConv2D in Keras 3.x doesn't support kernel_regularizer
        # Regularization can be applied at the model level if needed
        MHRSSA = tf.keras.layers.DepthwiseConv2D((1, 5), activation=None)(MHRSSA)
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
    
    # Check for NaN/Inf values
    val_nan_true = np.isnan(y_val_orig).sum() + np.isinf(y_val_orig).sum()
    val_nan_pred = np.isnan(y_val_pred).sum() + np.isinf(y_val_pred).sum()
    if val_nan_true > 0 or val_nan_pred > 0:
        logger.warning(f"Validation data contains NaN/Inf: true={val_nan_true}, pred={val_nan_pred}")
    
    # Remove NaN/Inf values for correlation calculation
    val_mask = np.isfinite(y_val_orig) & np.isfinite(y_val_pred)
    if val_mask.sum() < 2:
        logger.warning("Warning: Too few valid values for validation correlation calculation")
        val_corr = np.nan
        val_p = np.nan
    else:
        val_true_clean = y_val_orig[val_mask]
        val_pred_clean = y_val_pred[val_mask]
        # Check if data has variance
        if np.std(val_true_clean) < 1e-10:
            logger.warning("Validation true values have zero variance, correlation is undefined")
            val_corr = np.nan
            val_p = np.nan
        elif np.std(val_pred_clean) < 1e-10:
            logger.warning("Validation predicted values have zero variance, correlation is undefined")
            val_corr = np.nan
            val_p = np.nan
        else:
            val_corr, val_p = pearsonr(val_true_clean, val_pred_clean)
    
    val_mse = mean_squared_error(y_val_orig, y_val_pred)
    val_mae = mean_absolute_error(y_val_orig, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    
    val_corr_str = f"{val_corr:.6f}" if not np.isnan(val_corr) else "nan"
    logger.info(f"Validation: MSE={val_mse:.6f}, MAE={val_mae:.6f}, RMSE={val_rmse:.6f}, Corr={val_corr_str}")
    
    # Test metrics
    y_test_pred = model(X_test, training=False).numpy().squeeze()
    
    # Check for NaN/Inf values
    test_nan_true = np.isnan(y_test_orig).sum() + np.isinf(y_test_orig).sum()
    test_nan_pred = np.isnan(y_test_pred).sum() + np.isinf(y_test_pred).sum()
    if test_nan_true > 0 or test_nan_pred > 0:
        logger.warning(f"Test data contains NaN/Inf: true={test_nan_true}, pred={test_nan_pred}")
    
    # Remove NaN/Inf values for correlation calculation
    test_mask = np.isfinite(y_test_orig) & np.isfinite(y_test_pred)
    if test_mask.sum() < 2:
        logger.warning("Warning: Too few valid values for test correlation calculation")
        test_corr = np.nan
        test_p = np.nan
    else:
        test_true_clean = y_test_orig[test_mask]
        test_pred_clean = y_test_pred[test_mask]
        # Check if data has variance
        if np.std(test_true_clean) < 1e-10:
            logger.warning("Test true values have zero variance, correlation is undefined")
            test_corr = np.nan
            test_p = np.nan
        elif np.std(test_pred_clean) < 1e-10:
            logger.warning("Test predicted values have zero variance, correlation is undefined")
            test_corr = np.nan
            test_p = np.nan
        else:
            test_corr, test_p = pearsonr(test_true_clean, test_pred_clean)
    
    test_mse = mean_squared_error(y_test_orig, y_test_pred)
    test_mae = mean_absolute_error(y_test_orig, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    
    test_corr_str = f"{test_corr:.6f}" if not np.isnan(test_corr) else "nan"
    logger.info(f"Test:       MSE={test_mse:.6f}, MAE={test_mae:.6f}, RMSE={test_rmse:.6f}, Corr={test_corr_str}")
    
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
        # Keras 3.x requires .weights.h5 extension
        weights_path = os.path.join(models_dir, "best_weights.weights.h5")
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

def plot_fatigue_predictions(predictions, output_path, logger):
    """Plot comprehensive visualization of fatigue (composite label) predictions"""
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
        # Remove NaN/Inf values for plotting
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        
        if len(y_true_clean) == 0:
            ax1.text(0.5, 0.5, 'No valid data to plot', ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'No valid data to plot', ha='center', va='center', transform=ax2.transAxes)
            ax3.text(0.5, 0.5, 'No valid data to plot', ha='center', va='center', transform=ax3.transAxes)
            return
        
        # 1. Scatter plot with regression line
        ax1.scatter(y_true_clean, y_pred_clean, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Add diagonal line (perfect prediction)
        min_val = min(y_true_clean.min(), y_pred_clean.min())
        max_val = max(y_true_clean.max(), y_pred_clean.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Add regression line (with error handling)
        try:
            # Check if data has variance
            if np.std(y_true_clean) < 1e-10 or np.std(y_pred_clean) < 1e-10:
                logger.warning(f"{set_name}: Data has zero variance, skipping regression line")
                slope = 0.0
            else:
                z = np.polyfit(y_true_clean, y_pred_clean, 1)
                p = np.poly1d(z)
                ax1.plot(y_true_clean, p(y_true_clean), "b-", lw=2, alpha=0.8, label=f'Regression (slope={z[0]:.3f})')
                slope = z[0]
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"{set_name}: Could not fit regression line: {e}")
            slope = 0.0
        
        ax1.set_xlabel('True Fatigue', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Predicted Fatigue', fontsize=11, fontweight='bold')
        corr_str = f'{set_metrics["corr"]:.4f}' if not np.isnan(set_metrics["corr"]) else 'nan'
        ax1.set_title(f'{set_name.upper()} Set: Scatter Plot\n'
                      f'r = {corr_str}, RMSE = {set_metrics["rmse"]:.4f}',
                      fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        # 2. Time series comparison
        time_points = np.arange(len(y_true_clean))
        ax2.plot(time_points, y_true_clean, 'o-', label='True Fatigue', alpha=0.7, linewidth=2, markersize=3)
        ax2.plot(time_points, y_pred_clean, 's-', label='Predicted Fatigue', alpha=0.7, linewidth=2, markersize=3)
        ax2.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Fatigue', fontsize=11, fontweight='bold')
        ax2.set_title(f'{set_name.upper()} Set: Time Series', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. Residuals plot
        residuals = y_true_clean - y_pred_clean
        ax3.scatter(y_pred_clean, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('Predicted Fatigue', fontsize=11, fontweight='bold')
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
# Evaluation Function (can run separately)
# =============================================================================

def evaluate_model(data_dir, config, logger):
    """Load saved model and evaluate on preprocessed data"""
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)
    
    # Load preprocessed data
    processed_dir = os.path.join(data_dir, config.OUTPUT_DIR)
    de_path = os.path.join(processed_dir, "de_features.npy")
    comp_path = os.path.join(processed_dir, "composite_labels.npy")
    split_path = os.path.join(processed_dir, "train_val_test_split.npy")
    
    if not os.path.exists(de_path):
        raise FileNotFoundError(f"Preprocessed data not found: {de_path}. Please run training first.")
    if not os.path.exists(comp_path):
        raise FileNotFoundError(f"Composite labels not found: {comp_path}. Please run training first.")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Data split not found: {split_path}. Please run training first.")
    
    logger.info(f"Loading preprocessed data from: {processed_dir}")
    de_transposed = np.load(de_path)
    labels = np.load(comp_path)
    split_data = np.load(split_path, allow_pickle=True).item()
    train_idx = split_data['train_idx']
    val_idx = split_data['val_idx']
    test_idx = split_data['test_idx']
    
    logger.info(f"Loaded data: {de_transposed.shape}, labels: {labels.shape}")
    logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Load model and scaler
    log_dir = os.path.join(data_dir, config.LOG_DIR)
    models_dir = os.path.join(log_dir, "models")
    
    # Try to load model
    model_path = os.path.join(models_dir, "best_model")
    weights_path = os.path.join(models_dir, "best_weights.weights.h5")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}. Please run training first.")
    
    import pickle
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    logger.info(f"Loaded scaler from: {scaler_path}")
    
    # Initialize model and build with dummy input for weight loading
    model = VIGNetFP(mode='RGS')
    dummy_input = tf.constant(de_transposed[:1][..., np.newaxis], dtype=tf.float64)
    _ = model(dummy_input, training=False)
    logger.info("Model built for weight loading")
    
    # Try to load model
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded model from: {model_path}")
        except Exception as e:
            logger.warning(f"Could not load model from {model_path}: {e}")
            if os.path.exists(weights_path):
                logger.info(f"Loading weights from: {weights_path}")
                model.load_weights(weights_path)
            else:
                raise FileNotFoundError(f"Neither model nor weights found. Please run training first.")
    elif os.path.exists(weights_path):
        logger.info(f"Loading weights from: {weights_path}")
        model.load_weights(weights_path)
    else:
        raise FileNotFoundError(f"Model or weights not found. Please run training first.")
    
    # Prepare data
    X_train = de_transposed[train_idx]
    X_val = de_transposed[val_idx]
    X_test = de_transposed[test_idx]
    
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]
    
    # Normalize features
    train_shape = X_train.shape
    X_train = scaler.transform(X_train.reshape(train_shape[0], -1)).reshape(train_shape)
    X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
    
    # Add channel dimension
    X_train = np.expand_dims(X_train, -1)
    X_val = np.expand_dims(X_val, -1)
    X_test = np.expand_dims(X_test, -1)
    
    # Convert to tensors
    X_train = tf.constant(X_train, dtype=tf.float64)
    X_val = tf.constant(X_val, dtype=tf.float64)
    X_test = tf.constant(X_test, dtype=tf.float64)
    
    # Evaluate
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    
    # Train metrics
    y_train_pred = model(X_train, training=False).numpy().squeeze()
    train_mask = np.isfinite(y_train) & np.isfinite(y_train_pred)
    if train_mask.sum() < 2:
        logger.warning("Warning: Too few valid values for train correlation calculation")
        train_corr = np.nan
        train_p = np.nan
    else:
        train_corr, train_p = pearsonr(y_train[train_mask], y_train_pred[train_mask])
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_corr_str = f"{train_corr:.6f}" if not np.isnan(train_corr) else "nan"
    logger.info(f"Train:      MSE={train_mse:.6f}, MAE={train_mae:.6f}, RMSE={train_rmse:.6f}, Corr={train_corr_str}")
    
    # Validation metrics
    y_val_pred = model(X_val, training=False).numpy().squeeze()
    val_mask = np.isfinite(y_val) & np.isfinite(y_val_pred)
    if val_mask.sum() < 2:
        logger.warning("Warning: Too few valid values for validation correlation calculation")
        val_corr = np.nan
        val_p = np.nan
    else:
        val_corr, val_p = pearsonr(y_val[val_mask], y_val_pred[val_mask])
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_corr_str = f"{val_corr:.6f}" if not np.isnan(val_corr) else "nan"
    logger.info(f"Validation: MSE={val_mse:.6f}, MAE={val_mae:.6f}, RMSE={val_rmse:.6f}, Corr={val_corr_str}")
    
    # Test metrics
    y_test_pred = model(X_test, training=False).numpy().squeeze()
    test_mask = np.isfinite(y_test) & np.isfinite(y_test_pred)
    if test_mask.sum() < 2:
        logger.warning("Warning: Too few valid values for test correlation calculation")
        test_corr = np.nan
        test_p = np.nan
    else:
        test_corr, test_p = pearsonr(y_test[test_mask], y_test_pred[test_mask])
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_corr_str = f"{test_corr:.6f}" if not np.isnan(test_corr) else "nan"
    logger.info(f"Test:       MSE={test_mse:.6f}, MAE={test_mae:.6f}, RMSE={test_rmse:.6f}, Corr={test_corr_str}")
    
    # Save predictions
    predictions = {
        'train_true': y_train,
        'train_pred': y_train_pred,
        'val_true': y_val,
        'val_pred': y_val_pred,
        'test_true': y_test,
        'test_pred': y_test_pred,
        'metrics': {
            'train_mse': train_mse, 'train_mae': train_mae, 'train_rmse': train_rmse, 'train_corr': train_corr,
            'val_mse': val_mse, 'val_mae': val_mae, 'val_rmse': val_rmse, 'val_corr': val_corr,
            'test_mse': test_mse, 'test_mae': test_mae, 'test_rmse': test_rmse, 'test_corr': test_corr
        }
    }
    pred_path = os.path.join(models_dir, "predictions_eval.npy")
    np.save(pred_path, predictions)
    logger.info(f"\nSaved evaluation predictions to: {pred_path}")
    
    # Plot results
    plot_output_path = os.path.join(log_dir, "fatigue_prediction_results_eval.png")
    plot_fatigue_predictions(predictions, plot_output_path, logger)
    
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETED")
    logger.info("=" * 60)
    
    return predictions


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    """Run complete pipeline"""
    parser = argparse.ArgumentParser(description='EEG Data Processing and Modeling Pipeline')
    parser.add_argument('--data_dir', type=str, default='Weiyu_01',
                        help='Data directory containing merged_data.csv (default: Weiyu_01)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                        help='Pipeline mode: train (full pipeline) or evaluate (evaluate saved model)')
    args = parser.parse_args()
    
    # Get absolute path to data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '../data', args.data_dir)
    
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
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_path}")
    logger.info("")
    
    try:
        if args.mode == 'evaluate':
            # Evaluation mode: load saved model and evaluate
            predictions = evaluate_model(data_dir, config, logger)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("")
            logger.info("=" * 70)
            logger.info("EVALUATION COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Duration: {duration}")
            logger.info("")
            logger.info("Output files:")
            logger.info(f"  - Evaluation predictions: {config.LOG_DIR}/models/predictions_eval.npy")
            logger.info(f"  - Prediction plots: {config.LOG_DIR}/fatigue_prediction_results_eval.png")
            logger.info(f"  - Log file: {log_path}")
            return
        
        # Training mode: full pipeline
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
        
        # Step 8: Split data FIRST (based on perclos to avoid data leakage in normalization)
        # We split before computing composite label to ensure z-score normalization uses only training data
        train_idx, val_idx, test_idx = blockwise_random_split(perclos, config, logger)

        # Step 6b: Compute composite label (AFTER splitting, using train_idx for z-score normalization)
        composite_label, _norms = compute_composite_label(df, window_unix_timestamps, perclos, config, logger, train_idx=train_idx)
        
        # Step 7: Save preprocessed data
        de_transposed = save_preprocessed_data(data_dir, de_smoothed, perclos, composite_label, window_unix_timestamps, config, logger)
        
        # Save split indices for evaluation
        processed_dir = os.path.join(data_dir, config.OUTPUT_DIR)
        split_data = {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }
        split_path = os.path.join(processed_dir, "train_val_test_split.npy")
        np.save(split_path, split_data)
        logger.info(f"Saved data split to: {split_path}")
        
        # Step 9-10: Train model
        predictions = train_model(data_dir, de_transposed, composite_label, train_idx, val_idx, test_idx, config, logger)
        
        # Step 11: Plot prediction results
        plot_output_path = os.path.join(log_dir, "fatigue_prediction_results.png")
        plot_fatigue_predictions(predictions, plot_output_path, logger)
        
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

