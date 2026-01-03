"""
EDA Analysis for merged_data.csv
==========================================

Analyzes correlations between:
1. Subjective ratings (KSS, stress, frustration)
2. PERCLOS (eye closure percentage)
3. Task performance (accuracy, hit rate, false alarm rate, RT)
4. EEG features (DE: 25 bands x 2 channels, Bandpower: 5 bands x 2 channels)

Output:
- Correlation heatmaps
- Summary statistics

Usage:
    conda activate eeg
    cd /home/vivian/eeg/SEED_VIG/VIGNet
    python eda_analysis.py --data_dir Weiyu_01
    
    Or specify a different data directory:
    python eda_analysis.py --data_dir Subject_02
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """EDA configuration parameters"""
    
    # Input/Output paths
    INPUT_CSV = "merged_data.csv"
    OUTPUT_DIR = "eda_results"
    
    # Sampling and window parameters
    TARGET_FS = 200              # Target sampling rate (Hz)
    WINDOW_SEC = 8.0             # Window size for feature aggregation (seconds)
    PERCLOS_WINDOW_SEC = 60.0    # PERCLOS calculation window (seconds)
    
    # Data trimming parameters
    TRIM_START_SEC = 300.0       # Trim first 5 minutes (300 seconds)
    TRIM_END_SEC = 60.0          # Trim last 1 minute (60 seconds)
    
    # EEG columns
    EEG_COLUMNS = ['fp1_clean', 'fp2_clean']
    EYE_STATE_COLUMN = 'eye_state'
    TIMESTAMP_COLUMN = 'lsl_timestamp'
    UNIX_TIMESTAMP_COLUMN = 'unix_timestamp'
    
    # DE frequency bands (25 bands, 2Hz resolution, 1-50Hz)
    DE_FREQ_BANDS = [(i, i+2) for i in range(1, 50, 2)]
    N_DE_BANDS = 25
    
    # Traditional frequency bands
    TRADITIONAL_BANDS = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    # Task performance columns
    TARGET_COLUMN = 'is_target_double_jump'
    RESPONDED_COLUMN = 'responded'
    RT_COLUMN = 'rt'
    
    # Subjective rating columns
    SUBJECTIVE_COLUMNS = ['kss_rating', 'stress', 'frustration']


# =============================================================================
# Step 1: Data Loading and Preprocessing
# =============================================================================

def load_and_preprocess_data(data_dir, config):
    """Load merged CSV and preprocess"""
    print("=" * 60)
    print("STEP 1: Loading and Preprocessing Data")
    print("=" * 60)
    
    input_path = os.path.join(data_dir, config.INPUT_CSV)
    print(f"Loading data from: {input_path}")
    
    # Load CSV
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Sort by timestamp
    df = df.sort_values(by=config.TIMESTAMP_COLUMN).reset_index(drop=True)
    
    # Estimate sampling rate
    timestamps = df[config.TIMESTAMP_COLUMN].values
    time_diffs = np.diff(timestamps)
    original_fs = 1.0 / np.median(time_diffs)
    print(f"Estimated sampling rate: {original_fs:.2f} Hz")
    
    # Calculate total duration before trimming
    total_duration = timestamps[-1] - timestamps[0]
    print(f"Original duration: {total_duration:.2f}s ({total_duration/60:.2f} min)")
    
    # Trim start and end
    if config.TRIM_START_SEC > 0 or config.TRIM_END_SEC > 0:
        print(f"Trimming: first {config.TRIM_START_SEC}s, last {config.TRIM_END_SEC}s")
        
        start_time = timestamps[0] + config.TRIM_START_SEC
        end_time = timestamps[-1] - config.TRIM_END_SEC
        
        mask = (df[config.TIMESTAMP_COLUMN] >= start_time) & \
               (df[config.TIMESTAMP_COLUMN] <= end_time)
        df = df[mask].reset_index(drop=True)
        
        # Recalculate timestamps after trimming
        timestamps = df[config.TIMESTAMP_COLUMN].values
        new_duration = timestamps[-1] - timestamps[0]
        print(f"After trimming: {new_duration:.2f}s ({new_duration/60:.2f} min)")
        print(f"Samples after trimming: {len(df)}")
    
    return df, original_fs


# =============================================================================
# Step 2: PERCLOS Calculation
# =============================================================================

def calculate_perclos(df, config):
    """Calculate PERCLOS using 60-second sliding window (optimized)"""
    print("=" * 60)
    print("STEP 2: Calculating PERCLOS")
    print("=" * 60)
    
    # Get eye state data
    eye_states = df[config.EYE_STATE_COLUMN].values
    timestamps = df[config.UNIX_TIMESTAMP_COLUMN].values
    
    # Convert eye states to binary (1 = closed, 0 = open)
    eye_closed = np.isin(eye_states, ['closed', 'close']).astype(float)
    
    print(f"Eye state unique values: {np.unique(eye_states)}")
    print(f"Closed samples: {eye_closed.sum():.0f} / {len(eye_closed)} ({100*eye_closed.mean():.2f}%)")
    
    print(f"Computing PERCLOS with {config.PERCLOS_WINDOW_SEC}s window...")
    print("Using optimized vectorized method...")
    
    # Pre-compute window boundaries for all samples at once using searchsorted
    # This is O(n log n) instead of O(n²)
    window_starts = timestamps - config.PERCLOS_WINDOW_SEC
    start_indices = np.searchsorted(timestamps, window_starts, side='left')
    end_indices = np.arange(len(timestamps)) + 1
    
    # Calculate PERCLOS efficiently
    perclos = np.zeros(len(timestamps))
    
    # Process in chunks to show progress
    chunk_size = 50000
    for chunk_start in range(0, len(timestamps), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(timestamps))
        print(f"  Processing samples {chunk_start}-{chunk_end}/{len(timestamps)}")
        
        for i in range(chunk_start, chunk_end):
            start_idx = start_indices[i]
            end_idx = end_indices[i]
            if end_idx > start_idx:
                perclos[i] = np.mean(eye_closed[start_idx:end_idx])
    
    df['perclos'] = perclos
    print(f"PERCLOS range: [{perclos.min():.4f}, {perclos.max():.4f}]")
    print(f"PERCLOS mean: {perclos.mean():.4f}, std: {perclos.std():.4f}")
    
    return df


# =============================================================================
# Step 3: EEG Feature Extraction
# =============================================================================

def extract_de_for_window(signal_data, fs, freq_bands):
    """Extract DE features for a single window"""
    de_values = np.zeros(len(freq_bands))
    
    for i, (low_freq, high_freq) in enumerate(freq_bands):
        nyq = fs / 2.0
        low = max(0.01, low_freq / nyq)
        high = min(0.99, high_freq / nyq)
        
        if low < high:
            try:
                b, a = signal.butter(4, [low, high], btype='band')
                filtered = signal.filtfilt(b, a, signal_data)
                variance = np.var(filtered)
                de = 0.5 * np.log(2 * np.pi * np.e * (variance + 1e-10))
                de_values[i] = de
            except:
                de_values[i] = 0.0
        else:
            de_values[i] = 0.0
    
    return de_values


def extract_bandpower_for_window(signal_data, fs, bands):
    """Extract traditional bandpower features for a single window"""
    bandpower = {}
    
    # Compute PSD using Welch's method
    nperseg = min(len(signal_data), int(fs * 2))  # 2 second segments
    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=nperseg)
    
    for band_name, (low_freq, high_freq) in bands.items():
        # Find frequency indices
        idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        
        if np.sum(idx) > 0:
            # Calculate bandpower by integrating PSD
            bandpower[band_name] = np.trapz(psd[idx], freqs[idx])
        else:
            bandpower[band_name] = 0.0
    
    return bandpower


def extract_eeg_features(df, config, original_fs):
    """Extract DE and bandpower features for each time window"""
    print("=" * 60)
    print("STEP 3: Extracting EEG Features")
    print("=" * 60)
    
    # Get EEG data
    eeg_data = df[config.EEG_COLUMNS].values
    timestamps = df[config.TIMESTAMP_COLUMN].values
    
    # Calculate window parameters
    window_samples = int(config.WINDOW_SEC * original_fs)
    n_samples = len(df)
    
    # Create non-overlapping windows
    window_starts = list(range(0, n_samples - window_samples + 1, window_samples))
    n_windows = len(window_starts)
    
    print(f"Creating {n_windows} windows of {config.WINDOW_SEC}s each")
    print(f"Window samples: {window_samples}")
    
    # Initialize feature arrays
    n_channels = len(config.EEG_COLUMNS)
    n_de_bands = config.N_DE_BANDS
    n_trad_bands = len(config.TRADITIONAL_BANDS)
    
    de_features = np.zeros((n_windows, n_channels, n_de_bands))
    bandpower_features = np.zeros((n_windows, n_channels, n_trad_bands))
    window_timestamps = np.zeros(n_windows)
    window_indices = []
    
    print("Extracting DE features (25 bands x 2 channels)...")
    print("Extracting Bandpower features (5 bands x 2 channels)...")
    
    for w_idx, start in enumerate(window_starts):
        if (w_idx + 1) % 50 == 0 or w_idx == n_windows - 1:
            print(f"  Processing window {w_idx + 1}/{n_windows}")
        
        end = start + window_samples
        window_indices.append((start, end))
        window_timestamps[w_idx] = timestamps[start + window_samples // 2]
        
        for ch_idx in range(n_channels):
            window_signal = eeg_data[start:end, ch_idx]
            
            # Extract DE features
            de_features[w_idx, ch_idx, :] = extract_de_for_window(
                window_signal, original_fs, config.DE_FREQ_BANDS
            )
            
            # Extract bandpower features
            bp = extract_bandpower_for_window(
                window_signal, original_fs, config.TRADITIONAL_BANDS
            )
            bandpower_features[w_idx, ch_idx, :] = list(bp.values())
    
    print(f"DE features shape: {de_features.shape}")
    print(f"Bandpower features shape: {bandpower_features.shape}")
    
    return de_features, bandpower_features, window_timestamps, window_indices


# =============================================================================
# Step 4: Task Performance Metrics
# =============================================================================

def calculate_task_metrics(df, window_indices, config):
    """Calculate task performance metrics for each window"""
    print("=" * 60)
    print("STEP 4: Calculating Task Performance Metrics")
    print("=" * 60)
    
    n_windows = len(window_indices)
    
    # Initialize metrics
    accuracy = np.full(n_windows, np.nan)
    hit_rate = np.full(n_windows, np.nan)
    false_alarm_rate = np.full(n_windows, np.nan)
    rt_mean = np.full(n_windows, np.nan)
    
    # Get task data
    is_target = df[config.TARGET_COLUMN].values
    responded = df[config.RESPONDED_COLUMN].values
    rt = df[config.RT_COLUMN].values
    
    for w_idx, (start, end) in enumerate(window_indices):
        if (w_idx + 1) % 100 == 0 or w_idx == n_windows - 1:
            print(f"  Processing window {w_idx + 1}/{n_windows}")
        
        # Window data
        w_target = is_target[start:end]
        w_responded = responded[start:end]
        w_rt = rt[start:end]
        
        # Count targets and non-targets
        n_targets = np.nansum(w_target == 1)
        n_non_targets = np.nansum(w_target == 0)
        
        if n_targets > 0:
            # Hit rate: responded to targets / total targets
            target_mask = w_target == 1
            hits = np.nansum(w_responded[target_mask] == 1)
            hit_rate[w_idx] = hits / n_targets
            
        if n_non_targets > 0:
            # False alarm rate: responded to non-targets / total non-targets
            non_target_mask = w_target == 0
            false_alarms = np.nansum(w_responded[non_target_mask] == 1)
            false_alarm_rate[w_idx] = false_alarms / n_non_targets
        
        # Accuracy: correct responses / total events
        total_events = n_targets + n_non_targets
        if total_events > 0:
            # Correct = hits + correct rejections
            hits = np.nansum((w_target == 1) & (w_responded == 1))
            correct_rejections = np.nansum((w_target == 0) & (w_responded == 0))
            accuracy[w_idx] = (hits + correct_rejections) / total_events
        
        # Mean RT for responses
        valid_rt = w_rt[~np.isnan(w_rt) & (w_rt > 0)]
        if len(valid_rt) > 0:
            rt_mean[w_idx] = np.mean(valid_rt)
    
    task_metrics = {
        'accuracy': accuracy,
        'hit_rate': hit_rate,
        'false_alarm_rate': false_alarm_rate,
        'rt_mean': rt_mean
    }
    
    print(f"Task metrics calculated for {n_windows} windows")
    for name, values in task_metrics.items():
        valid = ~np.isnan(values)
        print(f"  {name}: {valid.sum()} valid windows, mean={np.nanmean(values):.4f}")
    
    return task_metrics


# =============================================================================
# Step 5: Handle Subjective Ratings
# =============================================================================

def process_subjective_ratings(df, window_indices, config):
    """Process subjective ratings with forward fill"""
    print("=" * 60)
    print("STEP 5: Processing Subjective Ratings")
    print("=" * 60)
    
    n_windows = len(window_indices)
    subjective_data = {}
    
    for col in config.SUBJECTIVE_COLUMNS:
        # Forward fill the column
        df[f'{col}_filled'] = df[col].ffill()
        
        # Get values for each window (use center value)
        values = np.full(n_windows, np.nan)
        
        for w_idx, (start, end) in enumerate(window_indices):
            center = (start + end) // 2
            val = df[f'{col}_filled'].iloc[center]
            if pd.notna(val):
                try:
                    values[w_idx] = float(val)
                except:
                    pass
        
        subjective_data[col] = values
        valid = ~np.isnan(values)
        print(f"  {col}: {valid.sum()} valid windows, mean={np.nanmean(values):.4f}")
    
    return subjective_data


# =============================================================================
# Step 6: Window Aggregation
# =============================================================================

def aggregate_all_metrics(df, window_indices, de_features, bandpower_features, 
                         task_metrics, subjective_data, config):
    """Aggregate all metrics to windows and create feature dataframe"""
    print("=" * 60)
    print("STEP 6: Aggregating All Metrics to Windows")
    print("=" * 60)
    
    n_windows = len(window_indices)
    
    # Calculate PERCLOS for each window
    perclos_values = df['perclos'].values
    window_perclos = np.zeros(n_windows)
    
    for w_idx, (start, end) in enumerate(window_indices):
        window_perclos[w_idx] = np.mean(perclos_values[start:end])
    
    # Create aggregated dataframe
    agg_data = {'perclos': window_perclos}
    
    # Add subjective ratings
    for col, values in subjective_data.items():
        agg_data[col] = values
    
    # Add task metrics
    for name, values in task_metrics.items():
        agg_data[name] = values
    
    # Add DE features (flatten: channels x bands)
    channel_names = ['fp1', 'fp2']
    for ch_idx, ch_name in enumerate(channel_names):
        for band_idx in range(config.N_DE_BANDS):
            freq_low = config.DE_FREQ_BANDS[band_idx][0]
            freq_high = config.DE_FREQ_BANDS[band_idx][1]
            col_name = f'de_{ch_name}_{freq_low}_{freq_high}Hz'
            agg_data[col_name] = de_features[:, ch_idx, band_idx]
    
    # Add bandpower features
    band_names = list(config.TRADITIONAL_BANDS.keys())
    for ch_idx, ch_name in enumerate(channel_names):
        for band_idx, band_name in enumerate(band_names):
            col_name = f'bp_{ch_name}_{band_name}'
            agg_data[col_name] = bandpower_features[:, ch_idx, band_idx]
    
    agg_df = pd.DataFrame(agg_data)
    
    print(f"Aggregated dataframe shape: {agg_df.shape}")
    print(f"Columns: {list(agg_df.columns[:15])}... ({len(agg_df.columns)} total)")
    
    return agg_df


# =============================================================================
# Step 7: Correlation Analysis
# =============================================================================

def calculate_correlations(agg_df, config):
    """Calculate correlation matrices"""
    print("=" * 60)
    print("STEP 7: Calculating Correlations")
    print("=" * 60)
    
    # Define variable groups
    behavior_cols = ['perclos'] + config.SUBJECTIVE_COLUMNS + ['accuracy', 'hit_rate', 'false_alarm_rate', 'rt_mean']
    
    de_cols = [c for c in agg_df.columns if c.startswith('de_')]
    bp_cols = [c for c in agg_df.columns if c.startswith('bp_')]
    
    # Filter to valid columns only
    behavior_cols = [c for c in behavior_cols if c in agg_df.columns]
    
    print(f"Behavior variables: {len(behavior_cols)}")
    print(f"DE features: {len(de_cols)}")
    print(f"Bandpower features: {len(bp_cols)}")
    
    # Print missing value statistics
    print("\nMissing value statistics:")
    for col in behavior_cols:
        if col in agg_df.columns:
            missing = agg_df[col].isnull().sum()
            valid = agg_df[col].notna().sum()
            print(f"  {col}: {valid}/{len(agg_df)} valid ({100*valid/len(agg_df):.1f}%)")
    
    # 1. Correlation between behavior variables
    behavior_df = agg_df[behavior_cols].dropna()
    behavior_corr = behavior_df.corr()
    
    print(f"\nBehavior correlation matrix shape: {behavior_corr.shape}")
    print(f"Valid samples for behavior correlation: {len(behavior_df)}/{len(agg_df)} ({100*len(behavior_df)/len(agg_df):.1f}%)")
    
    # 2. Correlation between EEG features and behavior
    # For each EEG feature, calculate correlation with each behavior variable
    # Lower minimum sample requirement from 10 to 3 to handle sparse data
    MIN_SAMPLES = 3
    eeg_behavior_corr = pd.DataFrame(index=de_cols + bp_cols, columns=behavior_cols)
    sample_counts = pd.DataFrame(index=de_cols + bp_cols, columns=behavior_cols)
    
    for eeg_col in de_cols + bp_cols:
        for beh_col in behavior_cols:
            # Get valid pairs
            valid = agg_df[[eeg_col, beh_col]].dropna()
            sample_counts.loc[eeg_col, beh_col] = len(valid)
            
            if len(valid) >= MIN_SAMPLES:
                try:
                    corr, _ = pearsonr(valid[eeg_col], valid[beh_col])
                    eeg_behavior_corr.loc[eeg_col, beh_col] = corr
                except:
                    eeg_behavior_corr.loc[eeg_col, beh_col] = np.nan
            else:
                eeg_behavior_corr.loc[eeg_col, beh_col] = np.nan
    
    eeg_behavior_corr = eeg_behavior_corr.astype(float)
    
    # Print sample count statistics
    print(f"\nEEG-Behavior correlation matrix shape: {eeg_behavior_corr.shape}")
    print(f"Minimum samples required: {MIN_SAMPLES}")
    print(f"Correlations calculated: {(eeg_behavior_corr.notna().sum().sum())}/{len(eeg_behavior_corr) * len(eeg_behavior_corr.columns)}")
    
    # Show sample counts for sparse variables
    sparse_vars = ['rt_mean', 'hit_rate']
    for var in sparse_vars:
        if var in behavior_cols:
            counts = sample_counts[var].astype(float)
            print(f"\nSample counts for {var}:")
            print(f"  Min: {counts.min():.0f}, Max: {counts.max():.0f}, Mean: {counts.mean():.1f}")
            print(f"  <{MIN_SAMPLES} samples: {(counts < MIN_SAMPLES).sum()}/{len(counts)}")
    
    return behavior_corr, eeg_behavior_corr, behavior_cols, de_cols, bp_cols


# =============================================================================
# Step 8: Visualization
# =============================================================================

def plot_behavior_correlation_heatmap(behavior_corr, output_dir):
    """Plot correlation heatmap for subjective ratings, PERCLOS, and task performance"""
    print("\nPlotting behavior correlation heatmap...")
    
    # Rename columns for better display
    display_names = {
        'perclos': 'PERCLOS',
        'kss_rating': 'KSS',
        'stress': 'Stress',
        'frustration': 'Frustration',
        'accuracy': 'Accuracy',
        'hit_rate': 'Hit Rate',
        'false_alarm_rate': 'False Alarm',
        'rt_mean': 'RT Mean'
    }
    
    corr_display = behavior_corr.copy()
    corr_display.index = [display_names.get(c, c) for c in corr_display.index]
    corr_display.columns = [display_names.get(c, c) for c in corr_display.columns]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr_display, dtype=bool), k=1)
    
    sns.heatmap(corr_display, annot=True, fmt='.3f', cmap='RdBu_r',
                vmin=-1, vmax=1, center=0, square=True, mask=mask,
                linewidths=0.5, cbar_kws={'shrink': 0.8},
                annot_kws={'size': 11}, ax=ax)
    
    ax.set_title('Correlation: Subjective Ratings, PERCLOS, and Task Performance',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'correlation_subjective_perclos_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_bandpower_correlation_heatmap(eeg_behavior_corr, bp_cols, behavior_cols, output_dir, config):
    """Plot correlation heatmap for traditional bandpower features"""
    print("\nPlotting bandpower correlation heatmap...")
    
    # Filter to bandpower columns only
    bp_corr = eeg_behavior_corr.loc[bp_cols, :].copy()
    
    # Rename for display
    display_names = {
        'perclos': 'PERCLOS',
        'kss_rating': 'KSS',
        'stress': 'Stress',
        'frustration': 'Frustration',
        'accuracy': 'Accuracy',
        'hit_rate': 'Hit Rate',
        'false_alarm_rate': 'False Alarm',
        'rt_mean': 'RT Mean'
    }
    
    row_names = []
    for col in bp_cols:
        parts = col.replace('bp_', '').split('_')
        ch = parts[0].upper()
        band = parts[1].capitalize()
        row_names.append(f'{ch} {band}')
    
    bp_corr.index = row_names
    bp_corr.columns = [display_names.get(c, c) for c in bp_corr.columns]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(bp_corr.astype(float), annot=True, fmt='.3f', cmap='RdBu_r',
                vmin=-1, vmax=1, center=0, linewidths=0.5,
                cbar_kws={'shrink': 0.8}, annot_kws={'size': 10}, ax=ax)
    
    ax.set_title('Correlation: Traditional Bandpower Features vs Behavior Variables',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Behavior Variables', fontsize=12)
    ax.set_ylabel('Bandpower Features', fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'correlation_bandpower.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_de_correlation_heatmap(eeg_behavior_corr, de_cols, behavior_cols, output_dir, config):
    """Plot correlation heatmap for DE features"""
    print("\nPlotting DE features correlation heatmap...")
    
    # Filter to DE columns only
    de_corr = eeg_behavior_corr.loc[de_cols, :].copy()
    
    # Rename for display
    display_names = {
        'perclos': 'PERCLOS',
        'kss_rating': 'KSS',
        'stress': 'Stress',
        'frustration': 'Frustration',
        'accuracy': 'Accuracy',
        'hit_rate': 'Hit Rate',
        'false_alarm_rate': 'False Alarm',
        'rt_mean': 'RT Mean'
    }
    
    # Create shorter row names
    row_names = []
    for col in de_cols:
        parts = col.replace('de_', '').split('_')
        ch = parts[0].upper()
        freq = f'{parts[1]}-{parts[2].replace("Hz", "")}'
        row_names.append(f'{ch} {freq}')
    
    de_corr.index = row_names
    de_corr.columns = [display_names.get(c, c) for c in de_corr.columns]
    
    fig, ax = plt.subplots(figsize=(14, 20))
    
    sns.heatmap(de_corr.astype(float), annot=True, fmt='.2f', cmap='RdBu_r',
                vmin=-1, vmax=1, center=0, linewidths=0.3,
                cbar_kws={'shrink': 0.5}, annot_kws={'size': 7}, ax=ax)
    
    ax.set_title('Correlation: DE Features (25 bands x 2 channels) vs Behavior Variables',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Behavior Variables', fontsize=12)
    ax.set_ylabel('DE Features (Channel Freq-Range)', fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'correlation_de_features.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def save_correlation_summary(behavior_corr, eeg_behavior_corr, output_dir):
    """Save correlation matrices to CSV"""
    print("\nSaving correlation summary to CSV...")
    
    # Save behavior correlation
    behavior_path = os.path.join(output_dir, 'correlation_behavior.csv')
    behavior_corr.to_csv(behavior_path)
    print(f"Saved: {behavior_path}")
    
    # Save EEG-behavior correlation
    eeg_path = os.path.join(output_dir, 'correlation_eeg_behavior.csv')
    eeg_behavior_corr.to_csv(eeg_path)
    print(f"Saved: {eeg_path}")
    
    # Create summary of strongest correlations
    summary_rows = []
    
    for eeg_col in eeg_behavior_corr.index:
        for beh_col in eeg_behavior_corr.columns:
            val = eeg_behavior_corr.loc[eeg_col, beh_col]
            if pd.notna(val):
                summary_rows.append({
                    'eeg_feature': eeg_col,
                    'behavior_variable': beh_col,
                    'correlation': val,
                    'abs_correlation': abs(val)
                })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('abs_correlation', ascending=False)
    
    summary_path = os.path.join(output_dir, 'correlation_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    
    # Print top correlations
    print("\nTop 10 EEG-Behavior Correlations:")
    print(summary_df.head(10).to_string(index=False))


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    """Run EDA analysis pipeline"""
    parser = argparse.ArgumentParser(description='EDA Analysis for merged_data.csv')
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
    
    # Create output directory in data directory
    output_dir = os.path.join(data_dir, config.OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = datetime.now()
    print("=" * 70)
    print("EDA ANALYSIS")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {output_dir}")
    print("")
    
    # Step 1: Load and preprocess data
    df, original_fs = load_and_preprocess_data(data_dir, config)
    
    # Step 2: Calculate PERCLOS
    df = calculate_perclos(df, config)
    
    # Step 3: Extract EEG features
    de_features, bandpower_features, window_timestamps, window_indices = extract_eeg_features(
        df, config, original_fs
    )
    
    # Step 4: Calculate task metrics
    task_metrics = calculate_task_metrics(df, window_indices, config)
    
    # Step 5: Process subjective ratings
    subjective_data = process_subjective_ratings(df, window_indices, config)
    
    # Step 6: Aggregate all metrics
    agg_df = aggregate_all_metrics(
        df, window_indices, de_features, bandpower_features,
        task_metrics, subjective_data, config
    )
    
    # Save aggregated data
    agg_path = os.path.join(output_dir, 'aggregated_features.csv')
    agg_df.to_csv(agg_path, index=False)
    print(f"\nSaved aggregated features to: {agg_path}")
    
    # Step 7: Calculate correlations
    behavior_corr, eeg_behavior_corr, behavior_cols, de_cols, bp_cols = calculate_correlations(
        agg_df, config
    )
    
    # Step 8: Create visualizations
    print("=" * 60)
    print("STEP 8: Creating Visualizations")
    print("=" * 60)
    
    plot_behavior_correlation_heatmap(behavior_corr, output_dir)
    plot_bandpower_correlation_heatmap(eeg_behavior_corr, bp_cols, behavior_cols, output_dir, config)
    plot_de_correlation_heatmap(eeg_behavior_corr, de_cols, behavior_cols, output_dir, config)
    
    # Save correlation summary
    save_correlation_summary(behavior_corr, eeg_behavior_corr, output_dir)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("")
    print("=" * 70)
    print("EDA ANALYSIS COMPLETED")
    print("=" * 70)
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print("")
    print("Output files:")
    print(f"  - {output_dir}/correlation_subjective_perclos_performance.png")
    print(f"  - {output_dir}/correlation_bandpower.png")
    print(f"  - {output_dir}/correlation_de_features.png")
    print(f"  - {output_dir}/correlation_summary.csv")
    print(f"  - {output_dir}/aggregated_features.csv")


if __name__ == "__main__":
    main()

