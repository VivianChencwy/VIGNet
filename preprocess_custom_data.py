"""
Preprocessing script for custom EEG + Eye state merged data
(Matching original SEED-VIG preprocessing pipeline)

This script processes the eeg_eye_merged.csv file to:
1. Downsample EEG to 200 Hz (matching SEED-VIG standard)
2. Segment into 8-second windows with 4-second stride
3. Compute Differential Entropy (DE) features using bandpass filter + variance
4. Apply Moving Average and LDS (Kalman) smoothing
5. Compute PERCLOS labels from eye state data
6. Save processed features and labels for training

Original SEED-VIG preprocessing parameters:
- Sampling rate: 200 Hz
- Window size: 8 seconds (1600 samples)
- Window stride: 4 seconds (800 samples, 50% overlap)
- Frequency bands: 25 sub-bands (2Hz resolution, 1-50Hz)
- DE calculation: bandpass filter + log(variance)
- Smoothing: Moving Average (window=5) + LDS (Kalman filter)
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import uniform_filter1d
import os
from datetime import datetime

# Optional: Kalman filter for LDS smoothing
try:
    from pykalman import KalmanFilter
    HAS_PYKALMAN = True
except ImportError:
    HAS_PYKALMAN = False
    print("Warning: pykalman not installed. LDS smoothing will be disabled.")


class EEGPreprocessor:
    """
    EEG Preprocessor matching SEED-VIG pipeline
    """
    
    def __init__(self, target_fs=200, window_sec=8.0, stride_sec=4.0):
        """
        Initialize preprocessor with SEED-VIG parameters
        
        Args:
            target_fs: Target sampling rate (200 Hz for SEED-VIG)
            window_sec: Window size in seconds (8s for SEED-VIG)
            stride_sec: Window stride in seconds (4s for SEED-VIG)
        """
        self.target_fs = target_fs
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.samples_per_window = int(window_sec * target_fs)  # 1600 samples
        self.stride_samples = int(stride_sec * target_fs)      # 800 samples
        
        # Define 25 frequency bands with 2Hz resolution (1-50Hz)
        # Same as original SEED-VIG: [(1,3), (3,5), (5,7), ..., (47,49), (49,51)]
        self.freq_bands = [(i, i+2) for i in range(1, 50, 2)]
        
        # Center frequencies for reference
        self.center_freqs = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 
                            22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
                            42, 44, 46, 48, 50]
    
    def resample(self, data, original_fs):
        """
        Resample signal to target sampling rate
        
        Args:
            data: numpy array of shape (n_samples, n_channels)
            original_fs: Original sampling rate
        
        Returns:
            Resampled data
        """
        if abs(original_fs - self.target_fs) < 1:
            return data  # No resampling needed
        
        n_samples = data.shape[0]
        duration = n_samples / original_fs
        target_samples = int(duration * self.target_fs)
        
        # Resample each channel
        resampled = np.zeros((target_samples, data.shape[1]))
        for ch in range(data.shape[1]):
            resampled[:, ch] = signal.resample(data[:, ch], target_samples)
        
        return resampled
    
    def segment_signal(self, data):
        """
        Segment signal into overlapping windows
        
        Args:
            data: numpy array of shape (n_samples, n_channels)
        
        Returns:
            segments: numpy array of shape (n_windows, samples_per_window, n_channels)
        """
        n_samples = data.shape[0]
        n_channels = data.shape[1]
        segments = []
        
        for start in range(0, n_samples - self.samples_per_window + 1, self.stride_samples):
            end = start + self.samples_per_window
            segments.append(data[start:end, :])
        
        return np.array(segments)
    
    def extract_de_bandpass(self, segment):
        """
        Extract Differential Entropy for each 2Hz band using bandpass filter + variance
        (Matching original SEED-VIG method)
        
        DE = 0.5 * log(2 * pi * e * variance)
        
        Args:
            segment: 1D array of shape (samples_per_window,)
        
        Returns:
            de_values: 1D array of shape (25,) - DE for each frequency band
        """
        de_values = np.zeros(25)
        
        for i, (low_freq, high_freq) in enumerate(self.freq_bands):
            # Design bandpass filter
            nyq = self.target_fs / 2.0
            low = max(0.01, low_freq / nyq)
            high = min(0.99, high_freq / nyq)
            
            if low < high:
                # Apply bandpass filter (4th order Butterworth)
                b, a = signal.butter(4, [low, high], btype='band')
                filtered = signal.filtfilt(b, a, segment)
                
                # Compute differential entropy: DE = 0.5 * log(2 * pi * e * variance)
                variance = np.var(filtered)
                de = 0.5 * np.log(2 * np.pi * np.e * (variance + 1e-10))
                de_values[i] = de
            else:
                de_values[i] = 0.0
        
        return de_values
    
    def extract_features(self, segments):
        """
        Extract DE features from all segments
        
        Args:
            segments: numpy array of shape (n_windows, samples_per_window, n_channels)
        
        Returns:
            de_features: numpy array of shape (n_channels, n_windows, 25)
        """
        n_windows = segments.shape[0]
        n_channels = segments.shape[2]
        
        # Initialize feature array: (n_channels, n_windows, 25)
        de_features = np.zeros((n_channels, n_windows, 25))
        
        print(f"Extracting DE features from {n_windows} windows...")
        
        for w in range(n_windows):
            if (w + 1) % 50 == 0:
                print(f"  Processing window {w + 1}/{n_windows}")
            
            for ch in range(n_channels):
                de_features[ch, w, :] = self.extract_de_bandpass(segments[w, :, ch])
        
        return de_features
    
    def apply_moving_average(self, features, window=5):
        """
        Apply moving average smoothing along the time axis
        (Matching original SEED-VIG smoothing)
        
        Args:
            features: 3D array of shape (n_channels, n_windows, n_freqs)
            window: Window size for moving average (default 5)
        
        Returns:
            smoothed: 3D array with same shape
        """
        n_channels, n_windows, n_freqs = features.shape
        smoothed = np.zeros_like(features)
        
        for ch in range(n_channels):
            for freq in range(n_freqs):
                smoothed[ch, :, freq] = uniform_filter1d(features[ch, :, freq], 
                                                         size=window, mode='nearest')
        
        return smoothed
    
    def apply_lds_smoothing(self, features):
        """
        Apply Linear Dynamic System (Kalman filter) smoothing along the time axis
        (Matching original SEED-VIG LDS smoothing)
        
        Args:
            features: 3D array of shape (n_channels, n_windows, n_freqs)
        
        Returns:
            smoothed: 3D array with same shape
        """
        if not HAS_PYKALMAN:
            print("Warning: pykalman not available, skipping LDS smoothing")
            return features
        
        n_channels, n_windows, n_freqs = features.shape
        smoothed = np.zeros_like(features)
        
        for ch in range(n_channels):
            for freq in range(n_freqs):
                # Time series for this channel and frequency
                ts = features[ch, :, freq]
                
                # Simple Kalman filter for smoothing
                kf = KalmanFilter(
                    initial_state_mean=ts[0],
                    n_dim_obs=1,
                    n_dim_state=1,
                    transition_matrices=[1],
                    observation_matrices=[1],
                    transition_covariance=0.1 * np.eye(1),
                    observation_covariance=1.0 * np.eye(1)
                )
                
                # Smooth the time series
                smoothed_ts, _ = kf.smooth(ts.reshape(-1, 1))
                smoothed[ch, :, freq] = smoothed_ts.flatten()
        
        return smoothed


class PERCLOSCalculator:
    """Calculate PERCLOS from eye state data"""
    
    def __init__(self, window_sec=60.0):
        """
        Initialize PERCLOS calculator
        
        Args:
            window_sec: Window size in seconds for PERCLOS calculation
                       (typically 60 seconds in fatigue detection literature)
        """
        self.window_sec = window_sec
    
    def compute_perclos(self, timestamps, eye_states, window_timestamps):
        """
        Compute PERCLOS for each window
        
        PERCLOS = percentage of time eyes are closed in the preceding window
        
        Args:
            timestamps: array of timestamps for eye state data
            eye_states: array of eye states ('open' or 'closed')
            window_timestamps: array of timestamps for each feature window (center)
        
        Returns:
            perclos: array of PERCLOS values for each window
        """
        perclos = np.zeros(len(window_timestamps))
        
        # Convert eye states to binary (1 = closed, 0 = open)
        eye_closed = (eye_states == 'closed').astype(float)
        
        for i, win_ts in enumerate(window_timestamps):
            # Find eye states in the preceding window
            window_start = win_ts - self.window_sec
            window_mask = (timestamps >= window_start) & (timestamps <= win_ts)
            
            if np.sum(window_mask) > 0:
                # PERCLOS = proportion of time eyes are closed
                perclos[i] = np.mean(eye_closed[window_mask])
            else:
                # If no data in window, use a default value
                perclos[i] = 0.0
        
        return perclos


def process_merged_data(input_csv, output_dir, perclos_window_sec=60.0, 
                        trim_start_sec=300.0, trim_end_sec=60.0):
    """
    Main function to process merged EEG + eye state data
    (Matching original SEED-VIG preprocessing pipeline)
    
    Args:
        input_csv: Path to eeg_eye_merged.csv
        output_dir: Directory to save processed features
        perclos_window_sec: Window size for PERCLOS calculation
        trim_start_sec: Seconds to trim from the start (default 300s = 5 minutes)
        trim_end_sec: Seconds to trim from the end (default 60s = 1 minute)
    """
    print("=" * 60)
    print("EEG + Eye State Data Preprocessing")
    print("(Matching SEED-VIG Pipeline)")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"Loaded {len(df)} samples")
    
    # Trim start and end of recording
    if trim_start_sec > 0 or trim_end_sec > 0:
        print(f"\n[Trimming] Removing first {trim_start_sec}s and last {trim_end_sec}s...")
        lsl_timestamps = df['lsl_timestamp'].values
        total_duration = lsl_timestamps[-1] - lsl_timestamps[0]
        print(f"  Original duration: {total_duration:.2f}s ({total_duration/60:.2f} min)")
        
        start_time = lsl_timestamps[0] + trim_start_sec
        end_time = lsl_timestamps[-1] - trim_end_sec
        
        # Filter data
        mask = (df['lsl_timestamp'] >= start_time) & (df['lsl_timestamp'] <= end_time)
        df = df[mask].reset_index(drop=True)
        
        new_duration = df['lsl_timestamp'].iloc[-1] - df['lsl_timestamp'].iloc[0]
        print(f"  Trimmed duration: {new_duration:.2f}s ({new_duration/60:.2f} min)")
        print(f"  Samples after trimming: {len(df)}")
    
    # Extract EEG columns (use cleaned data if available)
    eeg_columns = ['fp1_clean', 'fp2_clean']
    if 'fp1_clean' not in df.columns:
        if 'fp1_filtered' in df.columns:
            eeg_columns = ['fp1_filtered', 'fp2_filtered']
        else:
            eeg_columns = ['fp1_raw', 'fp2_raw']
    
    print(f"Using EEG columns: {eeg_columns}")
    
    # Extract EEG data
    eeg_data = df[eeg_columns].values
    print(f"EEG data shape: {eeg_data.shape}")
    
    # Get timestamps from LSL timestamp
    lsl_timestamps = df['lsl_timestamp'].values
    
    # Estimate original sampling rate
    time_diffs = np.diff(lsl_timestamps)
    original_fs = 1.0 / np.median(time_diffs)
    print(f"Original EEG sampling rate: {original_fs:.2f} Hz")
    
    # Initialize preprocessor with SEED-VIG parameters
    preprocessor = EEGPreprocessor(
        target_fs=200,       # SEED-VIG standard
        window_sec=8.0,      # 8-second windows
        stride_sec=4.0       # 4-second stride (50% overlap)
    )
    
    # Step 1: Resample to 200 Hz
    print(f"\n[Step 1] Resampling from {original_fs:.2f} Hz to 200 Hz...")
    eeg_resampled = preprocessor.resample(eeg_data, original_fs)
    print(f"  Resampled shape: {eeg_resampled.shape}")
    
    # Compute resampled timestamps
    duration = len(eeg_data) / original_fs
    resampled_timestamps = np.linspace(lsl_timestamps[0], lsl_timestamps[-1], len(eeg_resampled))
    
    # Step 2: Segment into windows
    print(f"\n[Step 2] Segmenting into {preprocessor.window_sec}s windows with {preprocessor.stride_sec}s stride...")
    segments = preprocessor.segment_signal(eeg_resampled)
    print(f"  Number of windows: {segments.shape[0]}")
    print(f"  Samples per window: {segments.shape[1]}")
    
    # Compute window center timestamps
    n_windows = segments.shape[0]
    window_center_indices = [
        int(i * preprocessor.stride_samples + preprocessor.samples_per_window / 2)
        for i in range(n_windows)
    ]
    window_timestamps = resampled_timestamps[window_center_indices]
    
    # Step 3: Extract DE features (bandpass + variance)
    print(f"\n[Step 3] Extracting DE features (bandpass filter + variance)...")
    de_raw = preprocessor.extract_features(segments)
    print(f"  DE features shape: {de_raw.shape}")  # (n_channels, n_windows, 25)
    
    # Step 4: Apply smoothing
    print(f"\n[Step 4] Applying smoothing...")
    print("  Applying moving average (window=5)...")
    de_movingAve = preprocessor.apply_moving_average(de_raw, window=5)
    
    print("  Applying LDS (Kalman filter) smoothing...")
    de_LDS = preprocessor.apply_lds_smoothing(de_raw)
    
    # Step 5: Compute PERCLOS labels
    print(f"\n[Step 5] Computing PERCLOS labels (window={perclos_window_sec}s)...")
    
    # Get eye state data
    eye_states = df['aria_eye_state'].values
    eye_timestamps_unix = df['aria_timestamp_sec'].values
    
    # Map LSL timestamps to UNIX timestamps for PERCLOS calculation
    lsl_to_unix_offset = df['aria_timestamp_sec'].iloc[0] - df['lsl_timestamp'].iloc[0]
    window_unix_timestamps = window_timestamps + lsl_to_unix_offset
    
    perclos_calc = PERCLOSCalculator(window_sec=perclos_window_sec)
    perclos = perclos_calc.compute_perclos(eye_timestamps_unix, eye_states, window_unix_timestamps)
    
    print(f"  PERCLOS shape: {perclos.shape}")
    print(f"  PERCLOS range: [{perclos.min():.4f}, {perclos.max():.4f}]")
    print(f"  PERCLOS mean: {perclos.mean():.4f}, std: {perclos.std():.4f}")
    
    # Create classification labels (same thresholds as SEED-VIG)
    clf_labels = np.zeros(len(perclos), dtype=int)
    clf_labels[perclos >= 0.35] = 1  # tired
    clf_labels[perclos >= 0.7] = 2   # drowsy
    
    # Print label distribution
    unique, counts = np.unique(clf_labels, return_counts=True)
    print(f"\nClassification label distribution:")
    label_names = ['awake (0-0.35)', 'tired (0.35-0.7)', 'drowsy (0.7+)']
    for u, c in zip(unique, counts):
        print(f"  {label_names[u]}: {c} ({100*c/len(clf_labels):.2f}%)")
    
    # Step 6: Save processed data
    print(f"\n[Step 6] Saving processed data to: {output_dir}")
    
    # Convert features to shape expected by VIGNet: (n_windows, n_channels, n_bands)
    # Original shape: (n_channels, n_windows, 25)
    # Need: (n_windows, n_channels, 25)
    de_LDS_transposed = np.moveaxis(de_LDS, 0, 1)  # (n_windows, n_channels, 25)
    de_movingAve_transposed = np.moveaxis(de_movingAve, 0, 1)
    
    # Save DE features (use de_LDS as primary, same as original VIGNet)
    features_path = os.path.join(output_dir, 'de_features.npy')
    np.save(features_path, de_LDS_transposed)
    print(f"  Saved DE features (LDS): {features_path}")
    print(f"    Shape: {de_LDS_transposed.shape}")
    
    # Also save moving average version
    features_ma_path = os.path.join(output_dir, 'de_features_movingAve.npy')
    np.save(features_ma_path, de_movingAve_transposed)
    print(f"  Saved DE features (MovingAve): {features_ma_path}")
    
    # Save PERCLOS labels (regression)
    perclos_path = os.path.join(output_dir, 'perclos_labels.npy')
    np.save(perclos_path, perclos)
    print(f"  Saved PERCLOS labels: {perclos_path}")
    
    # Save classification labels
    clf_path = os.path.join(output_dir, 'clf_labels.npy')
    np.save(clf_path, clf_labels)
    print(f"  Saved classification labels: {clf_path}")
    
    # Save timestamps
    ts_path = os.path.join(output_dir, 'timestamps.npy')
    np.save(ts_path, window_unix_timestamps)
    print(f"  Saved timestamps: {ts_path}")
    
    # Save metadata
    metadata = {
        'input_file': input_csv,
        'eeg_channels': eeg_columns,
        'original_sampling_rate': original_fs,
        'target_sampling_rate': 200,
        'window_sec': 8.0,
        'stride_sec': 4.0,
        'perclos_window_sec': perclos_window_sec,
        'n_windows': n_windows,
        'n_channels': de_LDS_transposed.shape[1],
        'n_bands': 25,
        'freq_bands': preprocessor.freq_bands,
        'smoothing': ['moving_average', 'LDS'],
        'de_method': 'bandpass_filter_variance',
        'processing_date': datetime.now().isoformat(),
        'pipeline': 'SEED-VIG compatible'
    }
    metadata_path = os.path.join(output_dir, 'metadata.npy')
    np.save(metadata_path, metadata)
    print(f"  Saved metadata: {metadata_path}")
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("Pipeline: SEED-VIG compatible")
    print(f"  - Sampling rate: 200 Hz")
    print(f"  - Window: 8s, Stride: 4s")
    print(f"  - DE: bandpass filter + variance")
    print(f"  - Smoothing: Moving Average + LDS")
    print(f"  - Output windows: {n_windows}")
    print("=" * 60)
    
    return de_LDS_transposed, perclos, clf_labels


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess EEG + Eye state merged data (SEED-VIG compatible)')
    parser.add_argument('--input', type=str, 
                        default='/home/vivian/eeg/SEED_VIG/Dec_09_experiments/experiment_20251208_155206/eeg_eye_merged.csv',
                        help='Path to merged CSV file')
    parser.add_argument('--output', type=str,
                        default='/home/vivian/eeg/SEED_VIG/Dec_09_experiments/experiment_20251208_155206/processed',
                        help='Output directory for processed data')
    parser.add_argument('--perclos-window', type=float, default=60.0,
                        help='Window size in seconds for PERCLOS calculation')
    parser.add_argument('--trim-start', type=float, default=300.0,
                        help='Seconds to trim from start of recording (default: 300s = 5 min)')
    parser.add_argument('--trim-end', type=float, default=60.0,
                        help='Seconds to trim from end of recording (default: 60s = 1 min)')
    
    args = parser.parse_args()
    
    process_merged_data(
        input_csv=args.input,
        output_dir=args.output,
        perclos_window_sec=args.perclos_window,
        trim_start_sec=args.trim_start,
        trim_end_sec=args.trim_end
    )
