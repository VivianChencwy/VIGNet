"""
Data loader for experiment dataset
Loads de_LDS EEG features and generates labels for 4 targets:
- PERCLOS: eye closure percentage (already available)
- KSS: Karolinska Sleepiness Scale, normalized to 0-1
- miss_rate: target miss rate per window (60s window, 30s overlap)
- false_alarm: false alarm rate per window (60s window, 30s overlap)
"""

import numpy as np
import pandas as pd
import os
from scipy.io import loadmat
from scipy import interpolate


class ExperimentDataLoader:
    """
    Data loader for experiment dataset with 4 separate targets.
    Each target is loaded independently for training separate models.
    
    For miss_rate and false_alarm:
    - Uses 60-second windows with 30-second overlap
    - Aggregates EEG features by averaging within each window
    """
    
    def __init__(self, 
                 eeg_feature_path,
                 perclos_path,
                 task_csv_path,
                 preprocessing_report_path=None,
                 feature_type="de_LDS",
                 train_ratio=0.7,
                 val_ratio=0.15,
                 test_ratio=0.15,
                 random_seed=970304):
        """
        Args:
            eeg_feature_path: Path to EEG feature .mat file
            perclos_path: Path to PERCLOS label .mat file
            task_csv_path: Path to task performance CSV file
            preprocessing_report_path: Path to preprocessing report (for window timing)
            feature_type: Feature type to use (default: de_LDS)
            train_ratio: Training set ratio (default: 0.7)
            val_ratio: Validation set ratio (default: 0.15)
            test_ratio: Test set ratio (default: 0.15)
            random_seed: Random seed for reproducibility
        """
        self.eeg_feature_path = eeg_feature_path
        self.perclos_path = perclos_path
        self.task_csv_path = task_csv_path
        self.preprocessing_report_path = preprocessing_report_path
        self.feature_type = feature_type
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # Window parameters for original EEG features (from preprocessing)
        self.window_size = 8.0  # seconds
        self.stride = 4.0  # seconds
        
        # Window parameters for task performance targets (miss_rate, false_alarm)
        self.task_window_size = 60.0  # 1 minute window
        self.task_stride = 10.0  # 10 second stride (83% overlap)
        
        # Load data
        self._load_eeg_features()
        self._load_perclos()
        self._compute_window_times()
        self._load_task_data()
        self._generate_all_labels()
    
    def _load_eeg_features(self):
        """Load EEG features from .mat file"""
        data = loadmat(self.eeg_feature_path)
        self.features = data[self.feature_type]  # Shape: (2, n_windows, 25)
        self.n_windows = self.features.shape[1]
        self.n_channels = self.features.shape[0]
        self.n_freqs = self.features.shape[2]
        print(f"Loaded EEG features: {self.features.shape}")
        print(f"  Channels: {self.n_channels}, Windows: {self.n_windows}, Frequencies: {self.n_freqs}")
    
    def _load_perclos(self):
        """Load PERCLOS labels from .mat file"""
        data = loadmat(self.perclos_path)
        self.perclos = np.squeeze(data['perclos'])
        print(f"Loaded PERCLOS: {self.perclos.shape}")
        print(f"  Range: [{self.perclos.min():.4f}, {self.perclos.max():.4f}]")
    
    def _compute_window_times(self):
        """Compute center timestamps for each window"""
        # Assume windows start from time 0
        # Window center = start + window_size/2
        # Start times: 0, stride, 2*stride, ...
        self.window_centers = np.array([
            i * self.stride + self.window_size / 2 
            for i in range(self.n_windows)
        ])
        print(f"Window time range: [{self.window_centers[0]:.1f}, {self.window_centers[-1]:.1f}] seconds")
    
    def _load_task_data(self):
        """Load task performance data from CSV"""
        self.task_df = pd.read_csv(self.task_csv_path)
        
        # Get formal task trials only
        self.formal_trials = self.task_df[self.task_df['phase'] == 'Mackworth_Formal'].copy()
        
        # Get the start time of formal task
        if len(self.formal_trials) > 0:
            self.task_start_time = self.formal_trials['lsl_timestamp'].min()
        else:
            self.task_start_time = self.task_df['lsl_timestamp'].min()
        
        print(f"Loaded task data: {len(self.formal_trials)} formal trials")
        print(f"  Task start time: {self.task_start_time:.2f}")
        
        # Load subjective ratings
        self.kss_data = self.task_df[self.task_df['event'] == 'KSS'][['lsl_timestamp', 'kss_rating']].copy()
        self.kss_data = self.kss_data[self.kss_data['kss_rating'].notna()]
        print(f"  KSS ratings: {len(self.kss_data)} data points")
    
    def _generate_all_labels(self):
        """Generate labels for all 4 targets"""
        self._generate_kss_labels()
        self._generate_task_performance_labels()
    
    def _generate_kss_labels(self):
        """Generate KSS labels by interpolation, normalized to 0-1"""
        if len(self.kss_data) > 0:
            # Convert window centers to absolute timestamps
            absolute_times = self.window_centers + self.task_start_time
            
            # Interpolate KSS values
            f_kss = interpolate.interp1d(
                self.kss_data['lsl_timestamp'].values,
                self.kss_data['kss_rating'].values,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            kss_raw = f_kss(absolute_times)
            
            # Clip to valid range [1, 9]
            kss_raw = np.clip(kss_raw, 1, 9)
            
            # Normalize to [0, 1]: (KSS - 1) / 8
            self.kss = (kss_raw - 1) / 8.0
            
            print(f"Generated KSS labels: {self.kss.shape}")
            print(f"  Raw range: [{kss_raw.min():.2f}, {kss_raw.max():.2f}]")
            print(f"  Normalized range: [{self.kss.min():.4f}, {self.kss.max():.4f}]")
        else:
            self.kss = np.full(self.n_windows, np.nan)
            print("Warning: No KSS data available")
    
    def _generate_task_performance_labels(self):
        """
        Generate miss_rate and false_alarm labels using 60-second windows with 30-second overlap.
        
        This creates a separate set of windows for task performance targets:
        - Window size: 60 seconds (1 minute)
        - Stride: 30 seconds (50% overlap)
        - EEG features are aggregated (averaged) within each window
        """
        # Compute total duration covered by EEG features
        total_duration = self.window_centers[-1] + self.window_size / 2
        
        # Create task performance windows (60s window, 30s stride)
        self.task_window_centers = []
        start_time = self.task_window_size / 2  # Start when we have a full window
        
        while start_time + self.task_window_size / 2 <= total_duration:
            self.task_window_centers.append(start_time)
            start_time += self.task_stride
        
        self.task_window_centers = np.array(self.task_window_centers)
        self.n_task_windows = len(self.task_window_centers)
        
        print(f"\nTask performance window configuration:")
        print(f"  Window size: {self.task_window_size}s, Stride: {self.task_stride}s")
        print(f"  Number of task windows: {self.n_task_windows}")
        print(f"  Window time range: [{self.task_window_centers[0]:.1f}, {self.task_window_centers[-1]:.1f}] seconds")
        
        # Initialize labels for task windows
        self.miss_rate = np.full(self.n_task_windows, np.nan)
        self.false_alarm = np.full(self.n_task_windows, np.nan)
        
        # Aggregate EEG features for task windows
        # Shape: (n_task_windows, n_channels, n_freqs)
        self.task_features = np.zeros((self.n_task_windows, self.n_channels, self.n_freqs))
        
        # Map original EEG windows to task windows
        self.task_window_eeg_indices = []  # For debugging: which EEG windows map to each task window
        
        valid_windows = 0
        
        for i in range(self.n_task_windows):
            task_win_start = self.task_window_centers[i] - self.task_window_size / 2
            task_win_end = self.task_window_centers[i] + self.task_window_size / 2
            
            # Find original EEG windows that fall within this task window
            # An EEG window is included if its center falls within the task window
            eeg_mask = (self.window_centers >= task_win_start) & (self.window_centers < task_win_end)
            eeg_indices = np.where(eeg_mask)[0]
            self.task_window_eeg_indices.append(eeg_indices)
            
            # Aggregate EEG features by averaging
            if len(eeg_indices) > 0:
                # features shape: (n_channels, n_windows, n_freqs)
                self.task_features[i] = np.mean(self.features[:, eeg_indices, :], axis=1)
            
            # Window time range (absolute) for trial matching
            abs_win_start = task_win_start + self.task_start_time
            abs_win_end = task_win_end + self.task_start_time
            
            # Find trials in this window
            mask = (self.formal_trials['lsl_timestamp'] >= abs_win_start) & \
                   (self.formal_trials['lsl_timestamp'] < abs_win_end)
            window_trials = self.formal_trials[mask]
            
            if len(window_trials) > 0:
                valid_windows += 1
                
                # Separate target and non-target trials
                target_trials = window_trials[window_trials['is_target_double_jump'] == 1]
                non_target_trials = window_trials[window_trials['is_target_double_jump'] == 0]
                
                n_targets = len(target_trials)
                n_non_targets = len(non_target_trials)
                
                # Compute miss rate (misses / targets)
                if n_targets > 0:
                    n_misses = target_trials['miss'].sum()
                    self.miss_rate[i] = n_misses / n_targets
                
                # Compute false alarm rate (false alarms / non-targets)
                if n_non_targets > 0:
                    n_false_alarms = non_target_trials['responded'].sum()
                    self.false_alarm[i] = n_false_alarms / n_non_targets
        
        print(f"\nGenerated task performance labels:")
        print(f"  Valid windows: {valid_windows}/{self.n_task_windows}")
        print(f"  Average EEG windows per task window: {np.mean([len(x) for x in self.task_window_eeg_indices]):.1f}")
        print(f"  miss_rate - valid: {np.sum(~np.isnan(self.miss_rate))}, "
              f"range: [{np.nanmin(self.miss_rate):.4f}, {np.nanmax(self.miss_rate):.4f}]")
        print(f"  false_alarm - valid: {np.sum(~np.isnan(self.false_alarm))}, "
              f"range: [{np.nanmin(self.false_alarm):.4f}, {np.nanmax(self.false_alarm):.4f}]")
    
    def _stratified_split(self, labels, valid_mask):
        """
        Stratified train/val/test split based on label distribution.
        Only uses samples where valid_mask is True.
        """
        valid_indices = np.where(valid_mask)[0]
        valid_labels = labels[valid_indices]
        
        n_samples = len(valid_indices)
        n_bins = 5  # Stratification bins
        
        # Create bins for stratification
        bin_edges = np.linspace(valid_labels.min(), valid_labels.max() + 0.001, n_bins + 1)
        bin_indices = np.digitize(valid_labels, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        train_idx, val_idx, test_idx = [], [], []
        
        # Stratified sampling per bin
        for bin_id in range(n_bins):
            bin_mask = (bin_indices == bin_id)
            bin_samples = valid_indices[bin_mask]
            
            if len(bin_samples) == 0:
                continue
            
            rng = np.random.RandomState(seed=self.random_seed + bin_id)
            shuffled = rng.permutation(bin_samples)
            
            n_train = int(len(shuffled) * self.train_ratio)
            n_val = int(len(shuffled) * self.val_ratio)
            
            train_idx.extend(shuffled[:n_train])
            val_idx.extend(shuffled[n_train:n_train + n_val])
            test_idx.extend(shuffled[n_train + n_val:])
        
        # Shuffle final indices
        rng = np.random.RandomState(seed=self.random_seed)
        train_idx = rng.permutation(train_idx)
        val_idx = rng.permutation(val_idx)
        test_idx = rng.permutation(test_idx)
        
        return np.array(train_idx), np.array(val_idx), np.array(test_idx)
    
    def get_dataset(self, target='perclos'):
        """
        Get train/val/test datasets for a specific target.
        
        Args:
            target: One of 'perclos', 'kss', 'miss_rate', 'false_alarm'
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
            X shape: (n_samples, n_channels, n_freqs, 1)
            y shape: (n_samples, 1)
            
        Note:
            For miss_rate and false_alarm targets, uses 60-second windows with 30-second overlap.
            EEG features are aggregated (averaged) within each window.
        """
        # Determine which feature set and labels to use based on target
        if target in ['miss_rate', 'false_alarm']:
            # Use task performance windows (60s window, 30s overlap)
            if target == 'miss_rate':
                labels = self.miss_rate
            else:
                labels = self.false_alarm
            
            # Use aggregated task features
            features = self.task_features  # Already (n_task_windows, n_channels, n_freqs)
            n_total = self.n_task_windows
            window_info = f"(60s window, {int(self.task_stride)}s stride)"
        else:
            # Use original EEG windows for perclos and kss
            if target == 'perclos':
                labels = self.perclos
            elif target == 'kss':
                labels = self.kss
            else:
                raise ValueError(f"Unknown target: {target}. "
                               f"Choose from: perclos, kss, miss_rate, false_alarm")
            
            # Prepare features: (n_channels, n_windows, n_freqs) -> (n_windows, n_channels, n_freqs)
            features = np.moveaxis(self.features, 0, 1)
            n_total = self.n_windows
            window_info = f"(8s window, 4s stride)"
        
        # Valid samples (non-NaN)
        valid_mask = ~np.isnan(labels)
        n_valid = np.sum(valid_mask)
        
        print(f"\nPreparing dataset for target: {target} {window_info}")
        print(f"  Valid samples: {n_valid}/{n_total}")
        
        if n_valid == 0:
            raise ValueError(f"No valid samples for target: {target}")
        
        # Stratified split
        train_idx, val_idx, test_idx = self._stratified_split(labels, valid_mask)
        
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        # Split data
        X_train = features[train_idx]
        X_val = features[val_idx]
        X_test = features[test_idx]
        
        y_train = labels[train_idx]
        y_val = labels[val_idx]
        y_test = labels[test_idx]
        
        # Add channel dimension for Conv2D: (n_samples, n_channels, n_freqs, 1)
        X_train = np.expand_dims(X_train, -1)
        X_val = np.expand_dims(X_val, -1)
        X_test = np.expand_dims(X_test, -1)
        
        # Add dimension to labels: (n_samples, 1)
        y_train = np.expand_dims(y_train, -1)
        y_val = np.expand_dims(y_val, -1)
        y_test = np.expand_dims(y_test, -1)
        
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
        print(f"  y range: [{y_train.min():.4f}, {y_train.max():.4f}]")
        
        return X_train, y_train, X_val, y_val, X_test, y_test


def load_experiment_dataset(target='perclos',
                           data_dir='data/experiment_20251124_140734',
                           feature_type='de_LDS',
                           random_seed=970304):
    """
    Convenience function to load experiment dataset for a specific target.
    
    Args:
        target: One of 'perclos', 'kss', 'miss_rate', 'false_alarm'
        data_dir: Base directory containing experiment data
        feature_type: EEG feature type (default: de_LDS)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Construct paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, data_dir)
    
    eeg_feature_path = os.path.join(data_path, 'processed', 'EEG_Feature_2Hz', 'eeg_eye_data.mat')
    perclos_path = os.path.join(data_path, 'processed', 'perclos_labels', 'eeg_eye_data.mat')
    
    # Find task CSV
    task_csv_files = [f for f in os.listdir(data_path) if f.startswith('S') and f.endswith('.csv')]
    if task_csv_files:
        task_csv_path = os.path.join(data_path, task_csv_files[0])
    else:
        raise FileNotFoundError(f"No task CSV found in {data_path}")
    
    print(f"Loading experiment data from: {data_path}")
    print(f"  EEG features: {eeg_feature_path}")
    print(f"  PERCLOS: {perclos_path}")
    print(f"  Task CSV: {task_csv_path}")
    
    # Create loader and get dataset
    loader = ExperimentDataLoader(
        eeg_feature_path=eeg_feature_path,
        perclos_path=perclos_path,
        task_csv_path=task_csv_path,
        feature_type=feature_type,
        random_seed=random_seed
    )
    
    return loader.get_dataset(target=target)


if __name__ == "__main__":
    # Test loading all targets
    for target in ['perclos', 'kss', 'miss_rate', 'false_alarm']:
        print(f"\n{'='*60}")
        print(f"Testing target: {target}")
        print('='*60)
        try:
            X_train, y_train, X_val, y_val, X_test, y_test = load_experiment_dataset(target=target)
            print(f"Successfully loaded {target} dataset")
        except Exception as e:
            print(f"Error loading {target}: {e}")


