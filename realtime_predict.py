#!/usr/bin/env python3
"""
Real-time prediction script for VIGNet FP1/FP2 model
Loads saved model and scaler to perform PERCLOS prediction on new EEG features
"""

import os
import sys
import pickle
import time
import statistics
import numpy as np
import tensorflow as tf
import argparse
from pathlib import Path
from typing import Dict

# Disable XLA to avoid compatibility issues
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"
os.environ["TF_DISABLE_XLA"] = "1"

try:
    tf.config.optimizer.set_jit(False)
except:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except:
    pass

# Force eager execution
tf.config.run_functions_eagerly(True)


class VIGNetPredictor:
    """
    Real-time predictor for VIGNet FP1/FP2 model
    """
    def __init__(self, model_dir, trial=None, gpu_idx=0):
        """
        Initialize predictor by loading model and scaler
        
        Args:
            model_dir: Directory containing saved models (e.g., './logs_fp_no_cv/models')
            trial: Trial number (1-21). If None, will use the first available model.
            gpu_idx: GPU index to use
        """
        # Configure GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Using GPU: {gpus[0]}")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("Warning: No GPU found, using CPU")
        
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.metadata = None
        
        # Timing statistics
        self.last_preprocess_time = 0.0
        self.last_inference_time = 0.0
        self.last_total_time = 0.0
        
        # Find model files
        if trial is None:
            # Find first available model
            model_dirs = [d for d in os.listdir(model_dir) if d.startswith('trial') and d.endswith('_best_model')]
            if not model_dirs:
                raise FileNotFoundError(f"No saved models found in {model_dir}")
            trial = int(model_dirs[0].replace('trial', '').replace('_best_model', ''))
            print(f"No trial specified, using first available: trial {trial}")
        
        self.trial = trial
        self._load_model()
    
    def _load_model(self):
        """Load model, scaler, and metadata"""
        # Load metadata
        metadata_path = os.path.join(self.model_dir, f"trial{self.trial}_metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"Loaded metadata for trial {self.trial}")
            print(f"  Task: {self.metadata['task']}")
            print(f"  Input shape: {self.metadata['input_shape']}")
            print(f"  Output shape: {self.metadata['output_shape']}")
        else:
            print(f"Warning: Metadata file not found: {metadata_path}")
            self.metadata = {'task': 'RGS', 'input_shape': (None, 2, 25, 1)}
        
        # Load model
        model_path = os.path.join(self.model_dir, f"trial{self.trial}_best_model")
        if os.path.exists(model_path):
            try:
                # Try loading as SavedModel
                self.model = tf.keras.models.load_model(model_path)
                print(f"Loaded model from: {model_path}")
            except Exception as e:
                print(f"Failed to load SavedModel: {e}")
                # Try loading weights (requires model architecture)
                print("Note: If using weights-only format, you need to reconstruct the model first")
                raise
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load scaler
        scaler_path = os.path.join(self.model_dir, f"trial{self.trial}_scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Loaded scaler from: {scaler_path}")
        else:
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    
    def preprocess(self, features):
        """
        Preprocess features for prediction
        
        Args:
            features: EEG features array, shape (N, 2, 25) or (2, 25) for single sample
                     N: number of samples
                     2: FP1 and FP2 channels
                     25: frequency bins
        
        Returns:
            Preprocessed features ready for model input, shape (N, 2, 25, 1)
        """
        # Convert to numpy array if needed
        features = np.array(features, dtype=np.float64)
        
        # Handle single sample
        if features.ndim == 2:
            features = features[np.newaxis, :, :]  # (1, 2, 25)
        
        # Ensure correct shape
        if features.ndim != 3 or features.shape[1] != 2 or features.shape[2] != 25:
            raise ValueError(f"Expected features shape (N, 2, 25), got {features.shape}")
        
        # Flatten for scaler
        original_shape = features.shape
        features_flat = features.reshape(original_shape[0], -1)
        
        # Normalize using saved scaler
        features_flat = self.scaler.transform(features_flat)
        
        # Reshape back
        features = features_flat.reshape(original_shape)
        
        # Add time dimension: (N, 2, 25) -> (N, 2, 25, 1)
        features = np.expand_dims(features, axis=-1)
        
        return features
    
    def predict(self, features, return_numpy=True, measure_time=False):
        """
        Predict PERCLOS scores from EEG features
        
        Args:
            features: EEG features, shape (N, 2, 25) or (2, 25) for single sample
            return_numpy: If True, return numpy array; if False, return TensorFlow tensor
            measure_time: If True, measure and store timing information
        
        Returns:
            Predictions: PERCLOS scores, shape (N,) for regression or (N, 3) for classification
        """
        total_start = time.perf_counter()
        
        # Preprocess
        preprocess_start = time.perf_counter()
        features_processed = self.preprocess(features)
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000  # ms
        
        # Convert to TensorFlow tensor
        features_tf = tf.constant(features_processed, dtype=tf.float64)
        
        # Predict
        inference_start = time.perf_counter()
        predictions = self.model(features_tf, training=False)
        inference_time = (time.perf_counter() - inference_start) * 1000  # ms
        
        # Post-process
        if self.metadata and self.metadata.get('task') == 'RGS':
            # Regression: squeeze to (N,)
            predictions = tf.squeeze(predictions)
        else:
            # Classification: apply softmax if not already applied
            if predictions.shape[-1] == 3:
                predictions = tf.nn.softmax(predictions)
        
        total_time = (time.perf_counter() - total_start) * 1000  # ms
        
        # Store timing information
        if measure_time:
            self.last_preprocess_time = preprocess_time
            self.last_inference_time = inference_time
            self.last_total_time = total_time
        
        if return_numpy:
            return predictions.numpy()
        else:
            return predictions
    
    def predict_single(self, features, measure_time=False):
        """
        Predict for a single sample (convenience method)
        
        Args:
            features: Single sample features, shape (2, 25)
            measure_time: If True, measure and store timing information
        
        Returns:
            Single prediction value (scalar for regression, array for classification)
        """
        predictions = self.predict(features, measure_time=measure_time)
        return predictions[0] if len(predictions.shape) > 0 else predictions
    
    def get_last_timing(self) -> Dict[str, float]:
        """
        Get timing information from the last prediction
        
        Returns:
            Dictionary with 'preprocess_ms', 'inference_ms', and 'total_ms'
        """
        return {
            'preprocess_ms': self.last_preprocess_time,
            'inference_ms': self.last_inference_time,
            'total_ms': self.last_total_time
        }
    
    def benchmark(self, num_iterations=1000, batch_sizes=[1, 8, 16, 32, 64], warmup_iterations=10) -> Dict:
        """
        Benchmark prediction performance with different batch sizes
        
        Args:
            num_iterations: Number of iterations per batch size
            batch_sizes: List of batch sizes to test
            warmup_iterations: Number of warmup iterations before timing
        
        Returns:
            Dictionary with benchmark results for each batch size
        """
        results = {}
        
        print("\n" + "="*80)
        print("Performance Benchmark")
        print("="*80)
        print(f"Warmup iterations: {warmup_iterations}")
        print(f"Test iterations per batch size: {num_iterations}")
        print(f"Batch sizes to test: {batch_sizes}")
        print("="*80)
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            # Prepare test data
            test_features = np.random.randn(batch_size, 2, 25).astype(np.float64)
            
            # Warmup (to avoid initialization overhead)
            print("  Warming up...", end="", flush=True)
            for _ in range(warmup_iterations):
                _ = self.predict(test_features, measure_time=False)
            print(" Done")
            
            # Benchmark
            print("  Running benchmark...", end="", flush=True)
            preprocess_times = []
            inference_times = []
            total_times = []
            
            for _ in range(num_iterations):
                _ = self.predict(test_features, measure_time=True)
                timing = self.get_last_timing()
                preprocess_times.append(timing['preprocess_ms'])
                inference_times.append(timing['inference_ms'])
                total_times.append(timing['total_ms'])
            
            print(" Done")
            
            # Calculate statistics
            def calc_stats(times):
                return {
                    'mean': statistics.mean(times),
                    'std': statistics.stdev(times) if len(times) > 1 else 0.0,
                    'min': min(times),
                    'max': max(times),
                    'p50': statistics.median(times),
                    'p95': np.percentile(times, 95),
                    'p99': np.percentile(times, 99)
                }
            
            results[batch_size] = {
                'preprocess': calc_stats(preprocess_times),
                'inference': calc_stats(inference_times),
                'total': calc_stats(total_times),
                'throughput_samples_per_sec': (batch_size * 1000) / statistics.mean(total_times)
            }
            
            # Print summary
            total_mean = results[batch_size]['total']['mean']
            throughput = results[batch_size]['throughput_samples_per_sec']
            print(f"  Mean total time: {total_mean:.3f} ms")
            print(f"  Throughput: {throughput:.2f} samples/second")
        
        return results
    
    def print_benchmark_results(self, results: Dict):
        """
        Print benchmark results in a formatted table
        
        Args:
            results: Results dictionary from benchmark() method
        """
        print("\n" + "="*80)
        print("Benchmark Results Summary")
        print("="*80)
        
        # Header
        print(f"{'Batch Size':<12} {'Mean (ms)':<12} {'Std (ms)':<12} {'P95 (ms)':<12} {'Throughput (samples/s)':<20}")
        print("-" * 80)
        
        # Data rows
        for batch_size in sorted(results.keys()):
            total_stats = results[batch_size]['total']
            throughput = results[batch_size]['throughput_samples_per_sec']
            print(f"{batch_size:<12} {total_stats['mean']:<12.3f} {total_stats['std']:<12.3f} "
                  f"{total_stats['p95']:<12.3f} {throughput:<20.2f}")
        
        print("\nDetailed Statistics:")
        print("-" * 80)
        for batch_size in sorted(results.keys()):
            print(f"\nBatch Size: {batch_size}")
            print(f"  Preprocessing:")
            print(f"    Mean: {results[batch_size]['preprocess']['mean']:.3f} ms")
            print(f"    Std:  {results[batch_size]['preprocess']['std']:.3f} ms")
            print(f"  Inference:")
            print(f"    Mean: {results[batch_size]['inference']['mean']:.3f} ms")
            print(f"    Std:  {results[batch_size]['inference']['std']:.3f} ms")
            print(f"  Total:")
            print(f"    Mean: {results[batch_size]['total']['mean']:.3f} ms")
            print(f"    Std:  {results[batch_size]['total']['std']:.3f} ms")
            print(f"    Min:  {results[batch_size]['total']['min']:.3f} ms")
            print(f"    Max:  {results[batch_size]['total']['max']:.3f} ms")
            print(f"    P50:  {results[batch_size]['total']['p50']:.3f} ms")
            print(f"    P95:  {results[batch_size]['total']['p95']:.3f} ms")
            print(f"    P99:  {results[batch_size]['total']['p99']:.3f} ms")
            print(f"  Throughput: {results[batch_size]['throughput_samples_per_sec']:.2f} samples/second")


def example_usage():
    """Example usage of the predictor"""
    parser = argparse.ArgumentParser(description='Real-time VIGNet prediction')
    parser.add_argument('--model-dir', type=str, default='./logs_fp_no_cv/models',
                       help='Directory containing saved models')
    parser.add_argument('--trial', type=int, default=None,
                       help='Trial number (1-21). If not specified, uses first available.')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU index to use')
    parser.add_argument('--test', action='store_true',
                       help='Run test prediction with dummy data')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--benchmark-iterations', type=int, default=1000,
                       help='Number of iterations for benchmark (default: 1000)')
    parser.add_argument('--benchmark-batch-sizes', type=int, nargs='+', default=[1, 8, 16, 32, 64],
                       help='Batch sizes to test in benchmark (default: 1 8 16 32 64)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    print("="*80)
    print("VIGNet Real-time Predictor (FP1/FP2)")
    print("="*80)
    
    try:
        predictor = VIGNetPredictor(args.model_dir, trial=args.trial, gpu_idx=args.gpu)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure you have trained and saved a model first.")
        print("Run: python experiment_fp_no_cv.py --trial 1")
        return
    
    if args.benchmark:
        # Run performance benchmark
        results = predictor.benchmark(
            num_iterations=args.benchmark_iterations,
            batch_sizes=args.benchmark_batch_sizes
        )
        predictor.print_benchmark_results(results)
        
    elif args.test:
        # Test with dummy data
        print("\n" + "="*80)
        print("Testing with dummy data")
        print("="*80)
        
        # Create dummy features: (10 samples, 2 channels, 25 frequency bins)
        dummy_features = np.random.randn(10, 2, 25).astype(np.float64)
        print(f"Input features shape: {dummy_features.shape}")
        
        # Predict with timing
        print("\nBatch prediction (10 samples):")
        predictions = predictor.predict(dummy_features, measure_time=True)
        timing = predictor.get_last_timing()
        print(f"Predictions shape: {predictions.shape}")
        print(f"Predictions: {predictions}")
        print(f"\nTiming:")
        print(f"  Preprocessing: {timing['preprocess_ms']:.3f} ms")
        print(f"  Inference:      {timing['inference_ms']:.3f} ms")
        print(f"  Total:          {timing['total_ms']:.3f} ms")
        
        # Single sample prediction with timing
        print("\nSingle sample prediction:")
        single_feature = dummy_features[0]  # (2, 25)
        single_pred = predictor.predict_single(single_feature, measure_time=True)
        timing = predictor.get_last_timing()
        print(f"Single prediction: {single_pred}")
        print(f"\nTiming:")
        print(f"  Preprocessing: {timing['preprocess_ms']:.3f} ms")
        print(f"  Inference:      {timing['inference_ms']:.3f} ms")
        print(f"  Total:          {timing['total_ms']:.3f} ms")
        
        print("\n" + "="*80)
        print("Test completed successfully!")
        print("="*80)
    else:
        print("\nPredictor loaded successfully!")
        print("\nUsage example:")
        print("  # Load predictor")
        print("  predictor = VIGNetPredictor('./logs_fp_no_cv/models', trial=1)")
        print("  ")
        print("  # Predict from features (shape: N, 2, 25)")
        print("  features = np.array([...])  # Your EEG features")
        print("  predictions = predictor.predict(features)")
        print("  ")
        print("  # Single sample prediction")
        print("  single_feature = features[0]  # shape: (2, 25)")
        print("  pred = predictor.predict_single(single_feature)")
        print("  ")
        print("\nFor real-time prediction:")
        print("  1. Extract EEG features from FP1/FP2 channels (2 channels, 25 frequency bins)")
        print("  2. Call predictor.predict(features) or predictor.predict_single(feature)")
        print("  3. Output is PERCLOS score (0-1 range) for regression task")
        print("  ")
        print("Input format requirements:")
        print("  - Features shape: (N, 2, 25) for batch or (2, 25) for single sample")
        print("  - 2 channels: FP1 and FP2")
        print("  - 25 frequency bins: DE features at 2Hz resolution")
        print("  - Features should be extracted using the same method as training")
        print("  - (de_LDS features from Forehead_EEG/EEG_Feature_2Hz)")


if __name__ == "__main__":
    example_usage()

