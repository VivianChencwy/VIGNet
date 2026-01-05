# Continuous Fatigue Prediction Pipeline

A complete pipeline for processing EEG data and training a VIGNet model to predict continuous fatigue levels using a composite label based on PERCLOS, KSS, stress, frustration, and accuracy metrics.

## Overview

This pipeline processes EEG data from forehead channels (FP1/FP2) and predicts fatigue levels using a composite label that combines multiple physiological and behavioral indicators. The pipeline follows a preprocessing approach similar to SEED-VIG for consistent comparison.

## Features

- **EEG Feature Extraction**: Differential Entropy (DE) features across 25 frequency bands
- **Composite Fatigue Label**: Weighted combination of PERCLOS, KSS, stress, frustration, and respond accuracy (weights: 3:3:1:1:1)
- **VIGNet Model**: Deep learning model with multi-head residual spectro-spatio attention
- **Comprehensive Evaluation**: Metrics, plots, and visualization of prediction results

## Requirements

- Python 3.9+
- TensorFlow 2.20.0
- Keras 3.10.0
- NumPy, Pandas, SciPy
- Matplotlib
- scikit-learn
- pykalman

## Data Format

The pipeline expects a CSV file named `merged_data.csv` in the data directory with the following columns:

- **EEG Channels**: `fp1_clean`, `fp2_clean`
- **Timestamps**: `lsl_timestamp`, `unix_timestamp`
- **Eye State**: `eye_state` (values: 'open', 'close')
- **Labels**: `kss_rating`, `stress`, `frustration`, `is_target_double_jump`, `responded` (used to compute accuracy as correctness of response vs target)

## Usage

### Training Mode (Full Pipeline)

Run the complete pipeline including data preprocessing, model training, and evaluation:

```bash
cd /root/vivian/eeg/Continuous_fatigue_prediction
python pipeline.py --data_dir Weiyu_01
```

### Arguments

- `--data_dir`: Data directory name (default: `Weiyu_01`)
  - The pipeline looks for data in `../data/{data_dir}/merged_data.csv`
- `--mode`: Pipeline mode (default: `train`)
  - `train`: Full pipeline (preprocessing + training)
  - `evaluate`: Load saved model and evaluate

## Pipeline Steps

### Step 1: Data Loading and Trimming
- Loads `merged_data.csv` from the data directory
- Removes rows with NaN timestamps
- Estimates sampling rate (robust to duplicates/outliers)
- Trims first 300 seconds (5 minutes) by default

### Step 2: Resampling
- Resamples EEG data from original sampling rate (~500-1000 Hz) to 200 Hz
- Uses scipy's `resample` function for each channel

### Step 3: Segmentation
- Segments resampled data into overlapping windows
- Window size: 8 seconds
- Stride: 4 seconds (50% overlap)

### Step 4: DE Feature Extraction
- Extracts Differential Entropy (DE) features for 25 frequency bands
- Frequency bands: 1-3 Hz, 3-5 Hz, ..., 49-51 Hz (2 Hz resolution)
- Uses 4th-order Butterworth bandpass filters

### Step 5: Smoothing
- Applies moving average smoothing (window=5)
- Optionally uses Kalman filter (LDS) if `pykalman` is available

### Step 6: PERCLOS Calculation
- Computes PERCLOS (Percentage of Eyelid Closure) for each window
- Uses a 60-second sliding window
- Binary classification: closed/close = 1, open = 0

### Step 6b: Composite Label Calculation
- Computes window-level means for KSS, stress, frustration, and accuracy
- Normalizes each component using z-score (based on training set only)
- Combines components with weights: 3:3:1:1:1 (perclos:KSS:stress:frustration:accuracy)
- Final composite label represents fatigue level

### Step 7: Save Preprocessed Data
- Saves DE features, PERCLOS labels, composite labels, and timestamps
- Output directory: `{data_dir}/processed/`

### Step 8: Data Splitting
- Block-wise stratified split to prevent data leakage
- Creates blocks of 4 consecutive windows (configurable)
- Stratifies blocks by composite label distribution
- Default split: 60% train, 20% validation, 20% test

### Step 9-10: Model Training
- VIGNet-FP model architecture
- Training parameters:
  - Learning rate: 0.01
  - Max epochs: 1000
  - Batch size: 64
  - Early stopping patience: 50
- Saves best model weights and scaler

### Step 11: Plot Prediction Results
- Generates comprehensive visualization plots:
  - Scatter plots with regression lines
  - Time series comparisons
  - Residual plots
- Output: `{data_dir}/logs/fatigue_prediction_results.png`

## Output Files

### Preprocessed Data (`{data_dir}/processed/`)
- `de_features.npy`: DE features (shape: n_windows, 2, 25)
- `perclos_labels.npy`: PERCLOS labels
- `composite_labels.npy`: Composite fatigue labels
- `timestamps.npy`: Window timestamps
- `train_val_test_split.npy`: Data split indices
- `metadata.npy`: Processing metadata

### Model and Results (`{data_dir}/logs/models/`)
- `best_model/` or `best_weights.weights.h5`: Trained model weights
- `scaler.pkl`: Feature scaler (StandardScaler)
- `predictions.npy`: Training predictions and metrics
- `predictions_eval.npy`: Evaluation predictions (if using evaluate mode)

### Logs and Plots (`{data_dir}/logs/`)
- `pipeline_{timestamp}.log`: Detailed execution log
- `fatigue_prediction_results.png`: Training prediction plots
- `fatigue_prediction_results_eval.png`: Evaluation prediction plots


## Model Architecture

VIGNet-FP (Vigilance Network for Forehead Channels):
- Input: (N, 2, 25, 1) - N windows, 2 channels, 25 frequency bands
- Multi-head Residual Spectro-Spatio Attention (MHRSSA)
- Three convolutional blocks with attention
- Spatial fusion layer
- Dense output layer (regression)

## Evaluation Metrics

The pipeline reports:
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **Correlation**: Pearson correlation coefficient