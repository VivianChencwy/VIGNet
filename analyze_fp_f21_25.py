#!/usr/bin/env python3
"""
Analysis script for FP1/FP2 VIGNet experiments (f21-25 frequency features)
Extracts metrics from logs and predictions, generates statistics and visualizations
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def parse_summary_log(log_path):
    """
    Parse training_summary log file to extract test metrics for all trials
    
    Args:
        log_path: Path to training_summary log file
        
    Returns:
        dict: Dictionary with trial numbers as keys and metrics as values
    """
    metrics = {}
    
    if not os.path.exists(log_path):
        print(f"Warning: Summary log not found: {log_path}")
        return metrics
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Match lines like: "Trial 1 - Test MSE: 0.000992, Test RMSE: 0.031494, Test Correlation: 0.993984"
            match = re.search(r'Trial (\d+) - Test MSE: ([\d.]+), Test RMSE: ([\d.]+), Test Correlation: ([\d.]+)', line)
            if match:
                trial = int(match.group(1))
                metrics[trial] = {
                    'test_mse': float(match.group(2)),
                    'test_rmse': float(match.group(3)),
                    'test_correlation': float(match.group(4))
                }
    
    return metrics


def parse_trial_log(log_path):
    """
    Parse individual trial log file to extract validation and test metrics
    
    Args:
        log_path: Path to individual trial log file
        
    Returns:
        dict: Dictionary with validation and test metrics
    """
    metrics = {
        'valid_mse': None,
        'valid_rmse': None,
        'valid_correlation': None,
        'test_mse': None,
        'test_rmse': None,
        'test_correlation': None
    }
    
    if not os.path.exists(log_path):
        return metrics
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # Parse validation metrics
        valid_match = re.search(
            r'Validation Set - MSE: ([\d.]+), MAE: ([\d.]+), RMSE: ([\d.]+), Pearson Correlation: ([\d.]+)',
            content
        )
        if valid_match:
            metrics['valid_mse'] = float(valid_match.group(1))
            metrics['valid_rmse'] = float(valid_match.group(3))
            metrics['valid_correlation'] = float(valid_match.group(4))
        
        # Parse test metrics
        test_match = re.search(
            r'Test Set - MSE: ([\d.]+), MAE: ([\d.]+), RMSE: ([\d.]+), Pearson Correlation: ([\d.]+)',
            content
        )
        if test_match:
            metrics['test_mse'] = float(test_match.group(1))
            metrics['test_rmse'] = float(test_match.group(3))
            metrics['test_correlation'] = float(test_match.group(4))
    
    return metrics


def load_predictions(predictions_dir):
    """
    Load all prediction files from predictions directory
    
    Args:
        predictions_dir: Path to predictions directory
        
    Returns:
        dict: Dictionary with trial numbers as keys, containing validation and test predictions
    """
    predictions = {}
    
    if not os.path.exists(predictions_dir):
        print(f"Warning: Predictions directory not found: {predictions_dir}")
        return predictions
    
    # Find all prediction files
    test_files = glob.glob(os.path.join(predictions_dir, 'trial*_test.npy'))
    valid_files = glob.glob(os.path.join(predictions_dir, 'trial*_validation.npy'))
    
    # Load test predictions
    for file_path in test_files:
        match = re.search(r'trial(\d+)_test\.npy', file_path)
        if match:
            trial = int(match.group(1))
            data = np.load(file_path, allow_pickle=True).item()
            if trial not in predictions:
                predictions[trial] = {}
            predictions[trial]['test'] = {
                'y_true': data['y_true'],
                'y_pred': data['y_pred']
            }
    
    # Load validation predictions
    for file_path in valid_files:
        match = re.search(r'trial(\d+)_validation\.npy', file_path)
        if match:
            trial = int(match.group(1))
            if trial not in predictions:
                predictions[trial] = {}
            data = np.load(file_path, allow_pickle=True).item()
            predictions[trial]['valid'] = {
                'y_true': data['y_true'],
                'y_pred': data['y_pred']
            }
    
    return predictions


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        dict: Dictionary with MSE, MAE, RMSE, and Correlation
    """
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    correlation, p_value = stats.pearsonr(y_true, y_pred)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation,
        'p_value': p_value
    }


def create_summary_statistics(summary_metrics, trial_metrics, predictions):
    """
    Create comprehensive statistics DataFrame
    
    Args:
        summary_metrics: Metrics from summary log
        trial_metrics: Metrics from individual trial logs
        predictions: Loaded predictions
        
    Returns:
        pd.DataFrame: Statistics for all trials
    """
    data = []
    
    # Get all trial numbers
    all_trials = set(summary_metrics.keys())
    all_trials.update(trial_metrics.keys())
    all_trials.update(predictions.keys())
    all_trials = sorted(all_trials)
    
    for trial in all_trials:
        row = {'Trial': trial}
        
        # Get metrics from summary log (test only)
        if trial in summary_metrics:
            row['Test_MSE'] = summary_metrics[trial]['test_mse']
            row['Test_RMSE'] = summary_metrics[trial]['test_rmse']
            row['Test_Correlation'] = summary_metrics[trial]['test_correlation']
        
        # Get metrics from individual trial log (validation and test)
        if trial in trial_metrics:
            if trial_metrics[trial]['valid_mse'] is not None:
                row['Valid_MSE'] = trial_metrics[trial]['valid_mse']
                row['Valid_RMSE'] = trial_metrics[trial]['valid_rmse']
                row['Valid_Correlation'] = trial_metrics[trial]['valid_correlation']
            if trial_metrics[trial]['test_mse'] is not None:
                # Overwrite with more detailed metrics if available
                row['Test_MSE'] = trial_metrics[trial]['test_mse']
                row['Test_RMSE'] = trial_metrics[trial]['test_rmse']
                row['Test_Correlation'] = trial_metrics[trial]['test_correlation']
        
        # Calculate metrics from predictions if available
        if trial in predictions:
            if 'test' in predictions[trial]:
                test_metrics = calculate_metrics(
                    predictions[trial]['test']['y_true'],
                    predictions[trial]['test']['y_pred']
                )
                # Use prediction-based metrics if log metrics not available
                if 'Test_MSE' not in row:
                    row['Test_MSE'] = test_metrics['mse']
                    row['Test_RMSE'] = test_metrics['rmse']
                    row['Test_Correlation'] = test_metrics['correlation']
            
            if 'valid' in predictions[trial]:
                valid_metrics = calculate_metrics(
                    predictions[trial]['valid']['y_true'],
                    predictions[trial]['valid']['y_pred']
                )
                if 'Valid_MSE' not in row:
                    row['Valid_MSE'] = valid_metrics['mse']
                    row['Valid_RMSE'] = valid_metrics['rmse']
                    row['Valid_Correlation'] = valid_metrics['correlation']
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def plot_rmse_comparison(df, output_path):
    """Plot RMSE comparison across all trials"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    trials = df['Trial'].values
    rmse = df['Test_RMSE'].values
    
    bars = ax.bar(trials, rmse, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Add mean line
    mean_rmse = rmse.mean()
    std_rmse = rmse.std()
    ax.axhline(mean_rmse, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rmse:.4f}')
    ax.axhline(mean_rmse + std_rmse, color='orange', linestyle=':', linewidth=1, alpha=0.7, label=f'±1 std: {std_rmse:.4f}')
    ax.axhline(mean_rmse - std_rmse, color='orange', linestyle=':', linewidth=1, alpha=0.7)
    
    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Test RMSE', fontsize=12)
    ax.set_title('Test RMSE Comparison Across All Trials (f21-25 Features)', fontsize=14, fontweight='bold')
    ax.set_xticks(trials)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved RMSE comparison plot: {output_path}")


def plot_correlation_comparison(df, output_path):
    """Plot Correlation comparison across all trials"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    trials = df['Trial'].values
    correlation = df['Test_Correlation'].values
    
    bars = ax.bar(trials, correlation, alpha=0.7, color='forestgreen', edgecolor='black', linewidth=0.5)
    
    # Add mean line
    mean_corr = correlation.mean()
    std_corr = correlation.std()
    ax.axhline(mean_corr, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_corr:.4f}')
    ax.axhline(mean_corr + std_corr, color='orange', linestyle=':', linewidth=1, alpha=0.7, label=f'±1 std: {std_corr:.4f}')
    ax.axhline(mean_corr - std_corr, color='orange', linestyle=':', linewidth=1, alpha=0.7)
    
    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Test Correlation', fontsize=12)
    ax.set_title('Test Correlation Comparison Across All Trials (f21-25 Features)', fontsize=14, fontweight='bold')
    ax.set_xticks(trials)
    ax.set_ylim([0.5, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Correlation comparison plot: {output_path}")


def plot_metrics_comparison(df, output_path):
    """Plot RMSE and Correlation side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    trials = df['Trial'].values
    rmse = df['Test_RMSE'].values
    correlation = df['Test_Correlation'].values
    
    # RMSE plot
    ax1.bar(trials, rmse, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    mean_rmse = rmse.mean()
    ax1.axhline(mean_rmse, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rmse:.4f}')
    ax1.set_xlabel('Trial Number', fontsize=12)
    ax1.set_ylabel('Test RMSE', fontsize=12)
    ax1.set_title('Test RMSE (f21-25 Features)', fontsize=14, fontweight='bold')
    ax1.set_xticks(trials)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Correlation plot
    ax2.bar(trials, correlation, alpha=0.7, color='forestgreen', edgecolor='black', linewidth=0.5)
    mean_corr = correlation.mean()
    ax2.axhline(mean_corr, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_corr:.4f}')
    ax2.set_xlabel('Trial Number', fontsize=12)
    ax2.set_ylabel('Test Correlation', fontsize=12)
    ax2.set_title('Test Correlation (f21-25 Features)', fontsize=14, fontweight='bold')
    ax2.set_xticks(trials)
    ax2.set_ylim([0.5, 1.0])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics comparison plot: {output_path}")


def plot_metrics_boxplot(df, output_path):
    """Plot boxplot for RMSE and Correlation distributions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # RMSE boxplot
    ax1.boxplot(df['Test_RMSE'].values, vert=True, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax1.set_ylabel('Test RMSE', fontsize=12)
    ax1.set_title('Test RMSE Distribution (f21-25 Features)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Correlation boxplot
    ax2.boxplot(df['Test_Correlation'].values, vert=True, patch_artist=True,
                boxprops=dict(facecolor='forestgreen', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Test Correlation', fontsize=12)
    ax2.set_title('Test Correlation Distribution (f21-25 Features)', fontsize=14, fontweight='bold')
    ax2.set_ylim([0.5, 1.0])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics boxplot: {output_path}")


def plot_scatter_all_trials(predictions, output_path):
    """Plot scatter plot of y_true vs y_pred for all trials combined (test set)"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    all_y_true = []
    all_y_pred = []
    
    for trial in sorted(predictions.keys()):
        if 'test' in predictions[trial]:
            all_y_true.extend(predictions[trial]['test']['y_true'])
            all_y_pred.extend(predictions[trial]['test']['y_pred'])
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    # Create scatter plot
    ax.scatter(all_y_true, all_y_pred, alpha=0.5, s=20, edgecolors='black', linewidths=0.3)
    
    # Add diagonal line (perfect prediction)
    min_val = min(all_y_true.min(), all_y_pred.min())
    max_val = max(all_y_true.max(), all_y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate and display metrics
    correlation = stats.pearsonr(all_y_true, all_y_pred)[0]
    rmse = np.sqrt(np.mean((all_y_true - all_y_pred) ** 2))
    
    ax.text(0.05, 0.95, f'Correlation: {correlation:.4f}\nRMSE: {rmse:.4f}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('True PERCLOS', fontsize=12)
    ax.set_ylabel('Predicted PERCLOS', fontsize=12)
    ax.set_title('Predicted vs True PERCLOS (All Trials, Test Set, f21-25 Features)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot: {output_path}")


def plot_residual_distribution(predictions, output_path):
    """Plot residual distribution histogram and Q-Q plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    all_y_true = []
    all_y_pred = []
    
    for trial in sorted(predictions.keys()):
        if 'test' in predictions[trial]:
            all_y_true.extend(predictions[trial]['test']['y_true'])
            all_y_pred.extend(predictions[trial]['test']['y_pred'])
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    residuals = all_y_true - all_y_pred
    
    # Histogram
    ax1.hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax1.axvline(residuals.mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {residuals.mean():.4f}')
    ax1.set_xlabel('Residual (True - Predicted)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Residual Distribution (f21-25 Features)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normal Distribution, f21-25 Features)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved residual distribution plot: {output_path}")


def plot_time_series_sample(predictions, output_path, num_trials=2):
    """Plot time series comparison for sample trials"""
    fig, axes = plt.subplots(num_trials, 1, figsize=(14, 4 * num_trials))
    
    if num_trials == 1:
        axes = [axes]
    
    # Select representative trials (best and worst correlation)
    trial_list = sorted(predictions.keys())
    if len(trial_list) >= num_trials:
        selected_trials = [trial_list[0], trial_list[-1]]  # First and last
    else:
        selected_trials = trial_list[:num_trials]
    
    for idx, trial in enumerate(selected_trials[:num_trials]):
        if 'test' in predictions[trial]:
            y_true = predictions[trial]['test']['y_true']
            y_pred = predictions[trial]['test']['y_pred']
            
            time_steps = np.arange(len(y_true))
            axes[idx].plot(time_steps, y_true, 'b-', label='True PERCLOS', linewidth=2, alpha=0.7)
            axes[idx].plot(time_steps, y_pred, 'r--', label='Predicted PERCLOS', linewidth=2, alpha=0.7)
            
            correlation = stats.pearsonr(y_true, y_pred)[0]
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            
            axes[idx].set_xlabel('Time Step', fontsize=12)
            axes[idx].set_ylabel('PERCLOS', fontsize=12)
            axes[idx].set_title(f'Trial {trial} - Time Series Comparison (Corr: {correlation:.4f}, RMSE: {rmse:.4f}, f21-25)',
                               fontsize=12, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved time series plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze FP1/FP2 VIGNet experiment results (f21-25 features)')
    parser.add_argument('--log-dir', type=str, default='./logs_fp_f21_25_1116',
                       help='Directory containing log files and predictions')
    parser.add_argument('--output-dir', type=str, default='./analysis_results/fp_f21_25',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    log_dir = args.log_dir
    output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("FP1/FP2 VIGNet Analysis (f21-25 Frequency Features)")
    print("="*80)
    
    # Find training_summary log
    summary_logs = glob.glob(os.path.join(log_dir, 'training_summary_*.log'))
    if not summary_logs:
        print(f"Error: No training_summary log found in {log_dir}")
        return
    
    summary_log = sorted(summary_logs)[-1]  # Use most recent
    print(f"Using summary log: {summary_log}")
    
    # Parse summary log
    print("\n1. Parsing summary log...")
    summary_metrics = parse_summary_log(summary_log)
    print(f"   Found metrics for {len(summary_metrics)} trials")
    
    # Parse individual trial logs
    print("\n2. Parsing individual trial logs...")
    trial_logs = glob.glob(os.path.join(log_dir, 'trial*_RGS_*.log'))
    trial_metrics = {}
    for log_path in trial_logs:
        match = re.search(r'trial(\d+)_RGS_', log_path)
        if match:
            trial = int(match.group(1))
            trial_metrics[trial] = parse_trial_log(log_path)
    print(f"   Found detailed metrics for {len(trial_metrics)} trials")
    
    # Load predictions
    print("\n3. Loading predictions...")
    predictions_dir = os.path.join(log_dir, 'predictions')
    predictions = load_predictions(predictions_dir)
    print(f"   Loaded predictions for {len(predictions)} trials")
    
    # Create summary statistics
    print("\n4. Creating summary statistics...")
    df = create_summary_statistics(summary_metrics, trial_metrics, predictions)
    
    # Save statistics to CSV
    csv_path = os.path.join(output_dir, 'summary_statistics.csv')
    df.to_csv(csv_path, index=False)
    print(f"   Saved statistics to: {csv_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    if 'Test_RMSE' in df.columns:
        print(f"Test RMSE - Mean: {df['Test_RMSE'].mean():.6f}, Std: {df['Test_RMSE'].std():.6f}")
        print(f"           Min: {df['Test_RMSE'].min():.6f}, Max: {df['Test_RMSE'].max():.6f}")
    if 'Test_Correlation' in df.columns:
        print(f"Test Correlation - Mean: {df['Test_Correlation'].mean():.6f}, Std: {df['Test_Correlation'].std():.6f}")
        print(f"                 Min: {df['Test_Correlation'].min():.6f}, Max: {df['Test_Correlation'].max():.6f}")
    
    # Generate visualizations
    print("\n5. Generating visualizations...")
    
    if 'Test_RMSE' in df.columns:
        plot_rmse_comparison(df, os.path.join(output_dir, 'rmse_comparison.png'))
    
    if 'Test_Correlation' in df.columns:
        plot_correlation_comparison(df, os.path.join(output_dir, 'correlation_comparison.png'))
    
    if 'Test_RMSE' in df.columns and 'Test_Correlation' in df.columns:
        plot_metrics_comparison(df, os.path.join(output_dir, 'metrics_comparison.png'))
        plot_metrics_boxplot(df, os.path.join(output_dir, 'metrics_boxplot.png'))
    
    if predictions:
        plot_scatter_all_trials(predictions, os.path.join(output_dir, 'scatter_all_trials.png'))
        plot_residual_distribution(predictions, os.path.join(output_dir, 'residual_distribution.png'))
        plot_time_series_sample(predictions, os.path.join(output_dir, 'time_series_sample.png'), num_trials=2)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

