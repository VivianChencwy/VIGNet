#!/usr/bin/env python3
"""
Comprehensive Analysis Script for All VIGNet Trials
Analyzes all trials in logs_fp/, generating individual plots per trial plus cross-trial systematic analysis.
"""

import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import sys

# Import parsing and plotting functions from visualize_log
import visualize_log as viz


def find_log_files(log_dir, trials='all', latest=True):
    """
    Find log files for specified trials.
    
    Args:
        log_dir: Directory containing log files
        trials: 'all' or comma-separated trial numbers
        latest: If True, use only the most recent log file for each trial
    
    Returns:
        dict: {trial_number: log_file_path}
    """
    log_dir = Path(log_dir)
    log_files = {}
    
    # Determine which trials to process
    if trials == 'all':
        trial_numbers = list(range(1, 22))  # 1-21
    else:
        trial_numbers = [int(t.strip()) for t in trials.split(',')]
    
    # Find log files for each trial
    for trial_num in trial_numbers:
        pattern = f"trial{trial_num}_RGS_*.log"
        matching_files = sorted(log_dir.glob(pattern))
        
        if matching_files:
            if latest:
                # Use the most recent file (sorted by name, which includes timestamp)
                log_files[trial_num] = matching_files[-1]
            else:
                # Use the first one found
                log_files[trial_num] = matching_files[0]
    
    return log_files


def collect_all_trial_data(log_files, log_stream):
    """
    Parse all log files and aggregate metrics.
    
    Args:
        log_files: dict of {trial_number: log_file_path}
        log_stream: File stream for logging
    
    Returns:
        dict: {trial_number: parsed_data}
    """
    all_data = {}
    
    for trial_num, log_path in sorted(log_files.items()):
        log_stream.write(f"Parsing trial {trial_num}: {log_path.name}\n")
        try:
            data = viz.parse_log_file(log_path)
            if data['trial_info'] and data['eval_metrics']:
                all_data[trial_num] = data
                log_stream.write(f"  ✓ Successfully parsed {len(data['eval_metrics'])} CV folds\n")
            else:
                log_stream.write(f"  ⚠ Warning: No data found in log file\n")
        except Exception as e:
            log_stream.write(f"  ✗ Error parsing: {str(e)}\n")
    
    log_stream.write(f"\nSuccessfully parsed {len(all_data)} trials\n")
    return all_data


def generate_individual_plots(all_data, log_files, output_dir, log_stream):
    """
    Generate individual plots for each trial.
    
    Args:
        all_data: dict of {trial_number: parsed_data}
        log_files: dict of {trial_number: log_file_path}
        output_dir: Path to output directory
        log_stream: File stream for logging
    """
    log_stream.write("\n" + "="*80 + "\n")
    log_stream.write("GENERATING INDIVIDUAL TRIAL PLOTS\n")
    log_stream.write("="*80 + "\n\n")
    
    for trial_num, data in sorted(all_data.items()):
        log_stream.write(f"Trial {trial_num}:\n")
        
        # 1. Training curves
        if data['training_losses']:
            output_path = output_dir / f"trial{trial_num}_training_curves.png"
            try:
                viz.plot_training_curves(data['training_losses'], output_path, trial_num)
                log_stream.write(f"  ✓ Training curves\n")
            except Exception as e:
                log_stream.write(f"  ✗ Training curves: {str(e)}\n")
        
        # 2. Regression metrics
        if data['eval_metrics']:
            output_path = output_dir / f"trial{trial_num}_regression_metrics.png"
            try:
                viz.plot_regression_metrics(data['eval_metrics'], data['cv_summary'], 
                                          output_path, trial_num)
                log_stream.write(f"  ✓ Regression metrics\n")
            except Exception as e:
                log_stream.write(f"  ✗ Regression metrics: {str(e)}\n")
        
        # 3. CV summary
        if data['cv_summary']:
            output_path = output_dir / f"trial{trial_num}_cv_summary.png"
            try:
                viz.plot_cv_summary(data['cv_summary'], output_path, trial_num, 'RGS')
                log_stream.write(f"  ✓ CV summary\n")
            except Exception as e:
                log_stream.write(f"  ✗ CV summary: {str(e)}\n")
        
        # 4. Regression visualization
        log_dir = log_files[trial_num].parent
        predictions_dir = log_dir / "predictions"
        if predictions_dir.exists():
            output_path = output_dir / f"trial{trial_num}_regression_visualization.png"
            try:
                viz.plot_regression_visualization(predictions_dir, output_path, trial_num)
                log_stream.write(f"  ✓ Regression visualization\n")
            except Exception as e:
                log_stream.write(f"  ✗ Regression visualization: {str(e)}\n")


def extract_summary_metrics(all_data):
    """
    Extract summary metrics for each trial.
    
    Returns:
        pd.DataFrame with columns: trial, test_mse, test_mae, test_rmse, test_corr, etc.
    """
    records = []
    
    for trial_num, data in sorted(all_data.items()):
        record = {'trial': trial_num}
        
        # Extract CV summary metrics if available
        cv_summary = data.get('cv_summary', {})
        
        if 'test_rmse' in cv_summary:
            record['test_rmse_mean'] = cv_summary['test_rmse'][0]
            record['test_rmse_std'] = cv_summary['test_rmse'][1]
        
        if 'test_corr' in cv_summary:
            record['test_corr_mean'] = cv_summary['test_corr'][0]
            record['test_corr_std'] = cv_summary['test_corr'][1]
        
        if 'test_mse' in cv_summary:
            record['test_mse_mean'] = cv_summary['test_mse'][0]
            record['test_mse_std'] = cv_summary['test_mse'][1]
        
        if 'test_mae' in cv_summary:
            record['test_mae_mean'] = cv_summary['test_mae'][0]
            record['test_mae_std'] = cv_summary['test_mae'][1]
        
        if 'valid_rmse' in cv_summary:
            record['valid_rmse_mean'] = cv_summary['valid_rmse'][0]
            record['valid_rmse_std'] = cv_summary['valid_rmse'][1]
        
        if 'valid_corr' in cv_summary:
            record['valid_corr_mean'] = cv_summary['valid_corr'][0]
            record['valid_corr_std'] = cv_summary['valid_corr'][1]
        
        # Find best and worst folds by test RMSE
        eval_metrics = data.get('eval_metrics', {})
        if eval_metrics:
            test_rmses = [(fold, metrics['test']['rmse']) 
                         for fold, metrics in eval_metrics.items() 
                         if 'test' in metrics and 'rmse' in metrics['test']]
            if test_rmses:
                best_fold = min(test_rmses, key=lambda x: x[1])
                worst_fold = max(test_rmses, key=lambda x: x[1])
                record['best_fold'] = best_fold[0]
                record['best_fold_rmse'] = best_fold[1]
                record['worst_fold'] = worst_fold[0]
                record['worst_fold_rmse'] = worst_fold[1]
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Add rankings
    if 'test_rmse_mean' in df.columns:
        df['rank_by_rmse'] = df['test_rmse_mean'].rank()
    if 'test_corr_mean' in df.columns:
        df['rank_by_corr'] = df['test_corr_mean'].rank(ascending=False)
    
    return df


def plot_cross_trial_comparison(all_data, output_dir, log_stream):
    """Generate cross-trial comparison plots."""
    log_stream.write("\n" + "="*80 + "\n")
    log_stream.write("GENERATING CROSS-TRIAL COMPARISON PLOTS\n")
    log_stream.write("="*80 + "\n\n")
    
    # Extract metrics
    df = extract_summary_metrics(all_data)
    
    # 1. RMSE comparison
    if 'test_rmse_mean' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        x = df['trial'].values
        y = df['test_rmse_mean'].values
        yerr = df['test_rmse_std'].values if 'test_rmse_std' in df.columns else None
        
        ax.bar(x, y, yerr=yerr, alpha=0.7, capsize=5, error_kw={'elinewidth': 2})
        ax.set_xlabel('Trial Number', fontsize=12)
        ax.set_ylabel('Test RMSE', fontsize=12)
        ax.set_title('RMSE Comparison Across All Trials', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = output_dir / "summary_rmse_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        log_stream.write("✓ RMSE comparison\n")
    
    # 2. Correlation comparison
    if 'test_corr_mean' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        x = df['trial'].values
        y = df['test_corr_mean'].values
        yerr = df['test_corr_std'].values if 'test_corr_std' in df.columns else None
        
        # Color bars based on correlation (positive = green, negative = red)
        colors = ['green' if c > 0 else 'red' for c in y]
        
        ax.bar(x, y, yerr=yerr, alpha=0.7, capsize=5, color=colors, 
               error_kw={'elinewidth': 2})
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Trial Number', fontsize=12)
        ax.set_ylabel('Pearson Correlation', fontsize=12)
        ax.set_title('Correlation Comparison Across All Trials', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = output_dir / "summary_correlation_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        log_stream.write("✓ Correlation comparison\n")
    
    # 3. All metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('All Metrics Comparison Across Trials', fontsize=16, fontweight='bold')
    
    metrics = [
        ('test_mse_mean', 'test_mse_std', 'MSE', axes[0, 0]),
        ('test_mae_mean', 'test_mae_std', 'MAE', axes[0, 1]),
        ('test_rmse_mean', 'test_rmse_std', 'RMSE', axes[1, 0]),
        ('test_corr_mean', 'test_corr_std', 'Correlation', axes[1, 1])
    ]
    
    for mean_col, std_col, title, ax in metrics:
        if mean_col in df.columns:
            x = df['trial'].values
            y = df[mean_col].values
            yerr = df[std_col].values if std_col in df.columns else None
            
            ax.bar(x, y, yerr=yerr, alpha=0.7, capsize=3, error_kw={'elinewidth': 1.5})
            ax.set_xlabel('Trial Number', fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(f'Test {title}', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "summary_all_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_stream.write("✓ All metrics comparison\n")
    
    # 4. Trial ranking
    if 'test_rmse_mean' in df.columns and 'test_corr_mean' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Trial Performance Rankings', fontsize=16, fontweight='bold')
        
        # Rank by RMSE (lower is better)
        df_sorted = df.sort_values('test_rmse_mean')
        ax = axes[0]
        ax.barh(range(len(df_sorted)), df_sorted['test_rmse_mean'].values, alpha=0.7)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels([f"Trial {t}" for t in df_sorted['trial'].values])
        ax.set_xlabel('Test RMSE', fontsize=12)
        ax.set_title('Ranked by RMSE (Best to Worst)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        # Rank by correlation (higher is better)
        df_sorted = df.sort_values('test_corr_mean', ascending=False)
        ax = axes[1]
        colors = ['green' if c > 0 else 'red' for c in df_sorted['test_corr_mean'].values]
        ax.barh(range(len(df_sorted)), df_sorted['test_corr_mean'].values, 
                alpha=0.7, color=colors)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels([f"Trial {t}" for t in df_sorted['trial'].values])
        ax.set_xlabel('Pearson Correlation', fontsize=12)
        ax.set_title('Ranked by Correlation (Best to Worst)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.invert_yaxis()
        
        plt.tight_layout()
        output_path = output_dir / "summary_trial_ranking.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        log_stream.write("✓ Trial ranking\n")


def plot_distribution_analysis(all_data, output_dir, log_stream):
    """Generate distribution analysis plots."""
    log_stream.write("\n" + "="*80 + "\n")
    log_stream.write("GENERATING DISTRIBUTION ANALYSIS PLOTS\n")
    log_stream.write("="*80 + "\n\n")
    
    df = extract_summary_metrics(all_data)
    
    # 1. RMSE distribution
    if 'test_rmse_mean' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('RMSE Distribution Across Trials', fontsize=14, fontweight='bold')
        
        # Histogram
        ax = axes[0]
        ax.hist(df['test_rmse_mean'].values, bins=15, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Test RMSE', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Histogram', fontsize=12)
        ax.axvline(df['test_rmse_mean'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {df["test_rmse_mean"].mean():.4f}')
        ax.axvline(df['test_rmse_mean'].median(), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {df["test_rmse_mean"].median():.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Box plot
        ax = axes[1]
        ax.boxplot([df['test_rmse_mean'].values], tick_labels=['Test RMSE'])
        ax.set_ylabel('RMSE', fontsize=11)
        ax.set_title('Box Plot', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = output_dir / "summary_rmse_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        log_stream.write("✓ RMSE distribution\n")
    
    # 2. Correlation distribution
    if 'test_corr_mean' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Correlation Distribution Across Trials', fontsize=14, fontweight='bold')
        
        # Histogram
        ax = axes[0]
        ax.hist(df['test_corr_mean'].values, bins=15, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Pearson Correlation', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Histogram', fontsize=12)
        ax.axvline(df['test_corr_mean'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {df["test_corr_mean"].mean():.4f}')
        ax.axvline(df['test_corr_mean'].median(), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {df["test_corr_mean"].median():.4f}')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Box plot
        ax = axes[1]
        ax.boxplot([df['test_corr_mean'].values], tick_labels=['Correlation'])
        ax.set_ylabel('Pearson Correlation', fontsize=11)
        ax.set_title('Box Plot', fontsize=12)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = output_dir / "summary_correlation_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        log_stream.write("✓ Correlation distribution\n")
    
    # 3. All metrics box plots
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics_to_plot = []
    labels = []
    
    for col, label in [('test_mse_mean', 'MSE'), ('test_mae_mean', 'MAE'), 
                       ('test_rmse_mean', 'RMSE'), ('test_corr_mean', 'Correlation')]:
        if col in df.columns:
            metrics_to_plot.append(df[col].values)
            labels.append(label)
    
    if metrics_to_plot:
        ax.boxplot(metrics_to_plot, tick_labels=labels)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Distribution of All Metrics Across Trials', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = output_dir / "summary_metrics_boxplots.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        log_stream.write("✓ Metrics box plots\n")


def generate_summary_table(all_data, output_dir, log_stream):
    """Generate summary statistics CSV table."""
    log_stream.write("\n" + "="*80 + "\n")
    log_stream.write("GENERATING SUMMARY STATISTICS TABLE\n")
    log_stream.write("="*80 + "\n\n")
    
    df = extract_summary_metrics(all_data)
    
    # Reorder columns for better readability
    column_order = ['trial', 'test_rmse_mean', 'test_rmse_std', 'test_corr_mean', 'test_corr_std',
                   'test_mse_mean', 'test_mse_std', 'test_mae_mean', 'test_mae_std',
                   'valid_rmse_mean', 'valid_rmse_std', 'valid_corr_mean', 'valid_corr_std',
                   'best_fold', 'best_fold_rmse', 'worst_fold', 'worst_fold_rmse',
                   'rank_by_rmse', 'rank_by_corr']
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    # Save to CSV
    output_path = output_dir / "summary_statistics.csv"
    df.to_csv(output_path, index=False, float_format='%.6f')
    log_stream.write(f"✓ Summary table saved to {output_path.name}\n")
    log_stream.write(f"  Trials analyzed: {len(df)}\n")
    
    # Print summary statistics
    log_stream.write("\n" + "-"*80 + "\n")
    log_stream.write("OVERALL STATISTICS\n")
    log_stream.write("-"*80 + "\n")
    
    if 'test_rmse_mean' in df.columns:
        log_stream.write(f"Test RMSE: {df['test_rmse_mean'].mean():.6f} ± {df['test_rmse_mean'].std():.6f}\n")
        log_stream.write(f"  Min: {df['test_rmse_mean'].min():.6f} (Trial {df.loc[df['test_rmse_mean'].idxmin(), 'trial']:.0f})\n")
        log_stream.write(f"  Max: {df['test_rmse_mean'].max():.6f} (Trial {df.loc[df['test_rmse_mean'].idxmax(), 'trial']:.0f})\n")
    
    if 'test_corr_mean' in df.columns:
        log_stream.write(f"\nTest Correlation: {df['test_corr_mean'].mean():.6f} ± {df['test_corr_mean'].std():.6f}\n")
        log_stream.write(f"  Min: {df['test_corr_mean'].min():.6f} (Trial {df.loc[df['test_corr_mean'].idxmin(), 'trial']:.0f})\n")
        log_stream.write(f"  Max: {df['test_corr_mean'].max():.6f} (Trial {df.loc[df['test_corr_mean'].idxmax(), 'trial']:.0f})\n")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive analysis of all VIGNet trial logs'
    )
    parser.add_argument('--trials', type=str, default='all',
                       help='Comma-separated trial numbers (e.g., "1,2,3") or "all" for all trials (default: all)')
    parser.add_argument('--log-dir', type=str, default='./logs_fp',
                       help='Directory containing log files (default: ./logs_fp)')
    parser.add_argument('--output-dir', type=str, default='./trial_analysis_results',
                       help='Output directory for all plots and tables (default: ./trial_analysis_results)')
    parser.add_argument('--latest', action='store_true',
                       help='Use only the most recent log file for each trial')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_path = output_dir / "analysis_log.txt"
    with open(log_path, 'w', encoding='utf-8') as log_stream:
        # Write header
        log_stream.write("="*80 + "\n")
        log_stream.write("COMPREHENSIVE TRIAL ANALYSIS\n")
        log_stream.write("="*80 + "\n")
        log_stream.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_stream.write(f"Log directory: {args.log_dir}\n")
        log_stream.write(f"Output directory: {output_dir}\n")
        log_stream.write(f"Trials: {args.trials}\n")
        log_stream.write(f"Latest logs only: {args.latest}\n")
        log_stream.write("="*80 + "\n\n")
        
        # Find log files
        log_stream.write("FINDING LOG FILES\n")
        log_stream.write("-"*80 + "\n")
        log_files = find_log_files(args.log_dir, args.trials, args.latest)
        log_stream.write(f"Found {len(log_files)} trial log files\n")
        for trial_num, trial_log_path in sorted(log_files.items()):
            log_stream.write(f"  Trial {trial_num}: {trial_log_path.name}\n")
        
        if not log_files:
            log_stream.write("\n✗ No log files found!\n")
            print("Error: No log files found. Check --log-dir and --trials arguments.")
            return
        
        log_stream.write("\n")
        
        # Collect all data
        all_data = collect_all_trial_data(log_files, log_stream)
        
        if not all_data:
            log_stream.write("\n✗ No valid data found!\n")
            print("Error: Could not parse any log files.")
            return
        
        # Generate individual plots
        generate_individual_plots(all_data, log_files, output_dir, log_stream)
        
        # Generate cross-trial comparison plots
        plot_cross_trial_comparison(all_data, output_dir, log_stream)
        
        # Generate distribution analysis plots
        plot_distribution_analysis(all_data, output_dir, log_stream)
        
        # Generate summary table
        df = generate_summary_table(all_data, output_dir, log_stream)
        
        # Final summary
        log_stream.write("\n" + "="*80 + "\n")
        log_stream.write("ANALYSIS COMPLETE\n")
        log_stream.write("="*80 + "\n")
        log_stream.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_stream.write(f"Total trials analyzed: {len(all_data)}\n")
        log_stream.write(f"Output directory: {output_dir.absolute()}\n")
        log_stream.write("="*80 + "\n")
    
    # Print to console
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Trials analyzed: {len(all_data)}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Analysis log: {log_path.absolute()}")
    print("="*80 + "\n")
    
    # Print summary statistics
    if 'test_rmse_mean' in df.columns:
        print(f"Overall Test RMSE: {df['test_rmse_mean'].mean():.6f} ± {df['test_rmse_mean'].std():.6f}")
    if 'test_corr_mean' in df.columns:
        print(f"Overall Test Correlation: {df['test_corr_mean'].mean():.6f} ± {df['test_corr_mean'].std():.6f}")
    print()


if __name__ == '__main__':
    main()

