#!/usr/bin/env python3
"""
Visualization script for VIGNet training logs.
Parses log files and generates training curves and performance metrics plots.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def parse_log_file(log_path):
    """Parse log file and extract training information."""
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Extract trial and task information
    trial_info = {}
    for line in lines:
        if 'TRIAL' in line and 'Task:' in line:
            match = re.search(r'TRIAL (\d+) - Task: (\w+)', line)
            if match:
                trial_info['trial'] = int(match.group(1))
                trial_info['task'] = match.group(2)
                break
    
    # Extract training losses for each CV fold (or single run if no CV)
    training_losses = defaultdict(list)
    current_fold = None
    has_cv = False
    
    # Extract evaluation metrics for each CV fold
    eval_metrics = defaultdict(dict)
    
    for i, line in enumerate(lines):
        # Detect CV fold start
        if 'CV Fold' in line:
            has_cv = True
            match = re.search(r'CV Fold (\d+)', line)
            if match:
                current_fold = int(match.group(1))
                # Initialize dict for this fold
                if current_fold not in eval_metrics:
                    eval_metrics[current_fold] = {}
        
        # If no CV detected yet, check if we should use fold 0 for single run
        if not has_cv and current_fold is None:
            # Check if this is a single run (no CV) by looking for START TRAINING
            if 'START TRAINING' in line:
                current_fold = 0
                if current_fold not in eval_metrics:
                    eval_metrics[current_fold] = {}
        
        # Extract training loss
        if current_fold is not None and 'Epoch:' in line and 'Training Loss:' in line:
            match = re.search(r'Epoch: (\d+), Training Loss: ([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                training_losses[current_fold].append((epoch, loss))
        
        # Extract evaluation metrics
        if 'EVALUATION RESULTS' in line and current_fold is not None:
            # Look ahead for metrics (within next 10 lines to be safe)
            found_valid = False
            found_test = False
            for j in range(i+1, min(i+11, len(lines))):
                eval_line = lines[j]
                # Stop if we hit the closing separator line (after both metrics should be found)
                if '='*60 in eval_line and j > i+3 and (found_valid and found_test):
                    break
                    
                if 'Validation Set' in eval_line and '-' in eval_line and not found_valid:
                    if trial_info.get('task') == 'RGS':
                        # Regression metrics - match line with MSE, MAE, RMSE, Correlation
                        match = re.search(
                            r'Validation Set - MSE: ([\d.]+), MAE: ([\d.]+), RMSE: ([\d.]+), Pearson Correlation: ([\d.]+)',
                            eval_line
                        )
                        if match:
                            eval_metrics[current_fold]['valid'] = {
                                'mse': float(match.group(1)),
                                'mae': float(match.group(2)),
                                'rmse': float(match.group(3)),
                                'correlation': float(match.group(4))
                            }
                            found_valid = True
                    else:
                        # Classification metrics
                        match = re.search(
                            r'Validation Set - Accuracy: ([\d.]+), Precision: ([\d.]+), Recall: ([\d.]+), F1-Score: ([\d.]+)',
                            eval_line
                        )
                        if match:
                            eval_metrics[current_fold]['valid'] = {
                                'accuracy': float(match.group(1)),
                                'precision': float(match.group(2)),
                                'recall': float(match.group(3)),
                                'f1': float(match.group(4))
                            }
                            found_valid = True
                
                if 'Test Set' in eval_line and '-' in eval_line and not found_test:
                    if trial_info.get('task') == 'RGS':
                        # Regression metrics - match line with MSE, MAE, RMSE, Correlation
                        match = re.search(
                            r'Test Set - MSE: ([\d.]+), MAE: ([\d.]+), RMSE: ([\d.]+), Pearson Correlation: ([\d.]+)',
                            eval_line
                        )
                        if match:
                            eval_metrics[current_fold]['test'] = {
                                'mse': float(match.group(1)),
                                'mae': float(match.group(2)),
                                'rmse': float(match.group(3)),
                                'correlation': float(match.group(4))
                            }
                            found_test = True
                    else:
                        # Classification metrics
                        match = re.search(
                            r'Test Set - Accuracy: ([\d.]+), Precision: ([\d.]+), Recall: ([\d.]+), F1-Score: ([\d.]+)',
                            eval_line
                        )
                        if match:
                            eval_metrics[current_fold]['test'] = {
                                'accuracy': float(match.group(1)),
                                'precision': float(match.group(2)),
                                'recall': float(match.group(3)),
                                'f1': float(match.group(4))
                            }
                            found_test = True
    
    # Extract CV summary
    cv_summary = {}
    in_summary = False
    current_set = None
    
    for line in lines:
        if 'CROSS-VALIDATION SUMMARY' in line:
            in_summary = True
            continue
        
        if in_summary and 'Validation Set' in line and 'Average' in line:
            current_set = 'valid'
            continue
        
        if in_summary and 'Test Set' in line and 'Average' in line:
            current_set = 'test'
            continue
        
        if in_summary and trial_info.get('task') == 'RGS' and current_set:
            if 'MSE:' in line and '±' in line:
                match = re.search(r'MSE: ([\d.]+) ± ([\d.]+)', line)
                if match:
                    cv_summary[f'{current_set}_mse'] = (float(match.group(1)), float(match.group(2)))
            
            if 'MAE:' in line and '±' in line:
                match = re.search(r'MAE: ([\d.]+) ± ([\d.]+)', line)
                if match:
                    cv_summary[f'{current_set}_mae'] = (float(match.group(1)), float(match.group(2)))
            
            if 'RMSE:' in line and '±' in line:
                match = re.search(r'RMSE: ([\d.]+) ± ([\d.]+)', line)
                if match:
                    cv_summary[f'{current_set}_rmse'] = (float(match.group(1)), float(match.group(2)))
            
            if 'Pearson Correlation:' in line and '±' in line:
                match = re.search(r'Pearson Correlation: ([\d.]+) ± ([\d.]+)', line)
                if match:
                    cv_summary[f'{current_set}_corr'] = (float(match.group(1)), float(match.group(2)))
        
        if in_summary and trial_info.get('task') != 'RGS' and current_set:
            if 'Accuracy:' in line and '±' in line:
                match = re.search(r'Accuracy: ([\d.]+) ± ([\d.]+)', line)
                if match:
                    cv_summary[f'{current_set}_acc'] = (float(match.group(1)), float(match.group(2)))
            
            if 'Precision:' in line and '±' in line:
                match = re.search(r'Precision: ([\d.]+) ± ([\d.]+)', line)
                if match:
                    cv_summary[f'{current_set}_prec'] = (float(match.group(1)), float(match.group(2)))
            
            if 'Recall:' in line and '±' in line:
                match = re.search(r'Recall: ([\d.]+) ± ([\d.]+)', line)
                if match:
                    cv_summary[f'{current_set}_rec'] = (float(match.group(1)), float(match.group(2)))
            
            if 'F1-Score:' in line and '±' in line:
                match = re.search(r'F1-Score: ([\d.]+) ± ([\d.]+)', line)
                if match:
                    cv_summary[f'{current_set}_f1'] = (float(match.group(1)), float(match.group(2)))
    
    return {
        'trial_info': trial_info,
        'training_losses': dict(training_losses),
        'eval_metrics': dict(eval_metrics),
        'cv_summary': cv_summary
    }


def plot_training_curves(training_losses, output_path, trial_num):
    """Plot training loss curves for all CV folds (or single run if no CV)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Determine if this is CV format (multiple folds) or single run
    is_cv = len(training_losses) > 1 or (len(training_losses) == 1 and list(training_losses.keys())[0] != 0)
    
    for fold, losses in sorted(training_losses.items()):
        epochs = [e for e, _ in losses]
        losses_values = [l for _, l in losses]
        label = f'CV Fold {fold}' if is_cv else 'Training Loss'
        ax.plot(epochs, losses_values, label=label, alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title(f'Trial {trial_num} - Training Loss Curves', fontsize=14, fontweight='bold')
    if len(training_losses) > 1 or is_cv:
        ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {output_path}")


def plot_regression_metrics(eval_metrics, cv_summary, output_path, trial_num):
    """Plot regression metrics comparison across CV folds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Trial {trial_num} - Regression Performance Metrics', fontsize=16, fontweight='bold')
    
    # Prepare data - only include folds that have both valid and test metrics
    folds = sorted(eval_metrics.keys())
    # Filter folds that have both valid and test metrics
    complete_folds = [f for f in folds if 'valid' in eval_metrics[f] and 'test' in eval_metrics[f]]
    
    if not complete_folds:
        print("Warning: No complete evaluation metrics found (missing valid or test data)")
        return
    
    valid_mse = [eval_metrics[f]['valid']['mse'] for f in complete_folds]
    test_mse = [eval_metrics[f]['test']['mse'] for f in complete_folds]
    valid_mae = [eval_metrics[f]['valid']['mae'] for f in complete_folds]
    test_mae = [eval_metrics[f]['test']['mae'] for f in complete_folds]
    valid_rmse = [eval_metrics[f]['valid']['rmse'] for f in complete_folds]
    test_rmse = [eval_metrics[f]['test']['rmse'] for f in complete_folds]
    valid_corr = [eval_metrics[f]['valid']['correlation'] for f in complete_folds]
    test_corr = [eval_metrics[f]['test']['correlation'] for f in complete_folds]
    
    x = np.arange(len(complete_folds))
    width = 0.35
    
    # MSE
    ax = axes[0, 0]
    ax.bar(x - width/2, valid_mse, width, label='Validation', alpha=0.8)
    ax.bar(x + width/2, test_mse, width, label='Test', alpha=0.8)
    ax.set_xlabel('CV Fold', fontsize=11)
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title('Mean Squared Error', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in complete_folds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # MAE
    ax = axes[0, 1]
    ax.bar(x - width/2, valid_mae, width, label='Validation', alpha=0.8)
    ax.bar(x + width/2, test_mae, width, label='Test', alpha=0.8)
    ax.set_xlabel('CV Fold', fontsize=11)
    ax.set_ylabel('MAE', fontsize=11)
    ax.set_title('Mean Absolute Error', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in complete_folds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # RMSE
    ax = axes[1, 0]
    ax.bar(x - width/2, valid_rmse, width, label='Validation', alpha=0.8)
    ax.bar(x + width/2, test_rmse, width, label='Test', alpha=0.8)
    ax.set_xlabel('CV Fold', fontsize=11)
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title('Root Mean Squared Error', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in complete_folds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Correlation
    ax = axes[1, 1]
    ax.bar(x - width/2, valid_corr, width, label='Validation', alpha=0.8)
    ax.bar(x + width/2, test_corr, width, label='Test', alpha=0.8)
    ax.set_xlabel('CV Fold', fontsize=11)
    ax.set_ylabel('Pearson Correlation', fontsize=11)
    ax.set_title('Pearson Correlation Coefficient', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in complete_folds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved regression metrics to {output_path}")


def plot_classification_metrics(eval_metrics, cv_summary, output_path, trial_num):
    """Plot classification metrics comparison across CV folds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Trial {trial_num} - Classification Performance Metrics', fontsize=16, fontweight='bold')
    
    # Prepare data
    folds = sorted(eval_metrics.keys())
    valid_acc = [eval_metrics[f]['valid']['accuracy'] for f in folds]
    test_acc = [eval_metrics[f]['test']['accuracy'] for f in folds]
    valid_prec = [eval_metrics[f]['valid']['precision'] for f in folds]
    test_prec = [eval_metrics[f]['test']['precision'] for f in folds]
    valid_rec = [eval_metrics[f]['valid']['recall'] for f in folds]
    test_rec = [eval_metrics[f]['test']['recall'] for f in folds]
    valid_f1 = [eval_metrics[f]['valid']['f1'] for f in folds]
    test_f1 = [eval_metrics[f]['test']['f1'] for f in folds]
    
    x = np.arange(len(folds))
    width = 0.35
    
    # Accuracy
    ax = axes[0, 0]
    ax.bar(x - width/2, valid_acc, width, label='Validation', alpha=0.8)
    ax.bar(x + width/2, test_acc, width, label='Test', alpha=0.8)
    ax.set_xlabel('CV Fold', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Accuracy', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in complete_folds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    # Precision
    ax = axes[0, 1]
    ax.bar(x - width/2, valid_prec, width, label='Validation', alpha=0.8)
    ax.bar(x + width/2, test_prec, width, label='Test', alpha=0.8)
    ax.set_xlabel('CV Fold', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in complete_folds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    # Recall
    ax = axes[1, 0]
    ax.bar(x - width/2, valid_rec, width, label='Validation', alpha=0.8)
    ax.bar(x + width/2, test_rec, width, label='Test', alpha=0.8)
    ax.set_xlabel('CV Fold', fontsize=11)
    ax.set_ylabel('Recall', fontsize=11)
    ax.set_title('Recall', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in complete_folds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    # F1-Score
    ax = axes[1, 1]
    ax.bar(x - width/2, valid_f1, width, label='Validation', alpha=0.8)
    ax.bar(x + width/2, test_f1, width, label='Test', alpha=0.8)
    ax.set_xlabel('CV Fold', fontsize=11)
    ax.set_ylabel('F1-Score', fontsize=11)
    ax.set_title('F1-Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in complete_folds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved classification metrics to {output_path}")


def plot_regression_visualization(predictions_dir, output_path, trial_num):
    """Plot regression curves and residual plots for all CV folds (or single run if no CV)."""
    predictions_dir = Path(predictions_dir)
    
    # Collect all predictions
    test_predictions = {}
    valid_predictions = {}
    
    # Try CV format first (trial{N}_cv{F}_*.npy)
    cv_files = list(predictions_dir.glob(f"trial{trial_num}_cv*_*.npy"))
    is_cv = len(cv_files) > 0
    
    if is_cv:
        # CV format: trial{N}_cv{F}_validation.npy or trial{N}_cv{F}_test.npy
        for pred_file in sorted(cv_files):
            data = np.load(pred_file, allow_pickle=True).item()
            cv = data['cv']
            set_name = data['set']
            
            if set_name == 'test':
                test_predictions[cv] = {'y_true': data['y_true'], 'y_pred': data['y_pred']}
            elif set_name == 'validation':
                valid_predictions[cv] = {'y_true': data['y_true'], 'y_pred': data['y_pred']}
    else:
        # No CV format: trial{N}_validation.npy or trial{N}_test.npy
        test_file = predictions_dir / f"trial{trial_num}_test.npy"
        valid_file = predictions_dir / f"trial{trial_num}_validation.npy"
        
        if test_file.exists():
            data = np.load(test_file, allow_pickle=True).item()
            test_predictions[0] = {'y_true': data['y_true'], 'y_pred': data['y_pred']}
        
        if valid_file.exists():
            data = np.load(valid_file, allow_pickle=True).item()
            valid_predictions[0] = {'y_true': data['y_true'], 'y_pred': data['y_pred']}
    
    if not test_predictions:
        print(f"Warning: No prediction files found in {predictions_dir}")
        return
    
    # Create figure with subplots: only test set regression and residual plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Trial {trial_num} - Test Set Regression Visualization', fontsize=16, fontweight='bold')
    
    # Test set regression plot (predicted vs actual)
    ax = axes[0]
    for cv in sorted(test_predictions.keys()):
        y_true = test_predictions[cv]['y_true']
        y_pred = test_predictions[cv]['y_pred']
        label = f'CV Fold {cv}' if is_cv else 'Single Run'
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, label=label)
    
    # Add perfect prediction line (y=x)
    min_val = min([min(test_predictions[cv]['y_true']) for cv in test_predictions])
    max_val = max([max(test_predictions[cv]['y_true']) for cv in test_predictions])
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
    
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title('Test Set - Predicted vs Actual', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Test set residual plot
    ax = axes[1]
    for cv in sorted(test_predictions.keys()):
        y_true = test_predictions[cv]['y_true']
        y_pred = test_predictions[cv]['y_pred']
        residuals = y_true - y_pred
        label = f'CV Fold {cv}' if is_cv else 'Single Run'
        ax.scatter(y_pred, residuals, alpha=0.5, s=20, label=label)
    
    # Add zero residual line
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Values', fontsize=12)
    ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax.set_title('Test Set - Residual Plot', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved regression visualization to {output_path}")


def plot_cv_summary(cv_summary, output_path, trial_num, task):
    """Plot cross-validation summary with error bars."""
    if not cv_summary:
        print("Warning: No CV summary data found in log file")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Trial {trial_num} - Cross-Validation Summary', fontsize=14, fontweight='bold')
    
    if task == 'RGS':
        # Regression summary - plot all available metrics
        metric_keys = ['mse', 'mae', 'rmse', 'corr']
        metric_labels = ['MSE', 'MAE', 'RMSE', 'Correlation']
        
        available_metrics = []
        valid_means = []
        valid_stds = []
        test_means = []
        test_stds = []
        
        for key, label in zip(metric_keys, metric_labels):
            valid_key = f'valid_{key}'
            test_key = f'test_{key}'
            if valid_key in cv_summary and test_key in cv_summary:
                available_metrics.append(label)
                valid_means.append(cv_summary[valid_key][0])
                valid_stds.append(cv_summary[valid_key][1])
                test_means.append(cv_summary[test_key][0])
                test_stds.append(cv_summary[test_key][1])
        
        if available_metrics:
            x = np.arange(len(available_metrics))
            width = 0.35
            
            ax = axes[0]
            ax.bar(x - width/2, valid_means, width, yerr=valid_stds, label='Validation', 
                   alpha=0.8, capsize=5, error_kw={'elinewidth': 2})
            ax.bar(x + width/2, test_means, width, yerr=test_stds, label='Test', 
                   alpha=0.8, capsize=5, error_kw={'elinewidth': 2})
            ax.set_xlabel('Metric', fontsize=11)
            ax.set_ylabel('Value', fontsize=11)
            ax.set_title('Average Performance (Mean ± Std)', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(available_metrics)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Correlation plot
            if 'valid_corr' in cv_summary and 'test_corr' in cv_summary:
                ax = axes[1]
                categories = ['Validation', 'Test']
                means = [cv_summary['valid_corr'][0], cv_summary['test_corr'][0]]
                stds = [cv_summary['valid_corr'][1], cv_summary['test_corr'][1]]
                ax.bar(categories, means, yerr=stds, alpha=0.8, capsize=10, 
                       error_kw={'elinewidth': 2}, color=['#1f77b4', '#ff7f0e'])
                ax.set_ylabel('Pearson Correlation', fontsize=11)
                ax.set_title('Average Correlation Coefficient', fontsize=12)
                ax.set_ylim([0, 1])
                ax.grid(True, alpha=0.3, axis='y')
            else:
                axes[1].axis('off')
        else:
            axes[0].axis('off')
            axes[1].axis('off')
    else:
        # Classification summary
        if 'valid_acc' in cv_summary and 'test_acc' in cv_summary:
            categories = ['Validation', 'Test']
            means = [cv_summary['valid_acc'][0], cv_summary['test_acc'][0]]
            stds = [cv_summary['valid_acc'][1], cv_summary['test_acc'][1]]
            
            ax = axes[0]
            ax.bar(categories, means, yerr=stds, alpha=0.8, capsize=10, 
                   error_kw={'elinewidth': 2}, color=['#1f77b4', '#ff7f0e'])
            ax.set_ylabel('Accuracy', fontsize=11)
            ax.set_title('Average Accuracy (Mean ± Std)', fontsize=12)
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
        
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved CV summary to {output_path}")


def main():
    # Log file path - modify this to change the log file
    log_file = 'trial2_RGS_20251116_175423.log'
    output_dir_str = './logs_fp_f21_25'
    prefix = None  # Set to None for auto-detection, or specify custom prefix
    
    # Parse log file
    log_path = Path(log_file)
    if not log_path.exists():
        # Try relative to logs directory
        log_path = Path('./logs_fp_f21_25') / log_file
        if not log_path.exists():
            print(f"Error: Log file not found: {log_file}")
            print(f"Tried: {log_file} and ./logs_fp_f21_25/{log_file}")
            return
    
    print(f"Parsing log file: {log_path}")
    data = parse_log_file(log_path)
    
    if not data['trial_info']:
        print("Error: Could not extract trial information from log file")
        return
    
    trial_num = data['trial_info']['trial']
    task = data['trial_info']['task']
    
    # Determine output prefix
    if prefix:
        output_prefix = prefix
    else:
        output_prefix = f"trial{trial_num}_{task.lower()}"
    
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print(f"\nGenerating plots for Trial {trial_num} ({task})...")
    
    # 1. Training curves
    if data['training_losses']:
        training_curve_path = output_dir / f"{output_prefix}_training_curves.png"
        plot_training_curves(data['training_losses'], training_curve_path, trial_num)
    
    # 2. Performance metrics
    if data['eval_metrics']:
        if task == 'RGS':
            metrics_path = output_dir / f"{output_prefix}_regression_metrics.png"
            plot_regression_metrics(data['eval_metrics'], data['cv_summary'], 
                                  metrics_path, trial_num)
        else:
            metrics_path = output_dir / f"{output_prefix}_classification_metrics.png"
            plot_classification_metrics(data['eval_metrics'], data['cv_summary'], 
                                       metrics_path, trial_num)
    
    # 3. CV summary
    if data['cv_summary']:
        summary_path = output_dir / f"{output_prefix}_cv_summary.png"
        plot_cv_summary(data['cv_summary'], summary_path, trial_num, task)
    
    # 4. Regression visualization (if predictions exist)
    if task == 'RGS':
        predictions_dir = output_dir / "predictions"
        if predictions_dir.exists():
            regression_viz_path = output_dir / f"{output_prefix}_regression_visualization.png"
            plot_regression_visualization(predictions_dir, regression_viz_path, trial_num)
        else:
            print(f"Note: Predictions directory not found: {predictions_dir}")
            print("      Run training with --save-predictions flag to generate regression plots.")
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()

