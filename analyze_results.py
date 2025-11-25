"""
Analyze and visualize VIGNet training results.
Generates plots and comprehensive reports for all trained models.

Usage:
    python analyze_results.py --log-dir logs_experiment
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class ResultsAnalyzer:
    """Analyze and visualize VIGNet training results."""
    
    TARGET_NAMES = {
        'perclos': 'PERCLOS',
        'kss': 'KSS (normalized)',
        'miss_rate': 'Miss Rate',
        'false_alarm': 'False Alarm Rate'
    }
    
    def __init__(self, log_dir):
        """
        Args:
            log_dir: Base directory containing training results
        """
        self.log_dir = log_dir
        self.summary_dir = os.path.join(log_dir, 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)
        
        self.results = {}
        self.available_targets = []
    
    def load_results(self):
        """Load results for all available targets."""
        print(f"\nLoading results from: {self.log_dir}")
        print('='*60)
        
        for target in ['perclos', 'kss', 'miss_rate', 'false_alarm']:
            target_dir = os.path.join(self.log_dir, target)
            
            if not os.path.exists(target_dir):
                print(f"  {target}: not found")
                continue
            
            try:
                # Load metrics
                metrics_path = os.path.join(target_dir, 'metrics.json')
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                # Load predictions
                predictions_path = os.path.join(target_dir, 'predictions.npy')
                predictions = np.load(predictions_path, allow_pickle=True).item()
                
                # Load training history
                history_path = os.path.join(target_dir, 'training_history.csv')
                history = pd.read_csv(history_path)
                
                # Load config
                config_path = os.path.join(target_dir, 'config.json')
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                self.results[target] = {
                    'metrics': metrics,
                    'predictions': predictions,
                    'history': history,
                    'config': config
                }
                self.available_targets.append(target)
                
                print(f"  {target}: loaded (test r={metrics['test']['pearson_r']:.4f})")
                
            except Exception as e:
                print(f"  {target}: error loading - {e}")
        
        print(f"\nLoaded {len(self.available_targets)} targets")
    
    def plot_scatter(self, target):
        """Generate scatter plot for a single target."""
        if target not in self.results:
            return
        
        predictions = self.results[target]['predictions']
        metrics = self.results[target]['metrics']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for idx, (pred_key, metric_key, ax) in enumerate(zip(['val', 'test'], ['validation', 'test'], axes)):
            y_true = predictions[pred_key]['y_true']
            y_pred = predictions[pred_key]['y_pred']
            
            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors='none')
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
            
            # Regression line
            z = np.polyfit(y_true, y_pred, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min_val, max_val, 100)
            ax.plot(x_line, p(x_line), 'g-', lw=2, alpha=0.7, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
            
            # Labels
            ax.set_xlabel('Actual', fontsize=11)
            ax.set_ylabel('Predicted', fontsize=11)
            ax.set_title(f'{metric_key.capitalize()} Set\n'
                        f'r={metrics[metric_key]["pearson_r"]:.4f}, '
                        f'RMSE={metrics[metric_key]["rmse"]:.4f}',
                        fontsize=12)
            ax.legend(loc='upper left', fontsize=9)
            ax.set_aspect('equal', adjustable='box')
        
        fig.suptitle(f'{self.TARGET_NAMES[target]} - Predicted vs Actual', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(self.log_dir, target, 'scatter_plot.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
    
    def plot_time_series(self, target):
        """Generate time series plot for a single target."""
        if target not in self.results:
            return
        
        predictions = self.results[target]['predictions']
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        for idx, (pred_key, display_name, ax) in enumerate(zip(['val', 'test'], ['Validation', 'Test'], axes)):
            y_true = predictions[pred_key]['y_true']
            y_pred = predictions[pred_key]['y_pred']
            
            x = np.arange(len(y_true))
            
            ax.plot(x, y_true, 'b-', alpha=0.7, lw=1.5, label='Actual')
            ax.plot(x, y_pred, 'r-', alpha=0.7, lw=1.5, label='Predicted')
            
            ax.set_xlabel('Sample Index', fontsize=11)
            ax.set_ylabel(self.TARGET_NAMES[target], fontsize=11)
            ax.set_title(f'{display_name} Set', fontsize=12)
            ax.legend(loc='upper right', fontsize=10)
            ax.set_xlim(0, len(y_true))
        
        fig.suptitle(f'{self.TARGET_NAMES[target]} - Time Series', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.log_dir, target, 'time_series_plot.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
    
    def plot_residuals(self, target):
        """Generate residual distribution plot for a single target."""
        if target not in self.results:
            return
        
        predictions = self.results[target]['predictions']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for idx, (pred_key, display_name, ax) in enumerate(zip(['val', 'test'], ['Validation', 'Test'], axes)):
            y_true = predictions[pred_key]['y_true']
            y_pred = predictions[pred_key]['y_pred']
            residuals = y_true - y_pred
            
            # Histogram
            ax.hist(residuals, bins=30, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.axvline(0, color='red', linestyle='--', lw=2, label='Zero')
            ax.axvline(np.mean(residuals), color='green', linestyle='-', lw=2, 
                      label=f'Mean: {np.mean(residuals):.4f}')
            
            ax.set_xlabel('Residual (Actual - Predicted)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'{display_name} Set\nStd: {np.std(residuals):.4f}', fontsize=12)
            ax.legend(loc='upper right', fontsize=9)
        
        fig.suptitle(f'{self.TARGET_NAMES[target]} - Residual Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.log_dir, target, 'residual_histogram.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
    
    def plot_learning_curve(self, target):
        """Generate learning curve plot for a single target."""
        if target not in self.results:
            return
        
        history = self.results[target]['history']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curve
        ax = axes[0]
        ax.plot(history['epoch'], history['train_loss'], 'b-', lw=2, label='Train Loss')
        ax.plot(history['epoch'], history['val_loss'], 'r-', lw=2, label='Validation Loss')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss (MSE)', fontsize=11)
        ax.set_title('Training Loss', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlim(1, len(history))
        
        # Correlation curve
        ax = axes[1]
        ax.plot(history['epoch'], history['val_correlation'], 'g-', lw=2, label='Validation Correlation')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Pearson Correlation', fontsize=11)
        ax.set_title('Validation Correlation', fontsize=12)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_xlim(1, len(history))
        ax.set_ylim(0, 1)
        
        fig.suptitle(f'{self.TARGET_NAMES[target]} - Learning Curves', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.log_dir, target, 'learning_curve.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
    
    def plot_all_targets(self):
        """Generate plots for all available targets."""
        print(f"\nGenerating individual target plots...")
        print('='*60)
        
        for target in self.available_targets:
            print(f"\n{target}:")
            self.plot_scatter(target)
            self.plot_time_series(target)
            self.plot_residuals(target)
            self.plot_learning_curve(target)
    
    def plot_comparison(self):
        """Generate comparison plots across all targets."""
        if len(self.available_targets) < 2:
            print("Not enough targets for comparison")
            return
        
        print(f"\nGenerating comparison plots...")
        print('='*60)
        
        # 1. Correlation comparison bar chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        targets = self.available_targets
        target_names = [self.TARGET_NAMES[t] for t in targets]
        
        # Test set correlations
        ax = axes[0]
        correlations = [self.results[t]['metrics']['test']['pearson_r'] for t in targets]
        colors = ['green' if c > 0.5 else 'orange' if c > 0.3 else 'red' for c in correlations]
        bars = ax.bar(range(len(targets)), correlations, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(target_names, rotation=15, ha='right')
        ax.set_ylabel('Pearson Correlation', fontsize=11)
        ax.set_title('Test Set Correlation by Target', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        for bar, corr in zip(bars, correlations):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{corr:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Test set RMSE
        ax = axes[1]
        rmses = [self.results[t]['metrics']['test']['rmse'] for t in targets]
        colors = ['green' if r < 0.1 else 'orange' if r < 0.2 else 'red' for r in rmses]
        bars = ax.bar(range(len(targets)), rmses, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(target_names, rotation=15, ha='right')
        ax.set_ylabel('RMSE', fontsize=11)
        ax.set_title('Test Set RMSE by Target', fontsize=12, fontweight='bold')
        for bar, rmse in zip(bars, rmses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                   f'{rmse:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        save_path = os.path.join(self.summary_dir, 'all_targets_comparison.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
        
        # 2. Scatter plot grid
        n_targets = len(targets)
        fig, axes = plt.subplots(1, n_targets, figsize=(5*n_targets, 5))
        if n_targets == 1:
            axes = [axes]
        
        for idx, (target, ax) in enumerate(zip(targets, axes)):
            predictions = self.results[target]['predictions']
            y_true = predictions['test']['y_true']
            y_pred = predictions['test']['y_pred']
            
            ax.scatter(y_true, y_pred, alpha=0.6, s=20, edgecolors='none')
            
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            r = self.results[target]['metrics']['test']['pearson_r']
            ax.set_xlabel('Actual', fontsize=10)
            ax.set_ylabel('Predicted', fontsize=10)
            ax.set_title(f'{self.TARGET_NAMES[target]}\nr = {r:.4f}', fontsize=11)
            ax.set_aspect('equal', adjustable='box')
        
        fig.suptitle('Test Set: Predicted vs Actual (All Targets)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_path = os.path.join(self.summary_dir, 'scatter_comparison.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
    
    def generate_report(self):
        """Generate comprehensive markdown report."""
        print(f"\nGenerating comprehensive report...")
        print('='*60)
        
        lines = []
        lines.append("# VIGNet Fatigue Prediction - Evaluation Report")
        lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"\n**Number of targets**: {len(self.available_targets)}")
        lines.append("\n---\n")
        
        # Summary table
        lines.append("## Summary\n")
        lines.append("| Target | Test Pearson r | Test Spearman r | Test RMSE | Test MAE |")
        lines.append("|--------|----------------|-----------------|-----------|----------|")
        
        for target in self.available_targets:
            m = self.results[target]['metrics']['test']
            lines.append(f"| {self.TARGET_NAMES[target]} | {m['pearson_r']:.4f} | "
                        f"{m['spearman_r']:.4f} | {m['rmse']:.4f} | {m['mae']:.4f} |")
        
        lines.append("\n---\n")
        
        # Individual target details
        for target in self.available_targets:
            lines.append(f"## {self.TARGET_NAMES[target]}\n")
            
            config = self.results[target]['config']
            metrics = self.results[target]['metrics']
            
            lines.append("### Configuration\n")
            lines.append(f"- Learning rate: {config['learning_rate']}")
            lines.append(f"- Epochs: {config['num_epochs']}")
            lines.append(f"- Batch size: {config['batch_size']}")
            lines.append(f"- Early stopping patience: {config['early_stopping_patience']}")
            lines.append(f"- Train samples: {config['train_samples']}")
            lines.append(f"- Validation samples: {config['val_samples']}")
            lines.append(f"- Test samples: {config['test_samples']}")
            lines.append(f"- Feature shape: {config['feature_shape']}")
            
            lines.append("\n### Validation Set Metrics\n")
            m = metrics['validation']
            lines.append(f"- MSE: {m['mse']:.6f}")
            lines.append(f"- MAE: {m['mae']:.6f}")
            lines.append(f"- RMSE: {m['rmse']:.6f}")
            lines.append(f"- Pearson r: {m['pearson_r']:.6f} (p={m['pearson_p']:.2e})")
            lines.append(f"- Spearman r: {m['spearman_r']:.6f} (p={m['spearman_p']:.2e})")
            
            lines.append("\n### Test Set Metrics\n")
            m = metrics['test']
            lines.append(f"- MSE: {m['mse']:.6f}")
            lines.append(f"- MAE: {m['mae']:.6f}")
            lines.append(f"- RMSE: {m['rmse']:.6f}")
            lines.append(f"- Pearson r: {m['pearson_r']:.6f} (p={m['pearson_p']:.2e})")
            lines.append(f"- Spearman r: {m['spearman_r']:.6f} (p={m['spearman_p']:.2e})")
            
            lines.append("\n### Visualizations\n")
            lines.append(f"- Scatter plot: `{target}/scatter_plot.png`")
            lines.append(f"- Time series: `{target}/time_series_plot.png`")
            lines.append(f"- Residuals: `{target}/residual_histogram.png`")
            lines.append(f"- Learning curve: `{target}/learning_curve.png`")
            
            lines.append("\n---\n")
        
        # Conclusions
        lines.append("## Conclusions\n")
        
        # Find best target
        best_target = max(self.available_targets, 
                         key=lambda t: self.results[t]['metrics']['test']['pearson_r'])
        best_r = self.results[best_target]['metrics']['test']['pearson_r']
        
        lines.append(f"- **Best performing target**: {self.TARGET_NAMES[best_target]} "
                    f"(r = {best_r:.4f})")
        
        lines.append("\n### Model Files\n")
        lines.append("Each target has the following saved files:")
        lines.append("- `model_weights.h5`: Trained model weights")
        lines.append("- `predictions.npy`: Validation and test predictions")
        lines.append("- `training_history.csv`: Training metrics per epoch")
        lines.append("- `metrics.json`: Final evaluation metrics")
        lines.append("- `config.json`: Training configuration")
        
        # Write report
        report_path = os.path.join(self.summary_dir, 'comprehensive_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"  Saved: {report_path}")
    
    def run(self):
        """Run complete analysis pipeline."""
        self.load_results()
        
        if len(self.available_targets) == 0:
            print("No results found!")
            return
        
        self.plot_all_targets()
        self.plot_comparison()
        self.generate_report()
        
        print(f"\n{'='*60}")
        print("Analysis Complete!")
        print(f"Results saved to: {self.summary_dir}")
        print('='*60)


def main():
    parser = argparse.ArgumentParser(description='Analyze VIGNet Training Results')
    parser.add_argument('--log-dir', type=str, default='./logs_experiment',
                        help='Directory containing training results')
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(log_dir=args.log_dir)
    analyzer.run()


if __name__ == "__main__":
    main()

