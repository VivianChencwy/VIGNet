"""
Visualize model predictions vs ground truth
Shows scatter plots, time series, and correlation analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from scipy.stats import pearsonr
import os
import argparse
from pathlib import Path


def load_predictions(predictions_dir, set_name='test'):
    """Load predictions from .npy file"""
    pred_file = os.path.join(predictions_dir, f"{set_name}_predictions.npy")
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    
    data = np.load(pred_file, allow_pickle=True).item()
    y_true = data['y_true'].flatten()
    y_pred = data['y_pred'].flatten()
    
    return y_true, y_pred


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    corr, p_value = pearsonr(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'correlation': corr,
        'p_value': p_value
    }


def plot_predictions(y_true, y_pred, metrics, output_path, set_name='test'):
    """Create comprehensive visualization of predictions"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Scatter plot with regression line
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Add diagonal line (perfect prediction)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Add regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax1.plot(y_true, p(y_true), "b-", lw=2, alpha=0.8, label=f'Regression Line (slope={z[0]:.3f})')
    
    ax1.set_xlabel('True PERCLOS', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted PERCLOS', fontsize=12, fontweight='bold')
    ax1.set_title(f'{set_name.upper()} Set: Scatter Plot\n'
                  f'Pearson r = {metrics["correlation"]:.4f} (p = {metrics["p_value"]:.4f})',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # 2. Time series comparison
    ax2 = plt.subplot(2, 3, 2)
    time_points = np.arange(len(y_true))
    ax2.plot(time_points, y_true, 'o-', label='True PERCLOS', alpha=0.7, linewidth=2, markersize=4)
    ax2.plot(time_points, y_pred, 's-', label='Predicted PERCLOS', alpha=0.7, linewidth=2, markersize=4)
    ax2.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PERCLOS', fontsize=12, fontweight='bold')
    ax2.set_title(f'{set_name.upper()} Set: Time Series Comparison', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Residuals plot
    ax3 = plt.subplot(2, 3, 3)
    residuals = y_true - y_pred
    ax3.scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Predicted PERCLOS', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Residuals (True - Predicted)', fontsize=12, fontweight='bold')
    ax3.set_title(f'{set_name.upper()} Set: Residuals Plot', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribution comparison
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(y_true, bins=15, alpha=0.6, label='True PERCLOS', color='blue', edgecolor='black')
    ax4.hist(y_pred, bins=15, alpha=0.6, label='Predicted PERCLOS', color='orange', edgecolor='black')
    ax4.set_xlabel('PERCLOS', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title(f'{set_name.upper()} Set: Distribution Comparison', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Error distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(residuals, bins=15, alpha=0.7, color='green', edgecolor='black')
    ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax5.set_xlabel('Residuals (True - Predicted)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax5.set_title(f'{set_name.upper()} Set: Error Distribution\n'
                  f'Mean Error = {residuals.mean():.4f}, Std = {residuals.std():.4f}',
                  fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Metrics summary text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    metrics_text = f"""
    METRICS SUMMARY ({set_name.upper()} SET)
    {'='*40}
    
    Correlation Metrics:
    • Pearson r: {metrics['correlation']:.4f}
    • p-value: {metrics['p_value']:.4f}
    • Significance: {'***' if metrics['p_value'] < 0.001 else '**' if metrics['p_value'] < 0.01 else '*' if metrics['p_value'] < 0.05 else 'ns'}
    
    Error Metrics:
    • MSE: {metrics['mse']:.6f}
    • RMSE: {metrics['rmse']:.4f}
    • MAE: {metrics['mae']:.4f}
    
    Data Statistics:
    • Sample size: {len(y_true)}
    • True PERCLOS range: [{y_true.min():.3f}, {y_true.max():.3f}]
    • Predicted PERCLOS range: [{y_pred.min():.3f}, {y_pred.max():.3f}]
    • True PERCLOS mean: {y_true.mean():.4f}
    • Predicted PERCLOS mean: {y_pred.mean():.4f}
    
    Interpretation:
    • r = {metrics['correlation']:.2f} indicates {'strong' if abs(metrics['correlation']) > 0.7 else 'moderate' if abs(metrics['correlation']) > 0.4 else 'weak'} correlation
    • {'Statistically significant' if metrics['p_value'] < 0.05 else 'Not statistically significant'} (p < 0.05)
    """
    
    ax6.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--predictions-dir', type=str, required=True,
                        help='Directory containing predictions .npy files')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: same as predictions-dir)')
    parser.add_argument('--set', type=str, default='test', choices=['test', 'valid'],
                        help='Which set to visualize (default: test)')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.predictions_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load predictions
    print(f"Loading {args.set} predictions from: {args.predictions_dir}")
    y_true, y_pred = load_predictions(args.predictions_dir, set_name=args.set)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    print(f"\nMetrics for {args.set.upper()} set:")
    print(f"  Pearson Correlation: {metrics['correlation']:.6f} (p = {metrics['p_value']:.6f})")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    
    # Create visualization
    output_path = os.path.join(args.output_dir, f"{args.set}_predictions_visualization.png")
    plot_predictions(y_true, y_pred, metrics, output_path, set_name=args.set)
    
    print(f"\nVisualization saved to: {output_path}")


if __name__ == "__main__":
    main()

