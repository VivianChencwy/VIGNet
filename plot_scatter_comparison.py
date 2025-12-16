"""
Generate scatter plots comparing actual vs predicted PERCLOS for all experiments
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.stats import pearsonr
import os

# Define experiments
experiments = [
    {
        'name': 'Weiyu',
        'path': '/home/vivian/eeg/SEED_VIG/Dec_09_experiments/experiment_20251208_155206/logs_custom/predictions',
        'color': '#2ecc71'
    },
    {
        'name': 'Qi Zhang',
        'path': '/home/vivian/eeg/SEED_VIG/Dec_09_experiments/experiment_20251209_144428/logs_custom/predictions',
        'color': '#3498db'
    },
    {
        'name': 'Shujia Chen',
        'path': '/home/vivian/eeg/SEED_VIG/Dec_09_experiments/experiment_20251209_154140/logs_custom/predictions',
        'color': '#e74c3c'
    },
    {
        'name': 'Xinyi',
        'path': '/home/vivian/eeg/SEED_VIG/Dec_09_experiments/experiment_20251209_163954/logs_custom/predictions',
        'color': '#9b59b6'
    }
]

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, exp in enumerate(experiments):
    ax = axes[idx]
    
    # Load predictions
    pred_file = os.path.join(exp['path'], 'test_predictions.npy')
    data = np.load(pred_file, allow_pickle=True).item()
    y_true = data['y_true'].flatten()
    y_pred = data['y_pred'].flatten()
    
    # Calculate metrics
    corr, p_value = pearsonr(y_true, y_pred)
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.7, s=80, c=exp['color'], 
               edgecolors='white', linewidth=1.5, label=f'n={len(y_true)}')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.1
    ax.plot([min_val - margin, max_val + margin], 
            [min_val - margin, max_val + margin], 
            'k--', lw=2, alpha=0.5, label='Perfect Prediction')
    
    # Regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min_val - margin, max_val + margin, 100)
    ax.plot(x_line, p(x_line), color=exp['color'], lw=2.5, alpha=0.8,
            label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
    
    # Labels and title
    ax.set_xlabel('True PERCLOS', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted PERCLOS', fontsize=13, fontweight='bold')
    
    # Significance stars
    if p_value < 0.001:
        sig = '***'
    elif p_value < 0.01:
        sig = '**'
    elif p_value < 0.05:
        sig = '*'
    else:
        sig = ' (ns)'
    
    ax.set_title(f'{exp["name"]}\nr = {corr:.3f}{sig}', 
                 fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([min_val - margin, max_val + margin])
    ax.set_ylim([min_val - margin, max_val + margin])
    ax.set_aspect('equal', adjustable='box')
    
    # Add text box with stats
    stats_text = f'RMSE: {np.sqrt(np.mean((y_true - y_pred)**2)):.3f}\nMAE: {np.mean(np.abs(y_true - y_pred)):.3f}'
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('PERCLOS Prediction: Actual vs Predicted (Test Set)\n4 Subjects Comparison', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# Save figure
output_path = '/home/vivian/eeg/SEED_VIG/Dec_09_experiments/all_subjects_scatter_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")

# Also create a combined scatter plot (all subjects on one)
fig2, ax2 = plt.subplots(figsize=(10, 10))

all_true = []
all_pred = []
for exp in experiments:
    pred_file = os.path.join(exp['path'], 'test_predictions.npy')
    data = np.load(pred_file, allow_pickle=True).item()
    y_true = data['y_true'].flatten()
    y_pred = data['y_pred'].flatten()
    
    corr, p_value = pearsonr(y_true, y_pred)
    sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
    
    ax2.scatter(y_true, y_pred, alpha=0.7, s=100, c=exp['color'], 
                edgecolors='white', linewidth=1.5, 
                label=f"{exp['name']} (r={corr:.2f}{sig}, n={len(y_true)})")
    
    all_true.extend(y_true)
    all_pred.extend(y_pred)

# Overall correlation
all_true = np.array(all_true)
all_pred = np.array(all_pred)
overall_corr, overall_p = pearsonr(all_true, all_pred)

# Perfect prediction line
ax2.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Perfect Prediction')

ax2.set_xlabel('True PERCLOS', fontsize=14, fontweight='bold')
ax2.set_ylabel('Predicted PERCLOS', fontsize=14, fontweight='bold')
ax2.set_title(f'All Subjects: PERCLOS Prediction (Test Set)\nOverall r = {overall_corr:.3f} (n={len(all_true)})', 
              fontsize=15, fontweight='bold')
ax2.legend(loc='upper left', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([-0.05, 1.05])
ax2.set_ylim([-0.05, 1.05])
ax2.set_aspect('equal', adjustable='box')

output_path2 = '/home/vivian/eeg/SEED_VIG/Dec_09_experiments/all_subjects_combined_scatter.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path2}")

print("\nDone!")

