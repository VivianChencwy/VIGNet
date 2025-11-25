# VIGNet Fatigue Prediction - Evaluation Report

**Generated**: 2025-11-24 22:44:22

**Number of targets**: 4

---

## Summary

| Target | Test Pearson r | Test Spearman r | Test RMSE | Test MAE |
|--------|----------------|-----------------|-----------|----------|
| PERCLOS | 0.9271 | 0.8936 | 0.0985 | 0.0735 |
| KSS (normalized) | 0.9874 | 0.9625 | 0.0240 | 0.0171 |
| Miss Rate | 0.8467 | 0.8159 | 0.1365 | 0.0913 |
| False Alarm Rate | 0.9385 | 0.8415 | 0.0102 | 0.0071 |

---

## PERCLOS

### Configuration

- Learning rate: 0.005
- Epochs: 200
- Batch size: 8
- Early stopping patience: 20
- Train samples: 401
- Validation samples: 84
- Test samples: 92
- Feature shape: [2, 25, 1]

### Validation Set Metrics

- MSE: 0.009822
- MAE: 0.068674
- RMSE: 0.099106
- Pearson r: 0.914416 (p=5.93e-34)
- Spearman r: 0.859113 (p=1.42e-25)

### Test Set Metrics

- MSE: 0.009709
- MAE: 0.073462
- RMSE: 0.098533
- Pearson r: 0.927090 (p=4.00e-40)
- Spearman r: 0.893554 (p=4.67e-33)

### Visualizations

- Scatter plot: `perclos/scatter_plot.png`
- Time series: `perclos/time_series_plot.png`
- Residuals: `perclos/residual_histogram.png`
- Learning curve: `perclos/learning_curve.png`

---

## KSS (normalized)

### Configuration

- Learning rate: 0.005
- Epochs: 200
- Batch size: 8
- Early stopping patience: 20
- Train samples: 401
- Validation samples: 85
- Test samples: 91
- Feature shape: [2, 25, 1]

### Validation Set Metrics

- MSE: 0.000333
- MAE: 0.014809
- RMSE: 0.018262
- Pearson r: 0.990151 (p=1.19e-72)
- Spearman r: 0.957536 (p=1.34e-46)

### Test Set Metrics

- MSE: 0.000576
- MAE: 0.017085
- RMSE: 0.023997
- Pearson r: 0.987437 (p=4.12e-73)
- Spearman r: 0.962518 (p=3.22e-52)

### Visualizations

- Scatter plot: `kss/scatter_plot.png`
- Time series: `kss/time_series_plot.png`
- Residuals: `kss/residual_histogram.png`
- Learning curve: `kss/learning_curve.png`

---

## Miss Rate

### Configuration

- Learning rate: 0.005
- Epochs: 200
- Batch size: 8
- Early stopping patience: 20
- Train samples: 122
- Validation samples: 25
- Test samples: 31
- Feature shape: [2, 25, 1]

### Validation Set Metrics

- MSE: 0.014496
- MAE: 0.077136
- RMSE: 0.120401
- Pearson r: 0.768112 (p=7.36e-06)
- Spearman r: 0.782248 (p=3.85e-06)

### Test Set Metrics

- MSE: 0.018637
- MAE: 0.091322
- RMSE: 0.136517
- Pearson r: 0.846659 (p=1.94e-09)
- Spearman r: 0.815892 (p=2.23e-08)

### Visualizations

- Scatter plot: `miss_rate/scatter_plot.png`
- Time series: `miss_rate/time_series_plot.png`
- Residuals: `miss_rate/residual_histogram.png`
- Learning curve: `miss_rate/learning_curve.png`

---

## False Alarm Rate

### Configuration

- Learning rate: 0.005
- Epochs: 200
- Batch size: 8
- Early stopping patience: 20
- Train samples: 124
- Validation samples: 25
- Test samples: 32
- Feature shape: [2, 25, 1]

### Validation Set Metrics

- MSE: 0.000046
- MAE: 0.004994
- RMSE: 0.006781
- Pearson r: 0.861960 (p=3.09e-08)
- Spearman r: 0.773326 (p=5.82e-06)

### Test Set Metrics

- MSE: 0.000105
- MAE: 0.007110
- RMSE: 0.010250
- Pearson r: 0.938500 (p=2.14e-15)
- Spearman r: 0.841484 (p=1.61e-09)

### Visualizations

- Scatter plot: `false_alarm/scatter_plot.png`
- Time series: `false_alarm/time_series_plot.png`
- Residuals: `false_alarm/residual_histogram.png`
- Learning curve: `false_alarm/learning_curve.png`

---

## Conclusions

- **Best performing target**: KSS (normalized) (r = 0.9874)

### Model Files

Each target has the following saved files:
- `model_weights.h5`: Trained model weights
- `predictions.npy`: Validation and test predictions
- `training_history.csv`: Training metrics per epoch
- `metrics.json`: Final evaluation metrics
- `config.json`: Training configuration