# VIGNet Experiment - FP1/FP2 Only Version

## Overview
This is a modified version of VIGNet that uses only FP1 and FP2 forehead channels (2 channels) instead of the full 17-channel EEG data.

## Key Differences from Original

### Data Loading (`utils_fp.py`)
- **Data Source**: `Forehead_EEG/EEG_Feature_2Hz` instead of `EEG_Feature_2Hz`
- **Channels**: 2 channels (FP1, FP2) instead of 17 channels
- **Shape**: `(N, 2, 25, 1)` instead of `(N, 17, 25, 1)`
- **Same Features**: Uses `de_LDS` (Differential Entropy with LDS smoothing)
- **Same Split**: 5-fold cross-validation with identical random seeds (970304)

### Model Architecture (`network_fp.py`)
- **Modified Layers**:
  - `conv4`: Changed from `(17, 1)` to `(2, 1)` kernel
  - `MHRSSA`: Changed `num_channel` default from 17 to 2
- **Same Architecture**: All other layers remain identical
- **Same Hyperparameters**: Regularization, activation functions unchanged

### Experiment Script (`experiment_fp.py`)
- **All training logic identical**: Same optimizer, learning rate, epochs, batch size
- **Same evaluation metrics**: MSE, MAE, RMSE, Pearson correlation for regression
- **Same logging format**: Compatible with existing visualization scripts
- **Output directory**: `./logs_fp` to separate from original results

## No Data Leakage
The implementation ensures no data leakage:
1. Uses same cross-validation splits (identical random seeds)
2. FP1/FP2 data comes from separate `Forehead_EEG` folder
3. No information from other channels
4. No temporal information leakage across splits

## Usage

### Run single trial
```bash
cd /home/vivian/eeg/SEED_VIG/VIGNet
python experiment_fp.py --trial 2 --task RGS
```

### Run all trials
```bash
python experiment_fp.py --task RGS
```

### Run classification task
```bash
python experiment_fp.py --trial 2 --task CLF
```

### Custom log directory
```bash
python experiment_fp.py --trial 2 --task RGS --log-dir ./my_logs
```

### Disable prediction saving
```bash
python experiment_fp.py --trial 2 --task RGS --no-save-predictions
```

## Expected Output

### Data Shapes
- Training: `(567, 2, 25, 1)`
- Validation: `(141, 2, 25, 1)`
- Test: `(177, 2, 25, 1)`

### Log Files
- `./logs_fp/trial{N}_RGS_{timestamp}.log`: Per-trial logs with all CV folds
- `./logs_fp/training_summary_{timestamp}.log`: Summary across all trials
- `./logs_fp/predictions/`: Prediction files for visualization

## Performance Expectations

Using only 2 frontal channels (FP1/FP2) instead of 17 channels will likely result in:
- **Lower correlation** with PERCLOS (frontal EEG captures less information)
- **Faster training** (fewer parameters in conv4 layer)
- **More practical** for real-world applications (fewer electrodes needed)

## Comparison with Original

To compare FP1/FP2-only vs full 17-channel model:

```bash
# Run FP1/FP2 version
python experiment_fp.py --trial 2 --task RGS --log-dir ./logs_fp

# Run original version
python experiment.py --trial 2 --task RGS --log-dir ./logs_full

# Compare results
# logs_fp/trial2_RGS_*.log vs logs_full/trial2_RGS_*.log
```

## Files Created

- `utils_fp.py`: Data loader for FP1/FP2 channels
- `network_fp.py`: VIGNet model adapted for 2 channels
- `experiment_fp.py`: Training script for FP1/FP2 version
- `README_FP.md`: This documentation

## Notes

1. **Same preprocessing**: Features are pre-extracted from SEED-VIG dataset
2. **Same evaluation**: Uses identical metrics and evaluation protocol
3. **Compatible with analysis**: Output format matches original for visualization scripts
4. **Independent**: Can run alongside original experiment.py without conflicts

