# VIGNet Paper Reproduction Guide

## ✅ All Fixes Applied

### Critical Bug Fixed
- ✅ Regression loss function bug fixed in `utils.py` and `utils_fp.py`
- ✅ Learning rate set to 1e-2 (0.01) for alignment with paper
- ✅ All hyperparameters match paper specifications
- ✅ Command-line interface ready for flexible execution

## Quick Start

### Run Single Trial (for testing)
```bash
cd /home/vivian/eeg/SEED_VIG/VIGNet
python experiment.py --trial 2 --task RGS
```

### Run All Trials (for paper reproduction)
```bash
cd /home/vivian/eeg/SEED_VIG/VIGNet
python experiment.py --task RGS
```

This will:
- Run all 23 trials automatically
- Perform 5-fold cross-validation per trial
- Save logs to `./logs/` directory
- Generate prediction files for visualization
- Create a summary log with all results

### Run FP1/FP2 Only Version
```bash
python experiment_fp.py --task RGS
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--trial N` | Run single trial (1-23) | None (runs all) |
| `--task TASK` | Task type: RGS or CLF | RGS |
| `--log-dir DIR` | Log directory | ./logs |
| `--no-save-predictions` | Don't save predictions | False (saves by default) |

## Expected Results

### Before Bug Fix
```
Trial 2 Results (with bug):
  Test RMSE: 0.148259 ± 0.040374
  Test Correlation: 0.424878 ± 0.298744
```

### After Bug Fix
Expected significant improvement:
- **RMSE**: Should decrease significantly (target: ~0.04 average across all trials)
- **Correlation**: Should increase significantly
- **Training loss**: Should converge smoothly using MSE instead of binary cross-entropy

## Understanding the Results

### Per-Trial Results
Each trial log file contains:
- Training loss per epoch (100 epochs)
- Validation metrics for each CV fold
- Test metrics for each CV fold
- Average metrics across 5 CV folds

### Overall Results (Paper Comparison)
The paper's reported RMSE=0.04 is likely:
1. **Average across all 23 trials**
2. **Test set performance** (not training or validation)
3. **Using the same 5-fold CV protocol**

To calculate:
```python
# After running all trials, compute:
all_trial_rmse = []  # Collect from each trial's summary
average_rmse = mean(all_trial_rmse)
std_rmse = std(all_trial_rmse)
```

## Hyperparameters (After Fix)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1e-2 | 0.01, Adam optimizer |
| Epochs | 100 | Standard for this task |
| Batch size | 5 | num_batches parameter |
| Loss function | MSE | ✅ **NOW CORRECT** for RGS |
| Features | de_LDS | Differential Entropy with LDS |
| Channels | 17 | Full EEG (or 2 for FP version) |
| CV folds | 5 | Within-subject validation |

## Data Details (from SEED-VIG)

- **Total trials**: 23 subjects
- **Samples per trial**: 885
- **Sampling rate**: 200 Hz → 2 Hz (after feature extraction)
- **Frequency bands**: 25 (1-50 Hz with 2 Hz resolution)
- **EEG channels**: 17 (or 4 forehead for FP version)
- **Labels**: PERCLOS values (0-1, continuous)

## File Structure After Running

```
VIGNet/
├── logs/
│   ├── training_summary_YYYYMMDD_HHMMSS.log   # Overall summary
│   ├── trial1_RGS_YYYYMMDD_HHMMSS.log         # Per-trial logs
│   ├── trial2_RGS_YYYYMMDD_HHMMSS.log
│   ├── ...
│   └── predictions/                            # Prediction files
│       ├── trial1_cv0_test.npy
│       ├── trial1_cv0_validation.npy
│       └── ...
└── logs_fp/                                     # FP1/FP2 version logs
```

## Visualization

After running experiments, visualize results:

```bash
# For a specific trial log
python visualize_log.py  # (if you update the hardcoded log path)

# Or use the analysis scripts in ../analysis/
```

## Common Issues and Solutions

### Issue: Training loss stays high (~0.6)
**Cause**: This was the bug! You were using binary cross-entropy for regression.
**Solution**: ✅ **Fixed** - Now uses MSE loss correctly.

### Issue: Poor correlation
**Cause**: Incorrect loss function preventing proper learning.
**Solution**: ✅ **Fixed** - Proper MSE loss should improve correlation significantly.

### Issue: Results differ from paper
**Possible reasons**:
1. Different random seed (currently 970304)
2. Need to average across all 23 trials
3. May need to tune learning rate further
4. Check if paper uses any data augmentation

## Verification Steps

1. **Check training loss**:
   - Should decrease smoothly
   - Should be < 0.1 by epoch 100
   - Should converge, not oscillate

2. **Check test RMSE**:
   - Single trial: Should be < 0.15
   - All trials average: Should approach 0.04

3. **Check correlation**:
   - Should be > 0.5 for good performance
   - Higher is better

## Next Steps

### Short-term (Testing)
```bash
# Quick test with one trial
python experiment.py --trial 2 --task RGS --log-dir ./logs_test

# Check the log file
tail -50 ./logs_test/trial2_RGS_*.log
```

### Long-term (Full Reproduction)
```bash
# Run all trials (takes several hours)
python experiment.py --task RGS --log-dir ./logs_paper_reproduction

# Analyze results
# Extract RMSE from all trial logs and compute average
```

## Important Notes

1. **Bug Impact**: The previous bug made it impossible to properly train for regression. With MSE loss, results should be dramatically better.

2. **Computation Time**: 
   - Single trial (~5 CV folds): ~15-30 minutes
   - All 23 trials: ~6-12 hours (depending on GPU)

3. **GPU Recommended**: Training on CPU will be much slower.

4. **Reproducibility**: Same random seed (970304) ensures reproducible CV splits.

## Modified Files Summary

| File | Lines Changed | Change Description |
|------|---------------|-------------------|
| `utils.py` | 149 | ✅ Fixed: regression_loss for RGS |
| `utils_fp.py` | 112 | ✅ Fixed: regression_loss for RGS |
| `experiment.py` | 58 | ✅ Changed: lr 1e-3 → 1e-2 |
| `experiment_fp.py` | 62-64 | ✅ Reset: lr, epochs, batches to paper values |

## Success Criteria

The bug fix is successful if:
- ✅ Training loss converges smoothly to < 0.1
- ✅ Test RMSE < 0.15 per trial (significantly better than 0.148)
- ✅ Test correlation > 0.5 (significantly better than 0.42)
- ✅ Average RMSE across 23 trials ≈ 0.04 (paper result)

## Contact & References

**Dataset**: https://bcmi.sjtu.edu.cn/~seed/seed-vig.html
**Paper**: VIGNet: A Deep Convolutional Neural Network for EEG-based Driver Vigilance Estimation
**Repository**: This implementation

For questions about the bug fix or reproduction, refer to `BUG_FIX_SUMMARY.md`.

---
**Status**: ✅ Ready for paper reproduction
**Last Updated**: November 14, 2025
**Bug Fix**: Regression loss function corrected

