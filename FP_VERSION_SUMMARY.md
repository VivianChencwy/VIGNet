# VIGNet FP1/FP2 Version - Implementation Summary

## Created Files

1. **utils_fp.py** (122 lines)
   - Data loader for FP1/FP2 channels only
   - Loads from `Forehead_EEG/EEG_Feature_2Hz` (4 channels)
   - Extracts only channels 2 and 3 (FP1, FP2)
   - Channel mapping (from paper): 0=AFz, 1=FPz, 2=FP1, 3=FP2
   - Output shape: `(N, 2, 25, 1)`

2. **network_fp.py** (77 lines)
   - VIGNet model adapted for 2 channels
   - Modified `conv4`: `(17, 1)` → `(2, 1)`
   - Modified `MHRSSA`: `num_channel=17` → `num_channel=2`
   - All other layers identical to original

3. **experiment_fp.py** (459 lines)
   - Training script using FP1/FP2 data
   - Same hyperparameters as original
   - Output to `./logs_fp/` directory
   - Full command-line interface

4. **test_fp_data.py** (120 lines)
   - Verification script for data loading
   - Tests data shape correctness
   - Verifies no data leakage

5. **README_FP.md**
   - Complete documentation
   - Usage examples
   - Comparison with original

6. **run_fp_example.sh**
   - Quick start script
   - Example commands

## Key Features

### ✓ Data Isolation
- **Independent data source**: `Forehead_EEG` folder (not `EEG_Feature_2Hz`)
- **No information leakage**: Only FP1/FP2, no data from other 15 channels
- **Same CV splits**: Uses identical random seeds (970304) for fair comparison

### ✓ Same Methodology
- **Same features**: de_LDS (Differential Entropy with LDS)
- **Same preprocessing**: Pre-extracted features from SEED-VIG
- **Same CV protocol**: 5-fold cross-validation
- **Same evaluation**: MSE, MAE, RMSE, Pearson correlation
- **Same training**: Adam optimizer, 100 epochs, learning rate 1e-2

### ✓ Compatible Output
- **Same log format**: Compatible with existing visualization scripts
- **Same prediction format**: Can use same analysis tools
- **Separate directory**: `logs_fp/` to avoid confusion with original results

## Data Shape Comparison

| Version | Data Source | Channels | Shape |
|---------|-------------|----------|-------|
| Original | `EEG_Feature_2Hz` | 17 | `(N, 17, 25, 1)` |
| FP Version | `Forehead_EEG/EEG_Feature_2Hz` | 2 | `(N, 2, 25, 1)` |

## Usage Examples

### Quick Test
```bash
cd /home/vivian/eeg/SEED_VIG/VIGNet

# Verify data loading
python test_fp_data.py

# Run single trial
python experiment_fp.py --trial 2 --task RGS
```

### Full Experiment
```bash
# Run all trials
python experiment_fp.py --task RGS

# Or run specific trials
python experiment_fp.py --trial 1 --task RGS
python experiment_fp.py --trial 2 --task RGS
python experiment_fp.py --trial 3 --task RGS
```

### Expected Output
```
Dataset shapes - Train: (567, 2, 25, 1), Valid: (141, 2, 25, 1), Test: (177, 2, 25, 1)
```

## Model Architecture Changes

### Original VIGNet (17 channels)
```python
conv4 = Conv2D(20, (17, 1))  # 17-channel spatial convolution
MHRSSA(x, 10, num_channel=17)  # 17-channel attention
```

### FP Version (2 channels)
```python
conv4 = Conv2D(20, (2, 1))   # 2-channel spatial convolution
MHRSSA(x, 10, num_channel=2)  # 2-channel attention
```

## Performance Expectations

Using only FP1/FP2 (2 channels):
- **Pros**: 
  - Faster training (fewer parameters)
  - More practical (only 2 electrodes needed)
  - Good for real-world applications
  
- **Cons**:
  - Lower correlation expected (less spatial information)
  - May miss information from temporal/posterior regions

## Comparison with Analysis Results

From the FP1/FP2 feature analysis (`analysis/fp_feature_analysis.py`):
- FP1/FP2 `de_LDS` features show **r ≈ -0.50** correlation with PERCLOS
- Top features: `de_LDS_f25` (49Hz), `de_LDS_f24` (47Hz), etc.
- This suggests FP1/FP2 alone contain significant fatigue information

## Verification

✓ **Data loading tested**: All tests passed
✓ **Syntax checked**: All files compile successfully  
✓ **Data isolation verified**: No leakage from 17-channel data
✓ **Shape verified**: Correct `(N, 2, 25, 1)` format
✓ **Compatible with TensorFlow**: Ready to run

## Next Steps

1. **Run training**: Execute `experiment_fp.py` for trials 1-3
2. **Compare results**: Check correlation vs. original 17-channel version
3. **Visualize**: Use same visualization tools on `logs_fp/` output
4. **Analyze**: Compare with feature correlation analysis results

## Files Location

```
VIGNet/
├── experiment_fp.py        # Main training script (FP1/FP2)
├── utils_fp.py              # Data loader (FP1/FP2)
├── network_fp.py            # Model (2 channels)
├── test_fp_data.py          # Verification script
├── README_FP.md             # Documentation
├── FP_VERSION_SUMMARY.md    # This file
├── run_fp_example.sh        # Quick start script
└── logs_fp/                 # Output directory (created on first run)
```

## Contact

For questions or issues with the FP1/FP2 version, refer to:
- `README_FP.md` for detailed usage
- `test_fp_data.py` for data verification
- Original `experiment.py` for comparison

---
**Created**: November 14, 2025
**Purpose**: Enable training VIGNet with only forehead (FP1/FP2) channels
**Status**: ✓ Tested and ready to use

