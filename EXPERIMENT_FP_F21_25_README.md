# Experiment FP F21-25 - 使用说明

## 概述

`experiment_fp_f21_25.py` 是专门使用 **de_LDS_f21-25** 特征（41Hz, 43Hz, 45Hz, 47Hz, 49Hz）的实验脚本。这些是分析中发现与疲劳相关性最强的高频gamma频段特征。

## 主要特点

1. **无交叉验证**: 使用固定的 70%/15%/15% train/val/test 分割
2. **仅使用高相关性特征**: 只使用 de_LDS_f21-25（5个频率特征）
3. **适配的网络架构**: 使用 `network_fp_f21_25.py`，针对5个频率特征优化

## 特征说明

### 使用的特征
- **de_LDS_f21**: 41Hz
- **de_LDS_f22**: 43Hz
- **de_LDS_f23**: 45Hz
- **de_LDS_f24**: 47Hz
- **de_LDS_f25**: 49Hz

### 特征相关性（来自分析）
根据 `analysis/outputs/fp_analysis_summary.md`:
- **de_LDS_f25** (49Hz): r = -0.5016
- **de_LDS_f24** (47Hz): r = -0.5001
- **de_LDS_f22** (43Hz): r = -0.4942
- **de_LDS_f23** (45Hz): r = -0.4906
- **de_LDS_f21** (41Hz): r = -0.4864

这些是**所有特征中与疲劳相关性最强的5个**。

## 数据分割

- **Train**: 70% (约620个样本)
- **Validation**: 15% (约133个样本)
- **Test**: 15% (约132个样本)

使用固定随机种子 `970304` 确保可重复性。

## 网络架构适配

由于输入从25个频率特征减少到5个，网络架构进行了适配：

### 主要修改 (`network_fp_f21_25.py`)
- **卷积核大小**: 从 `(1, 5)` 改为 `(1, 3)`，使用 `padding='same'`
- **DepthwiseConv2D**: 从 `(1, 5)` 改为 `(1, 3)`，使用 `padding='same'`
- 保持其他架构不变（2个通道，MHRSSA模块等）

## 使用方法

### 运行单个trial

```bash
cd /home/vivian/eeg/SEED_VIG/VIGNet
python experiment_fp_f21_25.py --trial 1
```

### 运行所有trials (1-21)

```bash
python experiment_fp_f21_25.py
```

### 自定义参数

```bash
# 指定任务类型
python experiment_fp_f21_25.py --trial 1 --task RGS  # 回归任务
python experiment_fp_f21_25.py --trial 1 --task CLF  # 分类任务

# 自定义日志目录
python experiment_fp_f21_25.py --log-dir ./my_logs

# 不保存预测结果
python experiment_fp_f21_25.py --no-save-predictions
```

## 输出文件

### 日志文件
- `trial{N}_{TASK}_{TIMESTAMP}.log` - 每个trial的详细训练日志
- `training_summary_{TIMESTAMP}.log` - 所有trials的汇总日志

### 预测文件（如果启用）
- `predictions/trial{N}_validation.npy` - 验证集预测
- `predictions/trial{N}_test.npy` - 测试集预测

## 超参数

与 `experiment_fp_no_cv.py` 保持一致：
- Learning rate: `0.005`
- Epochs: `500`
- Batch size: `8`
- Early stopping patience: `50`

## 文件结构

### 新增/修改的文件

1. **`utils_fp.py`**
   - 新增: `load_dataset_fp_no_cv_f21_25` 类
   - 功能: 加载数据并只提取 f21-25 特征

2. **`network_fp_f21_25.py`** (新建)
   - 适配的网络架构，针对5个频率特征优化

3. **`experiment_fp_f21_25.py`** (新建)
   - 主实验脚本

## 优势

1. **特征选择**: 只使用与疲劳相关性最强的5个特征
2. **更快的训练**: 输入维度从 (2, 25) 减少到 (2, 5)
3. **更小的模型**: 网络参数更少，训练更快
4. **生物学意义**: 专注于gamma频段（31-50Hz），与认知处理相关

## 预期性能

根据特征分析，这些特征与疲劳的相关系数约为 -0.49 到 -0.50，是**所有特征中相关性最强的**。预期模型性能应该与使用全部25个特征相当或更好。

## 与完整特征版本的对比

| 特性 | experiment_fp_no_cv | experiment_fp_f21_25 |
|------|---------------------|----------------------|
| 频率特征数 | 25 (f1-f25) | 5 (f21-f25) |
| 输入维度 | (2, 25) | (2, 5) |
| 网络架构 | network_fp | network_fp_f21_25 |
| 特征相关性 | 混合 | 最强 (r ≈ -0.49) |
| 训练速度 | 标准 | 更快 |

## 注意事项

1. **网络架构**: 必须使用 `network_fp_f21_25.py`，不能使用原始的 `network_fp.py`
2. **数据加载**: 使用 `load_dataset_fp_no_cv_f21_25` 类
3. **特征选择**: 只使用高频gamma频段特征，可能对某些trial效果不同

## 文件位置

- 脚本: `/home/vivian/eeg/SEED_VIG/VIGNet/experiment_fp_f21_25.py`
- 网络: `/home/vivian/eeg/SEED_VIG/VIGNet/network_fp_f21_25.py`
- 工具类: `/home/vivian/eeg/SEED_VIG/VIGNet/utils_fp.py` (新增类)
- 默认日志目录: `./logs_fp_f21_25/`

## 快速开始

```bash
cd /home/vivian/eeg/SEED_VIG/VIGNet
python experiment_fp_f21_25.py --trial 1
```

这将使用最强的5个频率特征训练模型，专注于高频gamma频段的differential entropy特征。

