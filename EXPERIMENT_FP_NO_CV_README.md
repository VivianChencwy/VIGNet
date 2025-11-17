# Experiment FP No CV - 使用说明

## 概述

`experiment_fp_no_cv.py` 是 `experiment_fp.py` 的简化版本，不使用5-fold交叉验证，而是使用固定的 train/val/test 分割（70%/15%/15%）。

## 主要区别

### experiment_fp.py (原版本)
- 使用5-fold交叉验证
- 每个trial运行5次（5个CV folds）
- 输出5个fold的平均结果
- 预测文件命名：`trial{N}_cv{F}_validation.npy`, `trial{N}_cv{F}_test.npy`

### experiment_fp_no_cv.py (新版本)
- 不使用交叉验证
- 每个trial只运行1次
- 使用固定的70%/15%/15%数据分割
- 预测文件命名：`trial{N}_validation.npy`, `trial{N}_test.npy`

## 数据分割

- **Train**: 70% (约620个样本)
- **Validation**: 15% (约133个样本)
- **Test**: 15% (约132个样本)

使用固定随机种子 `970304` 确保可重复性。

## 使用方法

### 运行单个trial

```bash
cd /home/vivian/eeg/SEED_VIG/VIGNet
python experiment_fp_no_cv.py --trial 1
```

### 运行所有trials (1-21)

```bash
python experiment_fp_no_cv.py
```

### 自定义参数

```bash
# 指定任务类型
python experiment_fp_no_cv.py --trial 1 --task RGS  # 回归任务
python experiment_fp_no_cv.py --trial 1 --task CLF  # 分类任务

# 自定义日志目录
python experiment_fp_no_cv.py --log-dir ./my_logs

# 不保存预测结果
python experiment_fp_no_cv.py --no-save-predictions
```

## 输出文件

### 日志文件
- `trial{N}_{TASK}_{TIMESTAMP}.log` - 每个trial的详细训练日志
- `training_summary_{TIMESTAMP}.log` - 所有trials的汇总日志

### 预测文件（如果启用）
- `predictions/trial{N}_validation.npy` - 验证集预测
- `predictions/trial{N}_test.npy` - 测试集预测

## 代码结构

### 新增类

**utils_fp.py**:
- `load_dataset_fp_no_cv` - 不使用CV的数据加载类

**experiment_fp_no_cv.py**:
- `experiment_fp_no_cv` - 不使用CV的实验类

## 超参数

与 `experiment_fp.py` 保持一致：
- Learning rate: `1e-3`
- Epochs: `500`
- Batch size: `5`
- Early stopping patience: `20`

## 优势

1. **更快的训练**：每个trial只运行1次，而不是5次
2. **更简单的分析**：不需要计算CV平均值
3. **适合快速实验**：快速验证模型和超参数

## 注意事项

1. **没有交叉验证**：结果可能对数据分割更敏感
2. **单次运行**：无法评估模型性能的稳定性（没有CV的标准差）
3. **固定分割**：使用固定的随机种子，每次运行使用相同的数据分割

## 适用场景

- 快速原型验证
- 超参数搜索
- 模型架构实验
- 不需要CV稳定性的场景

## 文件位置

- 脚本: `/home/vivian/eeg/SEED_VIG/VIGNet/experiment_fp_no_cv.py`
- 工具类: `/home/vivian/eeg/SEED_VIG/VIGNet/utils_fp.py` (新增 `load_dataset_fp_no_cv` 类)
- 默认日志目录: `./logs_fp_no_cv/`

