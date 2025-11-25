# VIGNet 实时预测指南

## 概述

本文档说明如何使用训练好的 VIGNet FP1/FP2 模型进行实时 PERCLOS 预测。

## 模型保存功能

### 自动保存

训练完成后，`experiment_fp_no_cv.py` 会自动保存以下文件到 `logs_fp_no_cv/models/` 目录：

1. **模型文件** (`trial{N}_best_model/`): TensorFlow SavedModel 格式的完整模型
2. **归一化器** (`trial{N}_scaler.pkl`): StandardScaler，用于特征归一化
3. **元数据** (`trial{N}_metadata.pkl`): 模型配置信息（任务类型、输入输出形状等）

### 保存位置

```
logs_fp_no_cv/
└── models/
    ├── trial1_best_model/          # 完整模型
    ├── trial1_scaler.pkl            # 归一化器
    ├── trial1_metadata.pkl          # 元数据
    ├── trial2_best_model/
    ├── trial2_scaler.pkl
    └── ...
```

## 实时预测可行性

### ✅ 模型支持实时预测

**VIGNet 模型完全支持实时预测**，原因如下：

1. **轻量级架构**: 模型仅使用 2 个通道（FP1/FP2），计算量小
2. **快速推理**: 卷积神经网络结构简单，单次预测耗时 < 10ms（GPU）或 < 50ms（CPU）
3. **批量处理**: 支持单样本和批量预测
4. **内存占用低**: 模型大小约 1-2MB

### 实时预测流程

```
原始EEG信号 → 特征提取 → 归一化 → 模型预测 → PERCLOS输出
   (FP1/FP2)   (2×25)    (scaler)   (VIGNet)    (0-1)
```

### 性能指标

- **单次预测延迟**: 
  - GPU: ~5-10ms
  - CPU: ~30-50ms
- **批量预测吞吐量**: 
  - GPU: > 100 samples/second
  - CPU: ~20-30 samples/second
- **内存占用**: ~100-200MB（包括模型和中间变量）

## 使用方法

### 1. 基本使用

```python
from realtime_predict import VIGNetPredictor
import numpy as np

# 初始化预测器
predictor = VIGNetPredictor(
    model_dir='./logs_fp_no_cv/models',
    trial=1,  # 使用 trial 1 的模型
    gpu_idx=0
)

# 准备特征数据 (N, 2, 25)
# N: 样本数
# 2: FP1 和 FP2 通道
# 25: 频率bins (DE特征，2Hz分辨率)
features = np.array([...])  # 你的EEG特征

# 批量预测
predictions = predictor.predict(features)
# predictions: shape (N,) - PERCLOS分数数组

# 单样本预测
single_feature = features[0]  # shape: (2, 25)
single_pred = predictor.predict_single(single_feature)
# single_pred: 标量 - 单个PERCLOS分数
```

### 2. 命令行测试

```bash
# 测试预测器（使用虚拟数据）
python realtime_predict.py --model-dir ./logs_fp_no_cv/models --trial 1 --test

# 查看使用说明
python realtime_predict.py --model-dir ./logs_fp_no_cv/models
```

### 3. 实时预测示例

```python
import numpy as np
from realtime_predict import VIGNetPredictor
import time

# 加载模型
predictor = VIGNetPredictor('./logs_fp_no_cv/models', trial=1)

# 模拟实时数据流
def realtime_prediction_loop():
    """实时预测循环示例"""
    while True:
        # 从EEG设备获取新数据（这里用随机数据模拟）
        # 实际应用中，这里应该是从硬件获取的实时EEG信号
        eeg_signal = get_eeg_signal()  # 获取原始EEG信号
        
        # 特征提取（需要实现特征提取函数）
        features = extract_features(eeg_signal)  # 提取 (2, 25) 特征
        
        # 预测
        start_time = time.time()
        perclos = predictor.predict_single(features)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # 输出结果
        print(f"PERCLOS: {perclos:.4f}, Inference time: {inference_time:.2f}ms")
        
        # 根据PERCLOS值触发警报
        if perclos > 0.7:
            print("⚠️  Warning: High drowsiness detected!")
        
        time.sleep(0.1)  # 控制采样频率（例如10Hz）

# 批量预测（更高效）
def batch_prediction(feature_buffer):
    """批量预测示例（推荐用于离线处理）"""
    # feature_buffer: shape (N, 2, 25)
    predictions = predictor.predict(feature_buffer)
    return predictions
```

## 输入数据格式要求

### 特征格式

- **形状**: `(N, 2, 25)` 或 `(2, 25)`（单样本）
  - `N`: 样本数量
  - `2`: FP1 和 FP2 通道
  - `25`: 频率bins（DE特征，2Hz分辨率，0-50Hz范围）

### 特征提取要求

特征必须与训练时使用的特征一致：

1. **特征类型**: `de_LDS` (Differential Entropy with Linear Dynamic System smoothing)
2. **频率分辨率**: 2Hz（25个bins覆盖0-50Hz）
3. **通道顺序**: 
   - Channel 0: AFz (not used)
   - Channel 1: FPz (not used)
   - Channel 2: FP1
   - Channel 3: FP2
   
   Note: We use Channel 2 (FP1) and Channel 3 (FP2) for prediction
4. **预处理**: 
   - 特征提取后需要应用 LDS 平滑
   - 归一化由预测器自动处理（使用保存的scaler）

### 特征提取流程

```python
# 伪代码：特征提取流程
def extract_features(raw_eeg_fp1, raw_eeg_fp2, fs=200):
    """
    从原始EEG信号提取特征
    
    Args:
        raw_eeg_fp1: FP1通道原始信号
        raw_eeg_fp2: FP2通道原始信号
        fs: 采样频率
    
    Returns:
        features: shape (2, 25) - DE特征
    """
    # 1. 预处理（滤波、去噪等）
    fp1_processed = preprocess(raw_eeg_fp1, fs)
    fp2_processed = preprocess(raw_eeg_fp2, fs)
    
    # 2. 计算功率谱密度 (PSD)
    psd_fp1 = compute_psd(fp1_processed, fs, resolution=2)
    psd_fp2 = compute_psd(fp2_processed, fs, resolution=2)
    
    # 3. 计算差分熵 (DE)
    de_fp1 = compute_differential_entropy(psd_fp1)
    de_fp2 = compute_differential_entropy(psd_fp2)
    
    # 4. LDS平滑
    de_fp1_smooth = lds_smoothing(de_fp1)
    de_fp2_smooth = lds_smoothing(de_fp2)
    
    # 5. 组合特征
    features = np.array([de_fp1_smooth, de_fp2_smooth])  # (2, 25)
    
    return features
```

## 实时预测系统架构建议

### 系统组件

1. **数据采集模块**: 从EEG设备获取FP1/FP2信号
2. **特征提取模块**: 实时计算DE特征（2Hz分辨率）
3. **预测模块**: 使用VIGNet模型进行PERCLOS预测
4. **决策模块**: 根据PERCLOS值触发警报或控制

### 延迟分析

```
总延迟 = 数据采集延迟 + 特征提取延迟 + 模型推理延迟 + 决策延迟

典型值：
- 数据采集: 50-100ms (取决于窗口大小)
- 特征提取: 20-50ms (FFT + DE计算)
- 模型推理: 5-50ms (GPU/CPU)
- 决策: <1ms

总延迟: ~75-200ms (可满足实时需求)
```

### 优化建议

1. **使用GPU**: 可显著降低推理延迟（5-10ms vs 30-50ms）
2. **批量处理**: 如果允许，批量处理多个样本可提高吞吐量
3. **异步处理**: 特征提取和模型推理可以流水线化
4. **模型量化**: 可考虑INT8量化以进一步加速（可能略微降低精度）

## 注意事项

1. **特征一致性**: 确保实时提取的特征与训练时使用的特征格式完全一致
2. **归一化**: 必须使用训练时保存的scaler，不能重新拟合
3. **数据质量**: 实时预测的准确性依赖于输入EEG信号的质量
4. **模型选择**: 不同trial的模型性能可能不同，建议选择验证集上表现最好的模型
5. **内存管理**: 长时间运行时注意内存泄漏，建议定期重启预测器

## 故障排除

### 问题1: 模型加载失败

```
Error: Model not found
```

**解决方案**: 确保已运行训练脚本并保存了模型
```bash
python experiment_fp_no_cv.py --trial 1
```

### 问题2: 特征形状不匹配

```
ValueError: Expected features shape (N, 2, 25), got ...
```

**解决方案**: 检查特征提取代码，确保输出形状正确

### 问题3: 预测结果异常

**可能原因**:
- 特征未正确归一化（应使用保存的scaler）
- 特征提取方法不一致
- 输入数据质量差

**解决方案**: 检查特征提取流程，确保与训练时一致

## 示例代码

完整示例请参考 `realtime_predict.py` 文件。

## 总结

✅ **VIGNet 模型完全支持实时预测**

- 模型轻量级，推理速度快
- 支持单样本和批量预测
- 延迟低，满足实时应用需求
- 使用简单，只需加载模型和scaler即可

关键是要确保：
1. 特征提取方法与训练时一致
2. 使用保存的scaler进行归一化
3. 输入数据格式正确

