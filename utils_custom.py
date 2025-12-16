# Import APIs
import numpy as np
import tensorflow as tf
import os


class load_dataset_custom():
    """
    Dataset loader for custom EEG + Eye state data
    Uses preprocessed DE features and PERCLOS labels from preprocess_custom_data.py
    Uses BLOCK-WISE RANDOM split to avoid data leakage while maintaining PERCLOS distribution
    """
    def __init__(self, data_dir, reg_label=False, train_ratio=0.7, val_ratio=0.15, 
                 test_ratio=0.15, block_size=8, gap_size=2, random_seed=42):
        """
        Initialize dataset loader
        
        Args:
            data_dir: Directory containing processed .npy files
            reg_label: If True, use PERCLOS for regression; else use classification labels
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            block_size: Number of consecutive windows per block (default 8 = ~32 seconds)
            gap_size: Number of windows to discard between blocks to prevent overlap (default 2)
            random_seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.reg_label = reg_label
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.block_size = block_size
        self.gap_size = gap_size
        self.random_seed = random_seed

    def _blockwise_random_split(self, n_samples, train_ratio, val_ratio):
        """
        Block-wise random split to avoid data leakage while preserving PERCLOS distribution.
        
        Why this works:
        - EEG windows overlap (8s window, 4s stride = 50% overlap)
        - Adjacent windows share data, so random split of individual windows causes leakage
        - Solution: Group windows into non-overlapping BLOCKS, then randomly assign blocks
        - Gaps between blocks ensure no data sharing between train/val/test
        
        Example with block_size=8, gap_size=2:
        Samples: [0,1,2,3,4,5,6,7] [gap] [10,11,12,13,14,15,16,17] [gap] [20,...]
                      Block 0                    Block 1                  Block 2
        
        Then blocks are randomly shuffled and assigned to train/val/test.
        
        Returns:
            trainIdx, valIdx, testIdx: arrays of indices
        """
        np.random.seed(self.random_seed)
        
        # Create blocks
        blocks = []
        unit_size = self.block_size + self.gap_size
        
        start_idx = 0
        while start_idx + self.block_size <= n_samples:
            block_indices = list(range(start_idx, start_idx + self.block_size))
            blocks.append(block_indices)
            start_idx += unit_size  # Skip gap
        
        n_blocks = len(blocks)
        print(f"Created {n_blocks} blocks (block_size={self.block_size}, gap={self.gap_size})")
        
        # Shuffle blocks
        block_order = np.random.permutation(n_blocks)
        
        # Assign blocks to train/val/test
        n_train_blocks = int(n_blocks * train_ratio)
        n_val_blocks = int(n_blocks * val_ratio)
        
        train_block_ids = block_order[:n_train_blocks]
        val_block_ids = block_order[n_train_blocks:n_train_blocks + n_val_blocks]
        test_block_ids = block_order[n_train_blocks + n_val_blocks:]
        
        # Collect indices from blocks
        trainIdx = np.array([idx for bid in train_block_ids for idx in blocks[bid]])
        valIdx = np.array([idx for bid in val_block_ids for idx in blocks[bid]])
        testIdx = np.array([idx for bid in test_block_ids for idx in blocks[bid]])
        
        # Sort indices within each split (for consistency, not required)
        trainIdx = np.sort(trainIdx)
        valIdx = np.sort(valIdx)
        testIdx = np.sort(testIdx)
        
        return trainIdx, valIdx, testIdx

    def call(self):
        """Load and split dataset"""
        # Load processed data
        features_path = os.path.join(self.data_dir, 'de_features.npy')
        perclos_path = os.path.join(self.data_dir, 'perclos_labels.npy')
        clf_path = os.path.join(self.data_dir, 'clf_labels.npy')
        
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")
        if not os.path.exists(perclos_path):
            raise FileNotFoundError(f"PERCLOS labels file not found: {perclos_path}")
        
        # Load features: shape (n_windows, n_channels, n_bands)
        feature = np.load(features_path)
        perclos = np.load(perclos_path)
        
        print(f"Loaded features: {feature.shape}, PERCLOS: {perclos.shape}")
        print(f"PERCLOS range: [{perclos.min():.4f}, {perclos.max():.4f}]")
        
        # Load or create classification labels
        if os.path.exists(clf_path):
            clf_labels = np.load(clf_path)
        else:
            # Create classification labels from PERCLOS
            clf_labels = np.zeros(len(perclos), dtype=int)
            clf_labels[perclos >= 0.35] = 1  # tired
            clf_labels[perclos >= 0.7] = 2   # drowsy
        
        # One-hot encode classification labels
        clfLabel = np.eye(3)[clf_labels]
        
        # BLOCK-WISE RANDOM split (preserves PERCLOS distribution, NO data leakage)
        n_samples = feature.shape[0]
        trainIdx, valIdx, testIdx = self._blockwise_random_split(
            n_samples, self.train_ratio, self.val_ratio
        )
        
        # Calculate discarded samples (gaps)
        total_used = len(trainIdx) + len(valIdx) + len(testIdx)
        discarded = n_samples - total_used
        
        print(f"Block-wise random split (no data leakage, balanced PERCLOS):")
        print(f"  Train: {len(trainIdx)} samples ({len(trainIdx)/total_used*100:.1f}%)")
        print(f"  Valid: {len(valIdx)} samples ({len(valIdx)/total_used*100:.1f}%)")
        print(f"  Test:  {len(testIdx)} samples ({len(testIdx)/total_used*100:.1f}%)")
        print(f"  Discarded (gaps): {discarded} samples")

        trainFeature = feature[trainIdx, :, :]
        validFeature = feature[valIdx, :, :]
        testFeature = feature[testIdx, :, :]
        
        trainLabel = clfLabel[trainIdx]
        validLabel = clfLabel[valIdx]
        testLabel = clfLabel[testIdx]
        
        trainReglabel = perclos[trainIdx]
        validReglabel = perclos[valIdx]
        testReglabel = perclos[testIdx]
        
        # Print PERCLOS distribution for each split
        print(f"  Train PERCLOS: mean={trainReglabel.mean():.4f}, std={trainReglabel.std():.4f}")
        print(f"  Valid PERCLOS: mean={validReglabel.mean():.4f}, std={validReglabel.std():.4f}")
        print(f"  Test PERCLOS:  mean={testReglabel.mean():.4f}, std={testReglabel.std():.4f}")

        if self.reg_label:
            trainLabel = np.expand_dims(trainReglabel, -1)
            validLabel = np.expand_dims(validReglabel, -1)
            testLabel = np.expand_dims(testReglabel, -1)

        # Add channel dimension: (n_windows, n_channels, n_bands) -> (n_windows, n_channels, n_bands, 1)
        trainFeature = np.expand_dims(trainFeature, -1)
        validFeature = np.expand_dims(validFeature, -1)
        testFeature = np.expand_dims(testFeature, -1)

        return trainFeature, trainLabel, validFeature, validLabel, testFeature, testLabel


def classification_loss(y, y_pred):
    return tf.keras.losses.binary_crossentropy(y, y_pred)


def regression_loss(y, y_pred):
    return tf.keras.losses.MSE(y, y_pred)


@tf.function
def grad(model, inputs, labels, mode):
    with tf.GradientTape() as tape:
        y_hat = model(inputs)

        if mode == "CLF":
            loss = classification_loss(y=labels, y_pred=y_hat)
        elif mode == "RGS":
            loss = regression_loss(y=labels, y_pred=y_hat)

    grad = tape.gradient(loss, model.trainable_variables)
    return loss, grad


