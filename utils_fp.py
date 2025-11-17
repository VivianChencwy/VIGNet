# Import APIs
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

from scipy.io import loadmat

class load_dataset_fp():
    """
    Dataset loader for FP1/FP2 forehead channels only
    Uses data from Forehead_EEG folder instead of full 17-channel EEG
    """
    def __init__(self, trial, cv, type="de_LDS", reg_label=False):
        self.trial = trial
        self.cv = cv
        self.type = type
        self.reg_label = reg_label

        self.basePath = "../SEED-VIG" # define the data path

    def _find_file(self, directory, prefix):
        """Find file starting with the given prefix in the directory"""
        pattern = os.path.join(self.basePath, directory, "{}_*.mat".format(prefix))
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError("No file found matching pattern: {}".format(pattern))
        return files[0]

    def call(self):
        # Load features from Forehead_EEG folder (4 channels, we use FP1 and FP2)
        feature_folder = "Forehead_EEG/EEG_Feature_2Hz"
        
        feature_file = self._find_file(feature_folder, self.trial)
        try:
            feature = loadmat(feature_file, struct_as_record=False)[self.type]
        except (OSError, KeyError):
            try:
                feature = loadmat(feature_file, struct_as_record=True)[self.type]
            except Exception as e:
                raise RuntimeError(f"Failed to load {feature_file}. Error: {e}")
        
        # Feature shape: (4, 885, 25) - 4 forehead channels
        # Extract only FP1 (channel 0) and FP2 (channel 1)
        feature = feature[[0, 1], :, :]  # Shape: (2, 885, 25)
        
        label_file = self._find_file("perclos_labels", self.trial)
        try:
            label = np.squeeze(loadmat(label_file, struct_as_record=False)["perclos"])
        except (OSError, KeyError):
            try:
                label = np.squeeze(loadmat(label_file, struct_as_record=True)["perclos"])
            except Exception as e:
                raise RuntimeError(f"Failed to load {label_file}. Error: {e}")

        # Create classification labels
        temp = np.zeros(shape=label.shape, dtype=int)
        for i in range(temp.shape[0]):
            if label[i] < 0.35:
                temp[i] = 0  # awake
            elif 0.35 <= label[i] < 0.7:
                temp[i] = 1  # tired
            else:
                temp[i] = 2  # drowsy

        clfLabel = np.eye(3)[temp]  # one-hot encoding / (885,3)
        feature = np.moveaxis(feature, 0, 1)  # feature.shape = (885, 2, 25)

        # We use five fold cross validation
        allIdx = np.random.RandomState(seed=970304).permutation(feature.shape[0])
        amount = int(feature.shape[0] / 5)

        testIdx = allIdx[self.cv * amount:(self.cv + 1) * amount]
        trainIdx = np.setdiff1d(allIdx, testIdx)

        amount = int(trainIdx.shape[0] / 5)
        randIdx = np.random.RandomState(seed=970304 + self.cv).permutation(trainIdx.shape[0])

        validIdx = trainIdx[randIdx[:amount]]
        trainIdx = np.setdiff1d(trainIdx, validIdx)

        trainFeature, validFeature, testFeature \
            = feature[trainIdx, :, :], feature[validIdx, :, :], feature[testIdx, :, :]
        trainLabel, validLabel, testLabel = clfLabel[trainIdx], clfLabel[validIdx], clfLabel[testIdx]
        trainReglabel, validReglabel, testReglabel = label[trainIdx], label[validIdx], label[testIdx]

        if self.reg_label == True:
            trainLabel, validLabel, testLabel = trainReglabel, validReglabel, testReglabel
            trainLabel, validLabel, testLabel \
                = np.expand_dims(trainLabel, -1), np.expand_dims(validLabel,-1), np.expand_dims(testLabel, -1)

        trainFeature, validFeature, testFeature \
            = np.expand_dims(trainFeature, -1), np.expand_dims(validFeature, -1), np.expand_dims(testFeature, -1)

        return trainFeature, trainLabel, validFeature, validLabel, testFeature, testLabel

class load_dataset_fp_no_cv():
    """
    Dataset loader for FP1/FP2 forehead channels only (no cross-validation)
    Uses fixed train/val/test split: 70%/15%/15%
    Uses data from Forehead_EEG folder instead of full 17-channel EEG
    """
    def __init__(self, trial, type="de_LDS", reg_label=False, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=970304):
        self.trial = trial
        self.type = type
        self.reg_label = reg_label
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        self.basePath = "../SEED-VIG" # define the data path

    def _find_file(self, directory, prefix):
        """Find file starting with the given prefix in the directory"""
        pattern = os.path.join(self.basePath, directory, "{}_*.mat".format(prefix))
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError("No file found matching pattern: {}".format(pattern))
        return files[0]

    def _stratified_split_regression(self, features, labels, train_ratio, val_ratio, random_seed):
        """Stratified split for regression by binning labels"""
        n_samples = features.shape[0]
        n_bins = 5  # Split PERCLOS into 5 bins
        
        # Create bins for stratification
        bin_edges = np.linspace(labels.min(), labels.max() + 0.001, n_bins + 1)
        bin_indices = np.digitize(labels, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        train_idx, val_idx, test_idx = [], [], []
        
        # Stratified sampling per bin
        for bin_id in range(n_bins):
            bin_mask = (bin_indices == bin_id)
            bin_samples = np.where(bin_mask)[0]
            
            if len(bin_samples) == 0:
                continue
                
            rng = np.random.RandomState(seed=random_seed + bin_id)
            shuffled = rng.permutation(bin_samples)
            
            n_train = int(len(shuffled) * train_ratio)
            n_val = int(len(shuffled) * val_ratio)
            
            train_idx.extend(shuffled[:n_train])
            val_idx.extend(shuffled[n_train:n_train + n_val])
            test_idx.extend(shuffled[n_train + n_val:])
        
        # Shuffle final indices
        rng = np.random.RandomState(seed=random_seed)
        train_idx = rng.permutation(train_idx)
        val_idx = rng.permutation(val_idx)
        test_idx = rng.permutation(test_idx)
        
        return np.array(train_idx), np.array(val_idx), np.array(test_idx)

    def call(self):
        # Load features from Forehead_EEG folder (4 channels, we use FP1 and FP2)
        feature_folder = "Forehead_EEG/EEG_Feature_2Hz"
        
        feature_file = self._find_file(feature_folder, self.trial)
        try:
            feature = loadmat(feature_file, struct_as_record=False)[self.type]
        except (OSError, KeyError):
            try:
                feature = loadmat(feature_file, struct_as_record=True)[self.type]
            except Exception as e:
                raise RuntimeError(f"Failed to load {feature_file}. Error: {e}")
        
        # Feature shape: (4, 885, 25) - 4 forehead channels
        # Extract only FP1 (channel 0) and FP2 (channel 1)
        feature = feature[[0, 1], :, :]  # Shape: (2, 885, 25)
        
        label_file = self._find_file("perclos_labels", self.trial)
        try:
            label = np.squeeze(loadmat(label_file, struct_as_record=False)["perclos"])
        except (OSError, KeyError):
            try:
                label = np.squeeze(loadmat(label_file, struct_as_record=True)["perclos"])
            except Exception as e:
                raise RuntimeError(f"Failed to load {label_file}. Error: {e}")

        # Create classification labels
        temp = np.zeros(shape=label.shape, dtype=int)
        for i in range(temp.shape[0]):
            if label[i] < 0.35:
                temp[i] = 0  # awake
            elif 0.35 <= label[i] < 0.7:
                temp[i] = 1  # tired
            else:
                temp[i] = 2  # drowsy

        clfLabel = np.eye(3)[temp]  # one-hot encoding / (885,3)
        feature = np.moveaxis(feature, 0, 1)  # feature.shape = (885, 2, 25)

        # Stratified train/val/test split: 70%/15%/15% (balanced across PERCLOS bins)
        trainIdx, valIdx, testIdx = self._stratified_split_regression(
            feature, label, self.train_ratio, self.val_ratio, self.random_seed
        )

        trainFeature, validFeature, testFeature \
            = feature[trainIdx, :, :], feature[valIdx, :, :], feature[testIdx, :, :]
        trainLabel, validLabel, testLabel = clfLabel[trainIdx], clfLabel[valIdx], clfLabel[testIdx]
        trainReglabel, validReglabel, testReglabel = label[trainIdx], label[valIdx], label[testIdx]

        if self.reg_label == True:
            trainLabel, validLabel, testLabel = trainReglabel, validReglabel, testReglabel
            trainLabel, validLabel, testLabel \
                = np.expand_dims(trainLabel, -1), np.expand_dims(validLabel,-1), np.expand_dims(testLabel, -1)

        trainFeature, validFeature, testFeature \
            = np.expand_dims(trainFeature, -1), np.expand_dims(validFeature, -1), np.expand_dims(testFeature, -1)

        return trainFeature, trainLabel, validFeature, validLabel, testFeature, testLabel

class load_dataset_fp_no_cv_f21_25():
    """
    Dataset loader for FP1/FP2 forehead channels only (no cross-validation)
    Uses only de_LDS_f21-25 features (41Hz, 43Hz, 45Hz, 47Hz, 49Hz)
    Uses fixed train/val/test split: 70%/15%/15%
    Uses data from Forehead_EEG folder instead of full 17-channel EEG
    """
    def __init__(self, trial, type="de_LDS", reg_label=False, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=970304):
        self.trial = trial
        self.type = type
        self.reg_label = reg_label
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        self.basePath = "../SEED-VIG" # define the data path

    def _find_file(self, directory, prefix):
        """Find file starting with the given prefix in the directory"""
        pattern = os.path.join(self.basePath, directory, "{}_*.mat".format(prefix))
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError("No file found matching pattern: {}".format(pattern))
        return files[0]

    def _stratified_split_regression(self, features, labels, train_ratio, val_ratio, random_seed):
        """Stratified split for regression by binning labels"""
        n_samples = features.shape[0]
        n_bins = 5  # Split PERCLOS into 5 bins
        
        # Create bins for stratification
        bin_edges = np.linspace(labels.min(), labels.max() + 0.001, n_bins + 1)
        bin_indices = np.digitize(labels, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        train_idx, val_idx, test_idx = [], [], []
        
        # Stratified sampling per bin
        for bin_id in range(n_bins):
            bin_mask = (bin_indices == bin_id)
            bin_samples = np.where(bin_mask)[0]
            
            if len(bin_samples) == 0:
                continue
                
            rng = np.random.RandomState(seed=random_seed + bin_id)
            shuffled = rng.permutation(bin_samples)
            
            n_train = int(len(shuffled) * train_ratio)
            n_val = int(len(shuffled) * val_ratio)
            
            train_idx.extend(shuffled[:n_train])
            val_idx.extend(shuffled[n_train:n_train + n_val])
            test_idx.extend(shuffled[n_train + n_val:])
        
        # Shuffle final indices
        rng = np.random.RandomState(seed=random_seed)
        train_idx = rng.permutation(train_idx)
        val_idx = rng.permutation(val_idx)
        test_idx = rng.permutation(test_idx)
        
        return np.array(train_idx), np.array(val_idx), np.array(test_idx)

    def call(self):
        # Load features from Forehead_EEG folder (4 channels, we use FP1 and FP2)
        feature_folder = "Forehead_EEG/EEG_Feature_2Hz"
        
        feature_file = self._find_file(feature_folder, self.trial)
        try:
            feature = loadmat(feature_file, struct_as_record=False)[self.type]
        except (OSError, KeyError):
            try:
                feature = loadmat(feature_file, struct_as_record=True)[self.type]
            except Exception as e:
                raise RuntimeError(f"Failed to load {feature_file}. Error: {e}")
        
        # Feature shape: (4, 885, 25) - 4 forehead channels
        # Extract only FP1 (channel 0) and FP2 (channel 1)
        feature = feature[[0, 1], :, :]  # Shape: (2, 885, 25)
        
        # Extract only f21-25 features (indices 20-24, 0-based)
        # f21=41Hz, f22=43Hz, f23=45Hz, f24=47Hz, f25=49Hz
        feature = feature[:, :, 20:25]  # Shape: (2, 885, 5)
        
        label_file = self._find_file("perclos_labels", self.trial)
        try:
            label = np.squeeze(loadmat(label_file, struct_as_record=False)["perclos"])
        except (OSError, KeyError):
            try:
                label = np.squeeze(loadmat(label_file, struct_as_record=True)["perclos"])
            except Exception as e:
                raise RuntimeError(f"Failed to load {label_file}. Error: {e}")

        # Create classification labels
        temp = np.zeros(shape=label.shape, dtype=int)
        for i in range(temp.shape[0]):
            if label[i] < 0.35:
                temp[i] = 0  # awake
            elif 0.35 <= label[i] < 0.7:
                temp[i] = 1  # tired
            else:
                temp[i] = 2  # drowsy

        clfLabel = np.eye(3)[temp]  # one-hot encoding / (885,3)
        feature = np.moveaxis(feature, 0, 1)  # feature.shape = (885, 2, 5)

        # Stratified train/val/test split: 70%/15%/15% (balanced across PERCLOS bins)
        trainIdx, valIdx, testIdx = self._stratified_split_regression(
            feature, label, self.train_ratio, self.val_ratio, self.random_seed
        )

        trainFeature, validFeature, testFeature \
            = feature[trainIdx, :, :], feature[valIdx, :, :], feature[testIdx, :, :]
        trainLabel, validLabel, testLabel = clfLabel[trainIdx], clfLabel[valIdx], clfLabel[testIdx]
        trainReglabel, validReglabel, testReglabel = label[trainIdx], label[valIdx], label[testIdx]

        if self.reg_label == True:
            trainLabel, validLabel, testLabel = trainReglabel, validReglabel, testReglabel
            trainLabel, validLabel, testLabel \
                = np.expand_dims(trainLabel, -1), np.expand_dims(validLabel,-1), np.expand_dims(testLabel, -1)

        trainFeature, validFeature, testFeature \
            = np.expand_dims(trainFeature, -1), np.expand_dims(validFeature, -1), np.expand_dims(testFeature, -1)

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

