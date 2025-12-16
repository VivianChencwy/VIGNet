"""
Experiment script for custom EEG + Eye state data

This script automatically preprocesses eeg_eye_merged.csv and trains the VIGNet model.
It handles the complete pipeline from raw data to trained model.

Usage:
    # Automatic preprocessing + training (if preprocessing data doesn't exist)
    python experiment_custom.py --input /path/to/eeg_eye_merged.csv --task RGS
    
    # Or use preprocessed data directly
    python experiment_custom.py --data-dir /path/to/processed --task RGS
"""

import utils_custom as utils
import network_custom as network
# Import preprocessing functions
from preprocess_custom_data import process_merged_data

# Import APIs
import os
import logging
from datetime import datetime
# Disable XLA before importing TensorFlow to avoid libdevice issues
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"
os.environ["TF_DISABLE_XLA"] = "1"

import tensorflow as tf
import numpy as np

# Disable XLA compilation at multiple levels
try:
    tf.config.optimizer.set_jit(False)
except:
    pass
try:
    tf.config.experimental.enable_op_determinism()
except:
    pass

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


class experiment_custom():
    """
    VIGNet experiment for custom EEG + Eye state data
    Uses preprocessed DE features and PERCLOS labels
    Uses block-wise random split (70%/15%/15%) to avoid data leakage
    """
    def __init__(self, data_dir, gpu_idx, task, logger=None, log_dir="./logs_custom",
                 block_size=8, gap_size=2, random_seed=42):
        # Assign GPU
        self.gpu_idx = gpu_idx
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_idx)
        
        # Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Using GPU: {gpus[0]}")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("Warning: No GPU found, using CPU")
        
        self.data_dir = data_dir
        self.task = task
        self.log_dir = log_dir
        self.block_size = block_size
        self.gap_size = gap_size
        self.random_seed = random_seed

        self.reg_label = False
        if self.task == "RGS":
            self.reg_label = True

        # Training hyperparameters
        self.learning_rate = 0.005
        self.num_epochs = 500
        self.num_batches = 8
        self.early_stopping_patience = 50
        
        # Create optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        tf.config.run_functions_eagerly(True)
        
        # Setup logging
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger("experiment_custom")
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                console_formatter = logging.Formatter('%(message)s')
                console_handler.setFormatter(console_formatter)
                self.logger.addHandler(console_handler)
        
        self.logger.info(f"START TRAINING - Task: {task}")
        self.logger.info(f"Data directory: {data_dir}")
        self.logger.info(f"Learning rate: {self.learning_rate}, Epochs: {self.num_epochs}, Batches: {self.num_batches}")
        self.logger.info(f"Data split: 70% train / 15% validation / 15% test (block-wise random)")
        self.logger.info(f"Block size: {self.block_size}, Gap size: {self.gap_size}, Random seed: {self.random_seed}")

    def training(self):
        # Load dataset
        self.logger.info("Loading dataset...")
        load_data = utils.load_dataset_custom(
            data_dir=self.data_dir, 
            reg_label=self.reg_label,
            block_size=self.block_size,
            gap_size=self.gap_size,
            random_seed=self.random_seed
        )
        Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest = load_data.call()
        self.logger.info(f"Dataset shapes - Train: {Xtrain.shape}, Valid: {Xvalid.shape}, Test: {Xtest.shape}")
        
        # Get feature dimensions
        n_channels = Xtrain.shape[1]
        n_bands = Xtrain.shape[2]
        self.logger.info(f"Feature dimensions - Channels: {n_channels}, Bands: {n_bands}")
        
        # Feature normalization
        train_shape = Xtrain.shape
        Xtrain_flat = Xtrain.reshape(train_shape[0], -1)
        Xvalid_flat = Xvalid.reshape(Xvalid.shape[0], -1)
        Xtest_flat = Xtest.reshape(Xtest.shape[0], -1)
        
        scaler = StandardScaler()
        Xtrain_flat = scaler.fit_transform(Xtrain_flat)
        Xvalid_flat = scaler.transform(Xvalid_flat)
        Xtest_flat = scaler.transform(Xtest_flat)
        
        Xtrain = Xtrain_flat.reshape(train_shape)
        Xvalid = Xvalid_flat.reshape(Xvalid.shape)
        Xtest = Xtest_flat.reshape(Xtest.shape)
        self.logger.info("Applied feature normalization (StandardScaler)")
        
        self.scaler = scaler
        
        # Convert to Tensor
        Xtrain = tf.constant(Xtrain, dtype=tf.float64)
        Ytrain = tf.constant(Ytrain, dtype=tf.float64)
        Xvalid = tf.constant(Xvalid, dtype=tf.float64)
        Yvalid_tf = tf.constant(Yvalid, dtype=tf.float64)
        Xtest = tf.constant(Xtest, dtype=tf.float64)
        Ytest_tf = tf.constant(Ytest, dtype=tf.float64)
        
        # Store original labels for evaluation
        if not self.reg_label:
            Yvalid_orig = np.argmax(Yvalid, axis=-1)
            Ytest_orig = np.argmax(Ytest, axis=-1)
        else:
            Yvalid_orig = Yvalid.squeeze()
            Ytest_orig = Ytest.squeeze()

        # Initialize model
        self.logger.info(f"Initializing VIGNet model (channels: {n_channels}, bands: {n_bands})...")
        
        # Choose model based on feature dimensions
        if n_bands >= 25:
            VIGNet = network.vignet_custom(mode=self.task, n_bands=n_bands, n_channels=n_channels)
        else:
            # Use simpler model for fewer bands
            VIGNet = network.vignet_custom_simple(mode=self.task, n_bands=n_bands, n_channels=n_channels)
            self.logger.info("Using simplified model architecture for small feature dimensions")

        # Optimization
        optimizer = self.optimizer
        num_batch_iter = int(Xtrain.shape[0] / self.num_batches)
        self.logger.info(f"Number of batch iterations per epoch: {num_batch_iter}")

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        random_seed = 42
        
        for epoch in range(self.num_epochs):
            loss_per_epoch = 0
            tf.random.set_seed(random_seed + epoch)
            rand_idx = tf.random.shuffle(tf.range(tf.shape(Xtrain)[0]))
            Xtrain_shuffled = tf.gather(Xtrain, rand_idx)
            Ytrain_shuffled = tf.gather(Ytrain, rand_idx)

            for batch in range(num_batch_iter):
                x = Xtrain_shuffled[batch * self.num_batches:(batch + 1) * self.num_batches, :, :, :]
                y = Ytrain_shuffled[batch * self.num_batches:(batch + 1) * self.num_batches, :]

                loss, grads = utils.grad(model=VIGNet, inputs=x, labels=y, mode=self.task)
                grads, global_norm = tf.clip_by_global_norm(grads, clip_norm=1.0)
                optimizer.apply_gradients(zip(grads, VIGNet.trainable_variables))
                loss_per_epoch += tf.reduce_mean(loss).numpy()

            avg_loss = loss_per_epoch / num_batch_iter
            
            # Evaluate on validation set
            Yvalid_pred_temp = VIGNet(Xvalid, training=False)
            if self.reg_label:
                Yvalid_orig_tf = tf.constant(Yvalid_orig, dtype=tf.float64)
                val_loss = tf.reduce_mean((Yvalid_orig_tf - tf.squeeze(Yvalid_pred_temp)) ** 2).numpy()
            else:
                val_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(Yvalid_tf, Yvalid_pred_temp)).numpy()
            
            self.logger.info("Epoch: {}, Training Loss: {:0.4f}, Validation Loss: {:0.4f}".format(
                epoch + 1, avg_loss, val_loss))
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = VIGNet.get_weights()
                self.logger.info("  -> New best validation loss: {:0.4f}".format(best_val_loss))
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    self.logger.info("Early stopping triggered at epoch {}".format(epoch + 1))
                    self.logger.info("Best validation loss: {:0.4f}".format(best_val_loss))
                    break
        
        # Restore best weights
        if best_weights is not None:
            VIGNet.set_weights(best_weights)
            self.logger.info("Restored best model weights")
        
        # Save model
        try:
            self.logger.info("Saving model...")
            self._save_model(VIGNet)
            self.logger.info("Model saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        # Final evaluation
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EVALUATION RESULTS")
        self.logger.info("=" * 60)
        
        Yvalid_pred = VIGNet(Xvalid, training=False)
        valid_metrics = self._evaluate(Yvalid_orig, Yvalid_pred, "Validation")
        
        Ytest_pred = VIGNet(Xtest, training=False)
        test_metrics = self._evaluate(Ytest_orig, Ytest_pred, "Test")
        
        self.logger.info("=" * 60 + "\n")
        
        # Save predictions
        predictions = {
            'valid': {
                'y_true': Yvalid_orig,
                'y_pred': Yvalid_pred.numpy().squeeze() if self.reg_label else np.argmax(Yvalid_pred.numpy(), axis=-1)
            },
            'test': {
                'y_true': Ytest_orig,
                'y_pred': Ytest_pred.numpy().squeeze() if self.reg_label else np.argmax(Ytest_pred.numpy(), axis=-1)
            }
        }
        
        return valid_metrics, test_metrics, predictions
    
    def _save_model(self, model):
        """Save trained model and scaler"""
        import pickle
        
        models_dir = os.path.join(self.log_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Save model
        model_save_path = os.path.join(models_dir, "best_model")
        try:
            model.save(model_save_path, save_format='tf')
            self.logger.info(f"Saved model to: {model_save_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save model: {e}")
            weights_save_path = os.path.join(models_dir, "best_weights.h5")
            try:
                model.save_weights(weights_save_path)
                self.logger.info(f"Saved model weights to: {weights_save_path}")
            except Exception as e2:
                self.logger.error(f"Failed to save model weights: {e2}")
        
        # Save scaler
        scaler_save_path = os.path.join(models_dir, "scaler.pkl")
        with open(scaler_save_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        self.logger.info(f"Saved scaler to: {scaler_save_path}")
    
    def _check_prediction_concentration(self, y_pred, set_name, threshold=0.02):
        """Check if predictions are concentrated around fixed values"""
        pred_std = np.std(y_pred)
        pred_range = np.max(y_pred) - np.min(y_pred)
        pred_unique_ratio = len(np.unique(np.round(y_pred, 3))) / len(y_pred)
        
        self.logger.info(f"{set_name} Prediction Stats:")
        self.logger.info(f"  Std: {pred_std:.6f}, Range: {pred_range:.6f}, Unique ratio: {pred_unique_ratio:.4f}")
        
        if pred_std < threshold:
            self.logger.warning(f"  WARNING: {set_name} predictions may be concentrated (std < {threshold})")
        
        return pred_std, pred_range, pred_unique_ratio
    
    def _evaluate(self, y_true, y_pred, set_name):
        """Evaluate model performance"""
        if self.reg_label:
            y_pred_np = y_pred.numpy().squeeze()
            y_true_np = y_true
            
            pred_std, pred_range, pred_unique_ratio = self._check_prediction_concentration(y_pred_np, set_name)
            
            mse = mean_squared_error(y_true_np, y_pred_np)
            mae = mean_absolute_error(y_true_np, y_pred_np)
            rmse = np.sqrt(mse)
            corr, p_value = pearsonr(y_true_np, y_pred_np)
            
            self.logger.info(f"{set_name} Set - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, "
                           f"Pearson Correlation: {corr:.6f} (p={p_value:.6f})")
            
            return {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'correlation': corr,
                'p_value': p_value,
                'pred_std': pred_std,
                'pred_range': pred_range,
                'pred_unique_ratio': pred_unique_ratio
            }
        else:
            y_pred_np = np.argmax(y_pred.numpy(), axis=-1)
            y_true_np = y_true
            
            accuracy = accuracy_score(y_true_np, y_pred_np)
            precision = precision_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
            recall = recall_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
            f1 = f1_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
            cm = confusion_matrix(y_true_np, y_pred_np)
            
            self.logger.info(f"{set_name} Set - Accuracy: {accuracy:.6f}, Precision: {precision:.6f}, "
                           f"Recall: {recall:.6f}, F1-Score: {f1:.6f}")
            self.logger.info(f"{set_name} Set Confusion Matrix:")
            self.logger.info(f"\n{cm}")
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm
            }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='VIGNet Training for Custom EEG Data (with automatic preprocessing)')
    parser.add_argument('--input', type=str, 
                        default='/home/vivian/eeg/SEED_VIG/Dec_09_experiments/experiment_20251208_155206/eeg_eye_merged.csv',
                        help='Path to eeg_eye_merged.csv file (for automatic preprocessing)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing preprocessed data (if None, will auto-generate from --input)')
    parser.add_argument('--task', type=str, default='RGS', choices=['RGS', 'CLF'],
                        help='Task type: RGS (regression) or CLF (classification)')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory for log files (if None, will auto-generate from --input)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index to use')
    parser.add_argument('--perclos-window', type=float, default=60.0,
                        help='Window size in seconds for PERCLOS calculation')
    parser.add_argument('--trim-start', type=float, default=300.0,
                        help='Seconds to trim from start of recording (default: 300s = 5 min)')
    parser.add_argument('--trim-end', type=float, default=60.0,
                        help='Seconds to trim from end of recording (default: 60s = 1 min)')
    parser.add_argument('--block-size', type=int, default=8,
                        help='Number of consecutive windows per block for data split (default: 8 = ~32s)')
    parser.add_argument('--gap-size', type=int, default=2,
                        help='Number of windows to discard between blocks to prevent overlap (default: 2)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for block-wise split (default: 42)')
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip preprocessing if data-dir is provided (assumes preprocessing already done)')
    
    args = parser.parse_args()
    
    # Auto-generate data-dir and log-dir from input path if not provided
    if args.data_dir is None:
        # Extract directory from input file path
        input_dir = os.path.dirname(os.path.abspath(args.input))
        args.data_dir = os.path.join(input_dir, 'processed')
    
    if args.log_dir is None:
        # Extract directory from input file path
        input_dir = os.path.dirname(os.path.abspath(args.input))
        args.log_dir = os.path.join(input_dir, 'logs_custom')
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup logging
    start_time = datetime.now()
    log_path = os.path.join(args.log_dir, f"training_{args.task}_{start_time.strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger("experiment_custom")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    logger.info("=" * 80)
    logger.info("VIGNet Training for Custom EEG Data")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Log file: {log_path}")
    logger.info("=" * 80)
    
    # Check if preprocessing is needed
    features_path = os.path.join(args.data_dir, 'de_features.npy')
    perclos_path = os.path.join(args.data_dir, 'perclos_labels.npy')
    
    need_preprocessing = False
    if not args.skip_preprocessing:
        if not os.path.exists(features_path) or not os.path.exists(perclos_path):
            need_preprocessing = True
            logger.info("Preprocessed data not found. Starting automatic preprocessing...")
        else:
            logger.info("Preprocessed data found. Skipping preprocessing.")
    else:
        logger.info("Skipping preprocessing (--skip-preprocessing flag set).")
    
    # Run preprocessing if needed
    if need_preprocessing:
        logger.info("=" * 80)
        logger.info("PREPROCESSING PHASE")
        logger.info("=" * 80)
        logger.info(f"Input file: {args.input}")
        logger.info(f"Output directory: {args.data_dir}")
        logger.info(f"PERCLOS window: {args.perclos_window}s")
        logger.info(f"Trim start: {args.trim_start}s, Trim end: {args.trim_end}s")
        
        try:
            # Run preprocessing
            process_merged_data(
                input_csv=args.input,
                output_dir=args.data_dir,
                perclos_window_sec=args.perclos_window,
                trim_start_sec=args.trim_start,
                trim_end_sec=args.trim_end
            )
            logger.info("Preprocessing completed successfully!")
            logger.info("=" * 80)
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    # Run training
    try:
        exp = experiment_custom(
            data_dir=args.data_dir,
            gpu_idx=args.gpu,
            task=args.task,
            logger=logger,
            log_dir=args.log_dir,
            block_size=args.block_size,
            gap_size=args.gap_size,
            random_seed=args.random_seed
        )
        valid_metrics, test_metrics, predictions = exp.training()
        
        # Save predictions
        predictions_dir = os.path.join(args.log_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        
        for set_name in ['valid', 'test']:
            pred_file = os.path.join(predictions_dir, f"{set_name}_predictions.npy")
            np.save(pred_file, {
                'set': set_name,
                'y_true': predictions[set_name]['y_true'],
                'y_pred': predictions[set_name]['y_pred']
            })
            logger.info(f"Saved {set_name} predictions to: {pred_file}")
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    end_time = datetime.now()
    logger.info("=" * 80)
    logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total duration: {end_time - start_time}")
    logger.info("=" * 80)


