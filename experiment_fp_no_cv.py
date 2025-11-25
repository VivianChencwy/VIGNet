import utils_fp as utils
import network_fp as network

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

class experiment_fp_no_cv():
    """
    VIGNet experiment using only FP1/FP2 forehead channels (2 channels)
    Uses fixed train/val/test split (70%/15%/15%) instead of cross-validation
    """
    def __init__(self, trial_idx, gpu_idx, task, logger=None, log_dir="./logs_fp_no_cv"):
        # Assign GPU
        self.gpu_idx = gpu_idx
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_idx)
        
        # Configure GPU memory growth to avoid allocating all memory at once
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
        
        self.trial_idx = trial_idx
        self.task = task # for controlling task of interests
        self.log_dir = log_dir  # Store log directory for model saving

        self.reg_label = False

        if self.task == "RGS":
            self.reg_label = True

        # Define learning schedules
        self.learning_rate = 0.005
        self.num_epochs = 200
        self.num_batches = 8
        self.early_stopping_patience = 20
        # Create optimizer with jit_compile=False to avoid XLA issues
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # Force eager execution to avoid XLA compilation
        tf.config.run_functions_eagerly(True)
        
        # Setup logging - use shared logger if provided
        if logger is not None:
            self.logger = logger
        else:
            # Fallback: create a simple logger if none provided
            self.logger = logging.getLogger(f"experiment_fp_no_cv_trial{trial_idx}")
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                console_formatter = logging.Formatter('%(message)s')
                console_handler.setFormatter(console_formatter)
                self.logger.addHandler(console_handler)
        
        self.logger.info(f"START TRAINING - Task: {task} (FP1/FP2 only, no CV)")
        self.logger.info(f"Learning rate: {self.learning_rate}, Epochs: {self.num_epochs}, Batches: {self.num_batches}")
        self.logger.info(f"Data split: 70% train / 15% validation / 15% test")

    def training(self):
        # Load dataset (FP1/FP2 only, no CV)
        self.logger.info("Loading dataset (FP1/FP2 channels only, fixed split)...")
        load_data = utils.load_dataset_fp_no_cv(trial=self.trial_idx, reg_label=self.reg_label)
        Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest = load_data.call()
        self.logger.info(f"Dataset shapes - Train: {Xtrain.shape}, Valid: {Xvalid.shape}, Test: {Xtest.shape}")
        
        # Feature normalization (StandardScaler on training data only)
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
        
        # Store scaler for model saving
        self.scaler = scaler
        
        # Convert to Tensor for better GPU utilization (optimize data transfer)
        Xtrain = tf.constant(Xtrain, dtype=tf.float64)
        Ytrain = tf.constant(Ytrain, dtype=tf.float64)
        Xvalid = tf.constant(Xvalid, dtype=tf.float64)
        Yvalid_tf = tf.constant(Yvalid, dtype=tf.float64)
        Xtest = tf.constant(Xtest, dtype=tf.float64)
        Ytest_tf = tf.constant(Ytest, dtype=tf.float64)
        
        # Store original labels for evaluation (keep as numpy for metrics)
        if not self.reg_label:
            Yvalid_orig = np.argmax(Yvalid, axis=-1)
            Ytest_orig = np.argmax(Ytest, axis=-1)
        else:
            Yvalid_orig = Yvalid.squeeze()
            Ytest_orig = Ytest.squeeze()

        # Call model (FP1/FP2 version)
        self.logger.info("Initializing VIGNet-FP model (2 channels)...")
        VIGNet = network.vignet_fp(mode=self.task)

        # Optimization
        optimizer = self.optimizer
        num_batch_iter = int(Xtrain.shape[0]/self.num_batches)
        self.logger.info(f"Number of batch iterations per epoch: {num_batch_iter}")

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        # Set random seed for reproducibility (required when determinism is enabled)
        random_seed = self.trial_idx * 100
        
        for epoch in range(self.num_epochs):
            loss_per_epoch = 0
            # Randomize the training dataset (using TensorFlow operations for better GPU utilization)
            tf.random.set_seed(random_seed + epoch)
            rand_idx = tf.random.shuffle(tf.range(tf.shape(Xtrain)[0]))
            Xtrain_shuffled = tf.gather(Xtrain, rand_idx)
            Ytrain_shuffled = tf.gather(Ytrain, rand_idx)

            for batch in range(num_batch_iter):
                # Sample minibatch (already on GPU as Tensor)
                x = Xtrain_shuffled[batch * self.num_batches:(batch + 1) * self.num_batches, :, :, :]
                y = Ytrain_shuffled[batch * self.num_batches:(batch + 1) * self.num_batches, :]

                # Estimate loss
                loss, grads = utils.grad(model=VIGNet, inputs=x, labels=y, mode=self.task)

                # Gradient clipping to prevent exploding gradients
                grads, global_norm = tf.clip_by_global_norm(grads, clip_norm=1.0)

                # Update the network
                optimizer.apply_gradients(zip(grads, VIGNet.trainable_variables))
                loss_per_epoch += tf.reduce_mean(loss).numpy()

            avg_loss = loss_per_epoch/num_batch_iter
            
            # Evaluate on validation set for early stopping
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
                self.logger.info("  → New best validation loss: {:0.4f}".format(best_val_loss))
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
        
        # Save model and scaler for inference
        try:
            self.logger.info("Starting model save process...")
            self._save_model(VIGNet)
            self.logger.info("Model save process completed")
        except Exception as e:
            self.logger.error(f"Error during model save process: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        # Final evaluation on validation and test sets
        self.logger.info("\n" + "="*60)
        self.logger.info("EVALUATION RESULTS")
        self.logger.info("="*60)
        
        # Validation set evaluation
        Yvalid_pred = VIGNet(Xvalid, training=False)
        valid_metrics = self._evaluate(Yvalid_orig, Yvalid_pred, "Validation")
        
        # Test set evaluation
        Ytest_pred = VIGNet(Xtest, training=False)
        test_metrics = self._evaluate(Ytest_orig, Ytest_pred, "Test")
        
        self.logger.info("="*60 + "\n")
        
        # Save predictions for visualization
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
        """Save trained model and scaler for inference"""
        import pickle
        
        # Create models directory
        models_dir = os.path.join(self.log_dir, "models")
        self.logger.info(f"Creating models directory: {models_dir}")
        try:
            os.makedirs(models_dir, exist_ok=True)
            self.logger.info(f"Models directory created/verified: {models_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create models directory: {e}")
            raise
        
        # Save model (SavedModel format for easy loading)
        model_save_path = os.path.join(models_dir, f"trial{self.trial_idx}_best_model")
        self.logger.info(f"Attempting to save model to: {model_save_path}")
        model_saved = False
        try:
            model.save(model_save_path, save_format='tf')
            self.logger.info(f"✓ Saved model to: {model_save_path}")
            model_saved = True
        except Exception as e:
            self.logger.warning(f"Failed to save model in SavedModel format: {e}")
            import traceback
            self.logger.warning(traceback.format_exc())
            # Fallback: save weights only
            weights_save_path = os.path.join(models_dir, f"trial{self.trial_idx}_best_weights.h5")
            self.logger.info(f"Attempting to save model weights to: {weights_save_path}")
            try:
                model.save_weights(weights_save_path)
                self.logger.info(f"✓ Saved model weights to: {weights_save_path}")
                model_saved = True
            except Exception as e2:
                self.logger.error(f"Failed to save model weights: {e2}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        if not model_saved:
            self.logger.error("WARNING: Model was not saved successfully!")
        
        # Save scaler (required for inference)
        scaler_save_path = os.path.join(models_dir, f"trial{self.trial_idx}_scaler.pkl")
        self.logger.info(f"Attempting to save scaler to: {scaler_save_path}")
        try:
            with open(scaler_save_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            self.logger.info(f"✓ Saved scaler to: {scaler_save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save scaler: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        # Save model metadata
        metadata = {
            'trial': self.trial_idx,
            'task': self.task,
            'input_shape': (None, 2, 25, 1),  # (batch, channels, frequency, time)
            'output_shape': (None, 1) if self.reg_label else (None, 3),
            'model_path': model_save_path if model_saved else None,
            'scaler_path': scaler_save_path
        }
        metadata_path = os.path.join(models_dir, f"trial{self.trial_idx}_metadata.pkl")
        self.logger.info(f"Attempting to save metadata to: {metadata_path}")
        try:
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            self.logger.info(f"✓ Saved model metadata to: {metadata_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save metadata: {e}")
            import traceback
            self.logger.warning(traceback.format_exc())
    
    def _check_prediction_concentration(self, y_pred, set_name, threshold=0.02):
        """Check if predictions are concentrated around fixed values"""
        pred_std = np.std(y_pred)
        pred_range = np.max(y_pred) - np.min(y_pred)
        pred_unique_ratio = len(np.unique(np.round(y_pred, 3))) / len(y_pred)
        
        self.logger.info(f"{set_name} Prediction Stats:")
        self.logger.info(f"  Std: {pred_std:.6f}, Range: {pred_range:.6f}, Unique ratio: {pred_unique_ratio:.4f}")
        
        if pred_std < threshold:
            self.logger.warning(f"  ⚠ WARNING: {set_name} predictions may be concentrated (std < {threshold})")
        
        if pred_unique_ratio < 0.1:
            self.logger.warning(f"  ⚠ WARNING: {set_name} predictions have low diversity (unique ratio < 0.1)")
        
        return pred_std, pred_range, pred_unique_ratio
    
    def _evaluate(self, y_true, y_pred, set_name):
        """Evaluate model performance and return metrics"""
        if self.reg_label:
            # Regression task
            y_pred_np = y_pred.numpy().squeeze()
            y_true_np = y_true
            
            # Check for prediction concentration issues
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
            # Classification task
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
    
    parser = argparse.ArgumentParser(description='VIGNet Training Script (FP1/FP2 only, no CV)')
    parser.add_argument('--trial', type=int, default=None, 
                       help='Specify a single trial number to run (1-21). If not specified, runs all trials.')
    parser.add_argument('--task', type=str, default='RGS', choices=['RGS', 'CLF'],
                       help='Task type: RGS (regression) or CLF (classification). Default: RGS')
    parser.add_argument('--log-dir', type=str, default='./logs_fp_no_cv',
                       help='Directory for log files. Default: ./logs_fp_no_cv')
    parser.add_argument('--no-save-predictions', action='store_true',
                       help='Do not save predictions to .npy files for visualization')
    
    args = parser.parse_args()
    
    # Create main log directory
    main_log_dir = args.log_dir
    os.makedirs(main_log_dir, exist_ok=True)
    
    # Save predictions by default (unless --no-save-predictions is specified)
    save_predictions = not args.no_save_predictions
    if save_predictions:
        predictions_dir = os.path.join(main_log_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
    
    task = args.task
    start_time = datetime.now()
    
    # Determine which trials to run
    if args.trial is not None:
        if args.trial < 1 or args.trial > 21:
            print(f"Error: Trial number must be between 1 and 21. Got: {args.trial}")
            exit(1)
        trials_to_run = [args.trial]
        print(f"Running single trial: {args.trial}")
    else:
        trials_to_run = list(range(1, 22))
        print(f"Running all trials: 1-21")
    
    # Overall summary log
    summary_log_path = os.path.join(main_log_dir, f"training_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.log")
    summary_logger = logging.getLogger("summary")
    summary_logger.setLevel(logging.INFO)
    summary_logger.handlers = []
    summary_handler = logging.FileHandler(summary_log_path, mode='w', encoding='utf-8')
    summary_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    summary_logger.addHandler(summary_handler)
    summary_logger.info("="*80)
    summary_logger.info("SEED-VIG Training Summary (FP1/FP2 channels only, no CV)")
    summary_logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary_logger.info("Data split: 70% train / 15% validation / 15% test")
    summary_logger.info("="*80)
    
    for trial in trials_to_run:
        # Create a log file for each trial
        trial_log_path = os.path.join(main_log_dir, f"trial{trial}_{task}_{start_time.strftime('%Y%m%d_%H%M%S')}.log")
        trial_logger = logging.getLogger(f"trial_{trial}")
        trial_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        trial_logger.handlers = []
        
        # File handler - one log file per trial
        file_handler = logging.FileHandler(trial_log_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        trial_logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        trial_logger.addHandler(console_handler)
        
        trial_logger.info("="*80)
        trial_logger.info(f"TRIAL {trial} - Task: {task} (FP1/FP2 only, no CV)")
        trial_logger.info(f"Log file: {trial_log_path}")
        trial_logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        trial_logger.info("="*80)
        
        summary_logger.info(f"Starting Trial {trial}")
        
        # Single training run (no CV)
        main = experiment_fp_no_cv(trial_idx=trial, gpu_idx=0, task=task, logger=trial_logger, log_dir=main_log_dir)
        try:
            valid_metrics, test_metrics, predictions = main.training()
            
            # Save predictions (default behavior)
            if save_predictions:
                predictions_dir = os.path.join(main_log_dir, "predictions")
                for set_name_key, set_name_file in [('valid', 'validation'), ('test', 'test')]:
                    pred_file = os.path.join(predictions_dir, 
                                           f"trial{trial}_{set_name_file}.npy")
                    np.save(pred_file, {
                        'trial': trial,
                        'set': set_name_file,
                        'y_true': predictions[set_name_key]['y_true'],
                        'y_pred': predictions[set_name_key]['y_pred']
                    })
            
            trial_logger.info("Trial completed successfully")
            summary_logger.info(f"  Trial {trial} completed successfully")
            
            # Log final metrics
            if task == "RGS":
                summary_logger.info(f"  Trial {trial} - Test MSE: {test_metrics['mse']:.6f}, "
                                   f"Test RMSE: {test_metrics['rmse']:.6f}, "
                                   f"Test Correlation: {test_metrics['correlation']:.6f}")
            else:
                summary_logger.info(f"  Trial {trial} - Test Accuracy: {test_metrics['accuracy']:.6f}, "
                                   f"Test F1: {test_metrics['f1']:.6f}")
        except Exception as e:
            error_msg = f"Error in Trial {trial}: {str(e)}"
            trial_logger.error(error_msg)
            summary_logger.error(error_msg)
            import traceback
            trial_logger.error(traceback.format_exc())
            summary_logger.error(traceback.format_exc())
        
        trial_logger.info("="*80)
        trial_logger.info(f"Trial {trial} completed. End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        trial_logger.info("="*80)
        
        # Remove handlers to avoid memory issues
        trial_logger.handlers = []
    
    end_time = datetime.now()
    summary_logger.info("="*80)
    summary_logger.info(f"All training completed. End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary_logger.info(f"Total duration: {end_time - start_time}")
    summary_logger.info("="*80)

