import utils
import network

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
from scipy.stats import pearsonr

class experiment():
    def __init__(self, trial_idx, cv_idx, gpu_idx, task, logger=None, log_dir="./logs"):
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
        self.cv_idx = cv_idx
        self.task = task # for controlling task of interests

        self.reg_label = False

        if self.task == "RGS":
            self.reg_label = True

        # Define learning schedules
        self.learning_rate = 1e-3
        self.num_epochs = 100
        self.num_batches = 5
        # Create optimizer with jit_compile=False to avoid XLA issues
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # Force eager execution to avoid XLA compilation
        tf.config.run_functions_eagerly(True)
        
        # Setup logging - use shared logger if provided
        if logger is not None:
            self.logger = logger
        else:
            # Fallback: create a simple logger if none provided
            self.logger = logging.getLogger(f"experiment_trial{trial_idx}_cv{cv_idx}")
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                console_formatter = logging.Formatter('%(message)s')
                console_handler.setFormatter(console_formatter)
                self.logger.addHandler(console_handler)
        
        self.logger.info(f"START TRAINING CV {cv_idx} - Task: {task}")
        self.logger.info(f"Learning rate: {self.learning_rate}, Epochs: {self.num_epochs}, Batches: {self.num_batches}")

    def training(self):
        # Load dataset
        self.logger.info("Loading dataset...")
        load_data = utils.load_dataset(trial=self.trial_idx, cv=self.cv_idx, reg_label=self.reg_label)
        Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest = load_data.call()
        self.logger.info(f"Dataset shapes - Train: {Xtrain.shape}, Valid: {Xvalid.shape}, Test: {Xtest.shape}")
        
        # Store original labels for evaluation
        if not self.reg_label:
            Yvalid_orig = np.argmax(Yvalid, axis=-1)
            Ytest_orig = np.argmax(Ytest, axis=-1)
        else:
            Yvalid_orig = Yvalid.squeeze()
            Ytest_orig = Ytest.squeeze()

        # Call model
        self.logger.info("Initializing VIGNet model...")
        VIGNet = network.vignet(mode=self.task)

        # Optimization
        optimizer = self.optimizer
        num_batch_iter = int(Xtrain.shape[0]/self.num_batches)
        self.logger.info(f"Number of batch iterations per epoch: {num_batch_iter}")

        # Training loop
        for epoch in range(self.num_epochs):
            loss_per_epoch = 0
            # Randomize the training dataset
            rand_idx = np.random.permutation(Xtrain.shape[0])
            Xtrain, Ytrain = Xtrain[rand_idx, :, :, :], Ytrain[rand_idx, :]

            for batch in range(num_batch_iter):
                # Sample minibatch
                x = Xtrain[batch * self.num_batches:(batch + 1) * self.num_batches, :, :, :]
                y = Ytrain[batch * self.num_batches:(batch + 1) * self.num_batches, :]

                # Estimate loss
                loss, grads = utils.grad(model=VIGNet, inputs=x, labels=y, mode=self.task)

                # Update the network
                optimizer.apply_gradients(zip(grads, VIGNet.trainable_variables))
                loss_per_epoch += np.mean(loss)

            avg_loss = loss_per_epoch/num_batch_iter
            self.logger.info("Epoch: {}, Training Loss: {:0.4f}".format(epoch + 1, avg_loss))
        
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
    
    def _evaluate(self, y_true, y_pred, set_name):
        """Evaluate model performance and return metrics"""
        if self.reg_label:
            # Regression task
            y_pred_np = y_pred.numpy().squeeze()
            y_true_np = y_true
            
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
                'p_value': p_value
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
    
    parser = argparse.ArgumentParser(description='VIGNet Training Script')
    parser.add_argument('--trial', type=int, default=None, 
                       help='Specify a single trial number to run (1-23). If not specified, runs all trials.')
    parser.add_argument('--task', type=str, default='RGS', choices=['RGS', 'CLF'],
                       help='Task type: RGS (regression) or CLF (classification). Default: RGS')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory for log files. Default: ./logs')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save predictions to .npy files for visualization')
    
    args = parser.parse_args()
    
    # Create main log directory
    main_log_dir = args.log_dir
    os.makedirs(main_log_dir, exist_ok=True)
    
    # Create predictions directory if needed
    if args.save_predictions:
        predictions_dir = os.path.join(main_log_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
    
    task = args.task
    start_time = datetime.now()
    
    # Determine which trials to run
    if args.trial is not None:
        if args.trial < 1 or args.trial > 23:
            print(f"Error: Trial number must be between 1 and 23. Got: {args.trial}")
            exit(1)
        trials_to_run = [args.trial]
        print(f"Running single trial: {args.trial}")
    else:
        trials_to_run = list(range(1, 24))
        print(f"Running all trials: 1-23")
    
    # Overall summary log
    summary_log_path = os.path.join(main_log_dir, f"training_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.log")
    summary_logger = logging.getLogger("summary")
    summary_logger.setLevel(logging.INFO)
    summary_logger.handlers = []
    summary_handler = logging.FileHandler(summary_log_path, mode='w', encoding='utf-8')
    summary_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    summary_logger.addHandler(summary_handler)
    summary_logger.info("="*80)
    summary_logger.info("SEED-VIG Training Summary")
    summary_logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
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
        trial_logger.info(f"TRIAL {trial} - Task: {task}")
        trial_logger.info(f"Log file: {trial_log_path}")
        trial_logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        trial_logger.info("="*80)
        
        summary_logger.info(f"Starting Trial {trial}")
        
        # Store metrics for all CV folds
        all_valid_metrics = []
        all_test_metrics = []
        
        for fold in range(5):
            trial_logger.info(f"\n{'-'*80}")
            trial_logger.info(f"CV Fold {fold}")
            trial_logger.info(f"{'-'*80}")
            
            main = experiment(trial_idx=trial, cv_idx=fold, gpu_idx=0, task=task, logger=trial_logger, log_dir=main_log_dir)
            try:
                valid_metrics, test_metrics, predictions = main.training()
                all_valid_metrics.append(valid_metrics)
                all_test_metrics.append(test_metrics)
                
                # Save predictions if requested
                if args.save_predictions:
                    predictions_dir = os.path.join(main_log_dir, "predictions")
                    for set_name_key, set_name_file in [('valid', 'validation'), ('test', 'test')]:
                        pred_file = os.path.join(predictions_dir, 
                                               f"trial{trial}_cv{fold}_{set_name_file}.npy")
                        np.save(pred_file, {
                            'trial': trial,
                            'cv': fold,
                            'set': set_name_file,
                            'y_true': predictions[set_name_key]['y_true'],
                            'y_pred': predictions[set_name_key]['y_pred']
                        })
                
                trial_logger.info(f"CV Fold {fold} completed successfully")
                summary_logger.info(f"  Trial {trial} CV {fold} completed successfully")
            except Exception as e:
                error_msg = f"Error in Trial {trial} CV {fold}: {str(e)}"
                trial_logger.error(error_msg)
                summary_logger.error(error_msg)
                import traceback
                trial_logger.error(traceback.format_exc())
                summary_logger.error(traceback.format_exc())
        
        # Calculate and log average metrics across all CV folds
        if all_valid_metrics and all_test_metrics:
            trial_logger.info("\n" + "="*80)
            trial_logger.info("CROSS-VALIDATION SUMMARY - TRIAL {}".format(trial))
            trial_logger.info("="*80)
            
            if task == "RGS":
                # Regression metrics
                avg_valid_mse = np.mean([m['mse'] for m in all_valid_metrics])
                avg_valid_mae = np.mean([m['mae'] for m in all_valid_metrics])
                avg_valid_rmse = np.mean([m['rmse'] for m in all_valid_metrics])
                avg_valid_corr = np.mean([m['correlation'] for m in all_valid_metrics])
                
                avg_test_mse = np.mean([m['mse'] for m in all_test_metrics])
                avg_test_mae = np.mean([m['mae'] for m in all_test_metrics])
                avg_test_rmse = np.mean([m['rmse'] for m in all_test_metrics])
                avg_test_corr = np.mean([m['correlation'] for m in all_test_metrics])
                
                trial_logger.info("Validation Set (Average across 5 CV folds):")
                trial_logger.info(f"  MSE: {avg_valid_mse:.6f} ± {np.std([m['mse'] for m in all_valid_metrics]):.6f}")
                trial_logger.info(f"  MAE: {avg_valid_mae:.6f} ± {np.std([m['mae'] for m in all_valid_metrics]):.6f}")
                trial_logger.info(f"  RMSE: {avg_valid_rmse:.6f} ± {np.std([m['rmse'] for m in all_valid_metrics]):.6f}")
                trial_logger.info(f"  Pearson Correlation: {avg_valid_corr:.6f} ± {np.std([m['correlation'] for m in all_valid_metrics]):.6f}")
                
                trial_logger.info("\nTest Set (Average across 5 CV folds):")
                trial_logger.info(f"  MSE: {avg_test_mse:.6f} ± {np.std([m['mse'] for m in all_test_metrics]):.6f}")
                trial_logger.info(f"  MAE: {avg_test_mae:.6f} ± {np.std([m['mae'] for m in all_test_metrics]):.6f}")
                trial_logger.info(f"  RMSE: {avg_test_rmse:.6f} ± {np.std([m['rmse'] for m in all_test_metrics]):.6f}")
                trial_logger.info(f"  Pearson Correlation: {avg_test_corr:.6f} ± {np.std([m['correlation'] for m in all_test_metrics]):.6f}")
                
                summary_logger.info(f"Trial {trial} - Test MSE: {avg_test_mse:.6f}, Test Correlation: {avg_test_corr:.6f}")
            else:
                # Classification metrics
                avg_valid_acc = np.mean([m['accuracy'] for m in all_valid_metrics])
                avg_valid_prec = np.mean([m['precision'] for m in all_valid_metrics])
                avg_valid_rec = np.mean([m['recall'] for m in all_valid_metrics])
                avg_valid_f1 = np.mean([m['f1'] for m in all_valid_metrics])
                
                avg_test_acc = np.mean([m['accuracy'] for m in all_test_metrics])
                avg_test_prec = np.mean([m['precision'] for m in all_test_metrics])
                avg_test_rec = np.mean([m['recall'] for m in all_test_metrics])
                avg_test_f1 = np.mean([m['f1'] for m in all_test_metrics])
                
                trial_logger.info("Validation Set (Average across 5 CV folds):")
                trial_logger.info(f"  Accuracy: {avg_valid_acc:.6f} ± {np.std([m['accuracy'] for m in all_valid_metrics]):.6f}")
                trial_logger.info(f"  Precision: {avg_valid_prec:.6f} ± {np.std([m['precision'] for m in all_valid_metrics]):.6f}")
                trial_logger.info(f"  Recall: {avg_valid_rec:.6f} ± {np.std([m['recall'] for m in all_valid_metrics]):.6f}")
                trial_logger.info(f"  F1-Score: {avg_valid_f1:.6f} ± {np.std([m['f1'] for m in all_valid_metrics]):.6f}")
                
                trial_logger.info("\nTest Set (Average across 5 CV folds):")
                trial_logger.info(f"  Accuracy: {avg_test_acc:.6f} ± {np.std([m['accuracy'] for m in all_test_metrics]):.6f}")
                trial_logger.info(f"  Precision: {avg_test_prec:.6f} ± {np.std([m['precision'] for m in all_test_metrics]):.6f}")
                trial_logger.info(f"  Recall: {avg_test_rec:.6f} ± {np.std([m['recall'] for m in all_test_metrics]):.6f}")
                trial_logger.info(f"  F1-Score: {avg_test_f1:.6f} ± {np.std([m['f1'] for m in all_test_metrics]):.6f}")
                
                summary_logger.info(f"Trial {trial} - Test Accuracy: {avg_test_acc:.6f}, Test F1: {avg_test_f1:.6f}")
            
            trial_logger.info("="*80)
        
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
