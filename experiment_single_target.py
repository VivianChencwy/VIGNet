"""
VIGNet training script for single-target fatigue prediction.
Trains one model per target indicator.

Usage:
    python experiment_single_target.py --target perclos
    python experiment_single_target.py --target kss
    python experiment_single_target.py --target miss_rate
    python experiment_single_target.py --target false_alarm
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime

# Disable XLA before importing TensorFlow
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"
os.environ["TF_DISABLE_XLA"] = "1"

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

# Configure GPU
try:
    tf.config.optimizer.set_jit(False)
except:
    pass

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU available: {gpus[0]}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found, using CPU")

# Force eager execution
tf.config.run_functions_eagerly(True)

# Import custom modules
from utils_experiment import load_experiment_dataset
from network_experiment import create_model


class SingleTargetTrainer:
    """Trainer for single-target VIGNet model."""
    
    def __init__(self, target, log_dir, data_dir='data/experiment_20251124_140734'):
        """
        Args:
            target: Target indicator ('perclos', 'kss', 'miss_rate', 'false_alarm')
            log_dir: Directory for logs and saved models
            data_dir: Directory containing experiment data
        """
        self.target = target
        self.data_dir = data_dir
        self.log_dir = os.path.join(log_dir, target)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Training hyperparameters
        self.learning_rate = 0.005
        self.num_epochs = 200
        self.batch_size = 8
        self.early_stopping_patience = 20
        
        # Setup logging
        self._setup_logging()
        
        # Training history
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'val_correlation': []
        }
    
    def _setup_logging(self):
        """Setup logging to file and console."""
        log_file = os.path.join(self.log_dir, 'training.log')
        
        self.logger = logging.getLogger(f"trainer_{self.target}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        
        # File handler
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(console_handler)
    
    def log(self, message):
        """Log message to file and console."""
        self.logger.info(message)
    
    def load_data(self):
        """Load and prepare dataset."""
        self.log(f"\n{'='*60}")
        self.log(f"Loading data for target: {self.target}")
        self.log('='*60)
        
        # Load dataset
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = \
            load_experiment_dataset(target=self.target, data_dir=self.data_dir)
        
        # Feature normalization (StandardScaler on training data only)
        train_shape = self.X_train.shape
        X_train_flat = self.X_train.reshape(train_shape[0], -1)
        X_val_flat = self.X_val.reshape(self.X_val.shape[0], -1)
        X_test_flat = self.X_test.reshape(self.X_test.shape[0], -1)
        
        self.scaler = StandardScaler()
        X_train_flat = self.scaler.fit_transform(X_train_flat)
        X_val_flat = self.scaler.transform(X_val_flat)
        X_test_flat = self.scaler.transform(X_test_flat)
        
        self.X_train = X_train_flat.reshape(train_shape)
        self.X_val = X_val_flat.reshape(self.X_val.shape)
        self.X_test = X_test_flat.reshape(self.X_test.shape)
        
        self.log("Applied StandardScaler normalization")
        
        # Convert to TensorFlow tensors
        self.X_train_tf = tf.constant(self.X_train, dtype=tf.float64)
        self.y_train_tf = tf.constant(self.y_train, dtype=tf.float64)
        self.X_val_tf = tf.constant(self.X_val, dtype=tf.float64)
        self.y_val_tf = tf.constant(self.y_val, dtype=tf.float64)
        self.X_test_tf = tf.constant(self.X_test, dtype=tf.float64)
        self.y_test_tf = tf.constant(self.y_test, dtype=tf.float64)
    
    def train(self):
        """Train the model."""
        self.log(f"\n{'='*60}")
        self.log(f"Training VIGNet for target: {self.target}")
        self.log('='*60)
        self.log(f"Learning rate: {self.learning_rate}")
        self.log(f"Epochs: {self.num_epochs}")
        self.log(f"Batch size: {self.batch_size}")
        self.log(f"Early stopping patience: {self.early_stopping_patience}")
        
        # Create model
        self.model = create_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Training loop
        num_batch_iter = int(self.X_train.shape[0] / self.batch_size)
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        for epoch in range(self.num_epochs):
            # Shuffle training data
            tf.random.set_seed(epoch)
            rand_idx = tf.random.shuffle(tf.range(tf.shape(self.X_train_tf)[0]))
            X_shuffled = tf.gather(self.X_train_tf, rand_idx)
            y_shuffled = tf.gather(self.y_train_tf, rand_idx)
            
            epoch_loss = 0
            
            for batch in range(num_batch_iter):
                # Get batch
                start_idx = batch * self.batch_size
                end_idx = (batch + 1) * self.batch_size
                x_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Compute gradients
                with tf.GradientTape() as tape:
                    y_pred = self.model(x_batch, training=True)
                    loss = tf.reduce_mean(tf.keras.losses.MSE(y_batch, y_pred))
                
                grads = tape.gradient(loss, self.model.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                
                epoch_loss += loss.numpy()
            
            avg_train_loss = epoch_loss / num_batch_iter
            
            # Validation
            y_val_pred = self.model(self.X_val_tf, training=False)
            val_loss = tf.reduce_mean(tf.keras.losses.MSE(self.y_val_tf, y_val_pred)).numpy()
            
            # Validation metrics
            y_val_np = self.y_val.squeeze()
            y_val_pred_np = y_val_pred.numpy().squeeze()
            val_mae = mean_absolute_error(y_val_np, y_val_pred_np)
            val_rmse = np.sqrt(mean_squared_error(y_val_np, y_val_pred_np))
            val_corr, _ = pearsonr(y_val_np, y_val_pred_np)
            
            # Record history
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            self.history['val_rmse'].append(val_rmse)
            self.history['val_correlation'].append(val_corr)
            
            self.log(f"Epoch {epoch+1:3d}: train_loss={avg_train_loss:.6f}, "
                    f"val_loss={val_loss:.6f}, val_corr={val_corr:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = self.model.get_weights()
                self.log(f"  -> New best validation loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    self.log(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        # Restore best weights
        if best_weights is not None:
            self.model.set_weights(best_weights)
            self.log("Restored best model weights")
    
    def evaluate(self):
        """Evaluate model on test set."""
        self.log(f"\n{'='*60}")
        self.log("Evaluation Results")
        self.log('='*60)
        
        # Predictions
        y_val_pred = self.model(self.X_val_tf, training=False).numpy().squeeze()
        y_test_pred = self.model(self.X_test_tf, training=False).numpy().squeeze()
        
        y_val_true = self.y_val.squeeze()
        y_test_true = self.y_test.squeeze()
        
        # Store predictions
        self.predictions = {
            'val': {'y_true': y_val_true, 'y_pred': y_val_pred},
            'test': {'y_true': y_test_true, 'y_pred': y_test_pred}
        }
        
        # Compute metrics
        self.metrics = {}
        
        for set_name, y_true, y_pred in [('Validation', y_val_true, y_val_pred),
                                          ('Test', y_test_true, y_test_pred)]:
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r_pearson, p_pearson = pearsonr(y_true, y_pred)
            r_spearman, p_spearman = spearmanr(y_true, y_pred)
            
            self.metrics[set_name.lower()] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'pearson_r': r_pearson,
                'pearson_p': p_pearson,
                'spearman_r': r_spearman,
                'spearman_p': p_spearman,
                'pred_mean': np.mean(y_pred),
                'pred_std': np.std(y_pred),
                'true_mean': np.mean(y_true),
                'true_std': np.std(y_true)
            }
            
            self.log(f"\n{set_name} Set:")
            self.log(f"  MSE:  {mse:.6f}")
            self.log(f"  MAE:  {mae:.6f}")
            self.log(f"  RMSE: {rmse:.6f}")
            self.log(f"  Pearson r:  {r_pearson:.6f} (p={p_pearson:.2e})")
            self.log(f"  Spearman r: {r_spearman:.6f} (p={p_spearman:.2e})")
            self.log(f"  Pred mean: {np.mean(y_pred):.4f}, std: {np.std(y_pred):.4f}")
            self.log(f"  True mean: {np.mean(y_true):.4f}, std: {np.std(y_true):.4f}")
    
    def save_results(self):
        """Save model, predictions, history, and metrics."""
        self.log(f"\n{'='*60}")
        self.log("Saving Results")
        self.log('='*60)
        
        # Save model weights
        weights_path = os.path.join(self.log_dir, 'model_weights.h5')
        self.model.save_weights(weights_path)
        self.log(f"Saved model weights: {weights_path}")
        
        # Save predictions
        predictions_path = os.path.join(self.log_dir, 'predictions.npy')
        np.save(predictions_path, self.predictions)
        self.log(f"Saved predictions: {predictions_path}")
        
        # Save training history
        history_path = os.path.join(self.log_dir, 'training_history.csv')
        pd.DataFrame(self.history).to_csv(history_path, index=False)
        self.log(f"Saved training history: {history_path}")
        
        # Save metrics
        metrics_path = os.path.join(self.log_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            # Convert numpy values to float
            metrics_serializable = {}
            for k, v in self.metrics.items():
                metrics_serializable[k] = {kk: float(vv) for kk, vv in v.items()}
            json.dump(metrics_serializable, f, indent=2)
        self.log(f"Saved metrics: {metrics_path}")
        
        # Save config
        config = {
            'target': self.target,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'early_stopping_patience': self.early_stopping_patience,
            'train_samples': int(self.X_train.shape[0]),
            'val_samples': int(self.X_val.shape[0]),
            'test_samples': int(self.X_test.shape[0]),
            'feature_shape': list(self.X_train.shape[1:]),
            'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        config_path = os.path.join(self.log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        self.log(f"Saved config: {config_path}")
    
    def run(self):
        """Run complete training pipeline."""
        start_time = datetime.now()
        
        self.log(f"\n{'='*60}")
        self.log(f"VIGNet Single-Target Training")
        self.log(f"Target: {self.target}")
        self.log(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log('='*60)
        
        # Pipeline
        self.load_data()
        self.train()
        self.evaluate()
        self.save_results()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.log(f"\n{'='*60}")
        self.log(f"Training Complete")
        self.log(f"Duration: {duration:.1f} seconds")
        self.log(f"Results saved to: {self.log_dir}")
        self.log('='*60)
        
        return self.metrics


def main():
    parser = argparse.ArgumentParser(description='VIGNet Single-Target Training')
    parser.add_argument('--target', type=str, required=True,
                        choices=['perclos', 'kss', 'miss_rate', 'false_alarm'],
                        help='Target indicator to predict')
    parser.add_argument('--log-dir', type=str, default='./logs_experiment',
                        help='Directory for logs and saved models')
    parser.add_argument('--data-dir', type=str, 
                        default='data/experiment_20251124_140734',
                        help='Directory containing experiment data')
    
    args = parser.parse_args()
    
    trainer = SingleTargetTrainer(
        target=args.target,
        log_dir=args.log_dir,
        data_dir=args.data_dir
    )
    
    metrics = trainer.run()
    
    print(f"\n[SUMMARY] Target: {args.target}")
    print(f"  Test RMSE: {metrics['test']['rmse']:.6f}")
    print(f"  Test Pearson r: {metrics['test']['pearson_r']:.6f}")


if __name__ == "__main__":
    main()



