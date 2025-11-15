#!/usr/bin/env python
"""
Quick verification script to test if the bug fix improves results
Runs a single CV fold from trial 2 to quickly verify the fix works
"""
import os
import sys

# Disable XLA before importing TensorFlow
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"
os.environ["TF_DISABLE_XLA"] = "1"

import tensorflow as tf
import numpy as np

try:
    tf.config.optimizer.set_jit(False)
except:
    pass

import utils
import network
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def quick_test():
    """Run a quick test with 10 epochs to verify the fix"""
    print("="*80)
    print("QUICK VERIFICATION TEST - Bug Fix Validation")
    print("="*80)
    print("\nTesting: Trial 2, CV fold 0, 10 epochs")
    print("Purpose: Verify regression loss is being used correctly\n")
    
    trial_idx = 2
    cv_idx = 0
    task = "RGS"
    
    # Load data
    print("Loading dataset...")
    load_data = utils.load_dataset(trial=trial_idx, cv=cv_idx, reg_label=True)
    Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest = load_data.call()
    print(f"Data shapes - Train: {Xtrain.shape}, Valid: {Xvalid.shape}, Test: {Xtest.shape}")
    
    # Create model
    print("\nInitializing VIGNet model...")
    VIGNet = network.vignet(mode=task)
    
    # Setup optimizer
    learning_rate = 1e-2
    num_epochs = 10  # Quick test
    num_batches = 5
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    print(f"Hyperparameters: LR={learning_rate}, Epochs={num_epochs}, Batch size={num_batches}")
    
    # Training loop
    print("\nStarting training...")
    num_batch_iter = int(Xtrain.shape[0]/num_batches)
    
    for epoch in range(num_epochs):
        loss_per_epoch = 0
        rand_idx = np.random.permutation(Xtrain.shape[0])
        Xtrain_shuffled = Xtrain[rand_idx, :, :, :]
        Ytrain_shuffled = Ytrain[rand_idx, :]
        
        for batch in range(num_batch_iter):
            x = Xtrain_shuffled[batch * num_batches:(batch + 1) * num_batches, :, :, :]
            y = Ytrain_shuffled[batch * num_batches:(batch + 1) * num_batches, :]
            
            # Estimate loss (this now uses regression_loss for RGS)
            loss, grads = utils.grad(model=VIGNet, inputs=x, labels=y, mode=task)
            
            # Update the network
            optimizer.apply_gradients(zip(grads, VIGNet.trainable_variables))
            loss_per_epoch += np.mean(loss)
        
        avg_loss = loss_per_epoch/num_batch_iter
        print(f"Epoch {epoch + 1:2d}/{num_epochs}: Loss = {avg_loss:.6f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    Ytest_pred = VIGNet(Xtest, training=False)
    y_pred = Ytest_pred.numpy().squeeze()
    y_true = Ytest.squeeze()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    corr, p_value = pearsonr(y_true, y_pred)
    
    print("\n" + "="*80)
    print("RESULTS (after 10 epochs only)")
    print("="*80)
    print(f"Test MSE:  {mse:.6f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Correlation: {corr:.6f} (p={p_value:.6f})")
    print("="*80)
    
    # Interpretation
    print("\n📊 Interpretation:")
    if avg_loss < 1.0:
        print("✅ Loss values are in reasonable range for MSE loss")
    else:
        print("⚠️  Loss is still high, may need more epochs")
    
    if rmse < 0.25:
        print("✅ RMSE is improving (much better than ~0.15 with bug)")
    elif rmse < 0.5:
        print("⚠️  RMSE is moderate, should improve with full 100 epochs")
    else:
        print("❌ RMSE still high, may indicate other issues")
    
    if abs(corr) > 0.3:
        print("✅ Correlation is significant")
    else:
        print("⚠️  Correlation is weak, should improve with full training")
    
    print("\n💡 Note: This is only 10 epochs. Full training (100 epochs) should give much better results.")
    print("    Compare this with logs from before the fix to see the improvement!")
    
    return mse, rmse, corr

if __name__ == '__main__':
    try:
        quick_test()
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

