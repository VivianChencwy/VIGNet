"""
VIGNet model adapted for experiment data
- 2 channels (FP1, FP2)
- 25 frequency bands (1-49Hz, 2Hz steps)
- Single regression output
"""

import tensorflow as tf
import numpy as np


class VIGNetExperiment(tf.keras.Model):
    """
    VIGNet for experiment data with 2 channels and 25 frequency bands.
    Uses Multi-Head Residual Spectro-Spatio Attention (MHRSSA) mechanism.
    """
    
    def __init__(self, num_channels=2, num_freqs=25):
        """
        Args:
            num_channels: Number of EEG channels (default: 2 for FP1, FP2)
            num_freqs: Number of frequency bands (default: 25)
        """
        tf.keras.backend.set_floatx("float64")
        super(VIGNetExperiment, self).__init__()
        
        self.num_channels = num_channels
        self.num_freqs = num_freqs
        
        # L1L2 regularization to prevent overfitting
        self.regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2=0.05)
        self.activation = tf.nn.leaky_relu
        
        # Convolutional layers with BatchNorm and Dropout
        # Use larger kernels for 25 frequency bands
        self.conv1 = tf.keras.layers.Conv2D(
            filters=16, 
            kernel_size=(1, 5), 
            padding='same', 
            kernel_regularizer=self.regularizer, 
            activation=None
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        
        self.conv2 = tf.keras.layers.Conv2D(
            filters=16, 
            kernel_size=(1, 5), 
            padding='same', 
            kernel_regularizer=self.regularizer, 
            activation=None
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        
        self.conv3 = tf.keras.layers.Conv2D(
            filters=16, 
            kernel_size=(1, 5), 
            padding='same', 
            kernel_regularizer=self.regularizer, 
            activation=None
        )
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(0.3)
        
        # Spatial convolution across channels
        self.conv4 = tf.keras.layers.Conv2D(
            filters=32, 
            kernel_size=(num_channels, 1), 
            kernel_regularizer=self.regularizer, 
            activation=self.activation
        )
        
        self.flatten = tf.keras.layers.Flatten()
        
        # Dense layers
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=self.regularizer)
        self.dropout_dense = tf.keras.layers.Dropout(0.3)
        
        # Output layer for regression
        self.output_layer = tf.keras.layers.Dense(1)
    
    def MHRSSA(self, x, out_filter):
        """
        Multi-Head Residual Spectro-Spatio Attention module.
        
        Args:
            x: Input tensor of shape (batch, channels, freqs, filters)
            out_filter: Number of output filters/heads
        
        Returns:
            Attention weights tensor
        """
        for i in range(out_filter):
            # Spatial attention across channels
            tmp = tf.keras.layers.Conv2D(
                self.num_channels, 
                (self.num_channels, 1), 
                kernel_regularizer=self.regularizer, 
                activation=None
            )(x)
            
            if i == 0:
                mhrssa = tmp
            else:
                mhrssa = tf.concat((mhrssa, tmp), 1)
        
        # Transpose for frequency attention
        mhrssa = tf.transpose(mhrssa, perm=[0, 3, 2, 1])
        
        # Depthwise convolution for spectral attention
        mhrssa = tf.keras.layers.DepthwiseConv2D(
            (1, 5), 
            padding='same', 
            kernel_regularizer=self.regularizer, 
            activation=None
        )(mhrssa)
        
        # Softmax attention weights
        mhrssa = tf.keras.activations.softmax(mhrssa)
        
        return mhrssa
    
    def call(self, x, training=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, freqs, 1)
            training: Boolean for training mode (affects dropout and batchnorm)
        
        Returns:
            Regression output of shape (batch, 1)
        """
        # First attention block
        att1 = self.MHRSSA(x, 16)
        hidden = self.conv1(x)
        hidden = self.bn1(hidden, training=training)
        hidden = self.activation(hidden)
        hidden *= att1
        hidden = self.dropout1(hidden, training=training)
        
        # Second attention block
        att2 = self.MHRSSA(hidden, 16)
        hidden = self.conv2(hidden)
        hidden = self.bn2(hidden, training=training)
        hidden = self.activation(hidden)
        hidden *= att2
        hidden = self.dropout2(hidden, training=training)
        
        # Third attention block
        att3 = self.MHRSSA(hidden, 16)
        hidden = self.conv3(hidden)
        hidden = self.bn3(hidden, training=training)
        hidden = self.activation(hidden)
        hidden *= att3
        hidden = self.dropout3(hidden, training=training)
        
        # Spatial convolution
        hidden = self.conv4(hidden)
        
        # Flatten and dense layers
        hidden = self.flatten(hidden)
        hidden = self.dense1(hidden)
        hidden = self.dropout_dense(hidden, training=training)
        
        # Output
        y_hat = self.output_layer(hidden)
        
        return y_hat


def create_model(num_channels=2, num_freqs=25):
    """
    Create a VIGNet model for experiment data.
    
    Args:
        num_channels: Number of EEG channels
        num_freqs: Number of frequency bands
    
    Returns:
        VIGNetExperiment model instance
    """
    return VIGNetExperiment(num_channels=num_channels, num_freqs=num_freqs)


if __name__ == "__main__":
    # Test model creation and forward pass
    print("Testing VIGNetExperiment model...")
    
    model = create_model()
    
    # Create dummy input
    batch_size = 8
    dummy_input = tf.random.normal((batch_size, 2, 25, 1), dtype=tf.float64)
    
    # Forward pass
    output = model(dummy_input, training=False)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {model.count_params()}")
    
    # Print model summary
    model.build(input_shape=(None, 2, 25, 1))
    model.summary()


