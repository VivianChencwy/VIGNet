# Import APIs
import tensorflow as tf
import numpy as np


class vignet_custom(tf.keras.Model):
    """
    VIGNet model adapted for custom EEG data
    
    Supports variable number of frequency bands (5 or 25)
    Uses 2 channels (FP1/FP2) by default
    """
    def __init__(self, mode, n_bands=25, n_channels=2):
        """
        Initialize VIGNet model
        
        Args:
            mode: 'CLF' for classification, 'RGS' for regression
            n_bands: Number of frequency bands (5 or 25)
            n_channels: Number of EEG channels (default 2 for FP1/FP2)
        """
        tf.keras.backend.set_floatx("float64")
        super(vignet_custom, self).__init__()

        self.mode = mode
        self.n_bands = n_bands
        self.n_channels = n_channels

        # Regularization to reduce overfitting
        self.regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2=0.05)
        self.activation = tf.nn.leaky_relu
        
        # Adjust kernel size based on number of bands
        # For 25 bands, use kernel size 5 (like original VIGNet)
        # For 5 bands, use kernel size 3 or 2
        if n_bands >= 25:
            kernel_size = 5
        elif n_bands >= 10:
            kernel_size = 3
        else:
            kernel_size = 2
        
        # Define convolution layers
        self.conv1 = tf.keras.layers.Conv2D(10, (1, kernel_size), 
                                            kernel_regularizer=self.regularizer, activation=None)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        
        self.conv2 = tf.keras.layers.Conv2D(10, (1, kernel_size), 
                                            kernel_regularizer=self.regularizer, activation=None)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        
        self.conv3 = tf.keras.layers.Conv2D(10, (1, kernel_size), 
                                            kernel_regularizer=self.regularizer, activation=None)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(0.3)
        
        # Final conv layer for channel mixing
        self.conv4 = tf.keras.layers.Conv2D(20, (n_channels, 1), 
                                            kernel_regularizer=self.regularizer, 
                                            activation=self.activation)

        self.flatten = tf.keras.layers.Flatten()

        if self.mode == "CLF":  # for 3-class classification task
            self.dense = tf.keras.layers.Dense(3)
        elif self.mode == "RGS":  # for PERCLOS score regression task
            self.dense = tf.keras.layers.Dense(1)

    def MHRSSA(self, x, out_filter):
        """
        Multi-head Residual Spectro-Spatio Attention module
        
        Args:
            x: Input tensor
            out_filter: Number of output filters
        
        Returns:
            Attention weights
        """
        for i in range(out_filter):
            tmp = tf.keras.layers.Conv2D(self.n_channels, (self.n_channels, 1), 
                                         kernel_regularizer=self.regularizer, activation=None)(x)
            if i == 0:
                MHRSSA = tmp
            else:
                MHRSSA = tf.concat((MHRSSA, tmp), 1)

        MHRSSA = tf.transpose(MHRSSA, perm=[0, 3, 2, 1])

        # Adjust depthwise conv kernel based on remaining spatial dimensions
        kernel_size = min(5, x.shape[2] if x.shape[2] is not None else 5)
        MHRSSA = tf.keras.layers.DepthwiseConv2D((1, kernel_size), 
                                                  kernel_regularizer=self.regularizer, 
                                                  activation=None)(MHRSSA)
        MHRSSA = tf.keras.activations.softmax(MHRSSA)
        return MHRSSA

    def call(self, x, training=False):
        att1 = self.MHRSSA(x, 10)
        hidden = self.conv1(x)
        hidden = self.bn1(hidden, training=training)
        hidden = self.activation(hidden)
        hidden *= att1
        hidden = self.dropout1(hidden, training=training)

        att2 = self.MHRSSA(hidden, 10)
        hidden = self.conv2(hidden)
        hidden = self.bn2(hidden, training=training)
        hidden = self.activation(hidden)
        hidden *= att2
        hidden = self.dropout2(hidden, training=training)

        att3 = self.MHRSSA(hidden, 10)
        hidden = self.conv3(hidden)
        hidden = self.bn3(hidden, training=training)
        hidden = self.activation(hidden)
        hidden *= att3
        hidden = self.dropout3(hidden, training=training)

        hidden = self.conv4(hidden)

        hidden = self.flatten(hidden)
        hidden = self.dense(hidden)

        if self.mode == "CLF":
            y_hat = tf.keras.activations.softmax(hidden)
        elif self.mode == "RGS":
            y_hat = hidden
        return y_hat


class vignet_custom_simple(tf.keras.Model):
    """
    Simplified VIGNet model for smaller feature dimensions
    
    Uses a simpler architecture when n_bands is small (e.g., 5)
    """
    def __init__(self, mode, n_bands=5, n_channels=2):
        tf.keras.backend.set_floatx("float64")
        super(vignet_custom_simple, self).__init__()

        self.mode = mode
        self.n_bands = n_bands
        self.n_channels = n_channels

        # Regularization
        self.regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2=0.05)
        self.activation = tf.nn.leaky_relu
        
        # Simpler architecture for small input
        self.conv1 = tf.keras.layers.Conv2D(16, (1, 2), padding='same',
                                            kernel_regularizer=self.regularizer, activation=None)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        
        self.conv2 = tf.keras.layers.Conv2D(32, (n_channels, 1),
                                            kernel_regularizer=self.regularizer, activation=None)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation=self.activation)
        self.dropout3 = tf.keras.layers.Dropout(0.3)

        if self.mode == "CLF":
            self.dense_out = tf.keras.layers.Dense(3)
        elif self.mode == "RGS":
            self.dense_out = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        hidden = self.conv1(x)
        hidden = self.bn1(hidden, training=training)
        hidden = self.activation(hidden)
        hidden = self.dropout1(hidden, training=training)
        
        hidden = self.conv2(hidden)
        hidden = self.bn2(hidden, training=training)
        hidden = self.activation(hidden)
        hidden = self.dropout2(hidden, training=training)
        
        hidden = self.flatten(hidden)
        hidden = self.dense1(hidden)
        hidden = self.dropout3(hidden, training=training)
        hidden = self.dense_out(hidden)

        if self.mode == "CLF":
            y_hat = tf.keras.activations.softmax(hidden)
        elif self.mode == "RGS":
            y_hat = hidden
        return y_hat


