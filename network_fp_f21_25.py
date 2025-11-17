# Import APIs
import tensorflow as tf
import numpy as np

# Define VIGNet for FP1/FP2 channels with only 5 frequency features (f21-25)
class vignet_fp_f21_25(tf.keras.Model):
    def __init__(self, mode):
        tf.keras.backend.set_floatx("float64")
        super(vignet_fp_f21_25, self).__init__()

        self.mode = mode

        self.regularizer = tf.keras.regularizers.L1L2(l1=0.05, l2=0.005)
        self.activation = tf.nn.leaky_relu
        
        # Define convolution layers adapted for 5 frequency features
        # Use smaller kernels and padding to handle reduced frequency dimension
        self.conv1 = tf.keras.layers.Conv2D(10, (1, 3), padding='same', kernel_regularizer=self.regularizer, activation=self.activation)
        self.conv2 = tf.keras.layers.Conv2D(10, (1, 3), padding='same', kernel_regularizer=self.regularizer, activation=self.activation)
        self.conv3 = tf.keras.layers.Conv2D(10, (1, 3), padding='same', kernel_regularizer=self.regularizer, activation=self.activation)
        # Changed from (17, 1) to (2, 1) for 2 channels
        self.conv4 = tf.keras.layers.Conv2D(20, (2, 1), kernel_regularizer=self.regularizer, activation=self.activation)

        self.flatten = tf.keras.layers.Flatten()

        if self.mode == "CLF": # for 3-class classification task
            self.dense = tf.keras.layers.Dense(3)
        elif self.mode == "RGS": # for PERCLOS score regression task
            self.dense = tf.keras.layers.Dense(1)

    # Define multi-head residual spectro-spatio attention module
    # Changed num_channel default from 17 to 2
    def MHRSSA(self, x, out_filter, num_channel=2):
        for i in range(out_filter):
            tmp = tf.keras.layers.Conv2D(num_channel, (num_channel, 1), kernel_regularizer=self.regularizer, activation=None)(x)
            if i == 0: MHRSSA = tmp
            else: MHRSSA = tf.concat((MHRSSA, tmp), 1)

        MHRSSA = tf.transpose(MHRSSA, perm=[0, 3, 2, 1])

        # Use smaller kernel for 5 frequency features
        MHRSSA = tf.keras.layers.DepthwiseConv2D((1, 3), padding='same', kernel_regularizer=self.regularizer, activation=None)(MHRSSA)
        MHRSSA = tf.keras.activations.softmax(MHRSSA)
        return MHRSSA


    def call(self, x):
        att1 = self.MHRSSA(x, 10)
        hidden = self.conv1(x)
        hidden *= att1

        att2 = self.MHRSSA(hidden, 10)
        hidden = self.conv2(hidden)
        hidden *= att2

        att3 = self.MHRSSA(hidden, 10)
        hidden = self.conv3(hidden)
        hidden *= att3

        hidden = self.conv4(hidden)

        hidden = self.flatten(hidden)
        hidden = self.dense(hidden)

        if self.mode == "CLF":
            y_hat = tf.keras.activations.softmax(hidden)
        elif self.mode == "RGS":
            y_hat = hidden
        return y_hat

