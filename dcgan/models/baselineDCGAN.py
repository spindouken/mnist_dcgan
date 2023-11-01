#!/usr/bin/env python3
from tensorflow.keras import layers
import tensorflow as tf

# why does the filter count decrease with each layer in GAN?
# imagine you're painting a picture, you start with a rough sketch of broad strokes and general shapes (high-level features),
#   then you add more and more detail and refine the picture (low-level features), however you don't need as many strokes to add detail
# each higher resolution layer adds finer details, and you might not need as many feature maps as in the lower
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(100,)), #  input shape is 100-dimensional noise vector, outputs a 7x7x256 tensor
        # normalize output to avoid internal covariate shift
        layers.BatchNormalization(),
        # activation function to introduce non-linearity
        layers.LeakyReLU(),
        # reshape the vector into a 7x7x256 tensor, prepare tensor for deconvolution(upsampling)
        layers.Reshape((7, 7, 256)),
        # upsample the tensor to 7x7x64 (deepen feature map from 256 to 64)
        layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        # upsample the tensor to 14x14x32
        layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        # final transpose convolutional layer, outputting a 28x28x1 tensor to generate final image (grayscale: output channel=1)
        # tanh activation so output is between -1 and 1
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        # takes in 28x28x1 image and applies n filters of size x * x (n, (x, x)) with strides = (y, y)
        layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        layers.LeakyReLU(),
        # randomly set 30% of input units to 0 at each update during training time, helps prevent overfitting
        layers.Dropout(0.3),
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        # randomly set 30% of input units to 0 at each update during training time, helps prevent overfitting
        layers.Dropout(0.3),
        # flatten the tensor into 1D vector
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model
