import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras import initializers
import numpy as np


def conv_layer(x, layer_idx, filters_root, kernel_size, dropout_rate=0.5):
    with tf.name_scope("conv_layer"):
        filters = 2 ** layer_idx * filters_root
        stddev = np.sqrt(2 / (kernel_size ** 2 * filters))

        x = layers.Conv2D(filters=filters // 2,
                                      kernel_size=(kernel_size, kernel_size),
                                      kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                      strides=1,
                                      padding="valid")(x)
        x = layers.Dropout(rate=dropout_rate)(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(filters=filters,
                                      kernel_size=(kernel_size, kernel_size),
                                      strides=1,
                                      padding="valid")(x)
        x = layers.Dropout(rate=dropout_rate)(x)
        x = layers.Activation("relu")(x)

        return x


def max_pool_layer(x, pool_size):
    x = layers.MaxPooling2D((pool_size, pool_size))(x)
    return x


def upconv_layer(x, layer_idx, filters_root, kernel_size):
    filters = 2 ** layer_idx * filters_root
    stddev = np.sqrt(2 / (kernel_size ** 2 * filters))
    x = layers.Conv2DTranspose(filters,
                               kernel_size=(kernel_size, kernel_size),
                               kernel_initializer=initializers.TruncatedNormal(
                                   stddev=stddev),
                               strides=kernel_size,
                               padding="valid"
                               )(x)
    x = layers.Activation("relu")(x)
    return x


def crop_layer(x, down_layer):
    with tf.name_scope("crop_layer"):
        x1_shape = tf.shape(down_layer)
        x2_shape = tf.shape(x)

        height_diff = (x1_shape[1] - x2_shape[1]) // 2
        width_diff = (x1_shape[2] - x2_shape[2]) // 2

        down_layer_cropped = down_layer[:, height_diff: -height_diff, width_diff: -width_diff, :]

        x = tf.concat([down_layer_cropped, x], axis=-1)
        return x


def build_model(channels:int, n_class:int, layer_depth:int, filters_root:int, kernel_size=3, pool_size=2) -> Model:
    inputs = Input(shape=(None, None, channels))

    x = inputs
    down_layers = {}

    with tf.name_scope("contracting"):
        for layer_idx in range(0, layer_depth - 1):
            with tf.name_scope(f"contracting_{layer_idx}"):
                x = conv_layer(x, layer_idx, filters_root, kernel_size)
                down_layers[layer_idx] = x
                x = max_pool_layer(x, pool_size)

    with tf.name_scope("bottom"):
        x = conv_layer(x, layer_idx + 1, filters_root, kernel_size)

    with tf.name_scope("expanding"):
        for layer_idx in range(layer_depth - 2, -1, -1):
            with tf.name_scope(f"expanding_{layer_idx}"):
                x = upconv_layer(x, layer_idx, filters_root, pool_size)
                x = crop_layer(x, down_layers[layer_idx])
                x = conv_layer(x, layer_idx, filters_root, kernel_size)

    stddev = np.sqrt(2 / (kernel_size ** 2 * filters_root))
    x = layers.Conv2D(filters=n_class,
                      kernel_size=(1, 1),
                      kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                      strides=1,
                      padding="valid"
                      )(x)

    outputs = layers.Activation("softmax")(x)
    model = Model(inputs, outputs)
    return model
