from typing import Optional, Union, Callable, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.optimizers import Adam

from unet.metrics import MeanIoU


def conv_block(layer_idx, filters_root, kernel_size, dropout_rate):
    def block(x):
        filters = 2 ** layer_idx * filters_root
        stddev = np.sqrt(2 / (kernel_size ** 2 * filters))

        x = layers.Conv2D(filters=filters,
                          kernel_size=(kernel_size, kernel_size),
                          kernel_initializer=TruncatedNormal(stddev=stddev),
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

    return block


def upconv_block(layer_idx, filters_root, kernel_size, pool_size):
    def block(x):
        filters = 2 ** (layer_idx + 1) * filters_root
        stddev = np.sqrt(2 / (kernel_size ** 2 * filters))
        x = layers.Conv2DTranspose(filters // 2,
                                   kernel_size=(pool_size, pool_size),
                                   kernel_initializer=TruncatedNormal(stddev=stddev),
                                   strides=pool_size,
                                   padding="valid"
                                   )(x)
        x = layers.Activation("relu")(x)

        return x

    return block


def crop_concat_block():
    def block(x, down_layer):
        x1_shape = tf.shape(down_layer)
        x2_shape = tf.shape(x)

        height_diff = (x1_shape[1] - x2_shape[1]) // 2
        width_diff = (x1_shape[2] - x2_shape[2]) // 2

        down_layer_cropped = down_layer[:, height_diff: -height_diff, width_diff: -width_diff, :]

        x = tf.concat([down_layer_cropped, x], axis=-1)
        return x

    return block


def build_model(channels:int,
                num_classes:int,
                layer_depth:int,
                filters_root:int,
                kernel_size:int=3,
                pool_size:int=2,
                dropout_rate:int=0.5) -> Model:

    inputs = Input(shape=(None, None, channels))

    x = inputs
    down_layers = {}

    with tf.name_scope("contracting"):
        for layer_idx in range(0, layer_depth - 1):
            with tf.name_scope(f"contracting_{layer_idx}"):
                x = conv_block(layer_idx, filters_root, kernel_size, dropout_rate)(x)
                down_layers[layer_idx] = x
                x = layers.MaxPooling2D((pool_size, pool_size))(x)

    with tf.name_scope("bottom"):
        x = conv_block(layer_idx + 1, filters_root, kernel_size, dropout_rate)(x)

    with tf.name_scope("expanding"):
        for layer_idx in range(layer_depth - 2, -1, -1):
            with tf.name_scope(f"expanding_{layer_idx}"):
                x = upconv_block(layer_idx, filters_root, kernel_size, pool_size)(x)
                x = crop_concat_block()(x, down_layers[layer_idx])
                x = conv_block(layer_idx, filters_root, kernel_size, dropout_rate)(x)

    stddev = np.sqrt(2 / (kernel_size ** 2 * filters_root * 2))
    x = layers.Conv2D(filters=num_classes,
                      kernel_size=(1, 1),
                      kernel_initializer=TruncatedNormal(stddev=stddev),
                      strides=1,
                      padding="valid"
                      )(x)
    x = layers.Activation("relu")(x)
    outputs = layers.Activation("softmax")(x)
    model = Model(inputs, outputs, name="unet")
    return model


def finalize_model(model,
                   loss: Optional[Union[Callable, str]]=losses.categorical_crossentropy,
                   optimizer: Optional= None,
                   metrics:Optional[List[Union[Callable,str]]]=None,
                   mean_iou: bool=True,
                   num_classes: Optional[int]=2,
                   **opt_kwargs):

    if optimizer is None:
        optimizer = Adam(**opt_kwargs)

    if metrics is None:
        metrics = ['categorical_crossentropy', 'categorical_accuracy']

    if mean_iou:
        metrics += [MeanIoU(num_classes=num_classes)]

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics,
                  )
