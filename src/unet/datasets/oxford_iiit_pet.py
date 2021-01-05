from typing import Tuple, Dict

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core import DatasetInfo

tfds.disable_progress_bar()

IMAGE_SIZE = (128, 128)
channels = 3
classes = 3


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], IMAGE_SIZE)
    input_mask = tf.image.resize(datapoint['segmentation_mask'], IMAGE_SIZE)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], IMAGE_SIZE)
    input_mask = tf.image.resize(datapoint['segmentation_mask'], IMAGE_SIZE)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_data(buffer_size=1000, **kwargs) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    dataset, info = _load_without_checksum_verification(**kwargs)
    train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test = dataset['test'].map(load_image_test)
    train_dataset = train.cache().shuffle(buffer_size).take(info.splits["train"].num_examples)
    return train_dataset, test


def _load_without_checksum_verification(**kwargs) -> Tuple[Dict, DatasetInfo]:
    builder = tfds.builder('oxford_iiit_pet:3.2.0')
    # by setting register_checksums as True to pass the check
    config = tfds.download.DownloadConfig(register_checksums=True)
    builder.download_and_prepare(download_config=config)
    dataset = builder.as_dataset()

    return dataset, (builder.info)
