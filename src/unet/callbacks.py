from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, Callback

from unet import utils


class TensorBoardImageSummary(Callback):

    def __init__(self, name,
                 logdir: str,
                 dataset: tf.data.Dataset,
                 max_outputs: int = None):
        self.name = name
        self.logdir = str(Path(logdir) / "summary")
        if max_outputs is None:
            max_outputs = self.images.shape[0]
        self.max_outputs = max_outputs

        self.dataset = dataset.take(self.max_outputs)

        self.file_writer = tf.summary.create_file_writer(self.logdir)
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.dataset.batch(batch_size=1))

        prediction_shape = predictions.shape[1:]

        cropped_images, cropped_labels = list(self.dataset
                                              .map(utils.crop_image_and_label_to_shape(prediction_shape))
                                              .take(self.max_outputs)
                                              .batch(self.max_outputs))[0]

        if prediction_shape[-1] == 2:
            predictions = predictions[..., :1]
            predictions = utils.to_rgb(predictions)
        else:
            predictions = np.argmax(predictions, axis=-1)[..., np.newaxis]
            predictions = np.tile(predictions, 3)

        output = np.concatenate((utils.to_rgb(cropped_images),
                                 utils.to_rgb(cropped_labels[..., :1]),
                                 predictions),
                                axis=2)

        with self.file_writer.as_default():
            tf.summary.image(self.name,
                             output,
                             step=epoch,
                             max_outputs=self.max_outputs)

            tf.summary.histogram(self.name + "_prediction_histograms",
                                 predictions,
                                 step=epoch,
                                 buckets=30,
                                 description=None)


class TensorBoardWithLearningRate(TensorBoard):
    def on_epoch_end(self, batch, logs=None):
        logs = logs or {}
        logs['learning_rate'] = K.get_value(self.model.optimizer.lr)
        super().on_epoch_end(batch, logs)
