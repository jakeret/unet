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
        self.logdir = str(Path(logdir) / "summaries")
        if max_outputs is None:
            max_outputs = self.images.shape[0]
        self.max_outputs = max_outputs

        self.dataset = dataset.take(self.max_outputs)

        self.file_writer = tf.summary.create_file_writer(self.logdir)

        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.dataset.batch(batch_size=1))

        self._log_histogramms(epoch, predictions)

        self._log_image_summaries(epoch, predictions)

        self.file_writer.flush()

    def _log_image_summaries(self, epoch, predictions):
        cropped_images, cropped_labels = list(self.dataset
                                              .map(utils.crop_image_and_label_to_shape(predictions.shape[1:]))
                                              .take(self.max_outputs)
                                              .batch(self.max_outputs))[0]

        output = self.combine_to_image(cropped_images.numpy(),
                                       cropped_labels.numpy(),
                                       predictions)

        with self.file_writer.as_default():
            tf.summary.image(self.name,
                             output,
                             step=epoch,
                             max_outputs=self.max_outputs)

    def combine_to_image(self, images: np.array, labels: np.array, predictions: np.array) -> np.array:
        """
        Concatenates the three tensors to one RGB image

        :param images: images tensor, shape [None, nx, ny, channels]
        :param labels: labels tensor, shape [None, nx, ny, 1] for sparse or [None, nx, ny, classes] for one-hot
        :param predictions: labels tensor, shape [None, nx, ny, classes]

        :return: image tensor, shape [None, nx, 3 x ny, 3]
        """

        if predictions.shape[-1] == 2:
            mask = predictions[..., :1]
        else:
            mask = np.argmax(predictions, axis=-1)[..., np.newaxis]

        output = np.concatenate((utils.to_rgb(images),
                                 utils.to_rgb(labels[..., :1]),
                                 utils.to_rgb(mask)),
                                axis=2)
        return output

    def _log_histogramms(self, epoch, predictions):
        with self.file_writer.as_default():
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
