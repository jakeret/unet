from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, Callback

from unet.utils import crop_to_shape, to_rgb


class TensorBoardImageSummary(Callback):

    def __init__(self, name,
                 logdir: str,
                 images: np.array,
                 labels: np.array,
                 max_outputs: int = None):
        self.name = name
        self.logdir = str(Path(logdir) / "summary")
        if max_outputs is None:
            max_outputs = self.images.shape[0]
        self.max_outputs = max_outputs

        self.images = images[:self.max_outputs]
        self.labels = labels[:self.max_outputs]
        self.file_writer = tf.summary.create_file_writer(self.logdir)
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model.predict(self.images, batch_size=1)
        cropped_images = crop_to_shape(self.images, prediction.shape)
        cropped_labels = crop_to_shape(self.labels, prediction.shape)

        output = np.concatenate((to_rgb(cropped_images),
                                 to_rgb(cropped_labels[..., :1]),
                                 to_rgb(prediction[..., :1])),
                                axis=2)

        with self.file_writer.as_default():
            tf.summary.image(self.name,
                             output,
                             step=epoch,
                             max_outputs=self.max_outputs)

            tf.summary.histogram(self.name + "_prediction_histograms",
                                 prediction,
                                 step=epoch,
                                 buckets=30,
                                 description=None)


class TensorBoardWithLearningRate(TensorBoard):
    def on_epoch_end(self, batch, logs=None):
        logs = logs or {}
        logs['learning_rate'] = K.get_value(self.model.optimizer.lr)
        super().on_epoch_end(batch, logs)
