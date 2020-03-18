import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.api._v2.keras import backend as K
from tensorflow_core.python.keras.callbacks import Callback, TensorBoard

from unet.utils import crop_to_shape, to_rgb


class TensorboardImageSummary(Callback):

    def __init__(self, logdir:str, images:np.array, labels:np.array, max_outputs:int=None):
        self.logdir = logdir
        self.images = images
        self.labels = labels
        if max_outputs is None:
            max_outputs = self.images.shape[0]
        self.max_outputs = max_outputs
        self.file_writer = tf.summary.create_file_writer(logdir)
        super(TensorboardImageSummary, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model.predict(self.images)
        cropped_images = crop_to_shape(self.images, prediction.shape)
        cropped_labels = crop_to_shape(self.labels, prediction.shape)

        output = np.concatenate((to_rgb(cropped_images),
                                 to_rgb(cropped_labels[..., :1]),
                                 to_rgb(prediction[..., :1])),
                                axis=2)

        with self.file_writer.as_default():
            tf.summary.image("Training data", output, step=epoch, max_outputs=self.max_outputs)


class TensorBoardWithLearningRate(TensorBoard):
    def on_epoch_end(self, batch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        super().on_epoch_end(batch, logs)
