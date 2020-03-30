# unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with unet.  If not, see <http://www.gnu.org/licenses/>.
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import unet
from unet import schedulers
from unet.callbacks import TensorBoardImageSummary, TensorBoardWithLearningRate
from unet.datasets import circles
from unet.utils import crop_to_shape

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class MeanIoU(tf.keras.metrics.Metric):
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.dtypes.float64)
        y_pred = tf.cast(y_pred, tf.dtypes.float64)
        I = tf.reduce_sum(y_pred * y_true, axis=(1, 2))
        U = tf.reduce_sum(y_pred + y_true, axis=(1, 2)) - I
        return tf.reduce_mean(I / U)



def train(unet_model, train_dataset, validation_dataset, batch_size, epochs, learning_rate=1e-3):
    x_train, y_train = train_dataset
    x_val, y_val = validation_dataset

    optimizer = Adam(learning_rate=learning_rate)
    num_classes = y_train.shape[-1]
    unet_model.compile(loss=losses.categorical_crossentropy,
                       optimizer=optimizer,
                       metrics=['categorical_crossentropy',
                                'categorical_accuracy',
                                MeanIoU(num_classes=num_classes)],
                       )

    prediction = unet_model.predict(x_train[:1])
    log_dir = Path("circles") / datetime.now().strftime("%Y-%m-%dT%H-%M_%S")
    train_samples = len(train_dataset[0])
    steps_per_epoch = (train_samples + batch_size - 1) // batch_size

    callbacks = [ModelCheckpoint(str(log_dir)),
                 TensorBoardImageSummary(log_dir,
                                         images=validation_dataset[0],
                                         labels=validation_dataset[1],
                                         max_outputs=6),
                 TensorBoardWithLearningRate(str(log_dir)),
                 schedulers.get(scheduler=schedulers.WARMUP_LINEAR_DECAY,
                                steps_per_epoch=steps_per_epoch, learning_rate=learning_rate,
                                batch_size=batch_size, epochs=epochs, warmup_proportion=0.1),
                 ]
    history = unet_model.fit(
        x=x_train,
        y=crop_to_shape(y_train, prediction.shape),
        validation_data=(x_val, crop_to_shape(y_val, prediction.shape)),
        epochs=epochs,
        batch_size=batch_size,
        # steps_per_epoch=steps_per_epoch,
        callbacks=callbacks
    )

    # unet_model.save(str(log_dir / "model.h5"))
    return history


def evaluate(unet_model, validation_dataset):
    x_val, y_val = validation_dataset
    prediction = unet_model.predict(x_val[:1])
    unet_model.evaluate(x_val,
                        crop_to_shape(y_val, prediction.shape))


if __name__ == '__main__':
    np.random.seed(98765)

    unet_model = unet.build_model(channels=circles.channels,
                                  num_classes=circles.classes,
                                  layer_depth=3,
                                  filters_root=16)

    print(unet_model.summary())
    train_dataset, validation_dataset = circles.load_data(100, nx=172, ny=172, circles=20)

    train(unet_model,
          train_dataset,
          validation_dataset,
          batch_size=1,
          epochs=25)

    evaluate(unet_model, validation_dataset)
