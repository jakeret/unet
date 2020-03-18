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

from datetime import datetime
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
import numpy as np
import unet
import tensorflow as tf

from unet.callbacks import TensorboardImageSummary, TensorBoardWithLearningRate
from unet.datasets import circles
from unet.utils import crop_to_shape

if __name__ == '__main__':
    np.random.seed(98765)

    unet_model = unet.build_model(channels=circles.channels,
                                  n_class=circles.classes,
                                  layer_depth=3,
                                  filters_root=16)

    print(unet_model.summary())
    train, validation = circles.load_data(10, nx=572, ny=572, circles=20)

    x_train, y_train = train
    # dataset_train = tf.data.Dataset.from_tensor_slices(train)
    # dataset_validation = tf.data.Dataset.from_tensor_slices(validation)

    optimizer = Adam()
    unet_model.compile(loss=losses.categorical_crossentropy,
                       optimizer=optimizer,
                       metrics=['categorical_crossentropy',
                                'categorical_accuracy',
                                tf.keras.metrics.MeanIoU(num_classes=2)],
                       )

    prediction = unet_model.predict(x_train[:1])

    log_dir = Path("circles") / datetime.now().strftime("%Y-%m-%dT%H-%M_%S")
    epochs = 10
    batch_size = 32
    steps_per_epoch = (len(train[0]) + batch_size - 1) // batch_size

    history = unet_model.fit(
        train[0],
        crop_to_shape(train[1], prediction.shape),
        # dataset_train.repeat(epochs),
        #                             validation_data=dataset_validation,
        epochs=epochs,
        batch_size=batch_size,
        # steps_per_epoch=steps_per_epoch,
        callbacks=[TensorBoardWithLearningRate(log_dir),
                   TensorboardImageSummary(str(log_dir),
                                           images=validation[0],
                                           labels=validation[1],
                                           max_outputs=3)]
    )

    unet_model.evaluate(validation[0],
                        crop_to_shape(validation[1], prediction.shape))
