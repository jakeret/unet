from unittest.mock import Mock

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

import unet
from unet.callbacks import TensorBoardWithLearningRate, TensorBoardImageSummary
from unet.schedulers import LearningRateScheduler


def _build_dataset(items=5, image_shape=(10, 10, 3), label_shape=(10, 10, 2)):
    images = np.ones((items, *image_shape))
    labels = np.ones((items, *label_shape))
    return tf.data.Dataset.from_tensor_slices((images, labels))


class TestTrainer:

    def test_fit(self, tmp_path):
        output_shape = (8, 8, 2)
        image_shape = (10, 10, 3)
        epochs = 5
        shuffle = True
        batch_size = 10

        model = Mock(name="model")
        model.predict().shape = (None, *output_shape)

        mock_callback = Mock()
        trainer = unet.Trainer(name="test",
                               log_dir_path=str(tmp_path),
                               checkpoint_callback=True,
                               tensorboard_callback=True,
                               tensorboard_images_callback=True,
                               callbacks=[mock_callback],
                               learning_rate_scheduler=unet.SchedulerType.WARMUP_LINEAR_DECAY,
                               warmup_proportion=0.1,
                               learning_rate=1.0)

        train_dataset = _build_dataset(image_shape=image_shape)
        validation_dataset = _build_dataset(image_shape=image_shape)
        test_dataset = _build_dataset(image_shape=image_shape)

        trainer.fit(model,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    test_dataset=test_dataset,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=shuffle)

        args, kwargs = model.fit.call_args
        train_dataset = args[0]
        validation_dataset = kwargs["validation_data"]

        assert tuple(train_dataset.element_spec[0].shape) == (None, *image_shape)
        assert tuple(train_dataset.element_spec[1].shape) == (None, *output_shape)
        assert train_dataset._batch_size.numpy() == batch_size

        assert validation_dataset._batch_size.numpy() == batch_size
        assert tuple(validation_dataset.element_spec[0].shape) == (None, *image_shape)
        assert tuple(validation_dataset.element_spec[1].shape) == (None, *output_shape)

        callbacks = kwargs["callbacks"]
        callback_types = [type(callback) for callback in callbacks]
        assert mock_callback in callbacks
        assert ModelCheckpoint in callback_types
        assert TensorBoardWithLearningRate in callback_types
        assert TensorBoardImageSummary in callback_types
        assert LearningRateScheduler in callback_types

        assert kwargs["epochs"] == epochs
        assert kwargs["shuffle"] == shuffle

        args, kwargs = model.evaluate.call_args
        test_dataset = args[0]
        assert tuple(test_dataset.element_spec[0].shape) == (None, *image_shape)
        assert tuple(test_dataset.element_spec[1].shape) == (None, *output_shape)
