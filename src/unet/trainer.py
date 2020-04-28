from datetime import datetime
from pathlib import Path
from typing import Union, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from unet import utils, schedulers
from unet.callbacks import TensorBoardWithLearningRate, TensorBoardImageSummary
from unet.schedulers import SchedulerType


class Trainer:
    """
    Fits a given model to a datasets and configres learning rate schedulers and
    various callbacks

    :param name: Name of the model, used to build the target log directory if no explicit path is given
    :param log_dir_path: Path to the directory where the model and tensorboard summaries should be stored
    :param checkpoint_callback: Flag if checkpointing should be enabled. Alternatively a callback instance can be passed
    :param tensorboard_callback: Flag if information should be stored for tensorboard. Alternatively a callback instance can be passed
    :param tensorboard_images_callback: Flag if intermediate predictions should be stored in Tensorboard. Alternatively a callback instance can be passed
    :param callbacks: List of additional callbacks
    :param learning_rate_scheduler: The learning rate to be used. Either None for a constant learning rate, a `Callback` or a `SchedulerType`
    :param scheduler_opts: Further kwargs passed to the learning rate scheduler
    """

    def __init__(self,
                 name: Optional[str]="unet",
                 log_dir_path: Optional[Union[Path, str]]=None,
                 checkpoint_callback: Optional[Union[TensorBoard, bool]] = True,
                 tensorboard_callback: Optional[Union[TensorBoard, bool]] = True,
                 tensorboard_images_callback: Optional[Union[TensorBoardImageSummary, bool]] = True,
                 callbacks: Union[List[Callback], None]=None,
                 learning_rate_scheduler: Optional[Union[SchedulerType, Callback]]=None,
                 **scheduler_opts,
                 ):
        self.checkpoint_callback = checkpoint_callback
        self.tensorboard_callback = tensorboard_callback
        self.tensorboard_images_callback = tensorboard_images_callback
        self.callbacks = callbacks
        self.learning_rate_scheduler = learning_rate_scheduler
        self.scheduler_opts=scheduler_opts

        if log_dir_path is None:
            log_dir_path = build_log_dir_path(name)
        if isinstance(log_dir_path, Path):
            log_dir_path = str(log_dir_path)

        self.log_dir_path = log_dir_path

    def fit(self,
            model: Model,
            train_dataset: tf.data.Dataset,
            validation_dataset: Optional[tf.data.Dataset]=None,
            test_dataset: Optional[tf.data.Dataset]=None,
            epochs=10,
            batch_size=1,
            **fit_kwargs):
        """
        Fits the model to the given data

        :param model: The model to be fit
        :param train_dataset: The dataset used for training
        :param validation_dataset: (Optional) The dataset used for validation
        :param test_dataset:  (Optional) The dataset used for test
        :param epochs: Number of epochs
        :param batch_size: Size of minibatches
        :param fit_kwargs: Further kwargs passd to `model.fit`
        """

        prediction_shape = self._get_output_shape(model, train_dataset)[1:]

        learning_rate_scheduler = self._build_learning_rate_scheduler(train_dataset=train_dataset,
                                                                      batch_size=batch_size,
                                                                      epochs=epochs,
                                                                      **self.scheduler_opts)

        callbacks = self._build_callbacks(train_dataset,
                                          validation_dataset)

        if learning_rate_scheduler:
            callbacks += [learning_rate_scheduler]

        train_dataset = train_dataset.map(utils.crop_labels_to_shape(prediction_shape)).batch(batch_size)

        if validation_dataset:
            validation_dataset = validation_dataset.map(utils.crop_labels_to_shape(prediction_shape)).batch(batch_size)

        history = model.fit(train_dataset,
                            validation_data=validation_dataset,
                            epochs=epochs,
                            callbacks=callbacks,
                            **fit_kwargs)

        self.evaluate(model, test_dataset, prediction_shape)

        return history

    def _get_output_shape(self,
                          model: Model,
                          train_dataset: tf.data.Dataset):
        return model.predict(train_dataset
                             .take(count=1)
                             .batch(batch_size=1)
                             ).shape

    def _build_callbacks(self,
                         train_dataset: Optional[tf.data.Dataset],
                         validation_dataset: Optional[tf.data.Dataset]) -> List[Callback]:
        if self.callbacks:
           callbacks = self.callbacks
        else:
            callbacks = []

        if isinstance(self.checkpoint_callback, Callback):
            callbacks.append(self.checkpoint_callback)
        elif self.checkpoint_callback:
            callbacks.append(ModelCheckpoint(self.log_dir_path,
                                             save_best_only=True))

        if isinstance(self.tensorboard_callback, Callback):
            callbacks.append(self.tensorboard_callback)
        elif self.tensorboard_callback:
            callbacks.append(TensorBoardWithLearningRate(self.log_dir_path))

        if isinstance(self.tensorboard_images_callback, Callback):
            callbacks.append(self.tensorboard_images_callback)
        elif self.tensorboard_images_callback:
            tensorboard_image_summary = TensorBoardImageSummary("train",
                                                                self.log_dir_path,
                                                                dataset=train_dataset,
                                                                max_outputs=6)
            callbacks.append(tensorboard_image_summary)

            if validation_dataset:
                tensorboard_image_summary = TensorBoardImageSummary("validation",
                                                                    self.log_dir_path,
                                                                    dataset=validation_dataset,
                                                                    max_outputs=6)
                callbacks.append(tensorboard_image_summary)

        return callbacks

    def _build_learning_rate_scheduler(self,
                                       train_dataset: tf.data.Dataset,
                                       **scheduler_opts
                                       ) -> Optional[Callback]:

        if self.learning_rate_scheduler is None:
            return None

        if isinstance(self.learning_rate_scheduler, Callback):
            return self.learning_rate_scheduler

        elif isinstance(self.learning_rate_scheduler, SchedulerType):
            train_dataset_size = tf.data.experimental.cardinality(train_dataset).numpy()
            learning_rate_scheduler = schedulers.get(
                scheduler=self.learning_rate_scheduler,
                train_dataset_size=train_dataset_size,
                **scheduler_opts)

            return learning_rate_scheduler

    def evaluate(self,
                 model:Model,
                 test_dataset: Optional[tf.data.Dataset]=None,
                 shape:Tuple[int, int, int]=None):

        if test_dataset:
            model.evaluate(test_dataset
                           .map(utils.crop_labels_to_shape(shape))
                           .batch(batch_size=1)
                           )


def build_log_dir_path(root: Optional[str]= "unet") -> str:
    return str(Path(root) / datetime.now().strftime("%Y-%m-%dT%H-%M_%S"))
