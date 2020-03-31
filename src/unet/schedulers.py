import logging
from enum import Enum
from typing import Callable

import tensorflow as tf
import tensorflow.keras.backend as K

logger = logging.getLogger(__name__)


class SchedulerType(Enum):
    WARMUP_LINEAR_DECAY = "warmup-linear-decay"


def get(scheduler:SchedulerType, train_dataset_size:int, learning_rate:float, **hyperparams):
    if scheduler == SchedulerType.WARMUP_LINEAR_DECAY:
        batch_size = hyperparams["batch_size"]
        steps_per_epoch = (train_dataset_size + batch_size - 1) // batch_size
        total_steps = steps_per_epoch * hyperparams["epochs"]
        warmup_steps = int(total_steps * hyperparams["warmup_proportion"])
        logger.info("Total steps %s, warum steps %s", total_steps, warmup_steps)

        schedule = WarmupLinearDecaySchedule(warmup_steps, total_steps, learning_rate)
        return LearningRateScheduler(schedule, steps_per_epoch, verbose=0)
    else:
        raise ValueError("Unknown scheduler %s"%scheduler)


class LearningRateScheduler(tf.keras.callbacks.Callback):
    # Currently, the optimizers in TF2 don't properly support LR schedulers as callable.
    # As alternative we have to use a Keras callback which only allows for updating the LR per batch instead per step

    """Learning rate scheduler.
    Arguments:
        schedule: a function that takes an step index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule:Callable[[int], float], steps_per_epoch:int, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.steps_per_epoch = steps_per_epoch
        self.verbose = verbose
        self._current_step = 0

    def on_train_batch_begin(self, batch, logs=None):
        new_lr = self.schedule(self._current_step)

        K.set_value(self.model.optimizer.lr, new_lr)

        self._current_step += 1

        if self.verbose > 0:
            logger.info('\nBatch %05d: LearningRateScheduler changing learning rate to %s.', batch + 1, new_lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['learning_rate'] = K.get_value(self.model.optimizer.lr)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['learning_rate'] = K.get_value(self.model.optimizer.lr)


class WarmupLinearDecaySchedule:
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, warmup_steps, total_steps, learning_rate, min_lr=0.0):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.initial_learning_rate = learning_rate
        self.min_lr = min_lr
        self.decay_steps = max(1.0, self.total_steps - self.warmup_steps)

    def __call__(self, step):
        if step < self.warmup_steps:
            learning_rate = self.initial_learning_rate * float(step) / max(1., self.warmup_steps)
        else:
            decay_factor = max(0, (self.total_steps - step) / self.decay_steps)
            learning_rate = self.min_lr + (self.initial_learning_rate - self.min_lr) * decay_factor

        return learning_rate
