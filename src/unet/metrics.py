import tensorflow as tf


class MeanIoU(tf.keras.metrics.MeanIoU):
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.dtypes.float64)
        y_pred = tf.cast(y_pred, tf.dtypes.float64)
        I = tf.reduce_sum(y_pred * y_true, axis=(1, 2))
        U = tf.reduce_sum(y_pred + y_true, axis=(1, 2)) - I
        return tf.reduce_mean(I / U)
