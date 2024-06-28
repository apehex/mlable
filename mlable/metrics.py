import math
import random

import keras
import tensorflow as tf

import mlable.utils

# ACCURACY ####################################################################

def token_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor, group: int=4) -> tuple:
    # category indexes
    __yt = tf.argmax(y_true, axis=-1)
    __yp = tf.argmax(y_pred, axis=-1)
    # matching
    __match = tf.equal(__yt, __yp)
    # group all the predictions for a given token
    if group and group > 1:
        __shape = mlable.utils.normalize_shape(list(__match.shape))
        # split
        __shape[-1] = mlable.utils.divide_dim(dim_l=__shape[-1], dim_r=group)
        __shape.append(group)
        # reshape
        __match = tf.reshape(__match, shape=__shape)
        # the token prediction is right if ALL its byte predictions are right
        __match = tf.reduce_all(__match, axis=-1)
    # count
    __total = math.prod([1 if not __d else __d for __d in list(__match.shape)])
    # cast
    __total = tf.convert_to_tensor(__total, dtype=tf.dtypes.int32)
    __match = tf.cast(__match, dtype=tf.dtypes.int32)
    # sum
    return (__total, tf.reduce_sum(__match))

class TokenAccuracy(tf.keras.metrics.Metric):
    def __init__(self, token_dim: int=4, name: str='token_accuracy', **kwargs):
        # init
        super(TokenAccuracy, self).__init__(name=name, **kwargs)
        # token size
        self._token = token_dim
        # state
        self._total = self.add_weight(shape=(), dtype=tf.dtypes.int32, name='total', initializer='zeros')
        self._correct = self.add_weight(shape=(), dtype=tf.dtypes.int32, name='correct', initializer='zeros')

    def reset_state(self) -> None:
        self._total.assign(0)
        self._correct.assign(0)

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor=None) -> None:
        __total, __correct = token_accuracy(y_true=y_true, y_pred=y_pred, group=self._token)
        # actually update
        self._total.assign_add(__total)
        self._correct.assign_add(__correct)

    def result(self) -> tf.Tensor:
        return tf.cast(self._correct, dtype=tf.dtypes.float32) / tf.cast(self._total, dtype=tf.dtypes.float32)

    def get_config(self) -> dict:
        return {"token_dim": self._token, "name": self.name,}

# LOSS ########################################################################
