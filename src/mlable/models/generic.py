import tensorflow as tf

import mlable.masking

# CONTRAST #####################################################################

@tf.keras.utils.register_keras_serializable(package='models')
class ContrastModel(tf.keras.models.Model):
    def compute_loss(
        self,
        x: tf.Tensor=None,
        y: tf.Tensor=None,
        y_pred: tf.Tensor=None,
        sample_weight: tf.Tensor=None,
    ):
        # weight according to the difference between x and y (reduced)
        __weights = mlable.masking.contrast(
            left=x,
            right=tf.cast(tf.argmax(y, axis=-1), dtype=x.dtype),
            weight=getattr(self, '_contrast_weight', 0.8),
            dtype=sample_weight.dtype)
        # combine with the sample weights
        __weights = __weights * sample_weight if (sample_weight is not None) else __weights
        # apply the original loss and reduction of the model
        return super(ContrastModel, self).compute_loss(x, y, y_pred, __weights)
