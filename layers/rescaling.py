import tensorflow as tf
from tensorflow.python.ops import math_ops

class Rescaling(tf.keras.layers.Layer):
    def __init__(self, scale, offset, name=None, **kwargs):
        self.scale = scale
        self.offset = offset
        super(Rescaling, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        dtype = self._compute_dtype
        return math_ops.cast(inputs, dtype) * math_ops.cast(self.scale, dtype) + math_ops.cast(self.offset, dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'scale': self.scale,
            'offset': self.offset,
        }
        base_config = super(Rescaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
