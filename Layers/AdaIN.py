import tensorflow
from tensorflow.python.layers.base import Layer

from Layers.EqualizedDense import EqualizedDense


class AdaIN(Layer):
    def __init__(self, gain=1, **kwargs):
        super(AdaIN, self).__init__(**kwargs)
        self.gain = gain

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        w_shape = input_shapes[1]

        self.w_channels = w_shape[-1]
        self.x_channels = x_shape[-1]

        self.dense_1 = EqualizedDense(self.x_channels, gain=1)
        self.dense_2 = EqualizedDense(self.x_channels, gain=1)

    def call(self, inputs):
        x, w = inputs
        ys = tensorflow.reshape(self.dense_1(w), (-1, 1, 1, self.x_channels))
        yb = tensorflow.reshape(self.dense_2(w), (-1, 1, 1, self.x_channels))
        return ys * x + yb