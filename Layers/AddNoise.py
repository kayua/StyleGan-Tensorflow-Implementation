from tensorflow import keras

from tensorflow.python.layers.base import Layer


class AddNoise(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.b = None

    def build(self, input_shape):
        n, h, w, c = input_shape[0]
        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
        self.b = self.add_weight(shape=[1, 1, 1, c], initializer=initializer, trainable=True, name="kernel")

    def call(self, inputs, **kwargs):
        x, noise = inputs
        output = x + self.b * noise
        return output
