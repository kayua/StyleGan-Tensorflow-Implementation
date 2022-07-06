from keras.layers import Reshape
from keras.utils import conv_utils
import tensorflow
from tensorflow.python.layers.base import Layer


class AdaIN(Layer):

    def __init__(self, data_format=None, eps=1e-7, **kwargs):
        super(AdaIN, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.spatial_axis = [1, 2] if self.data_format == 'channels_last' else [2, 3]
        self.eps = eps

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs, **kwargs):
        content = inputs[0]
        style = inputs[1]
        content_mean, content_std = self.get_mean_std(content)
        style_mean, style_std = self.get_mean_std(style)

        dimension_expanded = (content.shape[1], content.shape[1])
        style_std = tensorflow.keras.layers.UpSampling2D(size=dimension_expanded)(Reshape((1, 1, 1))(style_std))
        content_mean = tensorflow.keras.layers.UpSampling2D(size=dimension_expanded)(Reshape((1, 1, 256))(content_mean))
        content_std = tensorflow.keras.layers.UpSampling2D(size=dimension_expanded)(Reshape((1, 1, 256))(content_std))
        style_mean = tensorflow.keras.layers.UpSampling2D(size=dimension_expanded)(Reshape((1, 1, 1))(style_mean))

        out = style_std * (content - content_mean) / content_std + style_mean

        return out

    @staticmethod
    def get_mean_std(x, epsilon=1e-5):
        axes = [1, 2]
        mean, variance = tensorflow.nn.moments(x, axes=axes, keepdims=True)
        standard_deviation = tensorflow.sqrt(variance + epsilon)
        return mean, standard_deviation

    def get_config(self):
        config = {'eps': self.eps}
        base_config = super(AdaIN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
