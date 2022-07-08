from keras.layers import Reshape, UpSampling2D
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

        gradient_flow = inputs[0]
        style_flow = inputs[1]
        gradient_flow_means, gradient_flow_stander_deviation = self.get_mean_std(gradient_flow)
        style_mean, style_stander_deviation = self.get_mean_std(style_flow)

        dimension_expanded = (gradient_flow.shape[1], gradient_flow.shape[1])

        style_stander_deviation = Reshape((1, 1, 1))(style_stander_deviation)
        style_stander_deviation = UpSampling2D(size=dimension_expanded)(style_stander_deviation)

        gradient_flow_means = Reshape((1, 1, gradient_flow.shape[3]))(gradient_flow_means)
        gradient_flow_means = UpSampling2D(size=dimension_expanded)(gradient_flow_means)

        gradient_flow_stander_deviation = Reshape((1, 1, gradient_flow.shape[3]))(gradient_flow_stander_deviation)
        gradient_flow_stander_deviation = UpSampling2D(size=dimension_expanded)(gradient_flow_stander_deviation)

        style_mean = Reshape((1, 1, 1))(style_mean)
        style_mean = UpSampling2D(size=dimension_expanded)(style_mean)

        divergence = (gradient_flow - gradient_flow_means)
        return style_stander_deviation * divergence / gradient_flow_stander_deviation + style_mean

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
