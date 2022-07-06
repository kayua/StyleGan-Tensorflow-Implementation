
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

    def call(self, inputs):
        image = inputs[0]
        if len(inputs) == 2:
            style = inputs[1]
            style_mean, style_var = tensorflow.nn.moments(style, self.spatial_axis, keepdims=True)
        else:
            style_mean = tensorflow.expand_dims(tensorflow.expand_dims(inputs[1], self.spatial_axis[0]), self.spatial_axis[1])
            style_var = tensorflow.expand_dims(tensorflow.expand_dims(inputs[2], self.spatial_axis[0]), self.spatial_axis[1])
        image_mean, image_var = tensorflow.nn.moments(image, self.spatial_axis, keepdims=True)
        out = tensorflow.nn.batch_normalization(image, image_mean,
                                         image_var, style_mean,
                                         tensorflow.sqrt(style_var), self.eps)
        return out

    def get_config(self):
        config = {'eps': self.eps}
        base_config = super(AdaIN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))