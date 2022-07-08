import tensorflow
from keras import Input
from keras import Model
from keras.layers import AveragePooling2D
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.optimizer_v1 import Adam

from Neural import discriminator_loss

DEFAULT_LOSS_FUNCTION = discriminator_loss
DEFAULT_OPTIMIZER_FUNCTION = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
DEFAULT_VERBOSE_CONSTRUCTION = True
DEFAULT_NUMBER_CHANNELS = 3
DEFAULT_INITIAL_RESOLUTION = 4
DEFAULT_DIMENSION_CONVOLUTION_KERNELS = (3, 3)
DEFAULT_FILTER_PER_LAYER = [16, 16, 32, 32, 64, 64, 128, 128]
DEFAULT_LEVEL_FEATURE_DIMENSION = [1024, 512, 256, 128, 64, 32, 16, 8]
DEFAULT_THRESHOLD_ACTIVATION = 0.2


class Discriminator:

    def __init__(self, loss_function=DEFAULT_LOSS_FUNCTION, optimizer_function=DEFAULT_OPTIMIZER_FUNCTION,
                 number_channels=DEFAULT_NUMBER_CHANNELS, initial_resolution=DEFAULT_INITIAL_RESOLUTION,
                 threshold_activation=DEFAULT_THRESHOLD_ACTIVATION,
                 number_filters_per_layer=None, level_feature_dimension=None):

        if number_filters_per_layer is None: number_filters_per_layer = DEFAULT_FILTER_PER_LAYER

        if level_feature_dimension is None: level_feature_dimension = DEFAULT_LEVEL_FEATURE_DIMENSION

        self.loss_function = loss_function
        self.optimizer_function = optimizer_function
        self.initial_resolution = initial_resolution
        self.number_channels = number_channels
        self.number_filters_per_layer = number_filters_per_layer
        self.level_feature_dimension = level_feature_dimension
        self.size_kernel_filters = DEFAULT_DIMENSION_CONVOLUTION_KERNELS
        self.threshold_activation = threshold_activation
        self.discriminator_mapping = None
        self.discriminator = None
        self.discriminator_level = 1
        self.first_level_discriminator = None
        self.build_initial_block()

    @staticmethod
    def mini_batch_stander(input_tensor, epsilon=1e-8):
        batch_size, dimension_x, dimension_y, channels = tensorflow.shape(input_tensor)
        group_size = tensorflow.minimum(4, batch_size)
        dimension_shape = [group_size, -1, dimension_x, dimension_y, channels]
        gradient_flow = tensorflow.reshape(input_tensor, dimension_shape)
        group_mean, group_var = tensorflow.nn.moments(gradient_flow, axes=0, keepdims=False)
        group_std = tensorflow.sqrt(group_var + epsilon)
        average_stander = tensorflow.reduce_mean(group_std, axis=[1, 2, 3], keepdims=True)
        gradient_flow = tensorflow.tile(average_stander, [group_size, dimension_x, dimension_y, 1])
        return tensorflow.concat([input_tensor, gradient_flow], axis=-1)

    def color_mapping(self, resolution, number_features):
        input_layer = Input(shape=(resolution, resolution, self.number_channels))
        weight_kernels = tensorflow.keras.initializers.Ones()
        color_mapping = Conv2D(number_features, (1, 1), kernel_initializer=weight_kernels, trainable=False)(input_layer)
        color_mapping = Model(input_layer, color_mapping)
        return color_mapping

    def build_initial_block(self):

        number_filters = self.number_filters_per_layer[-2]
        resolution_mapping = self.color_mapping(self.initial_resolution, number_filters)
        input_feature = Input(shape=(self.initial_resolution, self.initial_resolution, number_filters))
        kernel_filters = self.size_kernel_filters

        gradient_flow = Conv2D(self.number_filters_per_layer[-1], kernel_filters, padding="same")(input_feature)
        gradient_flow = LeakyReLU(self.threshold_activation)(gradient_flow)
        gradient_flow = Conv2D(self.number_filters_per_layer[-1], (4, 4), padding="same")(gradient_flow)
        gradient_flow = LeakyReLU(self.threshold_activation)(gradient_flow)
        gradient_flow = self.fully_connected_block(gradient_flow)
        gradient_flow = Model(input_feature, gradient_flow)

        self.discriminator = gradient_flow
        self.discriminator_mapping = self.discriminator(resolution_mapping.output)
        self.discriminator_mapping = Model(resolution_mapping.input, self.discriminator_mapping)

    def add_level_discriminator(self, number_level):

        number_filters = self.number_filters_per_layer[-(number_level + 1)]
        resolution_feature = self.level_feature_dimension[-number_level]
        next_number_filters = self.number_filters_per_layer[-(number_level + 2)]
        resolution_mapping = self.color_mapping(resolution_feature, next_number_filters)
        input_feature = Input(shape=(resolution_feature, resolution_feature, next_number_filters))

        gradient_flow = Conv2D(number_filters, self.size_kernel_filters, padding="same")(input_feature)
        gradient_flow = LeakyReLU(self.threshold_activation)(gradient_flow)
        gradient_flow = Conv2D(number_filters, self.size_kernel_filters, padding="same")(gradient_flow)
        gradient_flow = LeakyReLU(self.threshold_activation)(gradient_flow)
        gradient_flow = AveragePooling2D()(gradient_flow)
        gradient_flow = self.discriminator(gradient_flow)
        gradient_flow = Model(input_feature, gradient_flow)

        self.discriminator = gradient_flow
        self.discriminator_mapping = self.discriminator(resolution_mapping.output)
        self.discriminator_mapping = Model(resolution_mapping.input, self.discriminator_mapping)
        self.discriminator_level += 1
        self.discriminator.summary()

    def get_discriminator(self, number_level):

        for i in range(self.discriminator_level, number_level):
            self.add_level_discriminator(i)

        return self.discriminator_mapping

    @staticmethod
    def fully_connected_block(input_layer):
        gradient_flow = Flatten()(input_layer)
        gradient_flow = Dense(1)(gradient_flow)
        return gradient_flow
