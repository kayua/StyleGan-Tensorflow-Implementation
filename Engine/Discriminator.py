from keras import Input
from keras import Model
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import LayerNormalization
from keras.layers import Flatten

DEFAULT_LOSS_FUNCTION = "binary_crossentropy"
DEFAULT_OPTIMIZER_FUNCTION = "adam"
DEFAULT_VERBOSE_CONSTRUCTION = False
DEFAULT_NUMBER_CHANNELS = 3
DEFAULT_INITIAL_RESOLUTION = 4
DEFAULT_DIMENSION_CONVOLUTION_KERNELS = (3, 3)
DEFAULT_FILTER_PER_LAYER = [16, 32, 64, 96, 128, 256, 512]
DEFAULT_LEVEL_FEATURE_DIMENSION = [512, 256, 128, 64, 32, 16, 8]
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
        self.input_discriminator = []
        self.discriminator_blocks = []
        self.first_level_discriminator = None
        self.build_discriminator()

    def convolutional_block(self, resolution_feature, number_filters):

        input_layer = Input(shape=(resolution_feature, resolution_feature, self.number_channels))
        self.input_discriminator.append(input_layer)

        gradient_flow = Conv2D(number_filters, self.size_kernel_filters, padding="same")(input_layer)
        gradient_flow = LeakyReLU(self.threshold_activation)(gradient_flow)

        gradient_flow = Conv2D(self.number_channels, self.size_kernel_filters, padding="same")(gradient_flow)
        gradient_flow = LeakyReLU(self.threshold_activation)(gradient_flow)

        gradient_flow = MaxPooling2D((2, 2))(gradient_flow)
        gradient_flow = Model(input_layer, gradient_flow)
        gradient_flow.compile(loss=self.loss_function, optimizer=self.optimizer_function)

        self.discriminator_blocks.append(gradient_flow)
        if DEFAULT_VERBOSE_CONSTRUCTION: gradient_flow.summary()

    def build_discriminator(self):

        input_layer = Input(shape=(self.initial_resolution, self.initial_resolution, self.number_channels))
        number_layer = self.number_filters_per_layer[-1]
        self.first_level_discriminator = LayerNormalization()(input_layer)
        self.first_level_discriminator = Conv2D(number_layer, self.size_kernel_filters,
                                                padding="same")(self.first_level_discriminator)
        self.first_level_discriminator = LeakyReLU(self.threshold_activation)(self.first_level_discriminator)

        self.first_level_discriminator = Conv2D(number_layer, self.size_kernel_filters,
                                                padding="same")(self.first_level_discriminator)
        self.first_level_discriminator = LeakyReLU(self.threshold_activation)(self.first_level_discriminator)

        self.first_level_discriminator = self.fully_connected_block(self.first_level_discriminator)
        self.first_level_discriminator = Model(input_layer, self.first_level_discriminator)

        if DEFAULT_VERBOSE_CONSTRUCTION: self.first_level_discriminator.summary()

        for i in range(len(self.level_feature_dimension)):
            self.convolutional_block(self.level_feature_dimension[i], self.number_filters_per_layer[i])

    @staticmethod
    def fully_connected_block(input_layer):
        gradient_flow = Flatten()(input_layer)
        gradient_flow = Dense(1)(gradient_flow)
        return gradient_flow

    def get_discriminator(self, number_level):

        number_level -= 1

        if number_level == 0:
            self.first_level_discriminator.summary()
            return self.first_level_discriminator

        discriminator_input = self.input_discriminator[-number_level]
        convolutional_blocks = self.discriminator_blocks[-number_level]
        convolutional_blocks = Model(discriminator_input, convolutional_blocks.output)
        convolutional_blocks.compile(loss=self.loss_function, optimizer=self.optimizer_function)

        for i in range(number_level - 1):
            convolutional_blocks = self.discriminator_blocks[-(number_level - (i + 1))](convolutional_blocks.output)
            convolutional_blocks = Model(discriminator_input, convolutional_blocks)

        convolutional_blocks.compile(loss=self.loss_function, optimizer=self.optimizer_function)
        discriminator_network = self.first_level_discriminator(convolutional_blocks.output)
        discriminator_network = Model(discriminator_input, discriminator_network, name="Discriminator")
        discriminator_network.summary()

        return discriminator_network

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def set_optimizer_function(self, optimizer_function):
        self.optimizer_function = optimizer_function

    def set_initial_resolution(self, initial_resolution):
        self.initial_resolution = initial_resolution

    def set_number_channels(self, number_channels):
        self.number_channels = number_channels

    def set_level_feature_dimension(self, level_feature_dimension):
        self.level_feature_dimension = level_feature_dimension
