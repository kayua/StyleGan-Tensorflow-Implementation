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
DEFAULT_VERBOSE_CONSTRUCTION = True
DEFAULT_NUMBER_CHANNELS = 3
DEFAULT_INITIAL_RESOLUTION = 4
DEFAULT_DIMENSION_CONVOLUTION_KERNELS = (3, 3)
DEFAULT_FILTER_PER_LAYER = [64, 64, 128, 128, 256, 256, 512, 512]
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
        self.input_discriminator = None
        self.discriminator_blocks = None
        self.first_level_discriminator = None


    def build_initial_block(self):


    @staticmethod
    def fully_connected_block(input_layer):
        gradient_flow = Flatten()(input_layer)
        gradient_flow = Dense(1)(gradient_flow)
        return gradient_flow



discriminator_instance = Discriminator()

discriminator_instance.get_discriminator(2)