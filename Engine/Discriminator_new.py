import tensorflow
from keras import Input
from keras import Model
from keras.layers import AveragePooling2D
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LeakyReLU

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
        self.input_discriminator_mapping = None
        self.discriminator = None
        self.first_level_discriminator = None
        #self.build_initial_block()

    @staticmethod
    def mini_batch_std(input_tensor, epsilon=1e-8):
        n, h, w, c = tensorflow.shape(input_tensor)
        group_size = tensorflow.minimum(4, n)
        x = tensorflow.reshape(input_tensor, [group_size, -1, h, w, c])
        group_mean, group_var = tensorflow.nn.moments(x, axes=0, keepdims=False)
        group_std = tensorflow.sqrt(group_var + epsilon)
        avg_std = tensorflow.reduce_mean(group_std, axis=[1, 2, 3], keepdims=True)
        x = tensorflow.tile(avg_std, [group_size, h, w, 1])
        return tensorflow.concat([input_tensor, x], axis=-1)



    def color_mapping(self, resolution, number_features):
        input_layer = Input(shape=(resolution, resolution, self.number_channels))
        weight_kernels = tensorflow.keras.initializers.Ones()
        color_mapping = Conv2D(number_features, (1, 1), kernel_initializer=weight_kernels, trainable=False)(input_layer)
        color_mapping = Model(input_layer, color_mapping)
        color_mapping.summary()
        return color_mapping



    def convolutional_block(self, number_filters, resolution):

        image_resolution = (resolution, resolution, 512)
        input_layer = Input(shape=image_resolution, name="s")
        gradient_flow = Conv2D(number_filters, self.size_kernel_filters, padding="same")(input_layer)
        gradient_flow = LeakyReLU(number_filters)(gradient_flow)
        gradient_flow = Conv2D(number_filters, self.size_kernel_filters, padding="same")(gradient_flow)
        gradient_flow = LeakyReLU(number_filters)(gradient_flow)
        gradient_flow = AveragePooling2D((2, 2))(gradient_flow)
        input_layer = Model(input_layer, input_layer, name="sss")
        modelo = self.discriminator.layers[1]
        modelo = modelo(input_layer)
        models = Model(modelo, self.discriminator)
        models.summary()
        exit()




    def build_initial_block(self):

        number_filters = self.number_filters_per_layer[-1]
        resolution_mapping = self.color_mapping(self.initial_resolution, number_filters)
        input_feature = Input(shape=(self.initial_resolution, self.initial_resolution, number_filters))

        self.input_discriminator = None
        self.input_discriminator_mapping = None
        self.discriminator = None
        self.first_level_discriminator = None

        number_filters = self.number_filters_per_layer[-1]

        gradient_flow = Conv2D(number_filters, self.size_kernel_filters, padding="same")()
        self.input_discriminator = gradient_flow
        gradient_flow = LeakyReLU(number_filters)(gradient_flow)
        gradient_flow = Conv2D(number_filters, (4, 4), padding="same")(gradient_flow)
        gradient_flow = LeakyReLU(number_filters)(gradient_flow)
        gradient_flow = self.fully_connected_block(gradient_flow)

        gradient_flow = Model(input_mapping, gradient_flow)

        gradient_flow.compile(loss=self.loss_function, optimizer=self.optimizer_function)
        self.discriminator = gradient_flow
        self.discriminator.summary()

    @staticmethod
    def fully_connected_block(input_layer):
        gradient_flow = Flatten()(input_layer)
        gradient_flow = Dense(1)(gradient_flow)
        return gradient_flow

    def get_discriminator(self, level_discriminator):
        return self.discriminator

discriminator_instance = Discriminator()

discriminator_instance.color_mapping(4)