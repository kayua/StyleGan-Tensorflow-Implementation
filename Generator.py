import logging

import tensorflow
from keras import Input
from keras import Model
from keras.layers import Conv2D, Flatten
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.layers import Reshape
from keras.layers import UpSampling2D

from Layers.AdaIN import AdaIN
from Layers.AddNoise import AddNoise

tensorflow.get_logger().setLevel(logging.ERROR)
level_size_feature_dimension = [4, 4, 8, 16, 32, 64, 128, 256, 512]

DEFAULT_VERBOSE_CONSTRUCTION = True


class Generator:

    def __init__(self):

        self.latent_dimension = 128
        self.num_neurons_mapping = 512
        self.num_mapping_blocks = 4
        self.initial_dimension = 4
        self.initial_num_channels = 256
        self.number_output_channels = 3
        self.mapping_neural_network = None
        self.size_kernel_filters = (3, 3)
        self.num_synthesis_block = 8
        self.constant_mapping_neural_network = None
        self.input_block = None
        self.list_block_synthesis = []
        self.list_level_noise_input = []
        self.loss_function = "binary_crossentropy"
        self.optimizer_function = "adam"
        self.num_filters_per_level = [256, 256, 256, 256, 256, 256, 256, 256, 256]
        self.latent_input = None
        self.initial_flow = None
        self.build_blocks()

    def block_mapping_network(self):

        latent_dimension_input = Input(shape=(self.latent_dimension, 1))
        gradient_flow = Flatten()(latent_dimension_input)
        gradient_flow = Dense(self.num_neurons_mapping)(gradient_flow)
        gradient_flow = LeakyReLU(0.2)(gradient_flow)

        for i in range(self.num_mapping_blocks - 2):
            gradient_flow = Dense(self.num_neurons_mapping)(gradient_flow)
            gradient_flow = LeakyReLU(0.2)(gradient_flow)

        gradient_flow = Dense(self.latent_dimension)(gradient_flow)
        network_model = Model(latent_dimension_input, gradient_flow, name="Mapping_Network")
        self.mapping_neural_network = network_model
        if DEFAULT_VERBOSE_CONSTRUCTION:
            self.mapping_neural_network.summary()


    def constant_mapping_block(self):

        dimension_latent_vector = 2 ** self.initial_dimension * self.initial_num_channels
        latent_input = Input(shape=(dimension_latent_vector, 1))
        mapping_format = (self.initial_dimension, self.initial_dimension, self.initial_num_channels)
        gradient_flow = Reshape(mapping_format)(latent_input)
        gradient_flow = Model(latent_input, gradient_flow, name="Constant_Block")
        gradient_flow.compile(loss=self.loss_function, optimizer=self.optimizer_function)
        self.constant_mapping_neural_network = gradient_flow
        if DEFAULT_VERBOSE_CONSTRUCTION:
            self.constant_mapping_neural_network.summary()

    def initial_block_synthesis(self, resolution_block, number_filters):

        input_flow = Input(shape=(resolution_block, resolution_block, number_filters))
        input_noise = Input(shape=(resolution_block, resolution_block, number_filters))
        input_latent = Input(shape=self.latent_dimension)
        input_latent = Reshape((self.latent_dimension, 1))(input_latent)
        gradient_flow = AddNoise()([input_flow, input_noise])
        gradient_flow = AdaIN()([gradient_flow, input_latent])
        gradient_flow = Conv2D(number_filters, self.size_kernel_filters, padding="same")(gradient_flow)
        gradient_flow = LeakyReLU(0.2)(gradient_flow)
        gradient_flow = AddNoise()([gradient_flow, input_noise])
        gradient_flow = AdaIN()([gradient_flow, input_latent])
        gradient_flow = Model([input_flow, input_noise, input_latent], gradient_flow)
        gradient_flow.compile(loss=self.loss_function, optimizer=self.optimizer_function)
        if DEFAULT_VERBOSE_CONSTRUCTION:
            gradient_flow.summary()
        return gradient_flow

    def non_initial_synthesis_block(self, resolution_block, number_filters):

        input_flow = Input(shape=(int(resolution_block), int(resolution_block), number_filters))
        input_noise = Input(shape=(resolution_block * 2, resolution_block * 2, number_filters))
        input_latent = Input(shape=self.latent_dimension)
        input_latent = Reshape((self.latent_dimension, 1))(input_latent)
        gradient_flow = UpSampling2D((2, 2))(input_flow)
        gradient_flow = AddNoise()([gradient_flow, input_noise])

        gradient_flow = AdaIN()([gradient_flow, input_latent])
        gradient_flow = Conv2D(number_filters, self.size_kernel_filters, padding="same")(gradient_flow)
        gradient_flow = LeakyReLU(0.2)(gradient_flow)
        gradient_flow = AddNoise()([gradient_flow, input_noise])
        gradient_flow = AdaIN()([gradient_flow, input_latent])
        gradient_flow = Model([input_flow, input_noise, input_latent], gradient_flow)
        gradient_flow.compile(loss=self.loss_function, optimizer=self.optimizer_function)
        if DEFAULT_VERBOSE_CONSTRUCTION:
            gradient_flow.summary()
        return gradient_flow

    def build_synthesis_block(self):

        input_flow = Input(shape=(self.initial_dimension, self.initial_dimension, self.initial_num_channels),
                           name="Input Mapping")
        input_latent = Input(shape=self.latent_dimension, name="Input Latent")
        self.latent_input = input_latent
        self.initial_flow = input_flow

        input_noise = Input(shape=(self.initial_dimension, self.initial_dimension, self.initial_num_channels),
                            name="Input Noise 1")
        first_level_block = self.initial_block_synthesis(self.initial_dimension, self.initial_num_channels)
        first_level_block = first_level_block([input_flow, input_noise, input_latent])
        self.list_block_synthesis.append(first_level_block)
        self.list_level_noise_input.append(input_noise)

        for i in range(self.num_synthesis_block - 1):
            resolution_feature = level_size_feature_dimension[i + 1]
            input_noise = Input(shape=(resolution_feature * 2, resolution_feature * 2, self.initial_num_channels),
                                name="Input Noise {}".format(i + 2))
            self.list_level_noise_input.append(input_noise)
            level_block = self.non_initial_synthesis_block(resolution_feature, self.initial_num_channels)
            level_block = level_block([self.list_block_synthesis[-1], self.list_level_noise_input[-1], input_latent])

            self.list_block_synthesis.append(level_block)

    def build_blocks(self):

        self.build_synthesis_block()
        self.block_mapping_network()
        self.constant_mapping_block()

    def color_mapping(self, resolution, number_channels_flow):

        input_color_mapping = Input(shape=(resolution, resolution, number_channels_flow))
        color_mapping = Conv2D(self.number_output_channels, (1, 1), padding="same")(input_color_mapping)
        color_mapping = Model(input_color_mapping, color_mapping)
        color_mapping.compile(loss=self.loss_function, optimizer=self.optimizer_function)
        if DEFAULT_VERBOSE_CONSTRUCTION:
            color_mapping.summary()
        return color_mapping

    def get_generator(self, number_level):

        neural_input_layer = [self.list_level_noise_input[i] for i in range(number_level)]
        neural_input_layer.append(self.initial_flow)
        neural_input_layer.append(self.latent_input)

        synthesis_model = Model(neural_input_layer, self.list_block_synthesis[number_level - 1])
        synthesis_model.compile(loss=self.loss_function, optimizer=self.optimizer_function)
        last_level_dimension = level_size_feature_dimension[number_level]
        last_level_filters = self.num_filters_per_level[number_level]
        neural_synthesis = self.color_mapping(last_level_dimension, last_level_filters)
        neural_synthesis = neural_synthesis([synthesis_model.output])
        neural_synthesis = Model(synthesis_model.inputs, neural_synthesis)

        neural_input_layer = [self.list_level_noise_input[i] for i in range(number_level)]
        neural_input_layer.append(self.initial_flow)
        neural_input_layer.append(self.mapping_neural_network.input)
        style_generator = neural_synthesis(neural_input_layer)
        style_generator = Model(neural_input_layer, style_generator)
        style_generator.summary()
        return style_generator


a = Generator()
a.get_generator(3)
