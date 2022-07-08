#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'All'
__email__ = '@unipampa.edu.br '
__version__ = '{2}.{0}.{1}'
__data__ = '2021/11/21'
__credits__ = ['All']

import json
import logging
import tensorflow
from keras import Input
from keras import activations
from keras import Model
from keras.layers import Add
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.layers import Reshape
from keras.layers import UpSampling2D
from keras.optimizer_v1 import Adam

from Engine.Layers.AdaIN import AdaIN
from Neural import generator_loss

tensorflow.get_logger().setLevel(logging.ERROR)

DEFAULT_VERBOSE_CONSTRUCTION = False
DEFAULT_LATENT_DIMENSION = 256
DEFAULT_NUMBER_NEURONS_MAPPING = 512
DEFAULT_NUMBER_MAPPING_BLOCKS = 4
DEFAULT_INITIAL_FEATURE_DIMENSION = 4
DEFAULT_INITIAL_NUMBER_CHANNELS = 64
DEFAULT_NUMBER_OUTPUT_CHANNELS = 3
DEFAULT_THRESHOLD_RELU = 0.2
DEFAULT_DIMENSION_CONVOLUTION_KERNELS = (3, 3)
DEFAULT_NUMBER_SYNTHESIS_BLOCKS = 8
DEFAULT_LEARNING_RATE = 0.0002
DEFAULT_BETA_1 = 0.5
DEFAULT_BETA_2 = 0.9
DEFAULT_LOSS_FUNCTION = generator_loss
DEFAULT_NUMBER_FILTERS_PER_LEVEL = [64, 64, 64, 64, 64, 64, 64, 64, 64]
DEFAULT_FEATURE_SIZE = [4, 4, 8, 16, 32, 64, 128, 256, 512]
DEFAULT_OPTIMIZER_FUNCTION = Adam(learning_rate=DEFAULT_LEARNING_RATE,
                                  beta_1=DEFAULT_BETA_1, beta_2=DEFAULT_BETA_2)


class Generator:

    def __init__(self, latent_dimension=DEFAULT_LATENT_DIMENSION, num_neurons_mapping=DEFAULT_NUMBER_NEURONS_MAPPING,
                 num_mapping_blocks=DEFAULT_NUMBER_MAPPING_BLOCKS, initial_dimension=DEFAULT_INITIAL_FEATURE_DIMENSION,
                 loss_function=DEFAULT_LOSS_FUNCTION, number_output_channels=DEFAULT_NUMBER_OUTPUT_CHANNELS,
                 optimizer_function=DEFAULT_OPTIMIZER_FUNCTION, level_verbose=DEFAULT_VERBOSE_CONSTRUCTION,
                 size_kernel_filters=DEFAULT_DIMENSION_CONVOLUTION_KERNELS, num_filters_per_level=None,
                 num_synthesis_block=DEFAULT_NUMBER_SYNTHESIS_BLOCKS, feature_size=None):

        if feature_size is None: feature_size = DEFAULT_FEATURE_SIZE
        if num_filters_per_level is None: num_filters_per_level = DEFAULT_NUMBER_FILTERS_PER_LEVEL

        self.latent_dimension = latent_dimension
        self.num_neurons_mapping = num_neurons_mapping
        self.num_mapping_blocks = num_mapping_blocks
        self.initial_dimension = initial_dimension
        self.initial_num_channels = DEFAULT_NUMBER_FILTERS_PER_LEVEL[-1]
        self.number_output_channels = number_output_channels
        self.size_kernel_filters = size_kernel_filters
        self.num_synthesis_block = num_synthesis_block
        self.loss_function = loss_function
        self.feature_size = feature_size
        self.optimizer_function = optimizer_function
        self.num_filters_per_level = num_filters_per_level
        self.level_verbose = level_verbose
        self.list_block_synthesis = []
        self.list_level_noise_input = []
        self.input_block = None
        self.latent_input = None
        self.initial_gradient_flow = None
        self.mapping_neural_network = None
        self.constant_mapping_neural_network = None
        self.__build_blocks()

    def set_latent_dimension(self, latent_dimension):
        self.latent_dimension = latent_dimension

    def set_number_neurons_mapping(self, number_neurons_mapping):
        self.num_neurons_mapping = number_neurons_mapping

    def set_number_mapping_block(self, number_mapping_blocks):
        self.num_mapping_blocks = number_mapping_blocks

    def set_initial_dimension(self, initial_dimension):
        self.initial_dimension = initial_dimension

    def set_initial_number_channels(self, initial_number_channels):
        self.initial_num_channels = initial_number_channels

    def set_number_output_channels(self, number_output_channels):
        self.number_output_channels = number_output_channels

    def set_dimension_kernel_filters(self, dimension_filter_kernels):
        self.size_kernel_filters = dimension_filter_kernels

    def set_number_synthesis_block(self, number_synthesis_block):
        self.num_synthesis_block = number_synthesis_block

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def set_optimizer_function(self, optimizer_function):
        self.optimizer_function = optimizer_function

    def set_number_filters_per_level(self, number_filter_per_level):
        self.num_filters_per_level = number_filter_per_level

    def get_latent_dimension(self):
        return self.latent_dimension

    def get_number_neurons_mapping(self):
        return self.num_neurons_mapping

    def get_number_mapping_block(self):
        return self.num_mapping_blocks

    def get_initial_dimension(self):
        return self.initial_dimension

    def get_initial_number_channels(self):
        return self.initial_num_channels

    def get_number_output_channels(self):
        return self.number_output_channels

    def get_dimension_kernel_filters(self):
        return self.size_kernel_filters

    def get_number_synthesis_block(self):
        return self.num_synthesis_block

    def get_loss_function(self):
        return self.loss_function

    def get_optimizer_function(self):
        return self.optimizer_function

    def get_number_filters_per_level(self):
        return self.num_filters_per_level

    def write_data_generator(self, discriminator_data_file):


        discriminator_data = {"latent_dimension": self.latent_dimension,
                              "num_neurons_mapping": self.num_neurons_mapping,
                              "num_mapping_blocks": self.num_mapping_blocks,
                              "initial_dimension": self.initial_dimension,
                              "initial_num_channels": self.initial_num_channels,
                              "number_output_channels": self.number_output_channels,
                              "size_kernel_filters": self.size_kernel_filters,
                              "num_synthesis_block": self.num_synthesis_block,
                              "loss_function": self.loss_function,

                              }


        self.feature_size = feature_size
        self.optimizer_function = optimizer_function
        self.num_filters_per_level = num_filters_per_level
        self.level_verbose = level_verbose








        with open("{}.json".format(discriminator_data_file), "w") as outfile:
            json.dump(discriminator_data, outfile)


    def __block_mapping_network(self):

        shape_latent_mapping = (self.latent_dimension, 1)
        latent_dimension_input = Input(shape=shape_latent_mapping, name="Latent Input")
        gradient_flow = Flatten()(latent_dimension_input)
        gradient_flow = Dense(self.num_neurons_mapping)(gradient_flow)
        gradient_flow = LeakyReLU(DEFAULT_THRESHOLD_RELU)(gradient_flow)

        for _ in range(self.num_mapping_blocks - 2):
            gradient_flow = Dense(self.num_neurons_mapping)(gradient_flow)
            gradient_flow = LeakyReLU(DEFAULT_THRESHOLD_RELU)(gradient_flow)

        gradient_flow = Dense(self.latent_dimension)(gradient_flow)
        network_model = Model(latent_dimension_input, gradient_flow, name="Mapping_Network")
        self.mapping_neural_network = network_model
        if self.level_verbose: self.mapping_neural_network.summary()

    def __constant_mapping_block(self):

        dimension_latent_vector = 2 ** self.initial_dimension * self.initial_num_channels
        latent_shape = (dimension_latent_vector, 1)
        latent_input = Input(shape=latent_shape)
        mapping_format = (self.initial_dimension, self.initial_dimension, self.initial_num_channels)
        gradient_flow = Reshape(mapping_format)(latent_input)
        gradient_flow = Model(latent_input, gradient_flow, name="Constant_Block")
        gradient_flow.compile(loss=self.loss_function, optimizer=self.optimizer_function)
        self.constant_mapping_neural_network = gradient_flow
        if self.level_verbose: self.constant_mapping_neural_network.summary()

    def __initial_block_synthesis(self, resolution_block, number_filters):

        resolution_features_block = (resolution_block, resolution_block, number_filters)

        input_flow = Input(shape=resolution_features_block)
        input_noise = Input(shape=(resolution_block, resolution_block, 1))
        input_latent = Input(shape=self.latent_dimension)
        input_latent = Reshape((self.latent_dimension, 1))(input_latent)

        gradient_flow = Add()([input_flow, input_noise])
        gradient_flow = AdaIN()([gradient_flow, input_latent])
        gradient_flow = Conv2D(number_filters, self.size_kernel_filters, padding="same")(gradient_flow)
        gradient_flow = LeakyReLU(DEFAULT_THRESHOLD_RELU)(gradient_flow)
        gradient_flow = Add()([gradient_flow, input_noise])
        gradient_flow = AdaIN()([gradient_flow, input_latent])

        initial_block_model = Model([input_flow, input_noise, input_latent], gradient_flow)
        initial_block_model.compile(loss=self.loss_function, optimizer=self.optimizer_function)

        if self.level_verbose: initial_block_model.summary()
        return initial_block_model

    def __non_initial_synthesis_block(self, resolution_block, number_filters):

        feature_input_resolution = (resolution_block, resolution_block, number_filters)
        (resolution_block * 2, resolution_block * 2, number_filters)

        input_flow = Input(shape=feature_input_resolution)
        input_noise = Input(shape=(resolution_block * 2, resolution_block * 2, 1))
        input_latent = Input(shape=self.latent_dimension)
        input_latent = Reshape((self.latent_dimension, 1))(input_latent)

        gradient_flow = UpSampling2D((2, 2))(input_flow)
        gradient_flow = Conv2D(number_filters, self.size_kernel_filters, padding="same")(gradient_flow)
        gradient_flow = LeakyReLU(DEFAULT_THRESHOLD_RELU)(gradient_flow)
        gradient_flow = Add()([gradient_flow, input_noise])
        gradient_flow = AdaIN()([gradient_flow, input_latent])
        gradient_flow = Conv2D(number_filters, self.size_kernel_filters, padding="same")(gradient_flow)
        gradient_flow = LeakyReLU(DEFAULT_THRESHOLD_RELU)(gradient_flow)
        gradient_flow = Add()([gradient_flow, input_noise])
        gradient_flow = AdaIN()([gradient_flow, input_latent])

        non_initial_block_model = Model([input_flow, input_noise, input_latent], gradient_flow)
        non_initial_block_model.compile(loss=self.loss_function, optimizer=self.optimizer_function)
        if self.level_verbose: non_initial_block_model.summary()
        return non_initial_block_model

    def __build_synthesis_block(self):

        dimension_input_flow = (self.initial_dimension, self.initial_dimension, self.initial_num_channels)
        dimension_input_flow_noise = (self.initial_dimension, self.initial_dimension, 1)
        input_flow = Input(shape=dimension_input_flow, name="Input Constant")
        input_latent = Input(shape=self.latent_dimension, name="Input Latent")
        input_noise = Input(shape=dimension_input_flow_noise, name="Input Noise 1")

        self.latent_input = input_latent
        self.initial_gradient_flow = input_flow

        first_level_block = self.__initial_block_synthesis(self.initial_dimension, self.initial_num_channels)
        first_level_block = first_level_block([input_flow, input_noise, input_latent])
        self.list_block_synthesis.append(first_level_block)
        self.list_level_noise_input.append(input_noise)

        for i in range(self.num_synthesis_block - 1):
            resolution_feature = self.feature_size[i + 1]
            output_resolution_feature = (resolution_feature * 2, resolution_feature * 2, 1)
            input_noise = Input(shape=output_resolution_feature, name="Input Noise {}".format(i + 2))
            self.list_level_noise_input.append(input_noise)
            level_block = self.__non_initial_synthesis_block(resolution_feature, self.initial_num_channels)
            level_block = level_block([self.list_block_synthesis[-1], self.list_level_noise_input[-1], input_latent])
            self.list_block_synthesis.append(level_block)

    def __build_blocks(self):

        self.__build_synthesis_block()
        self.__block_mapping_network()
        self.__constant_mapping_block()

    def __output_channels_mapping(self, resolution, number_channels_flow):

        input_channels_mapping = Input(shape=(resolution, resolution, number_channels_flow))
        output_mapping = Conv2D(self.number_output_channels, (1, 1), padding="same")(input_channels_mapping)
        output_mapping = Activation(activations.tanh)(output_mapping)
        output_mapping = Model(input_channels_mapping, output_mapping)
        output_mapping.compile(loss=self.loss_function, optimizer=self.optimizer_function)
        if self.level_verbose: output_mapping.summary()
        return output_mapping

    def get_generator(self, number_level):

        neural_input_layer = [self.list_level_noise_input[i] for i in range(number_level)]
        neural_input_layer.append(self.initial_gradient_flow)
        neural_input_layer.append(self.latent_input)

        synthesis_model = Model(neural_input_layer, self.list_block_synthesis[number_level - 1])
        synthesis_model.compile(loss=self.loss_function, optimizer=self.optimizer_function)
        if self.level_verbose: synthesis_model.summary()

        last_level_dimension = self.feature_size[number_level]
        last_level_filters = self.num_filters_per_level[number_level]

        neural_synthesis = self.__output_channels_mapping(last_level_dimension, last_level_filters)
        neural_synthesis = neural_synthesis([synthesis_model.output])
        neural_synthesis = Model(synthesis_model.inputs, neural_synthesis, name="Synthesis_Block")
        if self.level_verbose: neural_synthesis.summary()

        neural_input_layer = [self.list_level_noise_input[i] for i in range(number_level)]
        neural_input_layer.append(self.initial_gradient_flow)
        neural_input_layer.append(self.mapping_neural_network.input)

        style_generator = neural_synthesis(neural_input_layer)
        style_generator = Model(neural_input_layer, style_generator, name="Generator")
        if self.level_verbose: style_generator.summary()

        return style_generator
