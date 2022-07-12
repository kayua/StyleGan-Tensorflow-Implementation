#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'All'
__email__ = '@unipampa.edu.br '
__version__ = '{2}.{0}.{1}'
__data__ = '2021/11/21'
__credits__ = ['All']

import json
import logging
import os

import tensorflow
from keras import Input
from keras import Model
from keras.layers import AveragePooling2D
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.models import model_from_json

tensorflow.get_logger().setLevel(logging.ERROR)

DEFAULT_LOSS_FUNCTION = "binary_crossentropy"
DEFAULT_LEARNING_RATE = 0.0002
DEFAULT_BETA_1 = 0.5
DEFAULT_BETA_2 = 0.9
DEFAULT_VERBOSE_CONSTRUCTION = True
DEFAULT_NUMBER_CHANNELS = 3
DEFAULT_INITIAL_RESOLUTION = 4
DEFAULT_DIMENSION_CONVOLUTION_KERNELS = (3, 3)
DEFAULT_THRESHOLD_ACTIVATION = 0.2
DEFAULT_DISCRIMINATOR_LEVEL = 1
DEFAULT_FILTER_PER_LAYER = [16, 16, 32, 32, 64, 64, 128, 128]
DEFAULT_LEVEL_FEATURE_DIMENSION = [1024, 512, 256, 128, 64, 32, 16, 8]

DEFAULT_OPTIMIZER_FUNCTION = "adam"


class Discriminator:

    def __init__(self, loss_function=DEFAULT_LOSS_FUNCTION, optimizer_function=DEFAULT_OPTIMIZER_FUNCTION,
                 number_channels=DEFAULT_NUMBER_CHANNELS, initial_resolution=DEFAULT_INITIAL_RESOLUTION,
                 threshold_activation=DEFAULT_THRESHOLD_ACTIVATION, initial_level=DEFAULT_DISCRIMINATOR_LEVEL,
                 number_filters_per_layer=None, level_feature_dimension=None,
                 size_kernel_filters=DEFAULT_DIMENSION_CONVOLUTION_KERNELS,
                 level_verbose=DEFAULT_VERBOSE_CONSTRUCTION):

        if number_filters_per_layer is None: number_filters_per_layer = DEFAULT_FILTER_PER_LAYER

        if level_feature_dimension is None: level_feature_dimension = DEFAULT_LEVEL_FEATURE_DIMENSION

        self.loss_function = loss_function
        self.optimizer_function = optimizer_function
        self.initial_resolution = initial_resolution
        self.number_channels = number_channels
        self.number_filters_per_layer = number_filters_per_layer
        self.level_feature_dimension = level_feature_dimension
        self.size_kernel_filters = size_kernel_filters
        self.threshold_activation = threshold_activation
        self.discriminator_level = initial_level
        self.level_verbose = level_verbose
        self.discriminator_mapping = None
        self.discriminator = None
        self.first_level_discriminator = None

        self.__build_initial_block()

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def set_optimizer_function(self, optimizer_function):
        self.optimizer_function = optimizer_function

    def set_initial_resolution(self, initial_resolution):
        self.initial_resolution = initial_resolution

    def set_number_channels(self, number_channels):
        self.number_channels = number_channels

    def set_number_filters_per_layer(self, number_filters_per_layer):
        self.number_filters_per_layer = number_filters_per_layer

    def set_level_feature_dimension(self, level_feature_dimension):
        self.level_feature_dimension = level_feature_dimension

    def set_size_kernel_filters(self, size_kernel_filters):
        self.size_kernel_filters = size_kernel_filters

    def set_threshold_activation(self, threshold_activation):
        self.threshold_activation = threshold_activation

    def set_discriminator_level(self, initial_level):
        self.discriminator_level = initial_level

    def set_level_verbose(self, level_verbose):
        self.level_verbose = level_verbose

    def get_loss_function(self):
        return self.loss_function

    def get_optimizer_function(self):
        return self.optimizer_function

    def get_initial_resolution(self):
        return self.initial_resolution

    def get_number_channels(self):
        return self.number_channels

    def get_number_filters_per_layer(self):
        return self.number_filters_per_layer

    def get_level_feature_dimension(self):
        return self.level_feature_dimension

    def get_size_kernel_filters(self):
        return self.size_kernel_filters

    def get_threshold_activation(self):
        return self.threshold_activation

    def get_discriminator_level(self):
        return self.discriminator_level

    def get_level_verbose(self):
        return self.level_verbose

    def save_neural_network(self, path_models, prefix_model):

        model_json = self.discriminator_mapping.to_json()
        if not os.path.exists("{}/discriminator".format(path_models)):
            os.mkdir("{}/discriminator".format(path_models))

        with open("{}/discriminator/{}.json".format(path_models, prefix_model), "w") as json_file:
            json_file.write(model_json)

        self.discriminator_mapping.save_weights("{}/discriminator/{}.h5".format(path_models, prefix_model))
        print("Saved model to disk")

    def load_discriminator(self, model_file):

        self.load_data_discriminator("{}_data".format(model_file))
        self.load_neural_network("{}".format(model_file))

    def save_discriminator(self, path_model, model_file):


        self.save_neural_network(path_model, model_file)
        self.write_data_discriminator(path_model, model_file)

    def load_neural_network(self, file_output_neural_network):

        json_file = open('{}.json'.format(file_output_neural_network), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.discriminator_mapping = model_from_json(loaded_model_json)
        self.discriminator_mapping.load_weights("{}.h5".format(file_output_neural_network))
        print("Loaded model from disk")

    def load_data_discriminator(self, discriminator_data_file):

        with open("{}.json".format(discriminator_data_file)) as json_file:
            data = json.load(json_file)

            self.initial_resolution = data["initial_resolution"]
            self.number_channels = data["number_channels"]
            self.threshold_activation = data["threshold_activation"]
            self.discriminator_level = data["discriminator_level"]
            self.level_verbose = data["level_verbose"]
            self.number_filters_per_layer = data["number_filters_per_layer"]
            self.level_feature_dimension = data["level_feature_dimension"]

    def write_data_discriminator(self, path_models, prefix_model):

        discriminator_data = {"initial_resolution": self.initial_resolution,
                              "number_channels": self.number_channels,
                              "threshold_activation": self.threshold_activation,
                              "discriminator_level": self.discriminator_level,
                              "level_verbose": self.level_verbose,
                              "number_filters_per_layer": self.number_filters_per_layer,
                              "level_feature_dimension": self.level_feature_dimension}

        with open("{}/discriminator/{}_data.json".format(path_models, prefix_model), "w") as outfile:
            json.dump(discriminator_data, outfile)

    @staticmethod
    def __mini_batch_stander(input_tensor, epsilon=1e-8):
        batch_size, dimension_x, dimension_y, channels = tensorflow.shape(input_tensor)
        group_size = tensorflow.minimum(4, batch_size)
        dimension_shape = [group_size, -1, dimension_x, dimension_y, channels]
        gradient_flow = tensorflow.reshape(input_tensor, dimension_shape)
        group_mean, group_var = tensorflow.nn.moments(gradient_flow, axes=0, keepdims=False)
        group_std = tensorflow.sqrt(group_var + epsilon)
        average_stander = tensorflow.reduce_mean(group_std, axis=[1, 2, 3], keepdims=True)
        gradient_flow = tensorflow.tile(average_stander, [group_size, dimension_x, dimension_y, 1])
        return tensorflow.concat([input_tensor, gradient_flow], axis=-1)

    def __channels_mapping(self, resolution, number_features):

        input_layer = Input(shape=(resolution, resolution, self.number_channels))
        weight_kernels = tensorflow.keras.initializers.Ones()
        channels = Conv2D(number_features, (1, 1), kernel_initializer=weight_kernels, trainable=False)(input_layer)
        channels = Model(input_layer, channels)
        if self.level_verbose: channels.summary()
        return channels

    def __build_initial_block(self):

        number_filters = self.number_filters_per_layer[-2]
        resolution_mapping = self.__channels_mapping(self.initial_resolution, number_filters)
        input_feature = Input(shape=(self.initial_resolution, self.initial_resolution, number_filters))
        kernel_filters = self.size_kernel_filters
        gradient_flow = Conv2D(self.number_filters_per_layer[-1], kernel_filters, padding="same")(input_feature)
        gradient_flow = LeakyReLU(self.threshold_activation)(gradient_flow)
        gradient_flow = Conv2D(self.number_filters_per_layer[-1], (4, 4), padding="same")(gradient_flow)
        gradient_flow = LeakyReLU(self.threshold_activation)(gradient_flow)
        gradient_flow = self.__fully_connected_block(gradient_flow)
        gradient_flow = Model(input_feature, gradient_flow)
        if self.level_verbose: gradient_flow.summary()

        self.discriminator = gradient_flow
        self.discriminator_mapping = self.discriminator(resolution_mapping.output)
        self.discriminator_mapping = Model(resolution_mapping.input, self.discriminator_mapping)
        if self.level_verbose: self.discriminator_mapping.summary()

    def __add_level_discriminator(self, number_level):

        number_filters = self.number_filters_per_layer[-(number_level + 1)]
        resolution_feature = self.level_feature_dimension[-number_level]
        next_number_filters = self.number_filters_per_layer[-(number_level + 2)]
        resolution_mapping = self.__channels_mapping(resolution_feature, next_number_filters)
        input_feature = Input(shape=(resolution_feature, resolution_feature, next_number_filters))

        gradient_flow = Conv2D(number_filters, self.size_kernel_filters, padding="same")(input_feature)
        gradient_flow = LeakyReLU(self.threshold_activation)(gradient_flow)
        gradient_flow = Conv2D(number_filters, self.size_kernel_filters, padding="same")(gradient_flow)
        gradient_flow = LeakyReLU(self.threshold_activation)(gradient_flow)
        gradient_flow = AveragePooling2D()(gradient_flow)
        gradient_flow = self.discriminator(gradient_flow)
        gradient_flow = Model(input_feature, gradient_flow)
        if self.level_verbose: gradient_flow.summary()

        self.discriminator = gradient_flow
        self.discriminator_mapping = self.discriminator(resolution_mapping.output)
        self.discriminator_mapping = Model(resolution_mapping.input, self.discriminator_mapping)
        self.discriminator_level += 1
        if self.level_verbose: self.discriminator_mapping.summary()

    def get_discriminator(self, number_level):

        for i in range(self.discriminator_level, number_level): self.__add_level_discriminator(i)

        return self.discriminator_mapping

    @staticmethod
    def __fully_connected_block(input_layer):
        gradient_flow = Flatten()(input_layer)
        gradient_flow = Dense(1)(gradient_flow)
        return gradient_flow
