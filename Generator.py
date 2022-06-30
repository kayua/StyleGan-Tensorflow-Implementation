from keras import Model
from keras import Input
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import UpSampling2D

from Layers.AdaIN import AdaIN
from Layers.AddNoise import AddNoise

level_size_feature_dimension = [4, 8, 16, 32, 64]


class Generator:

    def __init__(self):

        self.latent_dimension = 128
        self.num_neurons_mapping = 512
        self.num_mapping_blocks = 4
        self.initial_dimension = 4
        self.initial_num_channels = 64
        self.number_output_channels = 3
        self.mapping_neural_network = None
        self.size_kernel_filters = (3, 3)
        self.num_synthesis_block = 3
        self.constant_mapping_neural_network = None
        self.input_block = None
        self.list_block_synthesis = []
        self.list_level_noise_input = []
        self.function_loss = "binary_crossentropy"
        self.num_filters_per_level = [32, 32, 32, 32]
        self.build_blocks()

    def block_mapping_network(self):

        latent_dimension_input = Input(shape=(self.latent_dimension, 1))
        if self.num_mapping_blocks > 1:

            gradient_flow = Dense(self.num_neurons_mapping)(latent_dimension_input)

            for i in range(self.num_mapping_blocks - 2):
                gradient_flow = Dense(self.num_neurons_mapping)(gradient_flow)
                gradient_flow = LeakyReLU(0.2)(gradient_flow)

            gradient_flow = Dense(self.latent_dimension)(gradient_flow)
            network_model = Model(latent_dimension_input, gradient_flow, name="Mapping_Network")
            network_model.summary()
            self.mapping_neural_network = network_model

    def constant_mapping_block(self):

        dimension_latent_vector = 2 ** self.initial_dimension * self.initial_num_channels
        latent_input = Input(shape=(dimension_latent_vector, 1))
        mapping_format = (self.initial_dimension, self.initial_dimension, self.initial_num_channels)
        gradient_flow = Reshape(mapping_format)(latent_input)
        gradient_flow = Model(latent_input, gradient_flow, name="Constant_Block")
        gradient_flow.compile(loss=self.function_loss, optimizer='adam', metrics=['accuracy'])
        self.constant_mapping_neural_network = gradient_flow

    def block_synthesis(self, resolution_block, number_filters, initial_block):

        if not initial_block:
            input_flow = Input(shape=(int(resolution_block / 2), int(resolution_block / 2), number_filters))
            input_noise = Input(shape=(resolution_block, resolution_block, number_filters))
            input_latent = Input(shape=(self.latent_dimension, 1))
            gradient_flow = UpSampling2D((2, 2))(input_flow)
            gradient_flow = AddNoise()([gradient_flow, input_noise])

        else:

            input_flow = Input(shape=(resolution_block, resolution_block, number_filters))
            input_noise = Input(shape=(resolution_block, resolution_block, number_filters))
            input_latent = Input(shape=(self.latent_dimension, 1))
            gradient_flow = AddNoise()([input_flow, input_noise])

        gradient_flow = AdaIN()([gradient_flow, input_latent])
        gradient_flow = Conv2D(number_filters, self.size_kernel_filters, padding="same")(gradient_flow)
        gradient_flow = LeakyReLU(0.2)(gradient_flow)
        gradient_flow = AddNoise()([gradient_flow, input_noise])
        gradient_flow = AdaIN()([gradient_flow, input_latent])

        gradient_flow = Model([input_flow, input_noise, input_latent], gradient_flow)

        gradient_flow.compile(loss=self.function_loss, optimizer='adam', metrics=['accuracy'])
        gradient_flow.summary()
        return gradient_flow

    def build_synthesis_block(self):

        input_flow = Input(shape=(self.initial_dimension, self.initial_dimension, self.initial_num_channels))

        input_latent = Input(shape=(self.latent_dimension, 1))

        if self.num_synthesis_block <= 1: return -1

        input_noise = Input(shape=(self.initial_dimension, self.initial_dimension, self.initial_num_channels))
        first_level_block = self.block_synthesis(self.initial_dimension, self.initial_num_channels, True)
        first_level_block = first_level_block([input_flow, input_noise, input_latent])
        self.list_block_synthesis.append(first_level_block)
        self.list_level_noise_input.append(input_noise)

        for i in range(self.num_synthesis_block - 1):

            resolution_feature = level_size_feature_dimension[i + 1]
            input_noise = Input(shape=(resolution_feature, resolution_feature, self.initial_num_channels))
            self.list_level_noise_input.append(input_noise)
            level_block = self.block_synthesis(level_size_feature_dimension[i], self.initial_num_channels, False)
            level_block = level_block([self.list_block_synthesis[-1], self.list_level_noise_input[-1], input_latent])
            self.list_block_synthesis.append(level_block)

    def build_blocks(self):

        self.build_synthesis_block()
        self.block_mapping_network()
        self.constant_mapping_block()

    def color_mapping(self, resolution, number_channels_flow):

        input_color_mapping = Input(shape=(resolution, resolution, number_channels_flow))
        color_mapping = Conv2D(self.number_output_channels, (1, 1), padding="same")(input_color_mapping)
        color_mapping = Model([input_color_mapping], color_mapping)
        color_mapping.compile(loss=self.function_loss, optimizer='adam', metrics=['accuracy'])

        return color_mapping

    def get_generator(self, number_level):

        constant_input = self.constant_mapping_neural_network
        list_input_noise = []








a = Generator()
