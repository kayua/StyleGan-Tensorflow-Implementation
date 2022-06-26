from keras import Model, Input
from keras.layers import Dense, LeakyReLU, Reshape, Conv2D, UpSampling2D
from tensorflow import keras
from tensorflow_addons.layers import InstanceNormalization

from Layers.AdaIN import AdaIN
from Layers.AddNoise import AddNoise


class Generator:

    def __init__(self):

        self.latent_dimension = 512
        self.num_neurons_mapping = 32
        self.num_mapping_blocks = 4
        self.initial_dimension = 4
        self.initial_num_channels = 128
        self.mapping_neural_network = None




    def build_blocks(self):
        latent_dimension_input = Input(shape=(self.latent_dimension, 1))
        self.mapping_neural_network = self.__block_mapping_network(latent_dimension_input)



    def __block_mapping_network(self, latent_dimension_input):

        if self.num_mapping_blocks > 1:

            gradient_flow = Dense(self.num_neurons_mapping)(latent_dimension_input)

            for i in range(self.num_mapping_blocks - 1):

                gradient_flow = Dense(self.num_neurons_mapping)(gradient_flow)
                gradient_flow = LeakyReLU(0.2)(gradient_flow)

            network_model = Model(latent_dimension_input, gradient_flow, name="Mapping_Network")
            network_model.summary()
            return network_model

    def __constant_block(self):

        dimension_latent_vector = 2**self.initial_dimension * self.initial_num_channels
        latent_input = Input(shape=(dimension_latent_vector, 1))
        mapping_format = (self.initial_dimension, self.initial_dimension, self.initial_num_channels, 1)
        gradient_flow = Reshape(mapping_format)(latent_input)
        network_model = Model(latent_input, gradient_flow, name="Constant_Block")
        network_model.summary()
        return network_model


    def __basic_block_synthesis(self, resolution_block, number_filters, random_noise, latent_noise):

        input_graph_flow = Input(shape=(resolution_block, resolution_block, number_filters))
        gradient_flow = AddNoise()([input_graph_flow, noise])
        gradient_flow = InstanceNormalization()(gradient_flow)
        gradient_flow = AdaIN()([gradient_flow, w])

        gradient_flow = EqualizedConv(filter_num, 3)(gradient_flow)
        gradient_flow = AddNoise()([gradient_flow, noise])
        gradient_flow = LeakyReLU(0.2)(gradient_flow)
        gradient_flow = InstanceNormalization()(gradient_flow)
        gradient_flow = AdaIN()([gradient_flow, w])
        return keras.Model([input_graph_flow, w, noise], gradient_flow, name=f"genblock_{res}x{res}")

        #input_tensor = Input(shape=input_shape)
        #noise = Input(shape=(res, res, 1))
        #w = Input(shape=512)
        #x = input_tensor





a = Generator()

