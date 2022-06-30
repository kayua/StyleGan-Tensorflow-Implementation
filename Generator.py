from keras import Model, Input
from keras.layers import Dense, LeakyReLU, Reshape, Conv2D, UpSampling2D

from Layers.AdaIN import AdaIN
from Layers.AddNoise import AddNoise


class Generator:

    def __init__(self):

        self.latent_dimension = 512
        self.num_neurons_mapping = 512
        self.num_mapping_blocks = 4
        self.initial_dimension = 4
        self.initial_num_channels = 128
        self.mapping_neural_network = None
        self.size_kernel_filters = (3, 3)
        self.num_synthesis_block = 3

        self.input_block = None
        self.list_level_block_output = []

        self.num_filters_per_level = [32, 32, 32, 32]

    def build_blocks(self):
        latent_dimension_input = Input(shape=(self.latent_dimension, 1))
        self.mapping_neural_network = self.__block_mapping_network(latent_dimension_input)

    def __block_mapping_network(self, latent_dimension_input):

        if self.num_mapping_blocks > 1:

            gradient_flow = Dense(self.num_neurons_mapping)(latent_dimension_input)

            for i in range(self.num_mapping_blocks - 2):
                gradient_flow = Dense(self.num_neurons_mapping)(gradient_flow)
                gradient_flow = LeakyReLU(0.2)(gradient_flow)

            gradient_flow = Dense(self.latent_dimension)(gradient_flow)
            network_model = Model(latent_dimension_input, gradient_flow, name="Mapping_Network")
            network_model.summary()
            return network_model

    def constant_mapping_block(self):

        dimension_latent_vector = 2 ** self.initial_dimension * self.initial_num_channels
        latent_input = Input(shape=(dimension_latent_vector, 1))
        mapping_format = (self.initial_dimension, self.initial_dimension, self.initial_num_channels)
        gradient_flow = Reshape(mapping_format)(latent_input)
        return Model(latent_input, gradient_flow, name="Constant_Block")

    def block_synthesis(self, resolution_block, number_filters, initial_block):

        input_flow = Input(shape=(resolution_block, resolution_block, number_filters))
        input_noise = Input(shape=(resolution_block, resolution_block, number_filters))
        input_latent = Input(shape=(self.latent_dimension, 1))

        if not initial_block:
            gradient_flow = UpSampling2D((2, 2))(input_flow)
            gradient_flow = AddNoise()([gradient_flow, input_noise])

        else:

            gradient_flow = AddNoise()([input_flow, input_noise])

        gradient_flow = AdaIN()([gradient_flow, input_latent])
        gradient_flow = Conv2D(number_filters, self.size_kernel_filters, padding="same")(gradient_flow)
        gradient_flow = LeakyReLU(0.2)(gradient_flow)
        gradient_flow = AddNoise()([gradient_flow, input_noise])
        gradient_flow = AdaIN()([gradient_flow, input_latent])

        gradient_flow = Model([input_flow, input_noise, input_latent], gradient_flow, name="Synthesis_Block_{}".format(resolution_block))
        gradient_flow.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        gradient_flow.summary()
        return gradient_flow

    def build_synthesis_block(self):


        input_flow = Input(shape=(self.initial_dimension, self.initial_dimension, self.initial_num_channels))
        input_noise = Input(shape=(self.initial_dimension, self.initial_dimension, self.initial_num_channels))
        input_latent = Input(shape=(self.latent_dimension, 1))


        if self.num_synthesis_block <= 1:

            return -1

        first_level_block = self.block_synthesis(4, 128, True)
        first_level_block = first_level_block([input_flow, input_noise, input_latent])
        self.list_level_block_output.append(first_level_block)
        last_resolution_feature = 2


        for i in range(self.num_synthesis_block-1):
            last_resolution_feature = last_resolution_feature * 2
            level_block = self.block_synthesis(last_resolution_feature, 128, False)
            level_block = level_block([self.list_level_block_output[-1], input_noise, input_latent])
            self.list_level_block_output.append(level_block)


        network_model = Model([input_flow, input_noise, input_latent], [self.list_level_block_output[-1]])
        network_model.summary()
        #input_flow = Input(shape=(4, 4, self.num_filters_per_level[0]))
        #input_noise = Input(shape=(4, 4, self.num_filters_per_level[0]))
        #input_latent = Input(shape=(self.latent_dimension, 1))
        #first_level_block = Model(first_level_block([input_flow, input_noise, input_latent]), first_level_block.output)
       # first_level_block.summary()


        #first_level_block = UpSampling2D()(first_level_block.output)

        #self.list_level_block_input.append(first_level_block)
        #last_resolution_block = self.num_mapping_blocks ** 2
        #for i in range(self.num_synthesis_block):
        #    last_level_block = self.list_level_block_input[-1]
        #    last_resolution_block = last_resolution_block ** 2
        #    level_block = self.block_synthesis(last_resolution_block, self.num_filters_per_level[i - 1])(last_level_block)
        #    level_block = UpSampling2D()(level_block.output)
        #    self.list_level_block_input.append(level_block)



a = Generator()
a.build_synthesis_block()
