from keras import Input, Model
from keras.layers import Conv2D, LeakyReLU, MaxPooling2D, Dense, LayerNormalization, Flatten

level_size_feature_dimension = [512, 256, 128, 64, 32, 16, 8, 4, 4]
number_filters_per_layer = [4, 8, 16, 32, 64, 128, 256, 512, 512]

number_channels = 3


class Discriminator:

    def __init__(self):

        self.loss_function = "binary_crossentropy"
        self.optimizer_function = "adam"
        self.input_discriminator = []
        self.discriminator_blocks = []
        self.first_level_discriminator = None
        pass


    def convolutional_block(self, resolution_feature, number_filters):

        input_layer = Input(shape=(resolution_feature, resolution_feature, number_channels))
        self.input_discriminator.append(input_layer)
        gradient_flow = Conv2D(number_filters, (3, 3), padding="same")(input_layer)
        gradient_flow = LeakyReLU(0.2)(gradient_flow)
        gradient_flow = Conv2D(number_filters, (3, 3), padding="same")(gradient_flow)
        gradient_flow = LeakyReLU(0.2)(gradient_flow)
        gradient_flow = MaxPooling2D((2, 2))(gradient_flow)
        gradient_flow = Model(input_layer, gradient_flow)
        gradient_flow.compile(loss=self.loss_function, optimizer=self.optimizer_function)
        self.discriminator_blocks.append(gradient_flow)
        gradient_flow.summary()

    def build_discriminator(self):
        resolution_feature = level_size_feature_dimension[-1]
        input_layer = Input(shape=(resolution_feature, resolution_feature, number_channels))
        number_layer = number_filters_per_layer[-1]
        self.first_level_discriminator = Conv2D(number_layer, (3, 3), padding="same")(input_layer)
        self.first_level_discriminator = Conv2D(number_layer, (3, 3), padding="same")(self.first_level_discriminator)
        self.first_level_discriminator = self.fully_connected_block(self.first_level_discriminator)
        self.first_level_discriminator = Model(input_layer, self.first_level_discriminator)
        self.first_level_discriminator.summary()

        for i in range(len(level_size_feature_dimension)):

            self.convolutional_block(level_size_feature_dimension[i], number_filters_per_layer[i])


    @staticmethod
    def fully_connected_block(input_layer):
        gradient_flow = Flatten()(input_layer)
        gradient_flow = LayerNormalization(gradient_flow)
        gradient_flow = Dense(1)(gradient_flow)
        gradient_flow = LeakyReLU(0.2)(gradient_flow)
        return gradient_flow





discriminator = Discriminator()
discriminator.build_discriminator()