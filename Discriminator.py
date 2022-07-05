from keras import Input, Model
from keras.layers import Conv2D, LeakyReLU, MaxPooling2D

level_size_feature_dimension = [512, 256, 128, 64, 32, 16, 8, 4, 4]
number_filters_per_layer = [8, 16, 32, 64, 128, 256, 512, 512]

number_channels = 3


class Discriminator:

    def __init__(self):

        self.loss_function = "binary_crossentropy"
        self.optimizer_function = "adam"
        self.input_discriminator = []
        self.discriminator_blocks = []
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

        for i in range(len(level_size_feature_dimension)):

            self.convolutional_block(level_size_feature_dimension[-i], number_filters_per_layer[-i])



discriminator = Discriminator()
discriminator.build_discriminator()