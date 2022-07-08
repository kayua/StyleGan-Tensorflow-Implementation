import numpy
import tensorflow
from cv2 import cv2
from abc import ABC
from keras import Model
from keras.optimizer_v1 import Adam

from Engine.Loss import discriminator_loss_function
from Engine.Loss import generator_loss_function

DEFAULT_DISCRIMINATOR = None
DEFAULT_GENERATOR = None
DEFAULT_LATENT_DIMENSION = 256
DEFAULT_DISCRIMINATOR_STEPS = 4
DEFAULT_GRADIENT_PENALTY_ALPHA = 10.0
DEFAULT_NETWORK_LEVEL = 2
DEFAULT_CONSTANT_VALUE_MAPPING = 0.5
DEFAULT_INITIAL_DIMENSION = 4
DEFAULT_NUMBER_FILTERS_PER_LAYER = [64, 64, 64, 64, 64, 64, 64, 64, 64]
DEFAULT_SIZE_FEATURE_DIMENSION = [512, 256, 128, 64, 32, 16, 8]
DEFAULT_GENERATOR_OPTIMIZER = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
DEFAULT_DISCRIMINATOR_OPTIMIZER = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
DEFAULT_DISCRIMINATOR_LOSS = discriminator_loss_function
DEFAULT_GENERATOR_LOSS = generator_loss_function
DEFAULT_DIMENSION_IMAGE_ALGORITHM = tensorflow.image.ResizeMethod.NEAREST_NEIGHBOR


class StyleGAN(Model, ABC):

    def __init__(self, discriminator=DEFAULT_DISCRIMINATOR, generator=DEFAULT_GENERATOR,
                 latent_dimension=DEFAULT_LATENT_DIMENSION, number_discriminator_steps=DEFAULT_DISCRIMINATOR_STEPS,
                 gradient_penalty_alpha=DEFAULT_GRADIENT_PENALTY_ALPHA, network_level=DEFAULT_NETWORK_LEVEL,
                 constant_mapping_value=DEFAULT_CONSTANT_VALUE_MAPPING, initial_dimension=DEFAULT_INITIAL_DIMENSION,
                 reduce_dimension_image_algorithm=DEFAULT_DIMENSION_IMAGE_ALGORITHM, number_filter_per_layer=None,
                 size_feature_dimension=None):

        super(StyleGAN, self).__init__()

        if size_feature_dimension is None: size_feature_dimension = DEFAULT_SIZE_FEATURE_DIMENSION
        if number_filter_per_layer is None: number_filter_per_layer = DEFAULT_NUMBER_FILTERS_PER_LAYER

        self.discriminator = discriminator
        self.network_level = network_level
        self.generator = generator
        self.number_discriminator_steps = number_discriminator_steps
        self.gradient_penalty_alpha = gradient_penalty_alpha
        self.latent_dimension = latent_dimension
        self.constant_mapping_value = constant_mapping_value
        self.initial_dimension = initial_dimension
        self.num_filters_per_level = number_filter_per_layer
        self.size_feature_dimension = size_feature_dimension
        self.reduce_dimension_image_method = reduce_dimension_image_algorithm
        self.discriminator_optimizer = None
        self.generator_optimizer = None
        self.discriminator_loss = None
        self.generator_loss = None

    def compile(self, discriminator_optimizer=DEFAULT_DISCRIMINATOR_OPTIMIZER,
                generator_optimizer=DEFAULT_GENERATOR_OPTIMIZER,
                discriminator_loss=DEFAULT_DISCRIMINATOR_LOSS,
                generator_loss=DEFAULT_GENERATOR_LOSS, **kwargs):
        super(StyleGAN, self).compile()
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.discriminator_loss = discriminator_loss
        self.generator_loss = generator_loss

    def set_discriminator(self, discriminator):
        self.discriminator = discriminator

    def set_network_level(self, network_level):
        self.network_level = network_level

    def set_generator(self, generator):
        self.generator = generator

    def set_number_discriminator_steps(self, number_discriminator_steps):
        self.number_discriminator_steps = number_discriminator_steps

    def set_gradient_penalty_alpha(self, gradient_penalty_alpha):
        self.gradient_penalty_alpha = gradient_penalty_alpha

    def set_latent_dimension(self, latent_dimension):
        self.latent_dimension = latent_dimension

    def set_constant_mapping_value(self, constant_mapping_value):
        self.constant_mapping_value = constant_mapping_value

    def set_initial_dimension(self, initial_dimension):
        self.initial_dimension = initial_dimension

    def set_number_filters_per_level(self, number_filters_per_level):
        self.num_filters_per_level = number_filters_per_level

    def set_size_feature_dimension(self, size_feature_dimension):
        self.size_feature_dimension = size_feature_dimension

    def get_discriminator(self):
        return self.discriminator

    def get_network_level(self):
        return self.network_level

    def get_generator(self):
        return self.generator

    def get_number_discriminator_steps(self):
        return self.number_discriminator_steps

    def get_gradient_penalty_alpha(self):
        return self.gradient_penalty_alpha

    def get_latent_dimension(self):
        return self.latent_dimension

    def get_constant_mapping_value(self):
        return self.constant_mapping_value

    def get_initial_dimension(self):
        return self.initial_dimension

    def get_number_filters_per_level(self):
        return self.num_filters_per_level

    def get_size_feature_dimension(self):
        return self.size_feature_dimension

    def __gradient_penalty(self, batch_size, real_images, fake_images):

        random_noise = tensorflow.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        divergence = fake_images - real_images
        update_score = real_images + random_noise * divergence

        with tensorflow.GradientTape() as gradient_penalty_reduce:
            gradient_penalty_reduce.watch(update_score)
            discriminator_result = self.discriminator(update_score, training=True)

        gradient_result = gradient_penalty_reduce.gradient(discriminator_result, [update_score])[0]
        stander_reduction = tensorflow.sqrt(tensorflow.reduce_sum(tensorflow.square(gradient_result), axis=[1, 2, 3]))
        return tensorflow.reduce_mean((stander_reduction - 1.0) ** 2)

    @staticmethod
    def __tensor_mapping(random_noise, constant_mapping, latent_input):

        input_mapping = {}

        for i in range(1, len(random_noise) + 1): input_mapping["Input Noise {}".format(i)] = random_noise[i - 1]

        input_mapping["Input Constant"] = constant_mapping
        input_mapping["Latent Input"] = latent_input

        return input_mapping

    def train_step(self, real_images):

        batch_size = tensorflow.shape(real_images)[0]

        for _ in range(self.number_discriminator_steps):
            random_latent_space = tensorflow.random.normal(shape=(batch_size, self.latent_dimension, 1))
            dimension = [batch_size, self.initial_dimension, self.initial_dimension, self.num_filters_per_level[0]]
            constant_mapping_tensor = tensorflow.fill(dimension, self.constant_mapping_value)
            random_noise_synthesis = self.__generate_random_noise(batch_size)
            input_mapping = self.__tensor_mapping(random_noise_synthesis, constant_mapping_tensor, random_latent_space)

            with tensorflow.GradientTape() as tape:
                synthetic_images_generated = self.generator(input_mapping, training=True)
                synthetic_discriminator_loss = self.discriminator(synthetic_images_generated, training=True)
                image_new_dimension = self.size_feature_dimension[-(self.network_level - 1)]
                real_image_resize = self.__resize_image(image_new_dimension, real_images)
                real_discriminator_loss = self.discriminator(real_image_resize, training=True)
                discriminator_loss = self.discriminator_loss(real_discriminator_loss, synthetic_discriminator_loss)
                gradient_update = self.__gradient_penalty(batch_size, real_image_resize, synthetic_images_generated)
                discriminator_loss = discriminator_loss + gradient_update * self.gradient_penalty_alpha

            discriminator_update = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
            gradient_apply = zip(discriminator_update, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(gradient_apply)

        random_latent_space = tensorflow.random.normal(shape=(batch_size, self.latent_dimension, 1))
        dimension = [batch_size, self.initial_dimension, self.initial_dimension, self.num_filters_per_level[0]]
        constant_mapping_tensor = tensorflow.fill(dimension, self.constant_mapping_value)
        random_noise_synthesis = self.__generate_random_noise(batch_size)
        input_mapping = self.__tensor_mapping(random_noise_synthesis, constant_mapping_tensor, random_latent_space)

        with tensorflow.GradientTape() as tape:
            synthetic_images_generated = self.generator(input_mapping, training=True)
            discriminator_loss = self.discriminator(synthetic_images_generated, training=True)
            generator_loss = self.generator_loss(discriminator_loss)

        generator_update = tape.gradient(generator_loss, self.generator.trainable_variables)
        gradient_apply = zip(generator_update, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(gradient_apply)

        return {"discriminator_loss_function": discriminator_loss, "generator_loss_function": generator_loss}

    def __resize_image(self, resolution_image, image_propagated):
        interpolation_operator = self.reduce_dimension_image_method
        shape_image = (resolution_image, resolution_image)
        image_propagated = tensorflow.image.resize(image_propagated, shape_image, method=interpolation_operator)
        image_propagated = tensorflow.cast(image_propagated, tensorflow.float32)
        return image_propagated

    def generate_images(self, latent_noise=None, noise_level=None, number_images=5, path_output="Results/"):

        if latent_noise is None:
            random_latent_space = tensorflow.random.normal(shape=(number_images, self.latent_dimension, 1))

        else:
            random_latent_space = latent_noise

        if noise_level is None:
            random_noise_synthesis = self.__generate_random_noise(number_images)

        else:
            random_noise_synthesis = noise_level

        dimension = [number_images, self.initial_dimension, self.initial_dimension, self.num_filters_per_level[0]]
        constant_mapping_tensor = tensorflow.fill(dimension, self.constant_mapping_value)

        input_mapping = self.tensor_mapping(random_noise_synthesis, constant_mapping_tensor, random_latent_space)

        images = self.generator(input_mapping)

        for i, img in enumerate(images):

            new_image = numpy.array(img*256.0)
            cv2.imwrite('{}/image_level_{}_id_{}.jpg'.format(path_output, self.network_level, i), new_image)

    def __generate_random_noise(self, batch_size):

        random_noise_vector = []

        resolution_feature = self.initial_dimension
        shape_feature = (batch_size, resolution_feature, resolution_feature, 1)
        random_noise = tensorflow.random.normal(shape=shape_feature)
        random_noise_vector.append(random_noise)

        for i in range(1, self.network_level):
            resolution_feature = self.size_feature_dimension[-i]
            shape_feature = (batch_size, resolution_feature, resolution_feature, 1)
            random_noise = tensorflow.random.normal(shape=shape_feature)
            random_noise_vector.append(random_noise)

        return random_noise_vector
