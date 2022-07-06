from abc import ABC
import tensorflow
from keras import Model

level_size_feature_dimension = [512, 256, 128, 64, 32, 16, 8]

DEFAULT_DISCRIMINATOR = None
DEFAULT_GENERATOR = None
DEFAULT_LATENT_DIMENSION = 256
DEFAULT_DISCRIMINATOR_STEPS = 4
DEFAULT_GRADIENT_PENALTY_ALPHA = 10.0
DEFAULT_NETWORK_LEVEL = 2


class StyleGAN(Model, ABC):

    def __init__(self, discriminator=DEFAULT_DISCRIMINATOR, generator=DEFAULT_GENERATOR,
                 latent_dimension=DEFAULT_LATENT_DIMENSION, number_discriminator_steps=DEFAULT_DISCRIMINATOR_STEPS,
                 gradient_penalty_alpha=DEFAULT_GRADIENT_PENALTY_ALPHA, network_level=DEFAULT_NETWORK_LEVEL):

        super(StyleGAN, self).__init__()
        self.discriminator = discriminator
        self.network_level = network_level
        self.generator = generator
        self.number_discriminator_steps = number_discriminator_steps
        self.gp_weight = gradient_penalty_alpha
        self.latent_dimension = latent_dimension
        self.constant_mapping_value = 0.5
        self.initial_dimension = 4
        self.num_filters_per_level = [256, 256, 256, 256, 256, 256, 256, 256, 256]

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(StyleGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):

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
    def tensor_mapping(random_noise, constant_mapping, latent_input):

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
            random_noise_synthesis = self.generate_random_noise(batch_size)
            input_mapping = self.tensor_mapping(random_noise_synthesis, constant_mapping_tensor, random_latent_space)

            with tensorflow.GradientTape() as tape:
                synthetic_images_generated = self.generator(input_mapping, training=True)
                synthetic_discriminator_loss = self.discriminator(synthetic_images_generated, training=True)

                real_image_resize = self.resize_image(8, real_images)
                real_discriminator_loss = self.discriminator(real_image_resize, training=True)

                discriminator_loss = self.d_loss_fn(real_img=real_discriminator_loss,
                                                    fake_img=synthetic_discriminator_loss)

                gradient_update = self.gradient_penalty(batch_size, real_image_resize, synthetic_images_generated)
                discriminator_loss = discriminator_loss + gradient_update * self.gp_weight

            discriminator_update = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
            gradient_apply = zip(discriminator_update, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(gradient_apply)

        random_latent_space = tensorflow.random.normal(shape=(batch_size, self.latent_dimension, 1))
        dimension = [batch_size, self.initial_dimension, self.initial_dimension, self.num_filters_per_level[0]]
        constant_mapping_tensor = tensorflow.fill(dimension, self.constant_mapping_value)
        random_noise_synthesis = self.generate_random_noise(batch_size)
        input_mapping = self.tensor_mapping(random_noise_synthesis, constant_mapping_tensor, random_latent_space)

        with tensorflow.GradientTape() as tape:
            synthetic_images_generated = self.generator(input_mapping, training=True)
            discriminator_loss = self.discriminator(synthetic_images_generated, training=True)
            g_loss = self.g_loss_fn(discriminator_loss)

        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        gradient_apply = zip(gen_gradient, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(gradient_apply)

        return {"discriminator_loss": discriminator_loss, "generator_loss": g_loss}

    @staticmethod
    def resize_image(resolution_image, image_propagated):
        interpolation_operator = tensorflow.image.ResizeMethod.NEAREST_NEIGHBOR
        shape_image = (resolution_image, resolution_image)
        image_propagated = tensorflow.image.resize(image_propagated, shape_image, method=interpolation_operator)
        image_propagated = tensorflow.cast(image_propagated, tensorflow.float32)
        return image_propagated

    def generate_random_noise(self, batch_size):

        random_noise_vector = []

        resolution_feature = self.initial_dimension
        shape_feature = (batch_size, resolution_feature, resolution_feature, 1)
        random_noise = tensorflow.random.normal(shape=shape_feature)
        random_noise_vector.append(random_noise)

        for i in range(1, self.network_level):
            resolution_feature = level_size_feature_dimension[-i]
            shape_feature = (batch_size, resolution_feature, resolution_feature, 1)
            random_noise = tensorflow.random.normal(shape=shape_feature)
            random_noise_vector.append(random_noise)

        return random_noise_vector
