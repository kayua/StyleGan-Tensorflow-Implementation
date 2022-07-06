import tensorflow
from keras import Model

level_size_feature_dimension = [512, 256, 128, 64, 32, 16, 8]


class StyleGAN(Model):

    def __init__(self, discriminator, generator, latent_dim, discriminator_extra_steps=3, gp_weight=10.0,
                 level_network=2):
        super(StyleGAN, self).__init__()
        self.discriminator = discriminator
        self.level_network = level_network
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.latent_dimension = 256
        self.constant_mapping_value = 0.5
        self.initial_dimension = 4
        self.num_filters_per_level = [256, 256, 256, 256, 256, 256, 256, 256, 256]

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn, **kwargs):
        super(StyleGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):

        alpha = tensorflow.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tensorflow.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tensorflow.sqrt(tensorflow.reduce_sum(tensorflow.square(grads), axis=[1, 2, 3]))
        gp = tensorflow.reduce_mean((norm - 1.0) ** 2)
        return gp

    @staticmethod
    def tensor_mapping(random_noise, constant_mapping, latent_input):

        input_mapping = {}

        for i in range(1, len(random_noise) + 1): input_mapping["Input Noise {}".format(i)] = random_noise[i - 1]

        input_mapping["Input Constant"] = constant_mapping
        input_mapping["Latent Input"] = latent_input

        return input_mapping

    def train_step(self, real_images):

        batch_size = tensorflow.shape(real_images)[0]

        for _ in range(self.d_steps):

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

                discriminator_loss = self.d_loss_fn(real_discriminator_loss, synthetic_discriminator_loss)

                gradient_update = self.gradient_penalty(batch_size, real_image_resize, synthetic_images_generated)
                discriminator_loss = discriminator_loss + gradient_update * self.gp_weight
                discriminator_update = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
                gradient_update_values = zip(discriminator_update, self.discriminator.trainable_variables)
                self.d_optimizer.apply_gradients(gradient_update_values)

        random_latent_space = tensorflow.random.normal(shape=(batch_size, self.latent_dimension, 1))
        dimension = [batch_size, self.initial_dimension, self.initial_dimension, self.num_filters_per_level[0]]
        constant_mapping_tensor = tensorflow.fill(dimension, self.constant_mapping_value)
        random_noise_synthesis = self.generate_random_noise(batch_size)
        input_mapping = self.tensor_mapping(random_noise_synthesis, constant_mapping_tensor, random_latent_space)

        with tensorflow.GradientTape() as tape:

            synthetic_images_generated = self.generator(input_mapping, training=True)
            discriminator_loss = self.discriminator(synthetic_images_generated, training=True)
            generator_loss = self.g_loss_fn(discriminator_loss)
            gradient_update_loss_generator = tape.gradient(generator_loss, self.generator.trainable_variables)
            gradient_update_values = zip(gradient_update_loss_generator, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(gradient_update_values)

        return {"discriminator_loss": discriminator_loss, "generator_loss": generator_loss}

    @staticmethod
    def resize_image(res, image):
        image = tensorflow.image.resize(image, (res, res), method=tensorflow.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = tensorflow.cast(image, tensorflow.float32)
        return image

    def generate_random_noise(self, batch_size):

        random_noise_vector = []

        resolution_feature = self.initial_dimension
        shape_feature = (batch_size, resolution_feature, resolution_feature, 1)
        random_noise = tensorflow.random.normal(shape=shape_feature)
        random_noise_vector.append(random_noise)

        for i in range(1, self.level_network):
            resolution_feature = level_size_feature_dimension[-i]
            shape_feature = (batch_size, resolution_feature, resolution_feature, 1)
            random_noise = tensorflow.random.normal(shape=shape_feature)
            random_noise_vector.append(random_noise)

        return random_noise_vector


