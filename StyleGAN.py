import cv2
import numpy
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
        self.latent_dimension = 128
        self.initial_dimension = 4
        self.num_filters_per_level = [256, 256, 256, 256, 256, 256, 256, 256, 256]

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
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

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        batch_size = tensorflow.shape(real_images)[0]

        random_noise_vector = []
        for i in range(self.d_steps):
            random_latent_vectors = numpy.array([self.generate_latent_noise()] for _ in range(batch_size))
            random_noise_vector.append(numpy.array([self.generate_random_noise()] for _ in range(batch_size)))

            with tensorflow.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tensorflow.random.normal(shape=(batch_size, self.latent_dim))
        with tensorflow.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}

    def generate_constant_mapping(self):

        number_filters = self.num_filters_per_level[-1]
        constant_mapping = numpy.array([0.5 for _ in range(number_filters * self.initial_dimension ** 2)])
        mapping_shape = (self.initial_dimension, self.initial_dimension, number_filters)
        constant_mapping = numpy.reshape(constant_mapping, mapping_shape)
        return constant_mapping

    def generate_random_noise(self):

        random_noise_vector = []

        for i in range(1, level_network):
            size_feature = level_size_feature_dimension[-i] ** 2
            resolution_feature = level_size_feature_dimension[-i]
            random_noise = numpy.random.uniform(0, 1, self.num_filters_per_level[-i] * size_feature)
            shape_feature = (resolution_feature, resolution_feature, num_filters_per_level[-i])
            random_noise_vector.append(numpy.reshape(random_noise, shape_feature))

        return random_noise_vector

    def generate_latent_noise(self):

        latent_input = numpy.random.uniform(0, 1, self.latent_dimension)
        latent_input = numpy.reshape(latent_input, (self.latent_dimension, 1))
        return latent_input

    def change_resolution_image(self, batch_image):

        batch_image_new_resolution = []
        size_image = level_size_feature_dimension[-self.level_network]
        tuple_shape_image = (size_image, size_image)

        for i in batch_image:
            new_image = cv2.resize(i, dsize=tuple_shape_image, interpolation=cv2.INTER_CUBIC)
            batch_image_new_resolution.append(new_image)

        return numpy.array(batch_image_new_resolution, dtype=numpy.float32)


level_network = 4
num_filters_per_level = [256, 256, 256, 256, 256, 256, 256, 256, 256]
