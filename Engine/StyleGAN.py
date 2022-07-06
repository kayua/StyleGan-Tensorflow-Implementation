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

    @staticmethod
    def tensor_mapping(random_noise, constant_mapping, latent_input):

        input_mapping = {}

        for i in range(1, len(random_noise) + 1):
            input_mapping["Input Noise {}".format(i)] = random_noise[i - 1]

        input_mapping["Input Constant"] = constant_mapping
        input_mapping["Latent Input"] = latent_input

        return input_mapping

    def train_step(self, real_images):

        if isinstance(real_images, tuple):
            real_images = real_images[0]

        batch_size = tensorflow.shape(real_images)[0]

        for i in range(self.d_steps):

            random_latent_vectors = tensorflow.random.normal(shape=(batch_size, self.latent_dimension, 1))
            dimension = [batch_size, self.initial_dimension, self.initial_dimension, self.num_filters_per_level[0]]
            constant_mapping = tensorflow.fill(dimension, 0.5)
            random_noise_synthesis = self.generate_random_noise(batch_size)
            input_mapping = self.tensor_mapping(random_noise_synthesis, constant_mapping, random_latent_vectors)
            with tensorflow.GradientTape() as tape:

                fake_images = self.generator(input_mapping, training=True)
                fake_logits = self.discriminator(fake_images, training=True)
                real_image_reshape = self.resize_image(8, real_images)
                real_logits = self.discriminator(real_image_reshape, training=True)
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                gp = self.gradient_penalty(batch_size, real_image_reshape, fake_images)
                d_loss = d_cost + gp * self.gp_weight
                d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
                self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        random_latent_vectors = tensorflow.random.normal(shape=(batch_size, self.latent_dimension, 1))
        dimension = [batch_size, self.initial_dimension, self.initial_dimension, self.num_filters_per_level[0]]
        constant_mapping = tensorflow.fill(dimension, 0.5)
        random_noise_synthesis = self.generate_random_noise(batch_size)
        input_mapping = self.tensor_mapping(random_noise_synthesis, constant_mapping, random_latent_vectors)
        print("STEP TRAINING")
        print("---------------------------------")
        print(input_mapping)
        print("---------------------------------")
        with tensorflow.GradientTape() as tape:

            generated_images = self.generator(input_mapping, training=True)
            gen_img_logits = self.discriminator(generated_images, training=True)
            g_loss = self.g_loss_fn(gen_img_logits)
            gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        print({"d_loss": d_loss, "g_loss": g_loss})

        return {"d_loss": d_loss, "g_loss": g_loss}

    def resize_image(self, res, image):
        image = tensorflow.image.resize(image, (res, res), method=tensorflow.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = tensorflow.cast(image, tensorflow.float32)
        return image

    def generate_random_noise(self, batch_size):

        random_noise_vector = []

        resolution_feature = self.initial_dimension
        shape_feature = (batch_size, resolution_feature, resolution_feature, self.num_filters_per_level[0])
        random_noise = tensorflow.random.normal(shape=shape_feature)
        random_noise_vector.append(random_noise)

        for i in range(1, self.level_network):
            resolution_feature = level_size_feature_dimension[-i]
            shape_feature = (batch_size, resolution_feature, resolution_feature, self.num_filters_per_level[0])
            random_noise = tensorflow.random.normal(shape=shape_feature)
            random_noise_vector.append(random_noise)

        return random_noise_vector

    def generate_latent_noise(self, batch_size):

        latent_input = numpy.random.uniform(0, 1, self.latent_dimension * batch_size)
        latent_input = numpy.reshape(latent_input, (batch_size, self.latent_dimension, 1))
        latent_input = numpy.array(latent_input, dtype=numpy.float32)
        return latent_input

    def change_resolution_image(self, batch_image):

        batch_image_new_resolution = []
        size_image = level_size_feature_dimension[-self.level_network]
        tuple_shape_image = (size_image, size_image)

        for i in batch_image:
            new_image = cv2.resize(i, dsize=tuple_shape_image, interpolation=cv2.INTER_CUBIC)
            batch_image_new_resolution.append(new_image)

        return numpy.array(batch_image_new_resolution, dtype=numpy.float32)
