import tensorflow
from tensorflow import keras

from Engine.Discriminator import Discriminator
from Engine.Generator import Generator
from Engine.StyleGAN import StyleGAN
from Tools.LoadImage import LoadImage

generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)


def discriminator_loss(real_img, fake_img):
    real_loss = tensorflow.reduce_mean(real_img)
    fake_loss = tensorflow.reduce_mean(fake_img)
    return fake_loss - real_loss


def generator_loss(fake_img):
    return -tensorflow.reduce_mean(fake_img)

level = 2
discriminator_instance = Discriminator()
generator_instance = Generator()
image_instance = LoadImage()
image_training = image_instance.get_dataset_image()
print(image_training.shape)
generator_model = generator_instance.get_generator(level)
discriminator_model = discriminator_instance.get_discriminator(level)


styleGan = StyleGAN(discriminator=discriminator_model, generator=generator_model, latent_dim=0, discriminator_extra_steps=1, level_network=level)
styleGan.compile(d_optimizer=discriminator_optimizer, g_optimizer=generator_optimizer, g_loss_fn=generator_loss, d_loss_fn=discriminator_loss)
styleGan.fit(image_training, batch_size=32, steps_per_epoch=1, epochs=100)
