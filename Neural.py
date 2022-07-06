import tensorflow
from tensorflow import keras

from StyleGAN import StyleGAN

generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)


def discriminator_loss(real_img, fake_img):
    real_loss = tensorflow.reduce_mean(real_img)
    fake_loss = tensorflow.reduce_mean(fake_img)
    return fake_loss - real_loss


def generator_loss(fake_img):
    return -tensorflow.reduce_mean(fake_img)

d_model = Discriminator()

wgan = StyleGAN(discriminator=d_model, generator=g_model, latent_dim=noise_dim, discriminator_extra_steps=3)