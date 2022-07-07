import tensorflow


def discriminator_loss_function(real_image, synthetic_image):
    real_loss = tensorflow.reduce_mean(real_image)
    synthetic_loss = tensorflow.reduce_mean(synthetic_image)
    return synthetic_loss - real_loss


def generator_loss_function(synthetic_image):
    return -tensorflow.reduce_mean(synthetic_image)