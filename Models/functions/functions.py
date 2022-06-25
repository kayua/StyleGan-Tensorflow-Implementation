import tensorflow
from keras import layers

from Models.Layers.EqualizedDense import EqualizedDense


def fade_in(alpha, a, b):
    return alpha * a + (1.0 - alpha) * b


def wasserstein_loss(y_true, y_pred):
    return -tensorflow.reduce_mean(y_true * y_pred)


def pixel_norm(x, epsilon=1e-8):
    return x / tensorflow.math.sqrt(tensorflow.reduce_mean(x ** 2, axis=-1, keepdims=True) + epsilon)


def minibatch_std(input_tensor, epsilon=1e-8):
    n, h, w, c = tensorflow.shape(input_tensor)
    group_size = tensorflow.minimum(4, n)
    x = tensorflow.reshape(input_tensor, [group_size, -1, h, w, c])
    group_mean, group_var = tensorflow.nn.moments(x, axes=(0), keepdims=False)
    group_std = tensorflow.sqrt(group_var + epsilon)
    avg_std = tensorflow.reduce_mean(group_std, axis=[1, 2, 3], keepdims=True)
    x = tensorflow.tile(avg_std, [group_size, h, w, 1])
    return tensorflow.concat([input_tensor, x], axis=-1)

def Mapping(num_stages, input_shape=512):
    z = layers.Input(shape=(input_shape))
    w = pixel_norm(z)
    for i in range(8):
        w = EqualizedDense(512, learning_rate_multiplier=0.01)(w)
        w = layers.LeakyReLU(0.2)(w)
    w = tensorflow.tile(tensorflow.expand_dims(w, 1), (1, num_stages, 1))
    return tensorflow.keras.Model(z, w, name="mapping")