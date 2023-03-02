# -
"""
модули сети

"""

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import generator_auto

#
orig_img_size = (224, 224)
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
#
# слой заполнения
class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")

# Резидуальный блок
def residual_block(
    x,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = layers.BatchNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = layers.BatchNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x


def downsample(
    x,
    filters,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
        kernel_regularizer='l2',
    )(x)
    x = layers.BatchNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def upsample(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    kernel_initializer=kernel_init,
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
        kernel_regularizer='l2',
    )(x)
    x = layers.BatchNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, num_img=4, S_int=[]):
        self.num_img = num_img

        self.S_int = S_int

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % 10) == 0:
            _, ax = plt.subplots(2, 2, figsize=(12, 12))

            print(self.S_int.shape)
            prediction = self.model.gen_F(self.S_int[:1, :, :, :]).numpy()

            for i, img in enumerate([self.S_int[0:1, :, :, :], self.S_int[1:2, :, :, :]]):
                print('       : ', img.shape)
                prediction = self.model.gen_F(img * 2 - 1).numpy()
                print(prediction.shape)
                prediction = ((prediction / 2 + 0.5) * 255).astype(np.uint8)
                img = (img[0, :, :, :3] * 255).astype(np.uint8)

                ax[i, 0].imshow(img)
                ax[i, 1].imshow(prediction[0, :, :, :3])
                ax[i, 0].set_title("Input image")
                ax[i, 1].set_title("Translated image")
                ax[i, 0].axis("off")
                ax[i, 1].axis("off")

                prediction = keras.preprocessing.image.array_to_img(prediction[0, :, :, :3])
                prediction.save(
                    "generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 1)
                )
            plt.show()
            plt.close()
