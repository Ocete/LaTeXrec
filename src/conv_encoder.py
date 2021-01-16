import tensorflow as tf

'''
Definitions for the convolutional part of the encoder.
'''


def conv_bn_elu(n_filters):
    """
    A Conv-BN-ELU block.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(n_filters, kernel_size=3,
                               strides=1, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ELU()
    ])


def resnet_encoder(d_model):
    """
    A simple residual network.
    """
    inputs = tf.keras.layers.Input(shape=(None, None, 1))
    block_output_1 = conv_bn_elu(64)(inputs)
    block_output_1 = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2))(block_output_1)

    block_output_2 = conv_bn_elu(64)(block_output_1)
    block_output_2 = tf.keras.layers.add([block_output_1, block_output_2])

    block_output_3 = conv_bn_elu(64)(block_output_2)
    block_output_3 = tf.keras.layers.add([block_output_2, block_output_3])
    block_output_3 = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2))(block_output_3)

    block_output_4 = conv_bn_elu(64)(block_output_3)
    block_output_4 = tf.keras.layers.add([block_output_3, block_output_4])
    block_output_4 = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2))(block_output_4)

    block_output_5 = conv_bn_elu(d_model)(block_output_4)
    return inputs, block_output_5


def resnet_decoder(enc_output):
    """
    A decoder matching `resnet_encoder`, for training it in an
    unsupervised/self-supervised manner.
    
    Params:
    - enc_output: the output of the encoder.
    """
    x = tf.keras.layers.Conv2DTranspose(
        64, activation='elu', kernel_size=3, padding='same', strides=(2, 2))(enc_output)
    x = tf.keras.layers.Conv2D(
        64, activation='elu', kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(
        1, activation='elu', padding='same', kernel_size=3, strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(
        64, activation='elu', kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(
        1, activation='elu', padding='same', kernel_size=3, strides=(2, 2))(x)
    return x

def convolutional_network(d_model):
    """
    A simple convolutional network for a baseline model.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ELU(),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ELU(),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ELU(),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    ])
