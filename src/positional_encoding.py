import numpy as np
import tensorflow as tf

'''
Functions for creating positional encoding tensors.
'''


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding_1d(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def positional_encoding_2d(d_model, height, width):
    """
    Returns a (1, height, width, d_model) tensor.
    Reference: https://github.com/wzlxjtu/PositionalEncoding2D/ \
               blob/master/positionalembedding2d.py

    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = np.zeros((d_model, height, width))
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = np.exp(np.arange(0., d_model, 2) *
                      -(np.log(10000.0) / d_model))
    pos_w = np.arange(0., width)[:, np.newaxis]
    pos_h = np.arange(0., height)[:, np.newaxis]
    pe[0:d_model:2, :, :] = np.tile(np.transpose(
        np.sin(pos_w * div_term), (1, 0))[:, np.newaxis, :], (1, height, 1))
    pe[1:d_model:2, :, :] = np.tile(np.transpose(
        np.cos(pos_w * div_term), (1, 0))[:, np.newaxis, :], (1, height, 1))
    pe[d_model::2, :, :] = np.tile(np.transpose(
        np.sin(pos_h * div_term), (1, 0))[:, :, np.newaxis], (1, 1, width))
    pe[d_model + 1::2, :, :] = np.tile(np.transpose(
        np.cos(pos_h * div_term), (1, 0))[:, :, np.newaxis], (1, 1, width))

    return np.transpose(pe, (1,2,0))[np.newaxis, ...]
