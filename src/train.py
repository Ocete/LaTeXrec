import datasets
import transformer
import conv_encoder

import argparse

'''
Main script used for training the model.
'''

parser = argparse.ArgumentParser(
    help='Main script used to train the model.'
)

# DATA LOADING ARGUMENTS
parser.add_argument(
    name='--dataset',
    choices=['toy_50k', 'im2latex'],
    required=True,
    help='Select the dataset used for training'
)

parser.add_argument(
    name='--samples',
    type=int,
    required=True,
    help='Number of samples to train on'
)

# MODEL ARGUMENTS

# Transformer arguments
parser.add_argument(
    name='--num-layers',
    type=int,
    required=True,
    help='Number of layers of the encoder and decoder'
)

parser.add_argument(
    name='--depth',
    type=int,
    required=True,
    help='Depth parameter of the model'
)

parser.add_argument(
    name='--feedforward-units',
    type=int,
    required=True,
    help='Units on pointwise feedforward network'
)

parser.add_argument(
    name='--num-heads',
    type=int,
    required=True,
    help='Number of heads on the multihead attention'
)

parser.add_argument(
    name='--dropout-rate',
    type=float,
    required=True,
    help='Dropout rate in the transformer'
)

parser.add_argument(
    name='--performer-attention-encoder',
    choices=['yes', 'no']
    required=True,
    help='Whether to employ the FAVOR+ mechanism for attention in the encoder'
)

parser.add_argument(
    name='--positional-encoding',
    choices=['standard', '2d']
    required=True,
    help='Positional encoding type'
)

# Convolutional encoder arguments
parser.add_argument(
    name='--conv-encoder',
    choices=['vanilla', 'resnet']
    required=True,
    help='Type of convolutional encoder to employ'
)

parser.add_argument(
    name='--pretrain-conv-encoder',
    choices=['yes', 'no']
    required=True,
    help='Whether to pretrain the convolutional encoder'
)

# TRAINING PARAMETERS
parser.add_argument(
    name='--epochs',
    type=int,
    required=True,
    help='Number of epochs to train'
)

parser.add_argument(
    name='--optimizer',
    choices=['adam', 'sgd', 'padam']
    required=True,
    help='Optimizer to employ'
)

parser.add_argument(
    name='--lr-schedule',
    choices=['constant', 'cyclical']
    required=True,
    help='Learning rate schedule'
)
