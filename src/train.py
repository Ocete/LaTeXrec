import datasets
import transformer
import conv_encoder

import argparse

'''
Main script used for training the model.
'''

parser = argparse.ArgumentParser(
    description='Main script used to train the model.'
)

data_loading = parser.add_argument_group(
    'Data loading',
    'Data loading arguments'
)

model_group = parser.add_argument_group(
    'Model',
    'Model arguments'
)

training_group = parser.add_argument_group(
    'Training',
    'Training arguments'
)

# DATA LOADING ARGUMENTS
data_loading.add_argument(
    '--dataset',
    choices=['toy_50k', 'im2latex'],
    required=True,
    help='Select the dataset used for training'
)

data_loading.add_argument(
    '--samples',
    type=int,
    required=True,
    help='Number of samples to train on'
)

# MODEL ARGUMENTS

# Transformer arguments
model_group.add_argument(
    '--num-layers',
    type=int,
    required=True,
    help='Number of layers of the encoder and decoder'
)

model_group.add_argument(
    '--depth',
    type=int,
    required=True,
    help='Depth parameter of the model'
)

model_group.add_argument(
    '--feedforward-units',
    type=int,
    required=True,
    help='Units on pointwise feedforward network'
)

model_group.add_argument(
    '--num-heads',
    type=int,
    required=True,
    help='Number of heads on the multihead attention'
)

model_group.add_argument(
    '--dropout-rate',
    type=float,
    required=True,
    help='Dropout rate in the transformer'
)

model_group.add_argument(
    '--performer-attention-encoder',
    choices=['yes', 'no'],
    required=True,
    help='Whether to employ the FAVOR+ mechanism for attention in the encoder'
)

model_group.add_argument(
    '--positional-encoding',
    choices=['standard', '2d'],
    required=True,
    help='Positional encoding type'
)

# Convolutional encoder arguments
model_group.add_argument(
    '--conv-encoder',
    choices=['vanilla', 'resnet'],
    required=True,
    help='Type of convolutional encoder to employ'
)

model_group.add_argument(
    '--pretrain-conv-encoder',
    choices=['yes', 'no'],
    required=True,
    help='Whether to pretrain the convolutional encoder'
)

# TRAINING PARAMETERS
training_group.add_argument(
    '--epochs',
    type=int,
    required=True,
    help='Number of epochs to train'
)

training_group.add_argument(
    '--optimizer',
    choices=['adam', 'sgd', 'padam'],
    required=True,
    help='Optimizer to employ'
)

training_group.add_argument(
    '--lr-schedule',
    choices=['constant', 'cyclical'],
    required=True,
    help='Learning rate schedule'
)

args = parser.parse_args()
