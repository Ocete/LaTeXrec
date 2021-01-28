import cli_arguments
import conv_encoder
import datasets
import latexrec_datasets
import log
import masks
import optimization
import pretrain_encoder
import transformer

import tensorflow as tf
import time
from pathlib import Path

'''
Main script used for training the model.
'''

# ARGUMENT PARSING
args = cli_arguments.parser.parse_args()

# LOG PARAMS AND INITIALIZE LOGGING

# - Get and create logging folder
log_folder = log.get_folder_path(args)

# - Log CLI arguments
log.log_params(log_folder, args)

# - Get logger
logger = log.get_logger(args, log_folder, mode=0)
logger.info('DEVELOPMENT:')

# LOAD DATA
remove_ambiguities = args.remove_ambiguities == 'yes'

# - Load dataset, ignore test dataset
logger.info('Loading data')

if args.dataset == 'im2latex':
    train_df, _ = latexrec_datasets.load_im2latex_dataset(remove_ambiguities)
    image_dir = latexrec_datasets.get_paths(1)[1]
elif args.dataset == 'toy_50k':
    train_df, _ = latexrec_datasets.load_toy_dataset(remove_ambiguities)
    image_dir = latexrec_datasets.get_paths(0)[1]

# - If 'samples' is an argument, take only those many samples
if hasattr(args, 'samples') and args.samples is not None:
    train_df = train_df[:args.samples]

# - Split in train/val and get tf.data.Dataset objects
train_df, val_df = latexrec_datasets.split_in_train_and_val(train_df)

train_dataset = latexrec_datasets.LaTeXrecDataset(
    train_df, image_dir)
val_dataset = latexrec_datasets.LaTeXrecDataset(
    val_df, image_dir)

# - Filter images that are too wide
train_dataset = train_dataset.filter(
    lambda im, _: tf.shape(im)[1] < 3840
)
val_dataset = val_dataset.filter(
    lambda im, _: tf.shape(im)[1] < 3840
)

# - Configure datasets for batching and prefetching. Pad images with ones (white
# - when normalized to [0,1]), formulas with end token.
train_dataset = train_dataset\
    .padded_batch(
        args.batch_size,
        padded_shapes=([-1, -1, 1], [-1]),
        padding_values=(
            tf.constant(1, dtype=tf.uint8),
            tf.constant(latexrec_datasets.LaTeXrecDataset.alph_size+1,
                        dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset\
    .padded_batch(
        args.batch_size,
        padded_shapes=([-1, -1, 1], [-1]),
        padding_values=(
            tf.constant(1, dtype=tf.uint8),
            tf.constant(latexrec_datasets.LaTeXrecDataset.alph_size+1,
                        dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

# BUILD MODEL
logger.info('Building model')

# - Set hyperparameters

# This should be the maximum allowed length of an input to the transformer encoder.
maximum_position_input = 5000
# This is the maximum allowed length of a target sequence.
maximum_position_target = args.maximum_target_length
# Size of the target vocabulary, plus 2 for start and end tokens.
target_vocab_size = latexrec_datasets.LaTeXrecDataset.alph_size+2

# For 2d positional encoding: set height and width. Width
# must be the maximum size of a picture, hardcoded for now
max_height = 50
max_width = 4000

# Set rest of model hyperparameters
num_layers = args.num_layers
d_model = args.depth
ff_units = args.feedforward_units
num_heads = args.num_heads
dropout = args.dropout_rate
performer_attention_encoder = args.performer_attention_encoder
pos_encoding = args.positional_encoding

# Transform as needed
performer_attention_encoder = performer_attention_encoder == 'yes'

# Checks on hyperparameters
if d_model % num_heads != 0:
    raise ValueError('Depth of the model must divide the number of heads')

if dropout < 0 or dropout > 1:
    raise ValueError('Dropout rate must be between 0 and 1')

# - Build convolutional encoder
if args.conv_encoder == 'vanilla':
    cnn_encoder = conv_encoder.vanilla_encoder(d_model,
                                               args.conv_filters)
    if args.pretrain_conv_encoder == 'yes':
        cnn_decoder = conv_encoder.vanilla_decoder(cnn_encoder.output,
                                                   args.conv_filters,
                                                   3)
elif args.conv_encoder == 'resnet':
    cnn_encoder = conv_encoder.resnet_encoder(d_model,
                                              args.conv_filters)
    if args.pretrain_conv_encoder == 'yes':
        cnn_decoder = conv_encoder.resnet_decoder(cnn_encoder.output,
                                                  args.conv_filters)

# - Pretrain encoder if requested
if args.pretrain_conv_encoder == 'yes':
    if args.conv_encoder_epochs is None:
        raise ValueError('--conv-encoder-epochs must be set for pretraining')

    logger.info('Pretraining encoder')
    pretrain_encoder.pretrain_conv_encoder(
        cnn_encoder,
        cnn_decoder,
        dataset,
        val_dataset,
        args.conv_encoder_epochs,
        8,
        logger
    )

    cnn_encoder.trainable = False

# - Finally build the Transformer model
model = transformer.Transformer(num_layers,
                                d_model,
                                num_heads,
                                ff_units,
                                target_vocab_size,
                                pe_input=maximum_position_input,
                                pe_target=maximum_position_target,
                                cnn_encoder=cnn_encoder,
                                pe_2d_height=max_height,
                                pe_2d_width=max_width,
                                pos_encoding=pos_encoding,
                                use_fast_attention_enc=performer_attention_encoder,
                                rate=dropout)

# - Build optimizer

# Learning rate schedule, optimizer and optimizer parameters follows Vaswani et
# al. (https://arxiv.org/abs/1706.03762).
# TODO: make these parameters configurable
if args.lr_schedule == 'vaswani':
    lr = optimization.VaswaniSchedule(d_model)
elif args.lr_schedule == 'recurrent-vaswani':
    lr = optimization.RecurrentVaswaniSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)

# SET UP CHECKPOINTING

checkpoint_path = str(Path(log_folder) / 'checkpoints')
ckpt = tf.train.Checkpoint(transformer=model,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# TRAIN MODEL
logger.info('Starting the model training')

# - Declare losses
train_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

val_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

# - Declare metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy'
)

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='val_accuracy'
)

bleu_metric = datasets.load_metric('bleu')
train_bleu = 0
val_bleu = 0

# - Functions implementing: loss, training step and evaluation


def loss_function(real, pred, loss_object):
    """
    Don't take into account masked out elements in loss calculation.
    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def train_step(inp, tar, evaluate_step):
    """
    A training step on a batch.
    """
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    mask = masks.create_look_ahead_mask(tf.shape(tar_inp)[1])

    with tf.GradientTape() as tape:
        predictions, _ = model(inp, tar_inp,
                               True,
                               look_ahead_mask=mask,
                               dec_padding_mask=None)
        loss = loss_function(tar_real, predictions, train_loss_object)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if evaluate_step:
        train_loss(loss)
        train_accuracy(tar_real, predictions)
        train_bleu = bleu_metric.compute(predictions=predictions,
                                         references=tar_real)

def evaluate():
    """
    Evaluate the model on the validation set.
    """
    val_loss.reset_states()
    for inp, tar in val_dataset:
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        mask = masks.create_look_ahead_mask(tf.shape(tar_inp)[1])

        predictions, _ = model(inp, tar_inp,
                               training=False,
                               look_ahead_mask=mask,
                               dec_padding_mask=None)

        loss = loss_function(tar_real, predictions, val_loss_object)
        val_loss(loss)
        val_accuracy(tar_real, predictions)

        val_bleu = bleu_metric.compute(predictions=predictions,
                                       references=tar_real)


    ckpt_manager.save()

# Params for early stopping
logger.info('Initializing early stop params')
es_params = { 
    'prev_val_acc': 0,
    'min_val_increment': 0.001, # 0.0001,
    'evals_without_increment': 0,
    'max_evals_without_incr': 10, # 10
    'early_stopping_triggered': False
}

def early_stopping():
    """
    Updates wether the algorithm should stop following
    an early stopping implementation.
    """
    global es_params

    val = val_accuracy.result()

    if val - es_params['prev_val_acc'] >= es_params['min_val_increment']:
        es_params['prev_val_acc'] = val
        es_params['evals_without_increment'] = 0
    else:
        es_params['evals_without_increment'] += 1

    es_params['early_stopping_triggered'] = \
        es_params['evals_without_increment'] >= \
            es_params['max_evals_without_incr']

# - Training history, for metrics
history = dict(loss=[], acc=[], val_loss=[], val_acc=[])
step = 0

# - Training loop
for epoch in range(args.epochs):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    for batch, (inp, tar) in enumerate(train_dataset):
        evaluate_step = batch % 50 == 0
        train_step(inp, tar, evaluate_step)

        if evaluate_step:
            evaluate()

            msg = ('Epoch {}\tbatch {}\t' +
                   'loss {:.4f}\taccuracy {:.4f}\t' +
                   'val. loss {:.4f}\tval. acc. {:.4f}' +
                   'bleu: {:.4f}\t val. bleu: {:.4f}')\
                .format(
                epoch + 1,
                batch,
                train_loss.result(),
                train_accuracy.result(),
                val_loss.result(),
                val_accuracy.result(),
                train_bleu,
                val_bleu
            )
            logger.info(msg)
            
            history['loss'].append(train_loss.result().numpy())
            history['acc'].append(train_accuracy.result().numpy())
            
            history['val_loss'].append(val_loss.result().numpy())
            history['val_acc'].append(val_accuracy.result().numpy())


            early_stopping()
            if es_params['early_stopping_triggered']:
                logger.info('Early stopping triggered.')
                break
            
    logger.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    if es_params['early_stopping_triggered']:
        logger.info('That was the last epoch due to early stopping.\n')
        break

# - Log history
log.log_history(log_folder, history)
