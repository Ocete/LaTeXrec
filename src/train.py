import datasets
import cli_arguments
import conv_encoder
import pretrain_encoder
import transformer
import optimization
import masks

import time

'''
Main script used for training the model.
'''

# ARGUMENT PARSING
args = cli_arguments.parser.parse_args()

# LOAD DATA
remove_ambiguities = args.remove_ambiguities == 'yes'

# - Load dataset, ignore test dataset
if args.dataset == 'im2latex':
    train_df, _ = datasets.load_im2latex_dataset(remove_ambiguities)
    image_dir = datasets.get_paths(1)[1]
elif args.dataset == 'toy_50k':
    train_df, _ = datasets.load_toy_dataset(remove_ambiguities)
    image_dir = datasets.get_paths(0)[1]

# - Split in train/val and get tf.data.Dataset objects
train_df, val_df = datasets.split_in_train_and_val(train_df)

train_dataset = datasets.LaTeXrecDataset(
    train_df, image_dir)
val_dataset = datasets.LaTeXrecDataset(
    val_df, image_dir)

# - Configure datasets for batching and prefetching
train_dataset = train_dataset\
    .padded_batch(args.batch_size,
                  padded_shapes=([-1, -1, 1], [-1]),
                  padding_values=(
                      tf.constant(1.0, dtype=tf.float16),
                      tf.constant(datasets.LaTeXrecDataset.alph_size+1)
                  )
                  ).prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset\
    .padded_batch(args.batch_size,
                  padded_shapes=([-1, -1, 1], [-1]),
                  padding_values=(
                      tf.constant(1.0, dtype=tf.float16),
                      tf.constant(datasets.LaTeXrecDataset.alph_size+1)
                  )
                  ).prefetch(tf.data.AUTOTUNE)

# BUILD MODEL

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

# - Build transformer

# This should be the maximum allowed length of an input to the transformer encoder.
maximum_position_input = 3000
# This is the maximum allowed length of a target sequence.
maximum_position_target = args.maximum_target_length
# Size of the target vocabulary, plus 2 for start and end tokens.
target_vocab_size = datasets.LaTeXrecDataset.alph_size

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


# - Pretrain encoder if requested
if args.pretrain_conv_encoder == 'yes':
    if args.conv_encoder_epochs is None:
        raise ValueError('--conv-encoder-epochs must be set for pretraining')

    pretrain_encoder.pretrain_conv_encoder(
        cnn_encoder,
        cnn_decoder,
        dataset,
        val_dataset,
        args.conv_encoder_epochs,
        8
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
                                rate=dropout_rate)

# - Build optimizer

# Learning rate schedule, optimizer and optimizer parameters follows Vaswani et
# al. (https://arxiv.org/abs/1706.03762).
# TODO: make these parameters configurable
lr = optimization.VaswaniSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)

# SET UP CHECKPOINTING

checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# TRAIN MODEL

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
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

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


def train_step(inp, tar):
    """
    A training step on a batch.
    """
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    mask = masks.create_look_ahead_mask(tf.shape(tar_inp)[1])

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask=None,
                                     look_ahead_mask=mask,
                                     dec_padding_mask=None)
        loss = loss_function(tar_real, predictions, train_loss_object)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


def evaluate():
    """
    Evaluate the model on the validation set.
    """
    val_loss.reset_states()
    for inp, tar in val_dataset:
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        mask = masks.create_look_ahead_mask(tf.shape(tar_inp)[1])

        predictions, _ = transformer(inp, tar_inp,
                                     training=False,
                                     enc_padding_mask=None,
                                     look_ahead_mask=mask,
                                     dec_padding_mask=None)

        loss = loss_function(tar_real, predictions, val_loss_object)
        val_loss(loss)
        val_accuracy(tar_real, predictions)


# - Training loop
for epoch in range(args.epochs):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    for batch, (inp, tar) in enumerate(train_dataset):
        train_step(inp, tar)

        if batch % 50 == 0:
            evaluate()
            print('Epoch {}\tbatch {}\t' +
                  'Loss {:.4f}\tAccuracy {:.4f}\t' +
                  'Val. loss {:.4f}\tVal. acc. {:.4f}'
                  .format(
                      epoch + 1,
                      batch,
                      train_loss.result(),
                      train_accuracy.result(),
                      val_loss.result(),
                      val_accuracy.result()
                  )
                  )

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
