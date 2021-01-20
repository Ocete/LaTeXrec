import datasets
import cli_arguments
import conv_encoder
import pretrain_encoder
import transformer

'''
Main script used for training the model.
'''

# ARGUMENT PARSING
args = cli_arguments.parser.parse_args()

# LOAD DATA
train_df, test_df = datasets.load_im2latex_dataset()
image_dir = datasets.get_paths(1)[1]

train_df, val_df = datasets.split_in_train_and_val(train_df)

train_dataset = datasets.LaTeXrecDataset(train_df, image_dir).prefetch(tf.data.AUTOTUNE)
val_dataset = datasets.LaTeXrecDataset(val_df, image_dir).prefetch(tf.data.AUTOTUNE)

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

# This should be the maximum length of an input to the transformer encoder.
maximum_position_input = None  # TODO
# This is the maximum allowed length of a target sequence.
maximum_position_target = args.maximum_target_length
# Size of the target vocabulary, plus 2 for start and end tokens.
target_vocab_size = None  # TODO

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
