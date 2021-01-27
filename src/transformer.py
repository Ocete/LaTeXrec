import tensorflow as tf
from attention import MultiHeadAttention
from conv_encoder import vanilla_encoder
import performer.fast_attention as fattn
from positional_encoding import positional_encoding_1d, positional_encoding_2d

'''
Implementation of the Transformer/Performer architecture.
'''


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(dff, activation='elu'),
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class PerformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 d_model,
                 num_heads,
                 dff,
                 rand_feat=100,
                 rate=0.1):
        super(PerformerEncoderLayer, self).__init__()

        self.mha = fattn.Attention(d_model, num_heads, rate,
                                   kernel_transformation=fattn.softmax_kernel_transformation,
                                   projection_matrix_type=True,
                                   nb_random_features=rand_feat)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        # (batch_size, input_seq_len, d_model)
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, fast_attn=False, rand_feat=100):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        # (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers,
                 d_model,
                 num_heads,
                 dff,
                 pos_encoding,
                 cnn_encoder=None,
                 use_fast_attention_enc=False,
                 rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = pos_encoding

        if cnn_encoder is None:
            self.cnn = vanilla_encoder(d_model, 32)
        else:
            self.cnn = cnn_encoder

        if use_fast_attention_enc:
            self.enc_layers = [PerformerEncoderLayer(d_model, num_heads, dff, rate=rate)
                               for _ in range(num_layers)]
        else:
            self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                               for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):

        x = self.cnn(x)
        x_s = tf.shape(x)
        x = tf.reshape(x, [x_s[0], -1, self.d_model])
        seq_len = tf.shape(x)[1]

        # Add position encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)

        return x  # (batch_size, cnn(input_seq_len), d_model)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # (batch_size, target_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        # (batch_size, target_seq_len, d_model)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        fn_output = self.dropout3(ffn_output, training=training)
        # (batch_size, target_seq_len, d_model)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 pos_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.bn = tf.keras.layers.BatchNormalization()

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = pos_encoding

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 target_vocab_size,
                 pe_input,
                 pe_target,
                 pe_2d_height,
                 pe_2d_width,
                 pos_encoding,
                 cnn_encoder=None,
                 use_fast_attention_enc=False,
                 rate=0.1):
        super(Transformer, self).__init__()

        self.pos_encoding_dec = positional_encoding_1d(pe_target, d_model)
        if pos_encoding == 'standard':
            self.pos_encoding_enc = positional_encoding_1d(pe_input, d_model)
        elif pos_encoding == '2d':
            self.pos_encoding_enc = positional_encoding_2d(d_model, pe_2d_height,
                                                           pe_2d_width)
            self.pos_encoding_enc = tf.reshape(self.pos_encoding_enc, [1, -1, d_model])

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, self.pos_encoding_enc,
                               cnn_encoder, use_fast_attention_enc, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size,
                               self.pos_encoding_enc, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training,
             look_ahead_mask, dec_padding_mask):

        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, training)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights
