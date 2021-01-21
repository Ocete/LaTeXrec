import tensorflow as tf
import time


def crop_batch_to_multiple(batch, multiple):
    h, w = batch.shape[1], batch.shape[2]
    h, w = h-h % multiple, w-w % multiple
    return tf.image.resize_with_crop_or_pad(batch, h, w)


def encoder_pretraining(encoder,
                        decoder,
                        dataset,
                        val_dataset,
                        epochs,
                        cnn_reduction):
    """
    Pretrain the convolutional encoder in an unsupervised fashion, by appending
    a decoder to it.

    Params:
    - encoder: convolutional encoder object.
    - decoder: convolutional decoder object.
    - dataset: `tf.data.Dataset` object for the dataset.
    - val_dataset: `tf.data.Dataset` object for the validation dataset.
    - cnn_reduction: dimensionality reduction factor applied by the encoder in
      each dimension. E.g.: if the encoder has three max pooling layers, it
      should be 8.
    """

    autoencoder = tf.keras.Model(encoder.input, decoder.output)

    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean()
    val_loss = tf.keras.metrics.Mean()

    def train_step(inp):
        with tf.GradientTape() as tape:
            inp = crop_batch_to_multiple(inp, cnn_reduction)
            reconstructed_inp = autoencoder(inp)
            loss = tf.keras.losses.MSE(inp, reconstructed_inp)

            gradients = tape.gradient(
                loss, autoencoder.trainable_variables
            )
            encoder_optimizer.apply_gradients(
                zip(gradients, autoencoder.trainable_variables)
            )

            train_loss(loss)

    def evaluate():
        val_loss.reset_states()
        for inp, _ in val_dataset:
            inp = crop_batch_to_multiple(inp, cnn_reduction)
            reconstructed_inp = autoencoder(inp, training=False)
            loss = tf.keras.losses.MSE(inp, reconstructed_inp)
            cnn_val_loss(loss)

    for epoch in range(epochs):
        start = time.time()
        train_loss.reset_states()

        for batch, (inp, _) in enumerate(dataset):
            train_step(inp)

            if batch % 20 == 0:
                evaluate()

                print('Epoch {} batch {} ' +
                      'loss {:.4f} val. loss: {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             train_loss.result(),
                                                             val_loss.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
