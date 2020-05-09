import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 0 all, 1 no info, 2 no warning, 3 no error
import tensorflow as tf
import numpy as np

def preprocess(filename, batch_size, seq_length):
    text = open('datasets/'+filename+'.txt', 'rb').read().decode(encoding='utf-8')
    print ('Length of text: {} characters'.format(len(text)))

    vocab = sorted(set(text))
    print ('{} unique characters'.format(len(vocab)))

    char_to_idx = {u:i for i, u in enumerate(vocab)}
    idx_to_char = np.array(vocab)

    text_as_int = np.array([char_to_idx[c] for c in text])

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)

    BUFFER_SIZE = 10000
    dataset = dataset.shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True)

    return dataset, idx_to_char, char_to_idx, vocab

def train_model(model, dataset, epochs, checkpoint_prefix):
    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    model.compile(optimizer='adam', loss=loss)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True,
        save_best_only=True,
        monitor='loss') # TODO monitor val_loss instead

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)

    return model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback, early_stopping])

def build_model(vocab_size, embedding_dim, rnn_units, batch_size, use_LSTM=False):
    model = tf.keras.Sequential()

    if embedding_dim != None:
        model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                            batch_input_shape=[batch_size, None]))
    else: # use one-hot encoding instead of embedding layer
        def one_hot(x):
            return tf.one_hot(tf.cast(x, 'uint8'), depth=vocab_size)
        model.add(tf.keras.layers.Lambda(one_hot, batch_input_shape=[batch_size,None]))

    # recurrent layer
    if use_LSTM:
        model.add(tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True))
    else:
        model.add(tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True, stateful=True))

    # output layer
    model.add(tf.keras.layers.Dense(vocab_size))

    return model

def generate_text(model, char_to_idx, idx_to_char,
                  start_string, num_generate=1000, temperature=1.0):
    # Converting our start string to numbers (vectorizing)
    input_eval = [char_to_idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    model.reset_states()
    for _ in range(num_generate):
        predictions = model(input_eval)
         # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx_to_char[predicted_id])

    return (start_string + ''.join(text_generated))
