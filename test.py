import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 0 all, 1 no info, 2 no warning, 3 no error
import tensorflow as tf
import numpy as np

from util import generate_text, build_model, train_model, preprocess

filename = 'shakespeare'

epochs = 1
batch_size = 64
seq_length = 100
rnn_units = 128 # 1024
embedding_dim = 256

dataset, idx_to_char, char_to_idx, vocab = preprocess(filename, batch_size, seq_length)

vocab_size = len(vocab)
checkpoint_dir = './training_checkpoints/' + filename
checkpoint_prefix = checkpoint_dir + '/ckpt'


### Train model (comment out if only generating)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)
# model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)) # use this to continue training where it left off
history = train_model(model, dataset, epochs=epochs, checkpoint_prefix=checkpoint_prefix)

### Generate sample (comment out if only training)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
print(generate_text(model,char_to_idx,idx_to_char,
                    start_string=u"Somebody"))


