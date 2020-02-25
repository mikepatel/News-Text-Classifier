"""
Michael Patel
February 2019

Python 3.6.5
TensorFlow 2.0.0

File description:

"""
################################################################################
# Imports
import tensorflow as tf

from parameters import *


################################################################################
# RNN
def build_rnn(vocab_size, num_categories):
    model = tf.keras.Sequential()

    # Embedding layer
    # convert sequences of integers to sequences of vectors
    model.add(tf.keras.layers.Embedding(
        input_dim=vocab_size,  # size of vocab
        output_dim=EMBEDDING_DIM  # dimension of dense embedding
    ))

    # GRU layers
    model.add(tf.keras.layers.GRU(
        units=NUM_RNN_UNITS
    ))

    # Fully connected output layer
    model.add(tf.keras.layers.Dense(
        units=num_categories,
        activation=tf.keras.activations.softmax
    ))

    return model


################################################################################
# FC
def build_fc(num_categories):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(
        units=512,
        input_shape=(MAX_WORDS, ),  # (batch, MAX_WORDS)
        activation=tf.keras.activations.relu
    ))

    model.add(tf.keras.layers.Dense(
        units=num_categories,
        activation=tf.keras.activations.softmax
    ))

    return model
