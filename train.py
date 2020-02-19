"""
Michael Patel
February 2019

Python 3.6.5
TensorFlow 2.0.0

File description:

"""
################################################################################
# Imports
import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from parameters import *
from model import *


################################################################################
# Main
if __name__ == "__main__":
    """
    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn", "Classify using a CNN", action="store_true")
    parser.add_argument("--rnn", "Classify using an RNN", action="store_true")
    args = parser.parse_args()
    
    if args.cnn:
        # use CNN
    elif args.rnn:
        # uss RNN
    else:
        print(f'Please provide an argument')
        parser.print_help()
        sys.exit(1)
    """

    # print out TF version
    print(f'TF version: {tf.__version__}')

    # eager execution is enabled by default in TF 2.0
    print(f'Using eager execution: {tf.executing_eagerly()}')

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    text = []
    labels = []

    # read in csv
    data_filepath = os.path.join(os.getcwd(), "data\\bbc-text.csv")
    df = pd.read_csv(data_filepath)

    # gather text sequences and cateogory labels
    text = list(df["text"])
    labels = list(df["category"])

    vocab = set(text)
    vocab_size = len(vocab)
    print(f'Vocab size: {vocab_size}')

    categories = set(labels)
    num_categories = len(categories)
    print(f'Categories: {categories}')
    print(f'Number of categories: {num_categories}')
    print(f'{df["category"].value_counts()}')

    # create mapping of class categories to integers
    cat2int = {}
    i = 0
    for c in categories:
        cat2int[c] = i
        i += 1

    # create reverse mapping of integers to class categories
    int2cat = {v: k for k, v in cat2int.items()}

    # convert all categorical labels to integers
    for i in range(len(labels)):
        labels[i] = cat2int[labels[i]]

    # 80% training set
    train_text = text[:int(0.8*len(text))]
    train_labels = labels[:int(0.8*len(labels))]
    train_labels = np.array(train_labels)

    # 10% validation set
    val_text = text[int(0.8*len(text)):int(0.9*len(text))]
    val_labels = labels[int(0.8*len(labels)):int(0.9*len(labels))]
    val_labels = np.array(val_labels)

    # 10% test set
    test_text = text[int(0.9*len(text)):]
    test_labels = labels[int(0.9*len(labels)):]
    test_labels = np.array(test_labels)

    print(f'Number of training text examples: {len(train_text)}')
    print(f'Number of training labels: {len(train_labels)}')
    print(f'Number of validation text examples: {len(val_text)}')
    print(f'Number of validation labels: {len(val_labels)}')
    print(f'Number of test text examples: {len(test_text)}')
    print(f'Number of text labels: {len(test_labels)}')

    # Tokenization
    tokenize = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_WORDS, char_level=False)  # word tokens
    tokenize.fit_on_texts(text)  # update internal vocabulary based on list of texts

    # Vectorization
    train_text = tokenize.texts_to_matrix(train_text)
    train_text = np.array(train_text)
    val_text = tokenize.texts_to_matrix(val_text)
    val_text = np.array(val_text)
    test_text = tokenize.texts_to_matrix(test_text)
    test_text = np.array(test_text)

    # truncate/pad input sequences so that they are all the same length

    # one-hot encode labels because the ordering of category values is not important
    # only 5 categories, so not a huge increase in the dimensionality
    # sequences of integers
    train_labels = tf.keras.utils.to_categorical(train_labels, num_categories)
    val_labels = tf.keras.utils.to_categorical(val_labels, num_categories)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_categories)

    # print shape
    print(f'Shape of train text: {train_text.shape}')
    print(f'Shape of train labels: {train_labels.shape}')
    print(f'Shape of validation text: {val_text.shape}')
    print(f'Shape of validation labels: {val_labels.shape}')
    print(f'Shape of test text: {test_text.shape}')
    print(f'Shape of test labels: {test_labels.shape}')

    # ----- MODEL ----- #
    # Embedding
    # GRU
    # Dense output
    m = build_rnn(vocab_size, num_categories)
    m.summary()

    m.compile(
        loss=tf.keras.losses.categorical_crossentropy,  # labels are one-hot encoded
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    # ----- ASSESSMENT ----- #
    # accuracy

    # precision

    # recall

    # ----- PREDICTION ----- #
