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
import numpy as np
import pandas as pd
import tensorflow as tf

from parameters import *


################################################################################
# Main
if __name__ == "__main__":
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

    text = list(df["text"])
    labels = list(df["category"])

    classes = set(labels)
    num_classes = len(classes)
    print(f'Classes: {classes}')
    print(f'Number of classes: {num_classes}')

    # create mapping of class categories to integers
    cat2int = {}
    i = 0
    for c in classes:
        cat2int[c] = i
        i += 1
    #print(cat2int)

    # create reverse mapping of integers to class categories
    int2cat = {v: k for k, v in cat2int.items()}
    #print(int2cat)

    for i in range(len(labels)):
        labels[i] = cat2int[labels[i]]

    # 80% training set
    train_text = text[:int(0.8*len(text))]
    train_labels = labels[:int(0.8*len(labels))]

    # 10% validation set
    val_text = text[int(0.8*len(text)):int(0.9*len(text))]
    val_labels = labels[int(0.8*len(labels)):int(0.9*len(labels))]

    # 10% test set
    test_text = text[int(0.9*len(text)):]
    test_labels = labels[int(0.9*len(labels)):]

    print(f'Number of training text examples: {len(train_text)}')
    print(f'Number of training labels: {len(train_labels)}')
    print(f'Number of validation text examples: {len(val_text)}')
    print(f'Number of validation labels: {len(val_labels)}')
    print(f'Number of test text examples: {len(test_text)}')
    print(f'Number of text labels: {len(test_labels)}')

    MAX_WORDS = len(set(text))  # limit data to top x words
    tokenize = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_WORDS, char_level=False)
    tokenize.fit_on_texts(text)
    t = tokenize.texts_to_matrix(text)
    print(np.array(t).shape)
    print(t)

    # truncate/pad input sequences so that they are all the same length

    # split data into training, validation, test sets

    # print shape

    # ----- MODEL ----- #
    # Embedding
    # GRU
    # Dense output

    # ----- ASSESSMENT ----- #
    # accuracy

    # precision

    # recall

    # ----- PREDICTION ----- #
