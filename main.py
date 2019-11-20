"""
author: Mario Grabovaj
description: Text classification with preprocessed text
date: 19.11.2019
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow import keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import matplotlib.pyplot as plt

BUFFER_SIZE = 1000


def show_graph(history_dict):
    """
    Show graph of accuracy and loss over time.

    :param history_dict: history dictionary
    :return: void show plot
    """
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.clf()  # clear figure

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.show()


if __name__ == '__main__':
    print('Text classification with preprocessed text')

    # Split data
    (train_data, test_data), info = tfds.load(
        'imdb_reviews/subwords8k',
        split=(tfds.Split.TRAIN, tfds.Split.TEST),
        as_supervised=True,
        with_info=True)

    # Encoder
    encoder = info.features['text'].encoder

    # Prepare data for training
    train_batches = (train_data.shuffle(BUFFER_SIZE).padded_batch(32, train_data.output_shapes))
    test_batches = (test_data.padded_batch(32, train_data.output_shapes))

    # Build the model
    model = keras.Sequential([
        keras.layers.Embedding(encoder.vocab_size, 16),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(1, activation='sigmoid')])


    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(train_batches,
                        epochs=10,
                        validation_data=test_batches,
                        validation_steps=30)

    # Evaluate the model
    loss, accuracy = model.evaluate(test_batches)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    # Create graph or accuracy and loss over time
    history_dict = history.history
    show_graph(history_dict)