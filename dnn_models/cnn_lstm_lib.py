# -*- coding: utf-8 -*-
"""
File Name: cnn_lstm_lib.py

Description:

Author: junghwan lee
Email: jhrrlee@gmail.com
Date Created: 2023.09.12
"""

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Concatenate, Dense, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 1e-10
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def create_CNN_LSTM(input_shape, cnn_filters, kernel_size, cnn_layers, lstm_units, lstm_dropout, lstm_layers, learning_rate, loss):
    model = Sequential()

    for i in range(cnn_layers):
        if input_shape[0] > 1:
            if i == 0:
                model.add(Conv1D(cnn_filters, kernel_size, strides=2, activation='relu', padding='same', input_shape=input_shape))
            else:
                model.add(Conv1D(cnn_filters, kernel_size, strides=2, activation='relu', padding='same'))
        else:
            break

    for i in range(lstm_layers):
        return_seq = (i < lstm_layers-1)
        model.add(LSTM(lstm_units, dropout=lstm_dropout, unroll=False, use_bias=True, return_sequences=return_seq))

    num_dense = lstm_units
    model.add(Dense(num_dense, activation='relu'))
    model.add(Dense(num_dense//2, activation='relu'))
    model.add(Dense(num_dense//4, activation='relu'))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=learning_rate, epsilon=1e-07)
    model.compile(optimizer=optimizer, loss=loss, metrics=['mean_absolute_error', 'mean_squared_error', mean_absolute_percentage_error, root_mean_squared_error])

    return model