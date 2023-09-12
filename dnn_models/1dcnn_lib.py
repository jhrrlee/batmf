# -*- coding: utf-8 -*-
"""
File Name: 1dcnn_lib.py

Description:

Author: junghwan lee
Email: jhrrlee@gmail.com
Date Created: 2023.09.12
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 1e-10
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def create_v2_1DCNN_FC(input_shape, num_filters, kernel_size, strides_size, activation, learning_rate, loss, num_layers, final_dropout, num_dense, final_activation):
    model = Sequential()

    for i in range(num_layers):
        if input_shape[0] > 1:
            if i == 0:
                model.add(Conv1D(num_filters, kernel_size, strides=strides_size, activation=activation, padding='same', input_shape=input_shape))
            else:
                model.add(Conv1D(num_filters, kernel_size, strides=strides_size, activation=activation, padding='same'))
            input_shape = (input_shape[0] // strides_size, num_filters)
        else:
            break

    model.add(GlobalAveragePooling1D())
    num_dense = num_filters
    model.add(Dropout(final_dropout))
    model.add(Dense(num_dense, activation=final_activation))
    model.add(Dropout(final_dropout))
    model.add(Dense(num_dense//2, activation=final_activation))
    model.add(Dropout(final_dropout))
    model.add(Dense(num_dense//2, activation=final_activation))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['mean_absolute_error', 'mean_squared_error', mean_absolute_percentage_error, root_mean_squared_error])

    return model