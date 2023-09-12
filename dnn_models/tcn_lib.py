# -*- coding: utf-8 -*-
"""
File Name: tcn_lib.py

Description:

Author: junghwan lee
Email: jhrrlee@gmail.com
Date Created: 2023.09.12
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mean_absolute_error, mean_squared_error
from tcn import TCN
import math

import tensorflow as tf
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 1e-10
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def v3_gen_TCN(input_shape, num_filters, kernel_size, activation, learning_rate, loss, nb_stacks, no_dilations, tcn_dropout):
    inputs = Input(shape=input_shape)

    dilations = [int(math.pow(2, i)) for i in range(no_dilations)]
    x = TCN(nb_filters=num_filters, kernel_size=kernel_size, nb_stacks=nb_stacks, dilations=dilations,
            activation=activation, padding='same', use_skip_connections=True, dropout_rate=tcn_dropout, return_sequences=True)(inputs)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1, activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate, epsilon=1e-07)
    model.compile(optimizer=optimizer, loss=loss, metrics=['mean_absolute_error', 'mean_squared_error', mean_absolute_percentage_error, root_mean_squared_error])

    return model