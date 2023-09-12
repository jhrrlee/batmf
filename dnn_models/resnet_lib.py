# -*- coding: utf-8 -*-
"""
File Name: resnet_lib.py

Description:

Author: junghwan lee
Email: jhrrlee@gmail.com
Date Created: 2023.09.12
"""

from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 1e-10  # some small constant
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# Function to create a Convolutional Block
def v2_conv_block(input_tensor, filters, kernel_size, activation='leaky_relu', strides=1):
    x = Conv1D(filters, kernel_size, activation=activation, strides=strides, padding='same')(input_tensor)
    x = Conv1D(filters, kernel_size, activation=activation, padding='same')(x)
    x = Conv1D(filters*4, kernel_size, activation=activation, padding='same')(x)
    shortcut = Conv1D(filters*4, kernel_size, activation=activation, strides=strides, padding='same')(input_tensor)
    x = Add()([x, shortcut])
    return x

# Function to create an Identity Block
def v2_identity_block(input_tensor, filters, kernel_size, activation='leaky_relu'):
    x = Conv1D(filters, kernel_size, activation=activation, padding='same')(input_tensor)
    x = Conv1D(filters, kernel_size, activation=activation, padding='same')(x)
    x = Conv1D(filters*4, kernel_size, activation=activation, padding='same')(x)
    x = Add()([x, input_tensor])
    return x

def v2_gen_resnet_model(input_shape, learning_rate, loss, no_resblock, no_filters, no_kernel_size, strides_size, pool_size):
    model_input = Input(shape=input_shape)

    x = Conv1D(filters=no_filters, kernel_size=no_kernel_size, strides=strides_size, padding='same', activation='relu')(model_input)
    x = MaxPooling1D(pool_size=pool_size, strides=strides_size, padding='same')(x)

    for _ in range(no_resblock):
        x = v2_conv_block(x, filters=no_filters, kernel_size=no_kernel_size, activation='leaky_relu', strides=strides_size)
        x = v2_identity_block(x, filters=no_filters, kernel_size=no_kernel_size, activation='leaky_relu')

    x = GlobalAveragePooling1D()(x)
    output = Dense(1, name='rul')(x)

    model = Model(inputs=model_input, outputs=output)

    optimizer = Adam(learning_rate=learning_rate, epsilon=1e-07)
    model.compile(optimizer=optimizer, loss=loss, metrics=['mean_absolute_error', 'mean_squared_error', mean_absolute_percentage_error, root_mean_squared_error])

    return model

