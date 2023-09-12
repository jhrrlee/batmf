# -*- coding: utf-8 -*-
"""
File Name: transformer_lib.py

Description:

Author: junghwan lee
Email: jhrrlee@gmail.com
Date Created: 2023.09.12
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.layers import MultiHeadAttention

def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 1e-10
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + epsilon))) * 100

import tensorflow as tf
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def positional_encoding(seq_len, d_model):
    angles = np.arange(seq_len)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :]//2)) / np.float32(d_model))
    pos_encoding = np.zeros(angles.shape)
    pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def create_v4_transformer_model(input_shape, learning_rate, loss, num_encoder, head_size, num_heads, ff_dim, d_model, dropout=0, final_dropout=0.1, final_activation='relu'):
    inputs = layers.Input(shape=input_shape)

    # Add position encoding
    pos_enc = positional_encoding(input_shape[0], d_model)
    x = layers.Add()([inputs, pos_enc])

    for _ in range(num_encoder):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Flatten the output
    x = layers.Flatten()(x)

    # Create a temporary model to calculate the output shape of the Flatten layer
    temp_model = Model(inputs=inputs, outputs=x)
    dummy_input = np.random.rand(1, *input_shape)
    output_shape = temp_model.predict(dummy_input).shape

    num_dense = output_shape[-1]

    x = layers.Dropout(final_dropout)(x)
    x = layers.Dense(num_dense, activation=final_activation)(x)
    x = layers.Dropout(final_dropout)(x)
    x = layers.Dense(num_dense//2, activation=final_activation)(x)
    x = layers.Dropout(final_dropout)(x)
    x = layers.Dense(num_dense//4, activation=final_activation)(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = Adam(learning_rate=learning_rate, epsilon=1e-07)
    model.compile(optimizer=optimizer, loss=loss, metrics=['mean_absolute_error', 'mean_squared_error', mean_absolute_percentage_error, root_mean_squared_error])
    return model