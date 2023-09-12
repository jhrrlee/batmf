# -*- coding: utf-8 -*-
"""
File Name: bilstm_attn_lib.py

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
    epsilon = 1e-10  # some small constant
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def v2_Bidirectional_LSTM_attention(input_shape, units, learning_rate, loss, num_layers, dropout_rate, final_activation):
  input_layer = Input(shape=input_shape)
  layer = input_layer

  for _ in range(num_layers-1):
      layer = Bidirectional(LSTM(units, return_sequences=True, dropout=dropout_rate))(layer)

  last_lstm = LSTM(units, return_state=True, dropout=dropout_rate)
  lstm_out, forward_h, forward_c, backward_h, backward_c = Bidirectional(last_lstm)(layer)

  context_vector = layers.Attention()([lstm_out, lstm_out])

  num_dense = units*2

  x = layers.Dense(num_dense, activation=final_activation)(context_vector)
  x = layers.Dense(num_dense // 2, activation=final_activation)(x)
  x = layers.Dense(num_dense // 4, activation=final_activation)(x)

  outputs = layers.Dense(1, activation="linear")(x)

  model = keras.Model(inputs=input_layer, outputs=outputs)

  optimizer = Adam(learning_rate=learning_rate, epsilon=1e-07)
  model.compile(optimizer=optimizer, loss=loss, metrics=['mean_absolute_error', 'mean_squared_error', mean_absolute_percentage_error, root_mean_squared_error])

  return model

