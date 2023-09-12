# -*- coding: utf-8 -*-
"""
File Name: lstm_lib.py

Description:

Author: junghwan lee
Email: jhrrlee@gmail.com
Date Created: 2023.09.12
"""

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, root_mean_squared_error
import tensorflow.keras.layers as layers

import tensorflow as tf
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 1e-10
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def v2_create_LSTM(input_shape, units, learning_rate, loss, num_layers, final_activation, dropout_rate):
  model = Sequential()


  model.add(LSTM(units, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, dropout=dropout_rate, unroll=False, use_bias=True, return_sequences=True if num_layers>1 else False, input_shape=input_shape))

  for i in range(1, num_layers):
      return_seq = (i < num_layers-1)
      model.add(LSTM(units, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, dropout=dropout_rate, unroll=False, use_bias=True, return_sequences=return_seq))

  num_dense = units
  x = layers.Dense(num_dense, activation=final_activation)(model.output)
  x = layers.Dense(num_dense // 2, activation=final_activation)(x)
  x = layers.Dense(num_dense // 4, activation=final_activation)(x)

  outputs = layers.Dense(1, activation="linear")(x)

  model = Model(inputs=model.input, outputs=outputs)


  optimizer = Adam(learning_rate=learning_rate, epsilon=1e-07)
  model.compile(optimizer=optimizer, loss=loss, metrics=['mean_absolute_error', 'mean_squared_error', mean_absolute_percentage_error, root_mean_squared_error])

  return model