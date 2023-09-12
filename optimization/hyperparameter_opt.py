#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
File Name: hyperparameter_opt.ipynb

Description: DNN hyperparameter optimization

Author: junghwan lee
Email: jhrrlee@gmail.com
Date Created: 2023.09.12
Todo:
1. Separation concerns between reshaping dataset and hyperparameter optimization
2. Refactor for enhanced flexibility.
3. Implement a function to resume training in case of kernel crash or stops
"""


# In[ ]:


import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import os


# In[ ]:


from google.colab import drive
drive.mount("/content/drive", force_remount=True)
sys.path.append('/content/drive/MyDrive/Colab_Notebooks')
sys.path.append('/content/drive/MyDrive/Colab_Notebooks/toyota_data')
sys.path.append('/content/drive/MyDrive/Colab_Notebooks/toyota_data/from pc/')
data_path = '/content/drive/MyDrive/Colab_Notebooks/toyota_data/data/'


# In[ ]:


data_path = './data/'
model_path = './models/Resnet/'
history_path = './history/Resnet/'
prediction_path = './prediction/Resnet/'
log_path = './log/'


# In[ ]:


get_ipython().system('pip install optuna')


# In[ ]:


get_ipython().system('pip install XlsxWriter')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import interpolate
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Add, Activation, Flatten, Input, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Conv1D, Flatten, Input, BatchNormalization, Activation, Add
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input, Dropout, MaxPooling1D, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import GRU
from keras import optimizers
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GlobalAveragePooling1D
import pickle
from tqdm import tqdm
import importlib
from tensorflow.keras.models import Model
import lib_analysis as ca
import optuna
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from tensorflow.keras import backend as K
import time
import math
import xlsxwriter


# In[ ]:


import lib_analysis as ca


# In[ ]:


from resnet_lib import v2_gen_resnet_model


# In[ ]:


importlib.reload(resnet_lib)


# In[ ]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# In[ ]:


physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
logical_devices = tf.config.list_logical_devices('GPU')
print("Num Logical GPUs Available: ", len(logical_devices))


# In[ ]:


def get_cyclesum_data_x(start_cycle, num_cycles, X):
  # Select only the data for the specified range of cycles
  X_selected = X[:, :, start_cycle:start_cycle+num_cycles]
  # Print shapes of the returned arrays
  print(f"Shape of X: {X_selected.shape}")

  return X_selected


# In[ ]:


def get_RUL(eol, num_cycles):
  return eol - num_cycles


# In[ ]:


import tensorflow as tf
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 1e-10  # some small constant
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + epsilon))) * 100


# In[ ]:


# Load the original data
"""
eol = np.load(data_path + 'valid_EOL_data.npy')
cycle_data_dis = np.load(data_path + 'valid_dis_data.npy')
cycle_data_ch = np.load(data_path + 'valid_ch_data.npy')
"""
# Load the original data
eol = np.load(data_path + 'eol.npy')
cycle_data_dis = np.load(data_path + 'cycle_data_dis.npy')
indices_train = np.load(data_path + 'train_ind.npy')
indices_val = np.load(data_path + 'test_ind.npy')
indices_test = np.load(data_path + 'secondary_test_ind.npy')
new_cycle_sum = np.load(data_path + 'new_cycle_sum.npy')

num_cells = eol.shape[0]
num_cycles = cycle_data_dis.shape[1]
cycle_count = np.arange(1, num_cycles+1)
RUL = eol - num_cycles

print("RUL shape:", RUL.shape)
print("cycle_data_dis shape:", cycle_data_dis.shape)
print("new_cycle_sum shape:", new_cycle_sum.shape)


# In[ ]:


indices_train = [0, 3, 4, 6, 9, 12, 15, 16, 19, 22, 28, 29, 32, 37, 39, 41, 44, 46, 48, 52, 55, 58, 61, 63, 66, 69, 73, 76, 79, 82, 85, 87, 89, 99, 102, 106, 109, 114, 117, 120, 123]
indices_val = [1, 5, 7, 8, 11, 13, 17, 20, 23, 25, 30, 33, 35, 38, 40, 45, 49, 50, 53, 56, 59, 64, 67, 70, 71, 74, 77, 80, 86, 90, 92, 94, 95, 98, 100, 104, 107, 110, 111, 113, 115, 118, 121]
indices_test = [2, 10, 14, 18, 21, 24, 26, 27, 31, 34, 36, 42, 43, 47, 51, 54, 57, 60, 62, 65, 68, 72, 75, 78, 81, 83, 84, 88, 91, 93, 96, 97, 101, 103, 105, 108, 112, 116, 119, 122]


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def rul_sumdata_preparation_with_fixed_indice(cycle_data, rul, indices_train, indices_val, indices_test):

  X = cycle_data.transpose(0,2,1)
  #X = cycle_data
  y = rul

  num_samples, num_cycles, num_features = X.shape
  # The length of each sequence is cycles * time_series
  sequence_length = num_cycles
  X_reshaped = X.reshape(num_samples, sequence_length, num_features)
  y_reshaped = y.reshape(-1, 1)

  print("X X_reshaped:", X.shape, X_reshaped.shape)
  print("y y_reshaped:", y.shape, y_reshaped.shape)
  print("num_features", num_features)
  # normalization
  y_scaler = MinMaxScaler()
  #scalers = [5]
  X_reshaped_norm = np.zeros_like(X_reshaped)
  for i in range(num_features):
    scalers= MinMaxScaler()
    X_reshaped_norm[:, :, i] = scalers.fit_transform(X_reshaped[:, :, i].reshape(-1, 1)).reshape(num_samples, sequence_length)

  y_scaler = MinMaxScaler()
  print("X_reshaped_norm", X_reshaped_norm.shape)
  #y_reshaped_norm = y_scaler.fit_transform(y_reshaped)
  y_reshaped_norm = y_reshaped


  # Split the data based on train, validatioin, test indices
  print("Indices of training set:", indices_train)
  print("Indices of validation set:", indices_val)
  print("Indices of test set:", indices_test)
  X_train, X_val, X_test = X_reshaped[indices_train], X_reshaped[indices_val], X_reshaped[indices_test]
  X_norm_train, X_norm_val, X_norm_test = X_reshaped_norm[indices_train], X_reshaped_norm[indices_val], X_reshaped_norm[indices_test]
  y_train, y_val, y_test  = y_reshaped[indices_train], y_reshaped[indices_val], y_reshaped[indices_test]
  y_norm_train, y_norm_val, y_norm_test = y_reshaped_norm[indices_train], y_reshaped_norm[indices_val], y_reshaped_norm[indices_test]
  print("X_train X_val X_test:", X_train.shape, X_val.shape, X_test.shape)
  print("X_norm_train X_norm_val X_norm_test:", X_norm_train.shape, X_norm_val.shape, X_norm_test.shape)
  print("y_train y_val y_test:", y_train.shape, y_val.shape, y_test.shape)
  print("y_norm_train y_norm_val y_norm_test:", y_norm_train.shape, y_norm_val.shape, y_norm_test.shape)

  return X_train, X_val, X_test, X_norm_train, X_norm_val, X_norm_test, y_train, y_val, y_test, y_norm_train, y_norm_val, y_norm_test


# In[ ]:


cycle_data_selected = get_cyclesum_data_x(40, 40, new_cycle_sum)
RUL_selected = get_RUL(eol, 80)


# In[ ]:


cycle_data_selected = get_cyclesum_data_x(1, 10, new_cycle_sum)
RUL_selected = get_RUL(eol, 10)


# In[ ]:


cycle_data_selected = get_cyclesum_data_x(40, 100, new_cycle_sum)
RUL_selected = get_RUL(eol, 100)


# In[ ]:


cycle_data_selected = get_cyclesum_data_x(1, 10, new_cycle_sum)
RUL_selected = get_RUL(eol, 10)


# In[ ]:


cycle_data_selected = get_cyclesum_data_x(11, 10, new_cycle_sum)
RUL_selected = get_RUL(eol, 10)


# In[ ]:


X_train, X_val, X_test, X_train_norm, X_val_norm, X_test_norm, y_train, y_val, y_test, y_train_norm, y_val_norm, y_test_norm = rul_sumdata_preparation_with_fixed_indice(cycle_data_selected, RUL_selected, indices_train, indices_val, indices_test)


# ## fixed parts regardless of architecture

# In[ ]:


def train_and_evaluate_model(model, no_epoch, X_train_norm, y_train_norm, X_val_norm, y_val_norm, X_test_norm, y_test_norm, batch_size, custom_objects, model_path, history_path, prediction_path, trial, test_name):
  start_time_train = time.time()
  try:
      history = model.fit(
          X_train_norm, y_train_norm,
          validation_data=(X_val_norm, y_val_norm),
          epochs=no_epoch,
          batch_size=int(batch_size),
          callbacks=[EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=2000),
                      ModelCheckpoint(model_path + f'{test_name}_{trial.number}.h5', monitor='val_loss', save_best_only=True)],
          verbose=0
      )
  except ValueError:
      return float('inf')

  training_time = time.time() - start_time_train
  model.save(model_path + f'final_{test_name}.h5')

  if os.path.exists(model_path + f'{test_name}_{trial.number}.h5'):
      model = load_model(model_path + f'{test_name}_{trial.number}.h5', custom_objects=custom_objects)
  else:
      return float('inf')

  start_time = time.time()
  y_train_pred = model.predict(X_train_norm)
  prediction_time = (time.time() - start_time) * 1000
  y_train = y_train_norm
  y_train_pred = np.squeeze(y_train_pred)
  y_train = np.squeeze(y_train)

  # Compute evaluation metrics on the unnormalized predictions
  train_rmse_rul = np.sqrt(mean_squared_error(y_train, y_train_pred))
  train_mae_rul = mean_absolute_error(y_train, y_train_pred)
  train_mape_rul = mean_absolute_percentage_error(y_train, y_train_pred).numpy().item()

  print(f"Train RMSE for RUL: {train_rmse_rul.item()} prediction_time: {prediction_time} training_time: {training_time} MAPE for RUL: {train_mape_rul} MAE for RUL: {train_mae_rul}")
  trial.set_user_attr('training_time', training_time)
  trial.set_user_attr('prediction_time', prediction_time)
  trial.set_user_attr('train_mape_rul', train_mape_rul)
  trial.set_user_attr('train_rmse_rul', train_rmse_rul.item())
  trial.set_user_attr('train_mae_rul', train_mae_rul.item())

  y_val_pred = model.predict(X_val_norm)
  # Add the prediction time to the trial
  # Unnormalize the predictions
  #y_pred = y_scaler.inverse_transform(y_pred_norm).flatten()
  y_val = y_val_norm
  y_val_pred = np.squeeze(y_val_pred)
  y_val = np.squeeze(y_val)

  # Compute evaluation metrics on the unnormalized predictions
  val_rmse_rul = np.sqrt(mean_squared_error(y_val, y_val_pred))
  val_mae_rul = mean_absolute_error(y_val, y_val_pred)
  val_mape_rul = mean_absolute_percentage_error(y_val, y_val_pred).numpy().item()

  print(f"Validation RMSE for RUL: {val_rmse_rul.item()} MAPE for RUL: {val_mape_rul} MAE for RUL: {val_mae_rul}")
  trial.set_user_attr('val_mape_rul', val_mape_rul)
  trial.set_user_attr('val_rmse_rul', val_rmse_rul.item())
  trial.set_user_attr('val_mae_rul', val_mae_rul.item())

  y_test_pred = model.predict(X_test_norm)
  # Add the prediction time to the trial
  # Unnormalize the predictions
  #y_pred = y_scaler.inverse_transform(y_pred_norm).flatten()
  y_test = y_test_norm
  y_test_pred = np.squeeze(y_test_pred)
  y_test = np.squeeze(y_test)

  # Compute evaluation metrics on the unnormalized predictions
  test_rmse_rul = np.sqrt(mean_squared_error(y_test, y_test_pred))
  test_mae_rul = mean_absolute_error(y_test, y_test_pred)
  test_mape_rul = mean_absolute_percentage_error(y_test, y_test_pred).numpy().item()

  print(f"Test RMSE for RUL: {test_rmse_rul.item()} MAPE for RUL: {test_mape_rul} MAE for RUL: {test_mae_rul}")
  trial.set_user_attr('test_mape_rul', test_mape_rul)
  trial.set_user_attr('test_rmse_rul', test_rmse_rul.item())
  trial.set_user_attr('test_mae_rul', test_mae_rul.item())

  # Save the history data for this trial as a separate CSV file
  history_df = pd.DataFrame(history.history)
  history_df.to_csv(history_path+test_name, index=False)

  # Create a Pandas Excel writer using openpyxl as the engine
  writer = pd.ExcelWriter(prediction_path + test_name + '.xlsx', engine='xlsxwriter')

  # Create separate DataFrames for each dataset
  train_data = pd.DataFrame({'y_train': y_train, 'y_train_pred': y_train_pred})
  val_data = pd.DataFrame({'y_val': y_val, 'y_val_pred': y_val_pred})
  test_data = pd.DataFrame({'y_test': y_test, 'y_test_pred': y_test_pred})

  # Write each DataFrame to a different sheet in the Excel file
  train_data.to_excel(writer, sheet_name='Train', index=False)
  val_data.to_excel(writer, sheet_name='Validation', index=False)
  test_data.to_excel(writer, sheet_name='Test', index=False)

  # Save the Excel file
  writer.save()
  mape_score = (train_mape_rul + val_mape_rul + test_mape_rul) / 3
  rmse_score = (train_rmse_rul + val_rmse_rul + test_rmse_rul) / 3
  print(f"rmse_score : {rmse_score} mape score : {mape_score}")

  return mape_score


# In[ ]:


def optimize_hyperparameters(n_trial, no_epoch, test_db, test_name, model_function, model_path, history_path, prediction_path, custom_objects):
    study = optuna.create_study(storage=f'sqlite:///{test_db}', study_name=test_name, direction='minimize', load_if_exists=True)
    study.optimize(lambda trial: objective(trial, no_epoch, test_name, model_function, model_path, history_path, prediction_path, custom_objects), n_trials=n_trial)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state', 'user_attrs'))
    df.rename(columns={'value': 'MAPE', 'user_attrs_prediction_time': 'Prediction Time (ms)', 'user_attrs_mape_rul': 'MAPE'}, inplace=True)

    best_params = study.best_params
    best_score = study.best_value
    print("Best Hyperparameters:", best_params)
    print("Best score:", best_score)
    return df


# ## Architecture dependency

# In[ ]:


def objective(trial, no_epoch, test_name, model_function, model_path, history_path, prediction_path, custom_objects):
    # Define the hyperparameter search space
    loss = trial.suggest_categorical('loss_function', ['mean_squared_error'])
    batch_size = trial.suggest_categorical('batch_size', [1, 41])
    learning_rate = trial.suggest_categorical('learning_rate', [0.001, 0.0001, 0.00001])
    no_resblock = trial.suggest_categorical('layers', [3, 4, 5])
    no_kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 10])
    strides_size = trial.suggest_categorical('strides_size', [1, 2])
    pool_size = trial.suggest_categorical('pool_size', [1, 2])
    no_filters = trial.suggest_categorical('filters', [128, 256])

    K.clear_session()
    input_shape = X_train.shape[1:]
    model = model_function(input_shape, learning_rate, loss, no_resblock, no_filters, no_kernel_size, strides_size, pool_size)

    mape_score = train_and_evaluate_model(model, no_epoch, X_train_norm, y_train_norm, X_val_norm, y_val_norm, X_test_norm, y_test_norm, batch_size, custom_objects, model_path, history_path, prediction_path, trial, test_name)

    return mape_score


# In[ ]:


"""
data_path = './data/'
model_path = '.'
history_path = '.'
prediction_path = '.'
log_path = './log/'
"""


# ## model dependency

# In[ ]:


test_db = 'resnet.db'
test_name = "v2_1_resnet_new2_sum_11_10"
no_epoch = 1
n_trial = 1
custom_objects = {"root_mean_squared_error": root_mean_squared_error}

optimize_hyperparameters(
    n_trial = n_trial,
    no_epoch = no_epoch,
    test_db = test_db,
    test_name = test_name,
    model_function=v2_gen_resnet_model,
    model_path=model_path,
    history_path=history_path,
    prediction_path=prediction_path,
    custom_objects=custom_objects
)


# In[ ]:


df.to_csv(log_path+test_name, index=False)

