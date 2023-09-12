# -*- coding: utf-8 -*-
"""cleansing_lib.ipynb


File Name: feature_gen.ipynb

Description: data visulization of original data.

Author: junghwan lee
Email: jhrrlee@gmail.com
Date Created: 2023.09.07

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.preprocessing import MinMaxScaler
from scipy import interpolate
import pickle
import itertools
import math
from tqdm import tqdm

"""This function is designed to visualize battery charge and discharge data and apply basic cleansing logic to identify accurate start and end points for the discharging phase.

Parameters:
bat_dict: A nested dictionary containing battery data. Each key represents a cell, and its value is another dictionary with cycle data and summary data.

cell_key: A string indicating the specific battery cell to be analyzed.

cycle_key: A string indicating the specific cycle of the battery cell to be analyzed.

max_gap (dQd/dt): An integer. It is the maximum allowable difference between consecutive indices in the discharge start determination logic. This helps to filter out noise or small fluctuations in the data and identify a reliable start point for the discharging phase. If the difference between indices exceeds this value, it will not be considered as a valid discharge start point.

validation_range: An integer. It defines the length of the sequence of indices being checked for consistent gaps when determining the start of the discharging phase. It's essentially the window size over which we're looking for consecutive or near-consecutive indices. A larger value means a stricter criterion, ensuring that the start point selected is consistent over a longer range.

Description:
The battery's discharging phase is identified by an increase in the Qd (discharge capacity) value. However, to ensure the identified start of the discharging phase isn't due to noise or minor fluctuations, the function applies a heuristic:

Look for points where the difference in Qd is positive (indicating a potential start of discharge).

From these points, identify sequences where the difference between consecutive indices (gaps) are all smaller than the max_gap value.

The validation_range parameter ensures that this consistent sequence of gaps is observed over a specified range, adding an extra layer of validation.
"""

# dataset-dependent function.
def get_cycle_data(bat_dict, cell_key):

    return bat_dict[cell_key]['cycles']

def configure_axis(ax, x_label, y_label, title=None):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title is not None:
      ax.set_title(title)

def find_charging_start(value, ch_current_threshold = 0):
    """Find the index of the charging start based on the current value."""
    arr_idx = np.where(value['I'] > ch_current_threshold)[0]
    return None if len(arr_idx) == 0 else arr_idx[0]

def find_discharging_start(value, idx_ch_s, max_gap, validation_range):
    """Find the index of the discharging start based on the max gap (unit: discharged capacity)."""
    arr_idx = np.where(np.diff(value['Qd']) > 0)[0]
    for count, idx in enumerate(arr_idx):
        if all(diff < max_gap for diff in np.diff(arr_idx[count:count+validation_range])):
            return idx
    return idx_ch_s - 1 if idx_ch_s is not None else None

def plot_cleansed_cycle(cycledata, cycle_key, ch_current_threshold=0, validation_range=20, max_gap=20):
    cycle_data = cycledata[cycle_key]
    fig, axes = plt.subplots(5, 2, figsize=(12,12), dpi=100)

    idx_ch_s = find_charging_start(cycle_data, ch_current_threshold)
    idx_dis_s = find_discharging_start(cycle_data, idx_ch_s, max_gap, validation_range)
    idx_ch_e = idx_dis_s - 1

    if idx_dis_s is not None:
        idx_ch_e = idx_dis_s - 1

        # Discharging end index
        idx_dis_e_list = np.where(np.diff(cycle_data['Qd'][idx_dis_s:]) == 0)[0] + idx_dis_s
        idx_dis_e = idx_dis_e_list[-1] if len(idx_dis_e_list) > 0 else None

    basic_plots = {
        'V': axes[0][1],
        'T': axes[1][0],
        'I': axes[1][1],
        'Qdlin': axes[2][0],
        'Tdlin': axes[2][1],
        'dQdV': axes[3][0],
        'Qc': axes[3][1],
        'Qd': axes[4][1]
    }

    for attr, ax in basic_plots.items():
        x_data = range(len(cycle_data[attr])) if attr in ['Qdlin', 'Tdlin', 'dQdV'] else cycle_data['t']
        ax.scatter(x_data, cycle_data[attr])
        x_label = 'Index' if attr in ['Qdlin', 'Tdlin', 'dQdV'] else 'Time'
        configure_axis(ax, x_label, attr)

    # charging and discharging in a plot
    if idx_ch_s is not None and idx_dis_s is not None:
        axes[0][0].scatter(cycle_data['t'][idx_ch_s:idx_ch_e], cycle_data['Qc'][idx_ch_s:idx_ch_e], color='red', label='Charging')
        axes[0][0].scatter(cycle_data['t'][idx_dis_s:idx_dis_e], cycle_data['Qd'][idx_dis_s:idx_dis_e], color='blue', label='Discharging')
        axes[0][0].legend(loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.show()

"""This function is more advanced for debugging of data cleasing with logs and plots. </br>
It is possible to verify multiple cycle
"""

def cycle_color_map(cycle_count):
    color_map = plt.cm.get_cmap('coolwarm')
    color_range = colors.Normalize(vmin=0, vmax=cycle_count)
    return color_map(color_range(np.arange(cycle_count)))

def plot_cleansed_cycles(cycledata, start_cycle, no_of_cycles, ch_current_threshold=0, validation_range=100, max_gap=20):
    cycles = cycledata
    fig, axes = plt.subplots(5, 2, figsize=(12,12), dpi=100)
    cycle_count = len(cycles.keys())
    cycle_colors = cycle_color_map(cycle_count)

    remove_cycle_count = 0

    for cy_count, (cy_key, cycle_data) in tqdm(itertools.islice(enumerate(cycles.items()), start_cycle, None), total=no_of_cycles):
        if cy_count > no_of_cycles + remove_cycle_count:
                break

        idx_ch_s = find_charging_start(cycle_data, ch_current_threshold)
        if idx_ch_s is None:
            print('no found discharging start with current at cycle', cy_key)
            remove_cycle_count = remove_cycle_count + 1
            continue

        idx_dis_s = find_discharging_start(cycle_data, idx_ch_s, max_gap, validation_range)
        idx_ch_e = idx_dis_s - 1

        arr_idx = np.where(np.diff(cycle_data['Qd'][idx_dis_s:])==0)[0] + idx_dis_s
        idx_dis_e = arr_idx[-1]

        arr_idx = np.where(cycle_data['t'] > 100)[0]
        if len(arr_idx) > 0:
            print('remove cycle from time', cy_key)
            remove_cycle_count = remove_cycle_count + 1
            continue

        basic_plots = {
            'V': axes[0][1],
            'T': axes[1][0],
            'I': axes[1][1],
            'Qdlin': axes[2][0],
            'Tdlin': axes[2][1],
            'dQdV': axes[3][0],
            'Qc': axes[3][1],
            'Qd': axes[4][1],
        }

        for attr, ax in basic_plots.items():
            x_data = range(len(cycle_data[attr])) if attr in ['Qdlin', 'Tdlin', 'dQdV'] else cycle_data['t']
            ax.scatter(x_data, cycle_data[attr], c=[cycle_colors[cy_count]])
            x_label = 'Index' if attr in ['Qdlin', 'Tdlin', 'dQdV'] else 'Time'
            #configure_axis(ax, x_label, attr, f'{attr} for {cell_key} {cy_key}')
            configure_axis(ax, x_label, attr)

        if idx_ch_s is not None and idx_dis_s is not None:
            axes[0][0].scatter(cycle_data['t'][idx_ch_s:idx_ch_e], cycle_data['Qc'][idx_ch_s:idx_ch_e], color='red')
            axes[0][0].scatter(cycle_data['t'][idx_dis_s:idx_dis_e], cycle_data['Qd'][idx_dis_s:idx_dis_e], color='blue')


    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=start_cycle, vmax=cycle_count))
    sm.set_array([])
    #fig.colorbar(sm, label='Cycle', ax=axes[:, -1], pad=0.1)
    fig.subplots_adjust(right=0.8, hspace=0.5)
    fig.colorbar(sm, ax=axes.ravel().tolist(), label='Cycle', pad=0.01)


    #plt.tight_layout()
    plt.show()

"""This function generates interpolated data for a complete cycle (charging, discharging cycle)"""

def inp(x, no_of_timeseries):
    t = np.arange(len(x))
    f = interpolate.interp1d(t, x, kind='linear')
    t_new = np.linspace(t.min(), t.max(), num=no_of_timeseries)
    return f(t_new)

def get_inp_cycles(bat_dict, cycle_start, no_of_cycles, max_data_size, no_of_timeseries):
    eol = []
    data_names = ['Qd', 'Qc', 'V', 'I', 'T', 'Qdlin', 'Tdlin', 'dQdV']
    cycle_data = []

    for cell_count, (cell_key, cell_value) in enumerate(bat_dict.items()):
        cycles = cell_value['cycles']
        eol.append(cell_value['cycle_life'][0][0])

        cell_data = []

        for _, cy_value in itertools.islice(cycles.items(), cycle_start, cycle_start + no_of_cycles):
            cycle_properties_data = []
            for idx, name in enumerate(data_names):
                interpolated_data = inp(cy_value[name], no_of_timeseries)
                if len(interpolated_data) > max_data_size:
                    print(f'{name} at cycle {cell_count} has length {len(interpolated_data)}, which is larger than max_data_size')
                cycle_properties_data.append(interpolated_data)

            cell_data.append(cycle_properties_data)

        cycle_data.append(cell_data)

    # Convert to numpy array
    cycle_data = np.array(cycle_data)

    return eol, cycle_data

def plot_inp_cycles(reshaped_data, cell_idx, feature_names=['Qd', 'Qc', 'V', 'I', 'T', 'Qdlin', 'Tdlin', 'dQdV']):

    no_of_cycles, no_of_features, no_of_timeseries = reshaped_data.shape[1:4]

    cols = 2
    rows = math.ceil(len(feature_names) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(10*cols, 6*rows))

    axes = axes.ravel()

    # Get cycle colors using the provided function
    cycle_colors = cycle_color_map(no_of_cycles)

    for feature_idx, feature_name in enumerate(feature_names):
        for cycle in range(no_of_cycles):
            axes[feature_idx].plot(reshaped_data[cell_idx, cycle, feature_idx, :], color=cycle_colors[cycle])
        axes[feature_idx].set_title(f"Feature: {feature_name}")
        axes[feature_idx].set_xlabel("Time Series Index")
        axes[feature_idx].set_ylabel(feature_name)
        axes[feature_idx].grid(True)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=no_of_cycles-1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation='vertical', label='Cycle', pad=0.01)

    plt.show()

# define the Interpolation function from [1]
def inp_with_time(x, t, no_of_timeseries):
    f = interpolate.interp1d(t,x, kind='linear')
    x_new=list(np.linspace(t.min(),t.max(),num=no_of_timeseries))
    return np.array(list(f(x_new)))

def get_cleaned_cycles(bat_dict, cycle_start, no_of_cycles, max_data_size, no_of_timeseries,
                       feature_names=['Qd', 'Qc', 'V', 'I', 'T'],
                       ch_current_threshold=0.01,
                       validation_range=20,
                       max_gap=20,
                       max_time=100):

    cycle_data_dis = []
    cycle_data_ch = []
    eol = []
    cleaned_cells = []  # List to store the IDs of cells that have been cleaned

    for cell_count, (cell_key, cell_value) in enumerate(bat_dict.items()):
        cycles = cell_value['cycles']
        eol.append(cell_value['cycle_life'][0][0])
        remove_cycle_count = 0
        max_dis = 0
        max_ch = 0

        for i, (cy_key, cy_value) in enumerate(itertools.islice(cycles.items(), cycle_start, None)):
            cy_count = i + 1
            if cy_count > no_of_cycles + remove_cycle_count:
                break

            time = cy_value['t']

            if max(time) > max_time: # unmanaged data if time is too long.
                print(f'remove cycle {cy_key} from time for cell {cell_key}')
                remove_cycle_count += 1
                cleaned_cells.append(cell_key)  # Add the cell ID to the cleaned_cells list
                continue

            # Identify charging and discharging start and end indices
            idx_ch_s = find_charging_start(cy_value, ch_current_threshold)
            if idx_ch_s is None:
                print(f'No found charging start with current at cycle {cy_key} for cell {cell_key}')
                remove_cycle_count += 1
                cleaned_cells.append(cell_key)  # Add the cell ID to the cleaned_cells list
                continue

            idx_dis_s = find_discharging_start(cy_value, idx_ch_s, max_gap, validation_range)
            idx_ch_e = idx_dis_s - 1
            #idx_dis_e = next((idx for idx in np.where(np.diff(cy_value['Qd'][idx_dis_s:]) == 0)[0] + idx_dis_s), None)
            arr_idx = np.where(np.diff(cy_value['Qd'][idx_dis_s:])==0)[0] + idx_dis_s
            idx_dis_e = arr_idx[-1]

            dis_time = time[idx_dis_s:idx_dis_e]
            ch_time = time[idx_ch_s:idx_ch_e]

            # Interpolate data for each feature
            for feature in feature_names:
                feature_value = cy_value[feature]
                ch_data = inp_with_time(feature_value[idx_ch_s:idx_ch_e], ch_time, no_of_timeseries)
                cycle_data_ch.append(ch_data)

                dis_data = inp_with_time(feature_value[idx_dis_s:idx_dis_e], dis_time, no_of_timeseries)
                cycle_data_dis.append(dis_data)

            # Update max sizes of time-series data for each cycle and check if they exceed allowed size
            max_dis = max(max_dis, len(dis_time))
            max_ch = max(max_ch, len(ch_time))

            if len(dis_time) > max_data_size:
                print(f'Cycle {cy_key} for cell {cell_key} exceeds max_data_size in discharge with data_size {len(dis_time)}')
            if len(ch_time) > max_data_size:
                print(f'Cycle {cy_key} for cell {cell_key} exceeds max_data_size in charge with data_size {len(ch_time)}')

            # Debugging
            if len(dis_time) <= 1:
                print(f'Issue detected in discharging time-series for cell {cell_key} at cycle {cy_key}.')
            if len(ch_time) <= 1:
                print(f'Issue detected in charging time-series for cell {cell_key} at cycle {cy_key}.')

    if cleaned_cells:
        cleaned_cells = list(set(cleaned_cells))  # Removing duplicate cell IDs
        print(f"Cycles cleaned for cells with IDs: {', '.join(cleaned_cells)}")

    cycle_data_dis = np.array(cycle_data_dis).reshape(-1, no_of_cycles, len(feature_names), no_of_timeseries)
    cycle_data_ch = np.array(cycle_data_ch).reshape(-1, no_of_cycles, len(feature_names), no_of_timeseries)
    eol=np.array(eol).reshape(-1,1)

    return eol, cycle_data_dis, cycle_data_ch

def getdata_from_summary(bat_dict, start_cycle, no_of_cycles, feature_names=['IR', 'QC', 'QD', 'Tavg', 'Tmin', 'Tmax', 'chargetime']):
    cycle_sum = []

    for cell in bat_dict.keys():
        summary = bat_dict[cell]['summary']
        cell_data = [summary[feature][start_cycle-1:start_cycle + no_of_cycles - 1] for feature in feature_names]
        cycle_sum.extend(cell_data)

    return cycle_sum

def calculate_current_statistics(cycle_data_dis, cycle_data_ch, current_index=2):
    no_of_cells, no_of_cycles = cycle_data_dis.shape[:2]

    # Initialize arrays to hold the statistics
    mean_current_dis = np.zeros((no_of_cells, no_of_cycles))
    max_current_dis = np.zeros((no_of_cells, no_of_cycles))
    min_current_dis = np.zeros((no_of_cells, no_of_cycles))

    mean_current_ch = np.zeros((no_of_cells, no_of_cycles))
    max_current_ch = np.zeros((no_of_cells, no_of_cycles))
    min_current_ch = np.zeros((no_of_cells, no_of_cycles))

    for i in range(no_of_cells):  # loop over cells
        for j in range(no_of_cycles):  # loop over cycles
            # For discharge data
            current_dis = np.abs(cycle_data_dis[i, j, current_index, :])  # get the current data and take the absolute value
            mean_current_dis[i, j] = np.mean(current_dis)
            max_current_dis[i, j] = np.max(current_dis)
            min_current_dis[i, j] = np.min(current_dis)

            # For charge data
            current_ch = np.abs(cycle_data_ch[i, j, current_index, :])  # get the current data and take the absolute value
            mean_current_ch[i, j] = np.mean(current_ch)
            max_current_ch[i, j] = np.max(current_ch)
            min_current_ch[i, j] = np.min(current_ch)

    return mean_current_dis, max_current_dis, min_current_dis, mean_current_ch, max_current_ch, min_current_ch

def get_human_data(bat_dict, cycle_start, no_of_cycles):

  human_data = []

  for cell_count, (cell_key, cell_value) in enumerate(bat_dict.items()):
    cycles = cell_value['cycles']
    max_dis = 0
    max_ch = 0
    remove_cycle_count = 0
    for i, (cy_key, cy_value) in enumerate(itertools.islice(cycles.items(), cycle_start, None)):
      cy_count = i + 1
      if cy_count > no_of_cycles:
          break
      human_data.append(cy_value['Qdlin'])
      human_data.append(cy_value['Tdlin'])
      human_data.append(cy_value['dQdV'])

  return human_data

def gen_statistical_data(cycle_sum, human_data, cycle_data_ch, cycle_data_dis):

  dqdv = human_data[:, :, 2, :]

  # Calculate the average, maximum, and minimum per cycle
  average_dqdv = np.mean(dqdv, axis=-1)
  maximum_dqdv = np.max(dqdv, axis=-1)
  minimum_dqdv = np.min(dqdv, axis=-1)

  average_dqdv = np.expand_dims(average_dqdv, axis=1)
  maximum_dqdv = np.expand_dims(maximum_dqdv, axis=1)
  minimum_dqdv = np.expand_dims(minimum_dqdv, axis=1)

  mean_current_dis, max_current_dis, _, mean_current_ch, max_current_ch, _ = calculate_current_statistics(cycle_data_dis, cycle_data_ch)

  mean_current_dis = np.expand_dims(mean_current_dis, axis=1)
  max_current_dis = np.expand_dims(max_current_dis, axis=1)
  mean_current_ch = np.expand_dims(mean_current_ch, axis=1)
  max_current_ch = np.expand_dims(max_current_ch, axis=1)

  #cycle_sum_reshaped = np.transpose(cycle_sum, (0, 2, 1))

  return np.concatenate((cycle_sum, average_dqdv, maximum_dqdv, minimum_dqdv, mean_current_dis, max_current_dis, mean_current_ch, max_current_ch), axis=1)