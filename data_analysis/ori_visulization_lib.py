# -*- coding: utf-8 -*-
"""

RUL_ML_Framework
File Name: original_data_visulization.py

Description: data visulization of original data.


Author: junghwan lee
Email: jhrrlee@gmail.com
Date Created: 2023.09.04

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from sklearn.preprocessing import MinMaxScaler

"""## Summary data visualization
From [1], summary and cycle keys </br>
- summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg': summary_TA, 'Tmin': summary_TM, 'Tmax': summary_TX, 'chargetime': summary_CT, 'cycle': summary_CY}  </br>
- cd = {'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 'Tdlin': Tdlin, 'V':V, 'dQdV': dQdV, 't':t}  </br>
</br>

Displays summary data for each cell in each plot per summary data.
"""

def plot_summary_all(bat_dict, n_cols):
  n_rows = (len(bat_dict.keys()) + n_cols - 1) // n_cols

  cells = list(bat_dict.keys())
  summary_keys = [key for key in bat_dict[cells[0]]['summary'] if key != 'cycle']

  for key in summary_keys:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_rows, 3 * n_cols))
    print(key)

    for j, cell in enumerate(cells):
      summary = bat_dict[cell]['summary']
      ax = axes[j // n_cols, j % n_cols]
      ax.plot(summary['cycle'], summary[key])
      ax.set_xlabel('Cycle Number')
      ax.set_ylabel(key)
      ax.set_title(f'{cell} {key}')

    plt.tight_layout()
    plt.show()

"""Display specific summary data for all cell"""

def filter_outliers(data, threshold=0.2):
    median_val = np.median(data)
    low_limit = median_val - median_val * threshold
    high_limit = median_val + median_val * threshold
    return [x for x in data if low_limit <= x <= high_limit], median_val

def get_cmap_and_norm(bat_dict):
    cycle_life_values = [int(value['cycle_life'][0][0]) for value in bat_dict.values()]
    norm = plt.Normalize(np.min(cycle_life_values), np.max(cycle_life_values))
    cmap = plt.get_cmap('coolwarm').reversed()
    return cmap, norm

def add_cycle_life_colorbar(ax, cmap, norm):
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Cycle Life', rotation=270, labelpad=20)

def plot_summary_per_summarykey(bat_dict, key, ylabel):
    fig, ax = plt.subplots()

    cmap, norm = get_cmap_and_norm(bat_dict)

    cells = list(bat_dict.keys())
    lines = []

    for cell in cells:
        summary = bat_dict[cell]['summary']
        data = [float(x) for x in summary[key]]
        filtered_data, _ = filter_outliers(data)
        line, = ax.plot(summary['cycle'][:len(filtered_data)], filtered_data)
        lines.append(line)

    cycle_life = [int(bat_dict[cell]['cycle_life'][0][0]) for cell in cells]
    colors = cmap(norm(np.array(cycle_life)))
    for line, color in zip(lines, colors):
        line.set_color(color)

    add_cycle_life_colorbar(ax, cmap, norm)

    ax.set_xlabel('Cycle Number')
    ax.set_ylabel(ylabel)
    plt.show()

def plot_all_cells_per_summary(bat_dict):
    cells = list(bat_dict.keys())
    summary_keys = [key for key in bat_dict[cells[0]]['summary'] if key != 'cycle']

    for key in summary_keys:
        plot_summary_per_summarykey(bat_dict, key, key)

"""Sorting cell with median values that are used to sort cells since IR, QC, QD has abnormal values for some cycles"""

def sort_scores_by_median_with_key(bat_dict, summary_key):
    cell_names = []
    scores = []

    for cell in bat_dict.keys():
        summary = bat_dict[cell]['summary']
        data = [float(x) for x in summary[summary_key]]
        filtered_data, median_val = filter_outliers(data)
        cell_names.append(cell)
        scores.append(median_val)

    sorted_scores, sorted_cell_names = zip(*sorted(zip(scores, cell_names)))

    return np.column_stack((sorted_cell_names, sorted_scores))

"""Sorting cycle life that is not in summary data"""

def sort_cycle_life(bat_dict):
    sorted_cycle_life = sorted(bat_dict.items(), key=lambda x: int(x[1]['cycle_life'][0][0]))

    sorted_cell_names, sorted_cell_data = zip(*sorted_cycle_life)
    cycle_life_values = [int(data['cycle_life'][0][0]) for data in sorted_cell_data]

    return np.column_stack((sorted_cell_names, cycle_life_values))

"""sorting cells using "sort_scores_by_median_with_key" and "plot_cycle_life" functions according to cycle life, IR, QC, etc"""

def gen_summary_df(bat_dict):
    parameters = ['IR', 'QC', 'QD', 'Tavg', 'Tmin', 'Tmax', 'chargetime', 'CL']
    sorted_data_dict = {}

    for parameter in parameters:
        if parameter == 'CL':
            sorted_data = sort_cycle_life(bat_dict)
        else:
            sorted_data = sort_scores_by_median_with_key(bat_dict, summary_key=parameter)
        #sort by name
        sorted_data = sorted(sorted_data, key=lambda x: x[0])
        sorted_data_dict[parameter] = np.array(sorted_data)

    df = pd.DataFrame({
        'Cell Name': sorted_data_dict['CL'][:, 0],
        'CL': sorted_data_dict['CL'][:, 1].astype(int),
        'IR': sorted_data_dict['IR'][:, 1],
        'QC': sorted_data_dict['QC'][:, 1],
        'QD': sorted_data_dict['QD'][:, 1],
        'Tavg': sorted_data_dict['Tavg'][:, 1],
        'Tmin': sorted_data_dict['Tmin'][:, 1],
        'Tmax': sorted_data_dict['Tmax'][:, 1],
        'Charge Time': sorted_data_dict['chargetime'][:, 1]
    })

    for column in df.columns:
        if column != 'Cell Name':
            df[column] = pd.to_numeric(df[column], errors='coerce')

    return df

def sorted_bar_chart(df, column_name):
    """
    Plot a sorted bar chart based on the column name.
    """
    sorted_df = df.sort_values(by=column_name)
    sorted_df.plot(x='Cell Name', y=column_name, kind='bar', figsize=(20,6), legend=False)
    plt.title(f'{column_name} by Cell Name')
    plt.ylabel(column_name)
    plt.xlabel('Cell Name')
    plt.show()

"""## ## Cycle data visualization
Display cycle data of all cells for a attribute at specific cycle </br>
cycle_keys = ['I', 'Qc', 'Qd', 'Qdlin', 'T', 'Tdlin', 'V', 'dQdV', 't']
"""

def plot_allcell_with_dkck(bat_dict, key, cycle_key):
    sorted_data = sort_cycle_life(bat_dict)
    cells = sorted_data[:, 0]
    cycle_life_values = sorted_data[:, 1].astype(int)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap, norm = get_cmap_and_norm(bat_dict)

    colors = cmap(norm(np.array(cycle_life_values)))

    for i, cell_key in enumerate(cells):
        ax.plot(bat_dict[cell_key]['cycles'][cycle_key][key], color=colors[i])

    ax.set_xlabel('Data point')
    ax.set_ylabel(key)
    ax.set_title(f'{key} for all Cells')
    add_cycle_life_colorbar(ax, cmap, norm)
    plt.show()

"""This function displays the discharge fade for a given cell in a plot, with a color bar representing the cycle number. The fade is shown using a scatter plot, with the capacity (Ah) on the x-axis and the voltage (V) on the y-axis. Each cycle is represented by a different color, with the color bar indicating the corresponding cycle number. This makes it easy to visually identify how the discharge capacity fades over time for each cycle of the cell."""

def cycle_color_map(cycle, cycle_count):
    color_map = matplotlib.colormaps['coolwarm']
    color_range = colors.Normalize(vmin=0, vmax=cycle_count)
    return color_map(color_range(cycle))

def plot_discharge_fade_bar(bat_dict, cell_name, start_cycle, start_point, end_point):
    """
    Plots the cycles of a given cell from a starting cycle and starting data point.

    Parameters:
    - bat_dict (dict): a dictionary containing battery data
    - cell_name (str): the name of the cell to plot
    - start_cycle (int): the starting cycle number to plot
    - start_point (int): the starting data point to plot for each cycle
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    n_lines = len(bat_dict[cell_name]['cycles'])
    for j, cycle_data in bat_dict[cell_name]['cycles'].items():
        if int(j) >= start_cycle:
            cycle_color = cycle_color_map(int(j), n_lines)
            ax.plot(cycle_data['Qd'][start_point:len(cycle_data['Qd'])-end_point], cycle_data['V'][start_point:len(cycle_data['Qd'])-end_point], color=cycle_color)

    ax.set_xlabel('Discharge Capacity (Ah)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title(f'{cell_name} Discharge Capacity Fade')

    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=1, vmax=n_lines))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Cycle')

    plt.show()

def plot_discharge_fade(bat_dict, cell_name, start_cycle, start_point, end_point):
    """
    Plots the cycles of a given cell from a starting cycle and starting data point.

    Parameters:
    - bat_dict (dict): a dictionary containing battery data
    - cell_name (str): the name of the cell to plot
    - start_cycle (int): the starting cycle number to plot
    - start_point (int): the starting data point to plot for each cycle
    - end_point (int): the ending data point to plot for each cycle, starting from the end of the cycle data

    Returns:
    - None: this function plots the data but does not return any values
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    cycle_count = 0
    for j, cycle_data in bat_dict[cell_name]['cycles'].items():
        if int(j) >= start_cycle:
            if int(j) % 100 == 0:
                cycle_label = f'{cycle_count+1}-{cycle_count+100}'
                cycle_count += 100
            else:
                cycle_label = None
            cycle_color = cycle_color_map(int(j), len(bat_dict[cell_name]['cycles']))
            ax.plot(cycle_data['Qd'][start_point:len(cycle_data['Qd'])-end_point], cycle_data['V'][start_point:len(cycle_data['Qd'])-end_point], label=cycle_label, color=cycle_color)

    ax.set_xlabel('Discharge Capacity (Ah)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title(f'{cell_name} Discharge Capacity Fade')
    ax.legend()
    plt.show()

"""From [2], cycle keys </br>
- cd = {'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 'Tdlin': Tdlin, 'V':V, 'dQdV': dQdV, 't':t}  </br>
</br>

Functions regarding cycle data will be used to validate original data of toyota.

"""

import itertools

def plot_cycledata_per_cell(bat_dict, cell_key, dk, start_cycle, no_of_cycles):
    cycles = bat_dict[cell_key]['cycles']
    fig, ax1 = plt.subplots(figsize=(8,6), dpi=100)

    cycle_count = len(cycles.keys())
    cycle_colors = cycle_color_map(cycle_count)

    for count, (cy_key, value) in itertools.islice(enumerate(cycles.items()), start_cycle, start_cycle + no_of_cycles):
        ax1.scatter(value['t'], value[dk], c=[cycle_colors[count]])

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=1, vmax=cycle_count))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1)
    cbar.set_label('Cycle')

    ax1.legend(loc='upper left', fontsize='small', ncol=1)

    # Set labels and titles
    ax1.set_xlabel('Time')
    ax1.set_ylabel(dk, color='r')
    ax1.set_title(f'{dk} {cell_key}')

    plt.show()

"""Displays the cycle data of a specified data key 'dk' for a given cell 'cell_key' in a scatter plot. The plot includes a color bar representing the cycle number, and displays data from a specified starting cycle 'start_cycle' up to a maximum number of cycles 'no_of_cycles'.

Parameters:
- bat_dict (dict): a dictionary containing battery data
- cell_key (str): the name of the cell to plot
- dk (str): the data key to plot, e.g. 'Qd', 'V', 'I'
- start_cycle (int): the starting cycle number to plot
- no_of_cycles (int): the maximum number of cycles to plot
"""

def plot_cycledata_per_cell(bat_dict, cell_key, dk, start_cycle, no_of_cycles):
  measured_dk_list = {'I', 'Qc', 'Qd', 'T', 'V'}
  cycles = bat_dict[cell_key]['cycles']
  fig, ax1 = plt.subplots(figsize=(8,6), dpi=100)

  cycle_count = len(cycles.keys())
  cycle_colors = cycle_color_map(cycle_count)

  for count, (cy_key, value) in itertools.islice(enumerate(cycles.items()), start_cycle, start_cycle + no_of_cycles):
    if dk in measured_dk_list:
      ax1.scatter(value['t'], value[dk], c=[cycle_colors[count]])
    else:
      ax1.scatter(range(len(value[dk])), value[dk], c=[cycle_colors[count]])

  sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=1, vmax=cycle_count))
  sm.set_array([])
  cbar = fig.colorbar(sm, ax=ax1)
  cbar.set_label('Cycle')

  ax1.legend(loc='upper left', fontsize='small', ncol=1)

  if dk in measured_dk_list:
    ax1.set_xlabel('Time')
  else:
    ax1.set_xlabel('Data Point')
  ax1.set_ylabel(dk, color='r')
  ax1.set_title(f'{dk} {cell_key}')

  ax1.autoscale(axis='y')

  plt.show()

def v2_cycle_color_map(start_cycle, no_of_cycles):
    color_map = plt.cm.get_cmap('coolwarm')
    color_range = colors.Normalize(vmin=start_cycle, vmax=start_cycle+no_of_cycles)
    return color_map(color_range(np.arange(start_cycle+no_of_cycles)))

#{'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 'Tdlin': Tdlin, 'V':V, 'dQdV': dQdV, 't':t}
def v2_plot_cycledata_per_cell(bat_dict, cell_key, dk, start_cycle, no_of_cycles):
  measured_dk_list = {'I', 'Qc', 'Qd', 'T', 'V'}
  cycles = bat_dict[cell_key]['cycles']
  fig, ax1 = plt.subplots(figsize=(8,6), dpi=100)

  cycle_count = len(cycles.keys())
  if cycle_count < start_cycle + no_of_cycles:
    no_of_cycles = cycle_count - start_cycle
  cycle_colors = v2_cycle_color_map(start_cycle, no_of_cycles)

  for count, (cy_key, value) in itertools.islice(enumerate(cycles.items()), start_cycle, start_cycle + no_of_cycles):
    if dk in measured_dk_list:
      ax1.scatter(value['t'], value[dk], c=[cycle_colors[count]])
    else:
      ax1.scatter(range(len(value[dk])), value[dk], c=[cycle_colors[count]])


  # Add colorbar
  sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=start_cycle, vmax=no_of_cycles))
  sm.set_array([])
  cbar = fig.colorbar(sm, ax=ax1)
  cbar.set_label('Cycle')

  ax1.legend(loc='upper left', fontsize='small', ncol=1)

  # Set labels and titles
  if dk in measured_dk_list:
    ax1.set_xlabel('Time')
  else:
    ax1.set_xlabel('Data Point')
  ax1.set_ylabel(dk, color='r')
  ax1.set_title(f'{dk} {cell_key}')

  # Autoscale y-axis
  ax1.autoscale(axis='y')

  plt.show()

"""This function generates scatter plots of a selected data key (dk) for multiple cells, each with different colors indicating the cycles. The data key is specified by dk_list, a list of data keys to be plotted. The start_cycle argument allows you to specify the starting cycle number, while the no_of_cycles argument specifies the total number of cycles to be plotted.

Parameters:
- bat_dict: a dictionary containing battery cycle data
- cell_keys: a list of cell keys to plot
- dk_list: a list of data keys to plot for each cell
- start_cycle: the cycle number to start plotting from
- no_of_cycles: the total number of cycles to plot

"""

def plot_cycledata_multiple_cells(bat_dict, cell_keys, dk_list, start_cycle, no_of_cycles):
    for cell_key in cell_keys:
        for dk in dk_list:
            plot_cycledata_per_cell(bat_dict, cell_key, dk, start_cycle, no_of_cycles)

def v2_plot_cycledata_multiple_cells(bat_dict, cell_keys, dk_list, start_cycle, no_of_cycles):
    for cell_key in cell_keys:
        for dk in dk_list:
            v2_plot_cycledata_per_cell(bat_dict, cell_key, dk, start_cycle, no_of_cycles)

"""This function generates scatter plots of a selected data key (dk) for a cell and a cycle."""

def v1_plot_cycledata_per_cellandcycle(bat_dict, cell_key, dk_list, cycle_num):
  """
  Plots cycle data for a given cell and cycle number from a battery dictionary.

  Parameters:
  bat_dict (dict): The battery dictionary containing cell and cycle data.
  cell_key (str): The key identifying the cell for which data should be plotted.
  dk_list (list): List of dataset keys specifying which datasets in the cycle data to plot.
  cycle_num (int): The cycle number for which data should be plotted.

  Returns:
  None. This function only creates a plot and does not return any value.
  """

  cycles = bat_dict[cell_key]['cycles']

  if len(dk_list) == 1:
    axs = [plt.subplots(figsize=(8, 6), dpi=100)]
  else:
    fig, axs = plt.subplots(len(dk_list), 1, figsize=(8, 6*len(dk_list)), dpi=100, sharex=True)

  for i, dk in enumerate(dk_list):
    cycle_data = cycles[str(cycle_num)]
    axs[i].scatter(range(len(cycle_data[dk])), cycle_data[dk])
    axs[i].set_ylabel(dk)
    axs[i].tick_params(axis='y')
    axs[i].set_xlabel('Time')
    axs[i].grid(True)
    print(len(cycle_data[dk]), cycle_data[dk][999], cycle_data[dk][-1])

  plt.suptitle(f'Cycle Data for Cell: {cell_key}, Cycle Number: {cycle_num}')

  plt.show()

"""This function, v2_plot_cycledata_per_cell, displays the cycle data with color-coded cycle numbers for a given cell and cycle number. The function takes in a battery dictionary bat_dict, a string cell_key which is the key for the cell in the dictionary, an optional list of data keys dk_list to plot on the right y-axis (if not provided, it defaults to an empty list), and an optional integer cycle_num representing the cycle number to plot (if not provided, it defaults to 2)."""

def v2_plot_cycledata_per_cellandcycle(bat_dict, cell_key, dk_list=None, cycle_num=2):

    cycles = bat_dict[cell_key]['cycles']

    color_list = ['g', 'c', 'y']
    line_colors = {}
    if dk_list is None:
      dk_list = []

    for i, dk in enumerate(dk_list):
      line_colors[dk] = color_list[i % len(color_list)]

    fig, ax1 = plt.subplots(figsize=(10,6), dpi=100)

    ax1.plot(range(len(cycles[str(cycle_num)]['Qc'])), cycles[str(cycle_num)]['Qc'], color='r', label='Qc')
    ax1.plot(range(len(cycles[str(cycle_num)]['Qd'])), cycles[str(cycle_num)]['Qd'], color='b', label='Qd')
    ax1.set_ylabel('Capacity (Ah)')

    for i, dk in enumerate(dk_list):
      ax = ax1.twinx()
      ax.spines[f"right"].set_position(("axes", 1 + i * 0.1/(len(dk_list)-2)))
      ax.plot(range(len(cycles[str(cycle_num)][dk])), cycles[str(cycle_num)][dk], color=line_colors[dk], label=dk)
      ax.set_ylabel(dk, color=line_colors[dk])
      ax.tick_params(axis='y', labelcolor=line_colors[dk])

    ax1.legend(loc='upper left', fontsize='small', ncol=1)

    ax1.set_xlabel('Data Point')
    ax1.set_title(f'{cell_key} Cycle {cycle_num}')

    plt.show()

"""[1] Severson, K.A.; Attia, P.M.; Jin, N.; Perkins, N.; Jiang, B.; Yang, Z.; Chen, M.H.; Aykol, M.; Herring, P.K.; Fraggedakis, D.; et al. Data-Driven Prediction of Battery Cycle Life before Capacity Degradation. Nat Energy 2019, 4, 383–391, doi:10.1038/s41560-019-0356-8. [Data-driven prediction of battery cycle life before capacity degradation](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation.git)"""