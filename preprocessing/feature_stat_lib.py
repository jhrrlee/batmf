# -*- coding: utf-8 -*-
"""
File Name: feature_stat_lib.py

Description: statistical analysis of features.

Author: junghwan lee
Email: jhrrlee@gmail.com
Date Created: 2023.09.07
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import skew
from scipy.stats import pearsonr

def plot_features_over_cycles(cell_data, features, cell_index):

    cycle_data = cell_data[cell_index]
    for i, feature in enumerate(features):
        plt.figure(figsize=(12, 6))
        plt.plot(cycle_data[i])
        plt.xlabel('Cycle number')
        plt.ylabel(feature)
        plt.title(f'{feature} over cycles for cell {cell_index}')
        plt.grid(True)
        plt.show()

def plot_3d_variance_by_index(new_cycle_sum, features):

    variance = np.var(new_cycle_sum, axis=2)

    epsilon = 1e-10

    log_variance_epsilon = np.log(variance + epsilon)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    batteries = np.arange(1, log_variance_epsilon.shape[0] + 1)
    features_idx = np.arange(1, log_variance_epsilon.shape[1] + 1)

    batteries_grid, features_grid = np.meshgrid(batteries, features_idx)

    sc = ax.scatter(batteries_grid, features_grid, log_variance_epsilon.T, c=log_variance_epsilon.T.flatten(), cmap='viridis')

    ax.set_xlabel('Battery')
    ax.set_ylabel('Feature')
    ax.set_zlabel('Log Variance')
    ax.set_yticks(features_idx)
    ax.set_yticklabels(features_idx)

    fig.tight_layout()

    fig.colorbar(sc, label='Log Variance', pad=0.1)

    plt.show()

def plot_3d_skewness(new_cycle_sum, features):


    skewness = skew(new_cycle_sum, axis=2)
    log_skewness_epsilon = skewness

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    batteries = np.arange(1, log_skewness_epsilon.shape[0] + 1)
    features_idx = np.arange(1, log_skewness_epsilon.shape[1] + 1)
    batteries_grid, features_grid = np.meshgrid(batteries, features_idx)

    sc = ax.scatter(batteries_grid, features_grid, log_skewness_epsilon.T, c=log_skewness_epsilon.T.flatten(), cmap='viridis')

    ax.set_xlabel('Battery')
    ax.set_ylabel('Feature')
    ax.set_zlabel('Skewness')
    ax.set_yticks(features_idx)
    ax.set_yticklabels(features_idx)

    fig.tight_layout()

    fig.colorbar(sc, label='Skewness', pad=0.1)

    plt.show()

def plot_variance_and_skewness(new_cycle_sum, features, epsilon=1e-10):

    variance = np.var(new_cycle_sum, axis=2)
    skewness = skew(new_cycle_sum, axis=2)

    log_variance_epsilon = np.log(variance + epsilon)

    rows = (len(features) + 1) // 2
    fig, axs = plt.subplots(rows, 2, figsize=(15, 4 * rows))

    if axs.ndim == 1:
        axs = axs.reshape(-1, 1)

    axs = axs.flatten()

    for i, feature in enumerate(features):
        ax2 = axs[i].twinx()

        l1, = axs[i].plot(log_variance_epsilon[:, i], label='Log Variance', color='b')
        axs[i].set_title(feature)
        axs[i].set_xlabel('Battery')
        axs[i].set_ylabel('Log Variance', color='b')

        l2, = ax2.plot(skewness[:, i], label='Skewness', color='r')
        ax2.set_ylabel('Skewness', color='r')

        if i == 0:
            lines = [l1, l2]
            labels = [line.get_label() for line in lines]
            axs[i].legend(lines, labels, loc='best')

    plt.tight_layout()
    plt.show()

def feature_correlations_dim_reduced(new_cycle_sum, eol, reduced_dim_cycle=10, method="var"):
    num_cells, num_features, num_cycles = new_cycle_sum.shape
    num_groups = num_cycles // reduced_dim_cycle

    reduced_data = np.zeros((num_cells, num_features, num_groups))

    # reduce the only cycle dimension
    for cell in range(num_cells):
        for feature in range(num_features):
            data = new_cycle_sum[cell, feature, :]
            if reduced_dim_cycle == 1:
                reduced_feature_data = data
            else:
                data_groups = np.array_split(data, num_groups)
                if method == "mean":
                    reduced_feature_data = [np.mean(group) for group in data_groups]
                elif method == "var":
                    reduced_feature_data = np.var(data_groups, axis=1)

            reduced_data[cell, feature, :len(reduced_feature_data)] = reduced_feature_data

    correlations = []
    for i in range(reduced_data.shape[1]):
        correlations_feature = []
        for j in range(reduced_data.shape[2]):
            feature_values = reduced_data[:, i, j]
            correlation = pearsonr(feature_values, eol.flatten())[0]
            correlations_feature.append(correlation)
        correlations.append(correlations_feature)

    return np.array(correlations)

def plot_3d_correlation(correlations, reduced_dimension=1):
    x = np.arange(reduced_dimension, (correlations.shape[1] + 1) * reduced_dimension, reduced_dimension)
    y = np.arange(correlations.shape[0])
    x, y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter(x, y, correlations, c=correlations.flatten(), cmap='viridis')

    ax.set_xlabel('Cycles')
    ax.set_ylabel('Features')
    ax.set_zlabel('Correlation Coefficient')

    ax.set_yticks(np.arange(correlations.shape[0]))

    fig.tight_layout()
    fig.colorbar(scat, ax=ax)

    plt.show()

def plot_2d_feature_correlations(correlations, feature_names):
    fig, axs = plt.subplots(7, 2, figsize=(15, 25))

    for i in range(7):
        for j in range(2):
            index = i * 2 + j
            if index < correlations.shape[0]:
                axs[i, j].scatter(range(correlations.shape[1]), correlations[index])
                axs[i, j].set_xlabel('Cycles')
                axs[i, j].set_ylabel('Correlation with EOL')
                axs[i, j].set_title(f'{feature_names[index]}')

    plt.tight_layout()
    plt.show()