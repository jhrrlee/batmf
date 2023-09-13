# -*- coding: utf-8 -*-
"""
File Name: partition_lib.py

Description: analysis of data distribution for partitions.

Author: junghwan lee
Email: jhrrlee@gmail.com
Date Created: 2023.09.07
"""

import numpy as np
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN, MeanShift, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.metrics import silhouette_score
from pandas.plotting import parallel_coordinates
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, data):
        self.data = data
        self.data_norm = self.normalize_and_reshape()

    def normalize_and_reshape(self):
        data_2d = self.data.reshape(self.data.shape[0], -1)
        scaler = StandardScaler()
        return scaler.fit_transform(data_2d)

    def get_label_embeddings(self, sizes):
        embeddings = []
        current_label = 0
        for size in sizes:
            embeddings.extend([current_label] * size)
            current_label += 1
        return embeddings

class DimensionReducer:
    def __init__(self, data_norm):
        self.data_norm = data_norm

    def reduce(self, method='PCA', n_components=2):
        if method == 'PCA':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 't-SNE':
            reducer = TSNE(n_components=n_components, random_state=42)
        elif method == 'UMAP':
            reducer = umap.UMAP(n_components=n_components, random_state=42)
        else:
            raise ValueError("Invalid method. Use 'PCA', 't-SNE', or 'UMAP'.")
        return reducer.fit_transform(self.data_norm)

    def visualize(self, reduced_data, labels=None):
        if reduced_data.shape[1] == 2:
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
            plt.xlabel('First Dimension')
            plt.ylabel('Second Dimension')
        elif reduced_data.shape[1] == 3:
            ax = plt.axes(projection='3d')
            ax.scatter3D(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=labels, cmap='viridis')
        else:
            print("Visualization is available for 2D or 3D data only.")
            return
        plt.colorbar()
        plt.show()

class Clusterer:
    def __init__(self, data_norm):
        self.data_norm = data_norm
        self.cluster_methods = {
            'KMeans': KMeans,
            'Agglomerative': AgglomerativeClustering,
            'Spectral': SpectralClustering,
            'DBSCAN': DBSCAN,
            'MeanShift': MeanShift,
            'OPTICS': OPTICS,
            'GMM': GaussianMixture
        }

    def cluster(self, method='KMeans', **kwargs):
        if method in self.cluster_methods:
            if method == 'GMM':
                model = self.cluster_methods[method](**kwargs).fit(self.data_norm)
                return model.predict(self.data_norm)
            else:
                model = self.cluster_methods[method](**kwargs).fit(self.data_norm)
                return model.labels_
        else:
            raise ValueError(f"Invalid clustering method. Available methods are: {', '.join(self.cluster_methods.keys())}")

    def determine_optimal_clusters(self, method='KMeans', max_clusters=10):
        scores = []
        range_values = range(2, max_clusters)

        for i in range_values:
            labels = self.cluster(method=method, n_clusters=i)
            score = silhouette_score(self.data_norm, labels)
            scores.append(score)

        plt.figure(figsize=(10, 6))
        plt.plot(range_values, scores, marker='o', linestyle='-', color='b')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title(f'Silhouette Scores for {method}')
        plt.grid(True)
        plt.show()

        return scores.index(max(scores)) + 2

class DataVisualizer:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def _get_labels(self, label_embeddings):
        if label_embeddings is None:
            if self.labels is None:
                raise ValueError("No labels provided during initialization or as an argument.")
            return self.labels
        return label_embeddings

    def plot_cluster_statistics(self, label_embeddings=None):
        labels = self._get_labels(label_embeddings)
        unique_labels = np.unique(labels)
        partitions = [self.data[labels == label] for label in unique_labels]

        means = []
        for partition in partitions:
            means.append(np.mean(partition[:, 0, :], axis=1))

        overall_means = [np.mean(mean) for mean in means]
        overall_stds = [np.std(mean) for mean in means]

        stats = list(zip(overall_means, overall_stds))

        plt.figure(figsize=(12, 6))
        plt.boxplot(means, labels=[f'Cluster {label}' for label in unique_labels])
        plt.title('Box Plots of Mean Values for First Feature Across Partitions')
        plt.ylabel('Mean Values')
        plt.show()

        return stats

    def plot2d(self, label_embeddings=None):
        labels = self._get_labels(label_embeddings)
        plt.figure(figsize=(8, 6))
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = np.where(labels == label)
            plt.scatter(self.data[idx, 0], self.data[idx, 1], alpha=0.6, label=f'Cluster {label}')
        plt.xlabel('First Dimension')
        plt.ylabel('Second Dimension')
        plt.title('2D Visualization of Data')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot3d(self, label_embeddings=None):
        labels = self._get_labels(label_embeddings)
        fig = plt.figure(figsize=(14, 14))
        ax = fig.add_subplot(111, projection='3d')
        unique_labels = np.unique(labels)

        colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            idx = np.where(labels == label)
            ax.scatter(self.data[idx, 0], self.data[idx, 1], self.data[idx, 2], color=colors[i], alpha=0.6, label=f'Cluster {label}')

        ax.set_xlabel('First Dimension')
        ax.set_ylabel('Second Dimension')
        ax.set_zlabel('Third Dimension')
        ax.set_title('3D Visualization of Data')
        ax.legend()
        plt.show()

    def plot_parallel_coordinates(self, label_embeddings=None):
        labels = self._get_labels(label_embeddings)
        df = pd.DataFrame(self.data)
        df['Cluster'] = labels
        plt.figure(figsize=(12, 6))
        parallel_coordinates(df, 'Cluster', colormap='viridis', alpha=0.5)
        plt.title('Parallel Coordinates Plot')
        plt.grid(True)
        plt.show()

    def plot_scatter_matrix(self, label_embeddings=None):
        labels = self._get_labels(label_embeddings)
        df = pd.DataFrame(self.data)
        df['Labels'] = labels
        sns.pairplot(df, hue='Labels', palette='viridis', plot_kws={'alpha': 0.6})
        plt.suptitle('Scatter Matrix of Multi-dimensional Data', y=1.02)
        plt.show()

class BatteryDataAnalyzer:
    def __init__(self, data):
        self.data = data
        self.preprocessor = DataPreprocessor(self.data)
        self.reducer = DimensionReducer(self.preprocessor.data_norm)
        self.clusterer = Clusterer(self.preprocessor.data_norm)

    def preprocess(self, sizes):
        return self.preprocessor.get_label_embeddings(sizes)

    def reduce_dimension(self, method='PCA', n_components=2):
        reduced_data = self.reducer.reduce(method=method, n_components=n_components)
        return reduced_data

    def determine_optimal_clusters(self, method='KMeans', max_clusters=10):
        return self.clusterer.determine_optimal_clusters(method=method, max_clusters=max_clusters)

    def cluster(self, method='KMeans', n_clusters=4):
        return self.clusterer.cluster(method=method, n_clusters=n_clusters)

    def visualize(self, reduced_data, labels=None, visualization_type="2d", title="Visualization"):
        visualizer = DataVisualizer(reduced_data, labels)

        visualization_methods = {
            "2d": visualizer.plot2d,
            "3d": visualizer.plot3d,
            "parallel_coordinates": visualizer.plot_parallel_coordinates,
            "scatter_matrix": visualizer.plot_scatter_matrix,
            "cluster_statistics": visualizer.plot_cluster_statistics
        }

        visualization_method = visualization_methods.get(visualization_type)
        if visualization_method:
            visualization_method(labels)
        else:
            print(f"Visualization type '{visualization_type}' not recognized.")

    def plot_confusion_matrix(self, original_labels, predicted_labels, title='Confusion Matrix'):
        cm = confusion_matrix(original_labels, predicted_labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Clusters')
        plt.ylabel('Original Clusters')
        plt.title(title)
        plt.show()
    def plot_cluster_distribution(self, labels):
        unique_labels = np.unique(labels)
        counts = np.bincount(labels)

        plt.figure(figsize=(8, 6))
        plt.bar(unique_labels, counts, color='blue', alpha=0.6)
        plt.xlabel('Cluster')
        plt.ylabel('Number of Samples')
        plt.title('Distribution of Samples in Each Cluster')
        plt.xticks(unique_labels)
        plt.show()

        cluster_indices = {}
        for label in unique_labels:
            cluster_indices[label] = [index for index, cluster_label in enumerate(labels) if cluster_label == label]

        return cluster_indices

    def stratified_data_split_fixed_size(self, data, labels, train_size, val_size, random_state=None):

        test_size = len(data) - train_size - val_size

        train_val_indices, test_indices, train_val_labels, _ = train_test_split(
            range(len(data)), labels, test_size=test_size, stratify=labels, random_state=random_state)

        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_size, stratify=train_val_labels, random_state=random_state)

        train_indices = sorted(train_indices)
        val_indices = sorted(val_indices)
        test_indices = sorted(test_indices)

        train_data = data[train_indices]
        val_data = data[val_indices]
        test_data = data[test_indices]

        return train_data, val_data, test_data, train_indices, val_indices, test_indices