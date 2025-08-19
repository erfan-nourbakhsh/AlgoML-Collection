# Import necessary libraries
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation (not used here but imported)
from scipy import ndimage  # Image processing utilities (not used directly)
from scipy.cluster import hierarchy  # Hierarchical clustering utilities
from scipy.spatial import distance_matrix  # Compute pairwise distance matrix
from matplotlib import pyplot as plt  # Plotting
from sklearn import manifold, datasets, preprocessing  # Manifold learning, datasets, preprocessing
from sklearn.cluster import AgglomerativeClustering  # Agglomerative clustering
from sklearn.datasets import make_blobs  # Generate synthetic cluster data

# Set random seed for reproducibility
np.random.seed(0)

# Define centers for synthetic clusters
centers = [[4, 4], [-2, -1], [1, 1], [10, 4]]

# Generate 500 samples around the defined centers with standard deviation 0.9
X, Y = make_blobs(n_samples=500, centers=centers, cluster_std=0.9)

# Initialize Agglomerative Clustering with 4 clusters and average linkage
agglomerative = AgglomerativeClustering(n_clusters=4, linkage='average')

# Fit the Agglomerative model to the data (Y is ignored internally, only X is used)
agglomerative.fit(X, Y)

# Standardize X (zero mean, unit variance) for comparison
Z = preprocessing.StandardScaler().fit(X).transform(X)

# Compute the minimum and maximum values of X along each feature dimension
x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)

# Get cluster labels assigned by the Agglomerative model
labels = agglomerative.labels_

# Normalize X to range [0, 1] for plotting
X = (X - x_min) / (x_max - x_min)

# Generate distinct colors for each cluster
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(labels))))

# Create a figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(2)

# Loop over each cluster to plot points
for k, col in zip(range(len(centers)), colors):
    # Mask for points belonging to cluster k
    my_members = (labels == k)
    
    # Plot normalized data in first subplot
    ax1.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plot standardized data in second subplot
    ax2.plot(Z[my_members, 0], Z[my_members, 1], 'w', markerfacecolor=col, marker='.')

# Set titles for subplots
ax1.set_title("Average Distance for X")
ax2.set_title("Using StandardScaler")

# Compute the full pairwise distance matrix of normalized X
dist_matrix = distance_matrix(X, X)

# Compute hierarchical clustering linkage using complete linkage method
completeLinkage = hierarchy.linkage(dist_matrix, 'complete')

# Generate dendrogram from the linkage
dendrogram = hierarchy.dendrogram(completeLinkage)

# Print dendrogram dictionary (coordinates, color info, etc.)
print(dendrogram)
