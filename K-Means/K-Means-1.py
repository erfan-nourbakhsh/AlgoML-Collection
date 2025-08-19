# Import necessary libraries
import random  # Random number generation
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation (not used directly here)
import matplotlib.pyplot as plt  # Plotting
from sklearn.cluster import KMeans  # K-Means clustering algorithm
from sklearn.datasets import make_blobs  # Synthetic dataset generation

# Set random seed for reproducibility
np.random.seed(0)

# Define the centers of synthetic clusters
centers = [[4, 4], [-2, -1], [2, -3], [1, 1]]

# Generate 5000 samples around the defined centers with standard deviation 0.9
X, Y = make_blobs(n_samples=5000, centers=centers, cluster_std=0.9)

# Initialize K-Means with 4 clusters, k-means++ initialization, and 12 runs for stability
k_means = KMeans(init="k-means++", n_clusters=4, n_init=12)

# Fit K-Means model to the data
k_means.fit(X)

# Extract labels assigned to each data point
k_means_labels = k_means.labels_

# Extract the coordinates of cluster centroids
k_means_cluster_centers = k_means.cluster_centers_

# Print the calculated centroids and the original centers for comparison
print(k_means_cluster_centers, centers)

# Set axis labels for the plot
plt.xlabel('X')
plt.ylabel('Y')

# Generate distinct colors for each cluster
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Plot each cluster and its centroid
for k, col in zip(range(len(centers)), colors):
    # Mask for points belonging to cluster k
    my_members = (k_means_labels == k)
    
    # Get the centroid coordinates for cluster k
    cluster_center = k_means_cluster_centers[k]
    
    # Plot all data points of cluster k with the assigned color
    plt.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plot the centroid with same color and black edge
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='black')

# Display the plot
plt.show()
