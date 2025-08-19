# Import necessary libraries
import numpy as np  # Numerical computations
from sklearn.cluster import DBSCAN  # DBSCAN clustering algorithm
from sklearn.datasets import make_blobs  # Generate synthetic cluster data
from sklearn.preprocessing import StandardScaler  # Standardize features
import matplotlib.pyplot as plt  # Plotting library
import warnings  # To handle warnings
from sklearn.cluster import KMeans  # K-Means clustering algorithm

# Set random seed for reproducibility
np.random.seed(150)

# Function to generate synthetic data points
def createDataPoints(centroidLocation, numSamples, clusterDeviation):
    # Generate synthetic blobs with specified centroids, number of samples, and cluster standard deviation
    X, Y = make_blobs(n_samples=numSamples, centers=centroidLocation, 
                      cluster_std=clusterDeviation)
    # Standardize features to have mean=0 and variance=1
    X = StandardScaler().fit_transform(X)
    # Return the feature matrix X and labels Y
    return X, Y

# Generate data points with 3 centroids and 1500 samples
X, Y = createDataPoints([[4, 3], [2, -1], [-1, 4]], 1500, 0.5)

# DBSCAN parameters
epsilon = 0.3  # Maximum distance between two samples for them to be considered as neighbors
minimumSamples = 7  # Minimum number of samples required to form a dense region

# Fit DBSCAN model to the data
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)

# Extract cluster labels for each point
labels = db.labels_

# Create a boolean mask to identify core samples
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# Count the number of clusters, ignoring noise (-1)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Get unique cluster labels
unique_labels = set(labels)

# Create a figure with 2 subplots (one for DBSCAN, one for K-Means)
fig, (ax1, ax2) = plt.subplots(2)

# Generate distinct colors for each unique label
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

# Plot DBSCAN clusters
for k, col in zip(unique_labels, colors):
    # Mask for points belonging to the current cluster
    class_member_mask = (labels == k)
    if k == -1:
        # Use black color for noise/outliers
        col = 'k'
    # Plot core points of the cluster
    xy = X[class_member_mask & core_samples_mask]
    ax1.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], marker='.', alpha=0.8)
    # Plot non-core points (border points or noise)
    xy = X[class_member_mask & ~core_samples_mask]
    ax1.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], marker='.', alpha=0.8)

# Fit K-Means clustering with 3 clusters
k_means = KMeans(init="k-means++", n_clusters=3, n_init=12)
k_means.fit(X)
k_means_labels = k_means.labels_

# Plot K-Means clusters
for k, col in zip(k_means_labels, colors):
    # Mask for points belonging to the current K-Means cluster
    my_members = (k_means_labels == k)
    ax2.scatter(X[my_members, 0], X[my_members, 1], c=col, marker='.', alpha=0.8)

# Set titles for the plots
ax1.set_title("DBSCAN")
ax2.set_title("K-Means")

# Display the plots
plt.show()
