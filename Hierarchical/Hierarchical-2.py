# Import necessary libraries
import numpy as np  # Numerical computations
import pandas as pd  # Data manipulation
from scipy import ndimage  # Image processing utilities (not used directly)
from scipy.cluster import hierarchy  # Hierarchical clustering utilities
from scipy.spatial import distance_matrix  # Pairwise distance computation
from matplotlib import pyplot as plt  # Plotting
from sklearn import manifold, datasets, preprocessing  # Preprocessing and datasets
from sklearn.cluster import AgglomerativeClustering  # Agglomerative clustering
from sklearn.datasets import make_blobs  # Synthetic data generation
import scipy  # For Euclidean distance
from scipy.cluster.hierarchy import fcluster  # Flat cluster extraction
from sklearn.metrics.pairwise import euclidean_distances  # Pairwise distance matrix

# Define function to format leaf labels in dendrogram
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])))

# Load the dataset and select first 10 rows
pdf = pd.read_csv('Hierarchical/cars_clus.csv')[0:10]

# Remove rows with missing values
pdf = pdf.dropna()

# Reset index after dropping rows
pdf = pdf.reset_index(drop=True)

# Replace string "$null$" with numeric 0
pdf = pdf.replace("$null$", 0)

# Select features for clustering
featureSet = pdf[['engine_s', 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

# Convert features to NumPy array
x = featureSet.values

# Scale features to range [0,1] using MinMaxScaler
min_max_scaler = preprocessing.MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)

# Get number of samples
leng = feature_mtx.shape[0]

# Compute pairwise Euclidean distance manually using nested loops (Scipy D)
D = np.zeros([leng, leng])
for i in range(leng):
    for j in range(leng):
        D[i, j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])

# Perform hierarchical clustering using complete linkage (Scipy)
Z = hierarchy.linkage(D, 'complete')

# Create a figure with two subplots to compare clustering methods
fig, (ax1, ax2) = plt.subplots(2)

# Plot dendrogram using the manually computed distance matrix
hierarchy.dendrogram(Z, leaf_label_func=llf, leaf_rotation=0, leaf_font_size=12,
                     orientation='right', ax=ax1)

# Compute pairwise Euclidean distances using scikit-learn
dist_matrix = euclidean_distances(feature_mtx, feature_mtx)

# Perform hierarchical clustering using distances from scikit-learn
Z_using_dist_matrix = hierarchy.linkage(dist_matrix, 'complete')

# Plot dendrogram using the scikit-learn distance matrix
hierarchy.dendrogram(Z_using_dist_matrix, leaf_label_func=llf, leaf_rotation=0, leaf_font_size=12,
                     orientation='right', ax=ax2)

# Set subplot titles
ax1.set_title("Scipy")
ax2.set_title("scikit-learn")

# Display dendrogram plots
plt.show()
