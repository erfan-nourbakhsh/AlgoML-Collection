# Import necessary libraries
import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
from sklearn.preprocessing import StandardScaler  # Standardization of features
from sklearn.cluster import KMeans  # K-Means clustering
import matplotlib.pyplot as plt  # 2D plotting
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting (not used in final plot)

# Load customer dataset
cust_df = pd.read_csv("K-Means/Cust_Segmentation.csv")

# Drop the 'Address' column as it is not useful for clustering
df = cust_df.drop('Address', axis=1)

# Extract feature values (excluding customer ID column)
X = df.values[:, 1:]

# Replace NaN values with 0
X = np.nan_to_num(X)

# Standardize features to have mean=0 and variance=1
standardize_dataset = StandardScaler().fit_transform(X)

# Initialize K-Means clustering with 4 clusters
k_means_2 = KMeans(init="k-means++", n_clusters=4, n_init=12)

# Fit K-Means model on the raw data (X)
k_means_2.fit(X)

# Get cluster labels assigned to each data point
labels_2 = k_means_2.labels_

# Add cluster labels to the DataFrame
df["Clus_km"] = labels_2

# Compute the mean of features for each cluster
divideByGroup = df.groupby('Clus_km').mean()

# 2D Visualization
# Size of scatter points proportional to second feature (e.g., Spending Score or similar)
area = np.pi * (X[:, 1])**2  

# Scatter plot of Age vs Income, colored by cluster labels
plt.scatter(X[:, 0], X[:, 3], c=labels_2.astype(float), alpha=0.5)

# Label axes
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

# Display the plot
plt.show()
