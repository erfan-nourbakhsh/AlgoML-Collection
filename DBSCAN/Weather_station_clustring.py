# Import necessary libraries
import csv  # For CSV file handling (not used directly here but imported)
import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Plotting
from sklearn.cluster import DBSCAN  # DBSCAN clustering
import sklearn.utils  # Utility functions for scikit-learn
from sklearn.preprocessing import StandardScaler  # Standardize features
from mpl_toolkits.basemap import Basemap  # Plotting geographic maps

# Read CSV file containing weather station data
pdf = pd.read_csv('DBSCAN/weather-stations20140101-20141231.csv')

# Keep only rows where the "Tm" column (temperature) is not null
pdf = pdf[pd.notnull(pdf["Tm"])]

# Reset the index after filtering
pdf = pdf.reset_index(drop=True)

# Define map boundaries: longitude (llon, ulon) and latitude (llat, ulat)
llon = -140  # Lower left longitude
ulon = -50   # Upper right longitude
llat = 40    # Lower left latitude
ulat = 65    # Upper right latitude

# Create a Basemap instance with Mercator projection and specified boundaries
my_map = Basemap(projection='merc',
                 resolution='l', area_thresh=1000.0,
                 llcrnrlon=llon, llcrnrlat=llat,
                 urcrnrlon=ulon, urcrnrlat=ulat)

# Convert latitude and longitude to x and y coordinates for plotting on the map
xs, ys = my_map(np.asarray(pdf.Long), np.asarray(pdf.Lat))

# Add the projected x and y coordinates to the DataFrame
pdf['xm'] = xs.tolist()
pdf['ym'] = ys.tolist()

# Set random state for reproducibility (DBSCAN does not use random directly but for consistency)
sklearn.utils.check_random_state(1000)

# Prepare dataset for clustering using only the map coordinates
Clus_dataSet = pdf[['xm', 'ym']]

# Replace NaN with 0 (if any)
Clus_dataSet = np.nan_to_num(Clus_dataSet)

# Standardize the features to have mean=0 and variance=1
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

# Compute DBSCAN clustering with epsilon=0.15 and minimum 10 samples per cluster
db = DBSCAN(eps=0.15, min_samples=10).fit(Clus_dataSet)

# Create a boolean mask to identify core samples
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# Extract cluster labels (-1 indicates noise)
labels = db.labels_

# Add the cluster labels to the DataFrame
pdf["Clus_Db"] = labels

# Count the number of real clusters (excluding noise)
realClusterNum = len(set(labels)) - (1 if -1 in labels else 0)

# Count total number of unique labels including noise
clusterNum = len(set(labels))

# Generate colors for each cluster using the 'jet' colormap
colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))

# Plot each cluster on the map
for clust_number in set(labels):
    # Assign gray color for noise, otherwise use a color from the colormap
    c = ([0.4, 0.4, 0.4] if clust_number == -1 else colors[int(clust_number)])
    
    # Select data points belonging to the current cluster
    clust_set = pdf[pdf.Clus_Db == clust_number]                    
    
    # Scatter plot the cluster points on the map
    my_map.scatter(clust_set.xm, clust_set.ym, color=c, marker='o', s=20, alpha=0.85)

# Show the plot
plt.show()
