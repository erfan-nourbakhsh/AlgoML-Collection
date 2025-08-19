# Import necessary libraries
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Plotting
import pandas as pd  # Data manipulation
from sklearn import preprocessing, metrics  # Scaling and evaluation metrics
from sklearn.model_selection import train_test_split  # Train-test split
from sklearn.neighbors import KNeighborsClassifier  # KNN classifier

# Load dataset
df = pd.read_csv('K-NearestNeigh/teleCust1000t.csv')

# Extract features for KNN
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 
        'ed', 'employ','retire', 'gender', 'reside']].values

# Alternative feature array excluding target column
X_same = df.drop('custcat', axis=1).values

# Extract target variable
y = df['custcat'].values

# Standardize features to zero mean and unit variance
NewX = preprocessing.StandardScaler().fit(X).transform(X)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Initialize KNN with 4 neighbors and fit to training data
neigh = KNeighborsClassifier(n_neighbors=4).fit(X_train, y_train)

# Predict on test data
y_predicted = neigh.predict(X_test)

# Find accuracy of initial KNN model
accuracy = metrics.accuracy_score(y_test, y_predicted)

# Find best K value
Ks = 8  # Maximum K to test
mean_acc = np.zeros((Ks-1))  # Store mean accuracy for each K
std_acc = np.zeros((Ks-1))   # Store standard deviation of accuracy for each K

# Loop over K values from 1 to Ks-1
for n in range(1, Ks):
    # Train KNN with n neighbors
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    
    # Predict on test set
    y_predicted = neigh.predict(X_test)
    
    # Compute mean accuracy
    mean_acc[n-1] = metrics.accuracy_score(y_test, y_predicted)
    
    # Compute standard deviation of accuracy
    std_acc[n-1] = np.std(y_predicted == y_test) / np.sqrt(y_predicted.shape[0])

# 2D Visualization of accuracy vs K
plt.plot(range(1, Ks), mean_acc, color="blue")

# Plot ±1 standard deviation as shaded area
plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, 
                 alpha=0.10, color="red")

# Plot ±3 standard deviations as shaded area
plt.fill_between(range(1, Ks), mean_acc - 3 * std_acc, mean_acc + 3 * std_acc, 
                 alpha=0.10, color="green")

# Add legend and axis labels
plt.legend(('Accuracy', '+/- 1xstd', '+/- 3xstd'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()

# Display plot
plt.show()
