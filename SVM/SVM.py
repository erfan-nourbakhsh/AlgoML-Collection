# Import necessary libraries
import pandas as pd  # Data manipulation
import pylab as pl  # Plotting utilities (not used directly here)
import numpy as np  # Numerical operations
import scipy.optimize as opt  # Optimization functions (not used directly here)
from sklearn import preprocessing, svm  # Preprocessing and SVM
from sklearn.model_selection import train_test_split  # Train-test split
import matplotlib.pyplot as plt  # Plotting
from sklearn.metrics import confusion_matrix, classification_report, jaccard_score, f1_score, log_loss  # Evaluation metrics

# Load dataset
cell_df = pd.read_csv("SVM/cell_samples.csv")

# Plot first 50 malignant cells (Class 4) as scatter points
ax = cell_df[cell_df['Class'] == 4][0:50].plot(
    kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant', s=50)

# Plot first 50 benign cells (Class 2) on the same axes
cell_df[cell_df['Class'] == 2][0:50].plot(
    kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax)

# Remove rows where 'BareNuc' is not numeric
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]

# Convert 'BareNuc' to integer type
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')

# Prepare feature matrix by dropping the target column 'Class'
feature_df = cell_df.drop('Class', axis=1).values
X = np.asarray(feature_df)

# Convert target column 'Class' to integer type
cell_df['Class'] = cell_df['Class'].astype('int')
Y = np.asarray(cell_df['Class'])

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=4)

# Define list of kernels to test in SVM
kernels = ['rbf', 'poly', 'sigmoid']

# Train and evaluate SVM for each kernel
for item in kernels:
    # Initialize SVM classifier with the given kernel
    clf = svm.SVC(kernel=item)
    
    # Fit SVM on training data
    clf.fit(X_train, Y_train)
    
    # Predict class labels for test data
    Y_predicted = clf.predict(X_test)
    
    # Compute confusion matrix with labels [2, 4]
    cnf_matrix = confusion_matrix(Y_test, Y_predicted, labels=[2, 4])
    
    # Compute F1-score (weighted)
    f1Score = f1_score(Y_test, Y_predicted, average='weighted')
    
    # Compute Jaccard similarity score (weighted)
    jaccard = jaccard_score(Y_test, Y_predicted, average='weighted')
    
    # Compute log loss
    logLoss = log_loss(Y_test, Y_predicted)
    
    # Print evaluation metrics
    print(cnf_matrix)
    print(f1Score)
    print(jaccard)
    print(logLoss)
    print("------------")
