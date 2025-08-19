# Import necessary libraries
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation
from sklearn.tree import DecisionTreeClassifier  # Decision Tree classifier
import sklearn.tree as tree  # Tree visualization utilities
from sklearn import preprocessing, metrics  # Data preprocessing and evaluation metrics
from sklearn.model_selection import train_test_split  # Train-test splitting
import matplotlib.pyplot as plt  # Plotting

# Load dataset from CSV file
my_data = pd.read_csv("DecisionTrees/drug200.csv")

# Select features for modeling: Age, Sex, BP, Cholesterol, Na_to_K
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

# Initialize label encoders for categorical features
le_sex = preprocessing.LabelEncoder()  # Encode Sex
le_bp = preprocessing.LabelEncoder()   # Encode BP
le_Cho = preprocessing.LabelEncoder()  # Encode Cholesterol

# Fit label encoders with possible values
le_sex.fit(['F', 'M'])  # Female/Male
le_bp.fit(["LOW", "NORMAL", "HIGH"])  # Blood Pressure levels
le_Cho.fit(['NORMAL', 'HIGH'])  # Cholesterol levels

# Transform categorical features to numeric values
X[:, 1] = le_sex.transform(X[:, 1])      # Encode Sex column
X[:, 2] = le_bp.transform(X[:, 2])       # Encode BP column
X[:, 3] = le_Cho.transform(X[:, 3])      # Encode Cholesterol column

# Set target variable
Y = my_data["Drug"]

# Split dataset into training (70%) and testing (30%) sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=3)

# Initialize Decision Tree classifier using entropy criterion and max depth of 4
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)

# Train the classifier on the training data
drugTree.fit(X_train, Y_train)

# Predict target labels for the test set
predTree = drugTree.predict(X_test)

# Create a DataFrame comparing actual vs predicted labels
newDF = pd.DataFrame({'Actual': Y_test.values, 'Predicted': predTree})

# Compute the accuracy of the predictions
accuracy = metrics.accuracy_score(Y_test, predTree)

# Display the plot (currently empty, placeholder for tree visualization if needed)
plt.show()
