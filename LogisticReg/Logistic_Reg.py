# Import necessary libraries
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation
from sklearn import preprocessing, metrics  # Preprocessing and evaluation metrics
from sklearn.model_selection import train_test_split  # Train-test split
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.metrics import jaccard_score, confusion_matrix, classification_report, log_loss, precision_score, recall_score  # Evaluation metrics
import matplotlib.pyplot as plt  # Plotting (not used here)

# Load churn dataset
churn_df = pd.read_csv('LogisticReg/ChurnData.csv')

# Select relevant features and target column
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',
                     'callcard', 'wireless', 'churn']]

# Convert churn column to integer type
churn_df['churn'] = churn_df['churn'].astype('int')

# Extract features for modeling
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])

# Extract target variable
Y = np.asarray(churn_df['churn'])

# Standardize features to zero mean and unit variance
X = preprocessing.StandardScaler().fit(X).transform(X)

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

# Initialize Logistic Regression with regularization C=0.01 and solver 'liblinear' and fit model
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, Y_train)

# Predict class labels for the test set
Y_predicted = LR.predict(X_test)

# Predict probabilities for each class
Y_prob = LR.predict_proba(X_test)

# Compute Jaccard index (similarity between predicted and true labels)
ja = jaccard_score(Y_test, Y_predicted, pos_label=0)

# Compute accuracy of predictions
acc = metrics.accuracy_score(Y_test, Y_predicted)

# Compute confusion matrix with labels ordered as [1,0]
conf = confusion_matrix(Y_test, Y_predicted, labels=[1,0])

# Generate classification report (precision, recall, f1-score)
report = classification_report(Y_test, Y_predicted)

# Compute log loss
logLass = log_loss(Y_test, Y_prob)

# Compute precision and recall
precision = precision_score(Y_test, Y_predicted)
recall = recall_score(Y_test, Y_predicted)

# Print evaluation results
print(report)
print(logLass)
print(precision)
print(recall)
print(conf)
