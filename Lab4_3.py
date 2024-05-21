# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:26:54 2024

@author: Olope
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "dataset_1.csv"
data = pd.read_csv(file_path)

# Separate features (X) and labels (y)
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # Last column

# Standardize the data using StandardScaler
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Split the standardized dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
clf = LogisticRegression(max_iter=10000)

# Fit the model on the training data
clf.fit(X_train, y_train)

# Predict labels on the testing set
y_pred = clf.predict(X_test)

# Calculate accuracy for Raw data
accuracy_raw = accuracy_score(y_test, y_pred)
print(f"Accuracy of Logistic Regression with Raw data: {accuracy_raw * 100:.2f}%")

#%% PCA
# Lists to store accuracy values
pca_accuracies = []

# Test PCA with different numbers of components (1 to 20)
for n in range(1, 21):
    # PCA
    pca = PCA(n_components=n)
    X_pca_train = pca.fit_transform(X_train)
    X_pca_test = pca.transform(X_test)
    
    # Logistic Regression
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_pca_train, y_train)
    y_pred = clf.predict(X_pca_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    pca_accuracies.append(accuracy)

# Plotting n_components vs Accuracy for PCA
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), pca_accuracies, marker='o', linestyle='-')
plt.title('PCA: n_components vs Accuracy')
plt.xlabel('Number of Components')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

#%% RFE

# Lists to store accuracy values
rfe_accuracies = []

# Test RFE with different numbers of features (1 to 20)
for n in range(1, 21):
    # RFE
    rfe = RFE(estimator=LogisticRegression(max_iter=10000), n_features_to_select=n)
    X_rfe_train = rfe.fit_transform(X_train, y_train)
    X_rfe_test = rfe.transform(X_test)
    
    # Logistic Regression
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_rfe_train, y_train)
    y_pred = clf.predict(X_rfe_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    rfe_accuracies.append(accuracy)

# Plotting n_features vs Accuracy for RFE
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), rfe_accuracies, marker='o', linestyle='-')
plt.title('RFE: n_features vs Accuracy')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

#%% LDA

# Lists to store accuracy values
lda_accuracies = []

# Test LDA with different numbers of components (1 to min(n_classes - 1, n_features))
n_classes = len(np.unique(y_train))
n_features = X_train.shape[21]
n_components_range = range(1, min(n_classes, n_features) + 1)

for n in n_components_range:
    # LDA
    lda = LDA(n_components=n)
    X_lda_train = lda.fit_transform(X_train, y_train)
    X_lda_test = lda.transform(X_test)
    
    # Logistic Regression
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_lda_train, y_train)
    y_pred = clf.predict(X_lda_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    lda_accuracies.append(accuracy)

# Plotting n_components vs Accuracy for LDA
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, lda_accuracies, marker='o', linestyle='-')
plt.title('LDA: n_components vs Accuracy')
plt.xlabel('Number of Components')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()