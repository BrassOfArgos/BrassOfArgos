# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 10:38:34 2024

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load the dataset
file_path = "dataset_1.csv"
data = pd.read_csv(file_path)

# Separate features (X) and labels (y)
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # Last column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
clf = LogisticRegression(max_iter=10000)

# Fit the model on the training data
clf.fit(X_train, y_train)

# Predict labels on the testing set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Logistic Regression: {accuracy * 100:.2f}%")

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


# Initialize lists to store accuracy values
lda_accuracy = []

for n in range(1, 21):
    # Apply LDA
    lda = LinearDiscriminantAnalysis()
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    
    # Train a classifier on the transformed training data
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train_lda, y_train)
    
    # Predict on the transformed testing data
    y_pred = clf.predict(X_test_lda)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    lda_accuracy.append(accuracy)

# Plotting n_components vs Accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), lda_accuracy, marker='o', linestyle='-')
plt.title('LDA Accuracy vs n_components')
plt.xlabel('n_components')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()