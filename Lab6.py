# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:09:51 2024

@author: Olope
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = pd.read_csv("C:/Users/Olope/.spyder-py3/data_science/Lab6/dataset.csv")

# Display basic information about the dataset
print("Number of instances:", len(data))
print("Number of attributes:", len(data.columns) - 1)  # Excluding the class column
print("Classes:", data.iloc[:, -1].unique())

# Visualize attribute distributions
num_attributes = len(data.columns) - 1
num_cols = 3  # Number of columns in the plot grid
num_rows = (num_attributes + num_cols - 1) // num_cols  # Calculate number of rows

plt.figure(figsize=(15, 5*num_rows))

for i, col in enumerate(data.columns[:-1]):
    plt.subplot(num_rows, num_cols, i+1)
    plt.hist(data[col], bins='auto', color='skyblue', edgecolor='black')
    plt.title(col)
    plt.xlabel("Value")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

#%% 2 Correlation Matrix

data2 = data.replace('?',0)

# Calculate the correlation matrix
correlation_matrix = data2.corr()

# Print correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.title("Correlation Matrix Heatmap")
plt.colorbar()

# Add annotation text
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(j, i, "{:.2f}".format(correlation_matrix.iloc[i, j]),
                 horizontalalignment='center', verticalalignment='center', color='white')

plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.tight_layout()
plt.show()

#%% 3

from sklearn.impute import SimpleImputer


# Identify missing values in the "age" attribute
missing_values = data[data['age'] == '?']

# Calculate the proportion of missing values
missing_proportion = len(missing_values)

print("Proportion of missing values in 'age' attribute:", missing_proportion)

data = data.replace('?','NaN')

# Evaluate whether imputation is necessary
if missing_proportion > 0:
    # If missing values exist, perform imputation
    imputer = SimpleImputer(missing_values=pd.NA, strategy='mean')  # You can choose different imputation strategies
    data['age'] = imputer.fit_transform(data[['age']])
    print("Missing values in 'age' attribute have been imputed.")
else:
    print("No missing values in 'age' attribute.")  
    
#%% 4

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

data2 = data.replace('?',0)
# Assume 'X' contains the features and 'y' contains the target variable
X = data2.drop(columns=['erythema']) 
y = data2['definite_borders']  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for grid search
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf', 'linear', 'poly']}

# Perform grid search with cross-validation
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train the SVM classifier with the best parameters
best_svm = SVC(**best_params)
best_svm.fit(X_train_scaled, y_train)

# Predictions
y_pred = best_svm.predict(X_test_scaled)

# Calculate evaluation metrics
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
kappa = cohen_kappa_score(y_test, y_pred)

# Print the evaluation metrics
print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Cohen's Kappa:", kappa)

#%% 5
# Assuming the class labels are in a column named 'class_label'
class_distribution = data2['melanin_incontinence'].value_counts()

# Plotting the bar chart
plt.figure(figsize=(8, 6))
class_distribution.plot(kind='bar', color='skyblue')
plt.title('Class Distribution')
plt.xlabel('Class Label')
plt.ylabel('Number of Instances')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#%% 6

from imblearn.over_sampling import SMOTE
# Load the dataset

data2 = data.replace('?',0)
X = data2.drop(columns=['itching'])  # Features
y = data2['koebner_phenomenon']  # Target

# Perform SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Plot the distribution of the original and balanced data
plt.figure(figsize=(10, 6))

# Original data
plt.subplot(1, 2, 1)
plt.title('Original Data')
data2['itching'].value_counts().plot(kind='bar', color='skyblue')
plt.xlabel('itching')
plt.ylabel('Number of Instances')
plt.xticks(rotation=45)

# Balanced data after SMOTE
plt.subplot(1, 2, 2)
plt.title('Balanced Data (SMOTE)')
pd.Series(y_smote).value_counts().plot(kind='bar', color='lightgreen')
plt.xlabel('itching')
plt.ylabel('Number of Instances')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

#%%7
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from imblearn.over_sampling import SMOTE


X = data2.drop(columns=['follicular_papules'])  # Features
y = data2['scaling']  # Target

# Perform SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Standardize the SMOTE data
scaler = StandardScaler()
X_smote_scaled = scaler.fit_transform(X_smote)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_smote_scaled, y_smote, test_size=0.2, random_state=42)

# Define the parameter grid for grid search
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf', 'linear', 'poly']}

# Perform grid search with cross-validation
svm = SVC(class_weight='balanced')
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train the SVM classifier with the best parameters
best_svm = SVC(**best_params, class_weight='balanced')
best_svm.fit(X_train, y_train)

# Predictions
y_pred = best_svm.predict(X_test)

# Calculate evaluation metrics
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
kappa = cohen_kappa_score(y_test, y_pred)

# Print the evaluation metrics
print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Cohen's Kappa:", kappa)



