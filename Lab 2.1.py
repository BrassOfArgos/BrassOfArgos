# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:25:15 2024

@author: Olope
"""

import numpy as np

# Create random matrices A and B
A = np.random.randint(-50, 50, size=(6, 6))
B = np.random.randint(-50, 50, size=(6, 4))

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)

# Find the common numbers in a matrix
common = np.intersect1d(A, B)
print("Common Numbers:")
print(common)
print("_____________________________________________________")

#Find numbers in A but not in B
Adif = np.setdiff1d(A, B)
print("These are the different numbers in A not found in B:")
print(Adif)
print("_____________________________________________________")

#Divide matrix A into two matrices: 
#A1 (2x6) contains the last two rows of A and 
#A2 (4x6) contains the remaining rows.
A_First = A[4:, :]
A_Second = A[:4, :]
print("\nThis is the First Matrix of A made of the last 2 rows:")
print(A_First)
print("\nThis is the Second Matrix of A made of the first 4 rows:")
print(A_Second)

# d. Add a column to matrix B that contains the maximum element of each row in A.
# Calculate the maximum element of each row in A
max_elements_of_A_rows = np.max(A, axis=1)  
# Add a new axis to match the shape of B
max_column = np.expand_dims(max_elements_of_A_rows, axis=1) 
# Add the new column to B
B_max_column = np.hstack((B, max_column))  
print("\nMatrix B with the maximum element of each row in A added as a new column:")
print(B_max_column)

# e. Add another column to matrix B that contains the minimum element of each row in A.
min_A_rows = np.min(A, axis=1)  # Calculate the minimum element of each row in A
min_column = np.expand_dims(min_A_rows, axis=1)  # Add a new axis to match the shape of B
B_total = np.hstack((B_max_column, min_column))  # Add the new column to B
print("\nMatrix B with the maximum and minimum elements of each row in A added as new columns:")
print(B_total)