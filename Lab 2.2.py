# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 16:51:00 2024

@author: Olope
"""

import pandas as pd

# Load Excel file into a DataFrame
df = pd.read_excel('passengers.xlsx')

# Display the DataFrame
print(df)

# Access specific columns or rows
# For example, to access the first column:
print(df['Column_Name'])

# Or to access a specific row:
print(df.iloc[0])  # Accesses the first row

# Iterate over rows
for index, row in df.iterrows():
    print(row['Column_Name'])

# Access specific cell
print(df.at[0, 'Column_Name'])  # Accesses the value of the first row in 'Column_Name'

# Filter data
filtered_data = df[df['Column_Name'] == 'desired_value']
print(filtered_data)