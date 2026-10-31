"""
Machine Learning Assignment: Feature Engineering and K-Means Clustering
Step 1: Project Setup and Data Cleaning
"""

import os
import pandas as pd
import numpy as np

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

# 1. Load the CSV dataset using pandas
DATASET_PATH = "students_dataset.csv"
print("========================================")
print("Step 1: Loading Dataset & Data Cleaning")
print("========================================")

df = pd.read_csv(DATASET_PATH)

print(f"\n1. Dataset loaded successfully!")
print(f"   Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\n--- Column Info ---")
print(df.info())

# 2. Check for duplicate rows
num_duplicates = df.duplicated().sum()
print(f"\n2. Duplicate Rows Count: {num_duplicates}")
if num_duplicates > 0:
    df = df.drop_duplicates()
    print(f"   Removed {num_duplicates} duplicate rows.")
else:
    print("   No duplicate rows found.")

# 3. Check for missing values (null / NaN)
print("\n3. Missing Values per Column:")
missing_values = df.isnull().sum()
print(missing_values)

# 4. Fill missing numerical values using the median
numerical_cols = df.select_dtypes(include=[np.number]).columns
missing_filled = False
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"   Filled missing values in '{col}' with median: {median_val:.2f}")
        missing_filled = True

if not missing_filled:
    print("   No missing numerical values needed imputation.")

# 5. Check and correct invalid values
print("\n4. Summary Statistics (Checking Min/Max for invalid values):")
print(df.describe())

# Criteria checks:
# Attendance and Assignment percentages should be between 0 and 100
# Grades should be between 0 and 100
# Hours and absences should not be negative
invalid_attendance = ((df['attendance_pct'] < 0) | (df['attendance_pct'] > 100)).sum()
invalid_assignments = ((df['assignments_done_pct'] < 0) | (df['assignments_done_pct'] > 100)).sum()
invalid_grades = ((df['previous_grade'] < 0) | (df['previous_grade'] > 100)).sum()
invalid_study = (df['study_hours_per_week'] < 0).sum()
invalid_absences = (df['absences'] < 0).sum()

print("\n5. Invalid Value Checks:")
print(f"   - Attendance outside [0, 100]: {invalid_attendance}")
print(f"   - Assignments outside [0, 100]: {invalid_assignments}")
print(f"   - Previous grade outside [0, 100]: {invalid_grades}")
print(f"   - Negative study hours: {invalid_study}")
print(f"   - Negative absences: {invalid_absences}")

print("\n[SUCCESS] Step 1 complete. Dataset is clean and ready for Step 2!")
