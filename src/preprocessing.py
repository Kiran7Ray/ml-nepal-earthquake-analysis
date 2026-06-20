import pandas as pd

# 1. Load the earthquake dataset from its relative path
df = pd.read_csv('../data/nepal_earthquakes_1990_2026.csv')

# 2. Display the first 5 rows to visually inspect the data structure
print(df.head())

# 3. Print the exact total count of rows and columns 
print("Rows and Columns:")
print(df.shape)

# 4. View a list of all column labels/features in the dataset
print("\nDataset Columns:")
print(df.columns)

# 5. Review data types, non-null counts, and memory usage
print("\nDataFrame Summary Info:")
df.info()

# 6. Generate summary statistics (mean, min, max, quartiles) for numeric columns
print("\nDescriptive Statistics:")
print(df.describe())

# 7. Count how many empty/NaN values exist in each column
print("\nMissing Values per Column:")
missing = df.isnull().sum()
print(missing)
