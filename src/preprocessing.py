##data understanding ......
import pandas as pd
import numpy as np

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
# 1. Handle Duplicates
duplicate = df.duplicated().sum()
print(f"Duplicate rows: {duplicate}")
df = df.drop_duplicates()  # Removed the duplicates

# 2. Impute Missing Values (Vectorized)
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
print("Missing values after imputation:\n", df.isnull().sum())

# 3. Rename and Drop Columns
df = df.rename(columns={
    'time': 'DateTime', 
    'mag': 'magnitude', 
    'magType': 'magnitudeType'
})
df = df.drop(columns=[
    'id', 'updated', 'net', 'status', 'locationSource', 'magSource'
])

# 4. Clean Place Names
print("Unique places before cleaning:\n", df['place'].unique())

# Fix literal '?' characters using regex=True
place_corrections = {
    'B\?glung': 'Baglung', 'Dh\?rchula': 'Dharchula', 'D\?rchul\?': 'Darchula', 
    'Kod\?ri\?\?': 'Kodari', 'Kh\?\?db\?ri\?\?': 'Khandbari', 'Il\?m': 'Ilam', 
    'Dhankut\?': 'Dhankuta', 'P\?tan': 'Patan', 'Panauti\?\?': 'Panauti', 
    'R\?mnagar': 'Ramnagar', 'B\?geshwar': 'Bageshwar', 'Naya B\?z\?r': 'Naya Bazar', 
    'Dadeldhur\?': 'Dadeldhura', 'Josh\?math': 'Joshimath', 'Dhar\n': 'Dharan', 
    '\?ik\?pur': 'Tikapur', 'D\?rjiling': 'Darjiling', 'Kh\?\?db\?ri': 'Khandbari', 
    'R\?jbir\?j': 'Rajbiraj', 'Lohagh\?t': 'Lohaghat', 'Tuls\?pur': 'Tulsipur', 
    'Pithor\?garh': 'Pithoragarh', 'Lah\?n': 'Lahan', 'Odl\?b\?ri': 'Odlabari'
}

for wrong, correct in place_corrections.items():
    df['place'] = df['place'].str.replace(wrong, correct, regex=True)

# 5. Save Cleaned Data
df.to_csv('../data/cleaned_data.csv', index=False)
print("CSV file saved successfully !!")