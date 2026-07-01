import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

#eda 

# Load and inspect dataset
data_set=pd.read_csv("../data/cleaned_data.csv") 
data_set.head() 

# Print row and column dimensions
print("ROWS AND COLUMN") 
print(data_set.shape) 

# Plot magnitude distribution with trend line
plt.figure(figsize=(8,5)) 
sns.histplot(data_set['magnitude'],bins=30,kde=True) 
plt.title("Earthquake Magnitude Distribution") 
plt.show() 

# Plot depth distribution with trend line
plt.figure(figsize=(8,5)) 
sns.histplot(data_set['depth'],bins=30,kde=True) #Kernel Density Estimate curve over a histogram. 
plt.title("Depth Distribution (in km)") 
plt.show() 

# Scatter plot: Magnitude vs Depth
plt.figure(figsize=(8,5)) 
sns.scatterplot( x='magnitude', y='depth', data=data_set ) 
plt.show() 

# Scatter plot: Magnitude vs Latitude
plt.figure(figsize=(8,5)) 
sns.scatterplot( y='latitude', x='magnitude', color="red", data=data_set ) 
plt.show() 

# Scatter plot: Magnitude vs Longitude
plt.figure(figsize=(8,5)) 
sns.scatterplot( y='longitude', x='magnitude', data=data_set ) 
plt.show() 

# Map coordinates with size/color reflecting magnitude
plt.figure(figsize=(10,7)) 
sns.scatterplot( x='longitude', y='latitude', hue='magnitude', size='magnitude', data=data_set ) 
plt.title("Earthquake Locations in Nepal") 
plt.show() 

# Convert to datetime and extract year
data_set['DateTime']=pd.to_datetime(data_set['DateTime']) 
data_set['year']=data_set['DateTime'].dt.year 
data_set.columns 

# Bar chart: Magnitude by year
plt.figure(figsize=(12,5)) 
plt.bar(data_set['year'],data_set['magnitude'],color="red",) 
plt.xlabel("years") 
plt.ylabel("magnitude") 
plt.title("magnitude by year") 
plt.show() 

# Boxplot for magnitude outliers
sns.boxplot(x=data_set['magnitude']) 
plt.show() 

# Boxplot for depth outliers
sns.boxplot(x=data_set['depth']) 
plt.show() 

# Extract numeric data and calculate correlation matrix
numeric_df = data_set.select_dtypes(include=np.number) 
corr = numeric_df.corr() 

# Plot correlation heatmap
plt.figure(figsize=(12,8)) 
sns.heatmap( corr, annot=True, cmap='coolwarm' ) 
plt.show() 

# Sort features by correlation with magnitude
corr['magnitude'].sort_values( ascending=False )


## feature engineering
#convert to date and time
data['DateTime']=pd.to_datetime(data['DateTime'])
data['DateTime']

#extract year ,month ,day, hour ,weekend
data['year']=data['DateTime'].dt.year
data['year']

data['month']=data['DateTime'].dt.month
data['month']

data['day'] = data['DateTime'].dt.day
data['day']

data['hour']=data['DateTime'].dt.hour
data['weekday'] = data['DateTime'].dt.dayofweek

#we keep only essential columns

#handle categorical columns
for cols in ['magnitudeType','type','place']:
    print(data[cols].value_counts())


cat_cols = [
    'magnitudeType',
    'type',
    'place'
]

data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
#nepal earthquack often cluster geographically ,create resional feature

def region(lat):
    
    if lat < 27.5:
        return "South"
    
    elif lat < 29:
        return "Central"
    
    else:
        return "North"

data['region'] = data['latitude'].apply(region)

data = pd.get_dummies(
    data,
    columns=['region'],
    drop_first=True
)

#feature selection
corr = data.corr(numeric_only=True)

corr['magnitude'].sort_values(
    ascending=False
)