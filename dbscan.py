import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# URL of the CSV file
csv_url = "https://raw.githubusercontent.com/kritzerenkrieg/python_dbscan/main/data_sekolah_mojokerto.csv" 
# Replace with the actual URL

# Importing the CSV into a DataFrame
df = pd.read_csv(csv_url)
df[['Latitude', 'Longitude']] = df['posisi geografis (Lintang, Bujur)'].str.split(', ', expand=True)

# Convert the new columns to numeric types
df['Latitude'] = pd.to_numeric(df['Latitude'])
df['Longitude'] = pd.to_numeric(df['Longitude'])

# Adjust the column names accordingly
X = df[['Latitude', 'Longitude']]

# Standardize the features
X = StandardScaler().fit_transform(X)

# DBSCAN clustering
dbscan = DBSCAN(eps=0.8, min_samples=3)  # Adjust parameters as needed
labels = dbscan.fit_predict(X)

# Add cluster labels to the DataFrame
df['Cluster'] = labels

# Displaying the clustered DataFrame
print(df)

# Visualize the clusters (for 2D data)
plt.scatter(df['Latitude'], df['Longitude'], c=df['Cluster'], cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()
