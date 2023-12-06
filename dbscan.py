import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min

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

# Calculating the centroid
centroid = X.mean()

# Displaying the centroid
print("Centroid:")
print(centroid)

# Standardize the features
X = StandardScaler().fit_transform(X)

# DBSCAN clustering
dbscan = DBSCAN(eps=0.8, min_samples=3)  # Adjust parameters as needed
labels = dbscan.fit_predict(X)

# Add cluster labels to the DataFrame
df['Cluster'] = labels

# Calculate the centroid of each cluster
centroids = df.groupby('Cluster')[['Latitude', 'Longitude']].mean()

# Calculate each data to data centroid
df['Distance_to_Centroid'] = ((df['Latitude'] - centroid['Latitude'])**2 + (df['Longitude'] - centroid['Longitude'])**2)**0.5

# Displaying the clustered DataFrame
print(df)

# Visualize the clusters (for 2D data)
plt.scatter(df['Latitude'], df['Longitude'], c=df['Cluster'], cmap='viridis')
plt.scatter(centroids['Latitude'], centroids['Longitude'], marker='X', c='red', label='Cluster Centroids')
plt.scatter(centroid['Latitude'], centroid['Longitude'], marker='*', c='blue', label='Data Centroid')
plt.title('DBSCAN Clustering')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()
