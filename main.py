import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Data Cleaning & Feature Selection
# Load the dataset
data = pd.read_csv("C:\\Users\\LENOVO\\Documents\\Python Projects\\Cluster House Project\\house.csv")

# Remove irrelevant columns (e.g., 'Unnamed: 0') and keep relevant features
relevant_columns = ['Sqft', 'Floor', 'TotalFloor', 'Bedroom', 'Living.Room', 'Bathroom', 'Price']
data = data[relevant_columns]

# Step 2: Finding the Optimal Value of K (Elbow Method)
# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Using the elbow method to find the optimal number of clusters (k)
inertia = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Optimistic Value of K')
plt.show()

#Based on the elbow method, choose the optimal value of k (e.g., from the plot)
optimal_k = 3
print(f"Optimal K value: {optimal_k}")

# Step 3: Clustering and Storing Cluster Assignments
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_assignments = kmeans.fit_predict(data_scaled)

# Add a 'Cluster' column to the original DataFrame to store cluster assignments
data['Cluster'] = cluster_assignments

# You now have a 'Cluster' column in the original DataFrame that indicates which cluster each house belongs to.
# You can explore and analyze the clusters further as needed.
print(data)
print(data.head())

import pandas as pd

# Assuming you have 'Cluster' column in your 'data' DataFrame
# If not, make sure you've performed clustering and added the 'Cluster' column as described in previous steps

# Specify the path where you want to save the new CSV file
output_csv_path = "C:\\Users\\LENOVO\\Documents\\Python Projects\\Cluster House Project\\house_with_clusters.csv"

# Save the DataFrame with the 'Cluster' column to the CSV file
data.to_csv(output_csv_path, index=False)

print(f"CSV file with 'Cluster' column saved to: {output_csv_path}")
