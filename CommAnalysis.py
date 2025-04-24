import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("datasets/SMA 1-3 Dataset.csv")

# Select numerical features for clustering
features = df[["Likes", "Comments", "Shares", "Followers"]]

# Normalize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Plot the clusters (using Likes and Followers for 2D plot)
plt.figure(figsize=(8,5))
plt.scatter(df['Likes'], df['Followers'], c=df['Cluster'], cmap='viridis')
plt.xlabel("Likes")
plt.ylabel("Followers")
plt.title("Community Detection via KMeans")
plt.colorbar(label="Cluster")
plt.show()

# Influential users: Top 3 from each cluster based on Followers
for cluster in df['Cluster'].unique():
    print(f"\nCluster {cluster} Influential Users:")
    print(df[df['Cluster'] == cluster].sort_values(by='Followers', ascending=False).head(3)[['Post_ID', 'Followers', 'Likes', 'Mentioned_Entities']])
