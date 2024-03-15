import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('crypto_data_clean.csv')

## Step 1: Separate the index column and the data
index_col = df.iloc[:, 0]  # The first column as index/identifier
data = df.iloc[:, 1:]  # The rest of the data excluding the first column
# Step 2: Standardize the data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data )
# Step 3: Apply PCA
n_components = 10
pca = PCA(n_components=n_components)
data_reduced = pca.fit_transform(data_standardized)

# Step 4: Reattach the index column to the PCA-reduced data

reduced_df = pd.DataFrame(data_reduced, index=index_col.index)
reduced_df.insert(0, 'Index', index_col)
# Print explained variance ratio for each component
print("Explained variance ratio by component:")
for i, variance in enumerate(pca.explained_variance_ratio_):
    print(f"PC-{i+1}: {variance:.4f}")

# Print cumulative explained variance
cumulative_variance = pca.explained_variance_ratio_.cumsum()
print("\nCumulative explained variance:")
for i, cum_variance in enumerate(cumulative_variance):
    print(f"PC-1 to PC-{i+1}: {cum_variance:.4f}")

plt.figure(figsize=(10, 7))
plt.plot(range(1, n_components + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

num_components = data_reduced.shape[1]
print(f"Number of principal components: {num_components}")

# Apply K-means++ clustering to the reduced data
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(data_reduced)

# Add the cluster labels to our reduced DataFrame
reduced_df['Cluster'] = clusters
reduced_df.to_csv('crypto_data_reduced.csv')
# Convert data_reduced to a DataFrame for easier plotting
reduced_df_for_plotting = pd.DataFrame(data_reduced, columns=[f'PC{i}' for i in range(1, data_reduced.shape[1] + 1)])
# Add the cluster labels to this DataFrame
reduced_df_for_plotting['Cluster'] = clusters

# Now plot using seaborn
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=reduced_df_for_plotting, palette='viridis', s=100, alpha=0.7)

plt.title('PCA Components and K-means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

sns.scatterplot(x='PC1', y='PC3', hue='Cluster', data=reduced_df_for_plotting, palette='viridis', s=100, alpha=0.7)

plt.title('PCA Components and K-means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 3')  # Updated to PC3
plt.legend(title='Cluster')
plt.show()


sns.scatterplot(x='PC2', y='PC3', hue='Cluster', data=reduced_df_for_plotting, palette='viridis', s=100, alpha=0.7)

plt.title('PCA Components and K-means Clustering')
plt.xlabel('Principal Component 2')
plt.ylabel('Principal Component 3')
plt.legend(title='Cluster')
plt.show()

# The silhouette score ranges from -1 (incorrect clustering) to +1 (highly dense clustering), with scores around zero indicating overlapping clusters.
from sklearn.metrics import silhouette_score
score = silhouette_score(data, clusters)
print('silhouette_score:', score)

# Davies-Bouldin Index: A lower score indicates that the clusters are more compact and well-separated
from sklearn.metrics import davies_bouldin_score
score = davies_bouldin_score(data, clusters)
print('davies_bouldin_score:', score)

reduced_df.to_csv('crypto_data_reduced.csv', header=True)
# print(df)