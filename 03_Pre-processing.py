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
data_standardized = scaler.fit_transform(data)

# Step 3: Apply PCA
pca = PCA(n_components=10)  # Adjust n_components to our desired number of features
data_reduced = pca.fit_transform(data_standardized)

# Step 4: Reattach the index column to the PCA-reduced data
reduced_df = pd.DataFrame(data_reduced, index=index_col.index)
reduced_df.insert(0, 'Index', index_col)
print(reduced_df)
# Apply K-means clustering to the reduced data
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(data_reduced)

# Add the cluster labels to our reduced DataFrame
reduced_df['Cluster'] = clusters

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

reduced_df.to_csv('crypto_data_reduced.csv', header=True)
# print(df)