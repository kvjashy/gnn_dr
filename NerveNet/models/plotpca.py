import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Load data from npy file
data_path = 'ankle_embeddings1.npy'  # Replace with the path to your .npy file
data = np.load(data_path)

# Initialize PCA with a single component
pca = PCA(n_components=1)

plt.figure(figsize=(15, 8))

for node_index in range(4):
    # Extracting all feature values for the given node across all time steps
    feature_data = data[:, 0, node_index, :]
    
    # Applying PCA transformation
    transformed_data = pca.fit_transform(feature_data)
    
    plt.plot(transformed_data, label=f"Node {node_index+1}")

plt.xlabel('Time Step')
plt.ylabel('PCA Transformed Value')
plt.title('PCA Transformed Features over Time for All Nodes')
plt.legend()
plt.tight_layout()
plt.savefig('pca2time.png')

plt.show()


# Load data from npy file
data = np.load(data_path)

# Concatenate the features across nodes for each timestep
concatenated_data = data.squeeze().reshape(10, -1)  # This will be of shape (10, 256)

# Apply t-SNE with only one component and adjusted perplexity
tsne = TSNE(n_components=1, perplexity=5)
embedded_data = tsne.fit_transform(concatenated_data)

# Plotting the t-SNE reduced representation
plt.figure(figsize=(15, 6))
plt.plot(embedded_data, 'o-')
plt.xlabel('Timestep')
plt.ylabel('t-SNE Value')
plt.title('t-SNE Visualization of Concatenated Node Features across Timesteps')
plt.grid(True)
plt.show()

data = np.load(data_path)

# Reshape data
reshaped_data = data.reshape(-1, 64)  # This will be of shape (40, 64)

# Apply PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(reshaped_data)

# For visualization purposes, we can use colors to represent each node's data points
colors = ['red', 'green', 'blue', 'purple'] * 10  # Assuming 4 nodes, repeated for 10 timesteps

plt.figure(figsize=(12, 8))
for i, color in enumerate(colors):
    plt.scatter(pca_data[i, 0], pca_data[i, 1], color=color, label=f"Node {i % 4 + 1}" if i < 4 else "")

plt.legend()
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.title('t-SNE plot of root node embeddings against last 10 timestep in an episode')
plt.savefig('pcatime.png')
plt.grid(True)
plt.show()