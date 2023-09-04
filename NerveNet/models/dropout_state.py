import torch.nn as nn
import torch.nn.functional as F
from NerveNet.models.nerve_net_opt import SimulatedAnnealingDropout
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

class DropoutState:
    def __init__(self, initial_dropout_rate=0.5):
        self.current_dropout_rate = initial_dropout_rate

    def set_rate(self, rate):
        self.current_dropout_rate = rate

    def get_rate(self):
        return self.current_dropout_rate
    

class DropoutLayer(nn.Module):
    def __init__(self, p=0.1):
        super(DropoutLayer, self).__init__()

        self.p = p 

    def forward(self, x):
        return F.dropout(x, self.p, self.training)

    def set_dropout_rate(self, p):
        self.p = p

class DropoutManager:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def update(self, rewards):
        new_dropout_rate = self.optimizer.update_dropout_rate(rewards)
        self.model.policy.mlp_extractor.set_dropout_rate(new_dropout_rate)
        return new_dropout_rate


def plot_tsne(embeddings, title='t-SNE plot of GNN messages'):
    # Applying t-SNE to reduce embeddings to 2 dimensions
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(embeddings)
    
    # Use timesteps as color information
    timesteps = np.arange(embeddings.shape[0])
    
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=timesteps, cmap='viridis', alpha=0.5)
    
    # Adding a colorbar to represent the timesteps
    cbar = plt.colorbar(scatter)
    cbar.set_label('Timestep')
    
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')





