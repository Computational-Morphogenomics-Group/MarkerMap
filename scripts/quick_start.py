import numpy as np
import scanpy as sc

from markermap.vae_models import MarkerMap, train_model
from markermap.utils import (
    new_model_metrics,
    parse_adata,
    plot_confusion_matrix,
    split_data_into_dataloaders,
)

# Set parameters
# z_size is the dimensionality of the latent space of the VAE
z_size = 16
# hidden_layer_size is the size of the hidden layers in the VAE before and after latent space
hidden_layer_size = 64
# k is the number of markers to extract
k=50

file_path = 'data/cite_seq/CITEseq.h5ad'

#get data
adata = sc.read_h5ad(file_path)
adata.obs['annotation'] = adata.obs['names']
X, y, encoder = parse_adata(adata)

# we will use 70% training data, 10% vaidation data during the training of the marker map, then 20% for testing
train_dataloader, val_dataloader, _, train_indices, val_indices, test_indices = split_data_into_dataloaders(
    X,
    y,
    train_size=0.7,
    val_size=0.1,
)

# define the marker map, we use a loss_tradeoff of 0 to only use the supervised training loss
supervised_marker_map = MarkerMap(X.shape[1], hidden_layer_size, z_size, len(np.unique(y)), k, loss_tradeoff=0)

train_model(supervised_marker_map, train_dataloader, val_dataloader)

# use only the k markers to train a simple model, in this case a random forest, then evaluate it with the test data
misclass_rate, test_rep, cm = new_model_metrics(
    X[np.concatenate([train_indices, val_indices]), :],
    y[np.concatenate([train_indices, val_indices])],
    X[test_indices, :],
    y[test_indices],
    markers = supervised_marker_map.markers().clone().cpu().detach().numpy(),
)

# print the results
print(misclass_rate)
print(test_rep['weighted avg']['f1-score'])
plot_confusion_matrix(cm, encoder.classes_)
