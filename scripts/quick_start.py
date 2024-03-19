import numpy as np
import scanpy as sc

from markermap.vae_models import MarkerMap, train_model
from markermap.utils import (
    new_model_metrics,
    plot_confusion_matrix,
    split_data,
)

# Set parameters
# z_size is the dimensionality of the latent space of the VAE
z_size = 16
# hidden_layer_size is the size of the hidden layers in the VAE before and after latent space
hidden_layer_size = 64
# k is the number of markers to extract
k=50
# model batch_size
batch_size = 64

# change the file path to point at your data
file_path = 'data/cite_seq/CITEseq.h5ad'

#get data
group_by = 'annotation'
adata = sc.read_h5ad(file_path)
adata.obs[group_by] = adata.obs['names']

# we will use 70% training data, 10% vaidation data during the training of the marker map, then 20% for testing
train_indices, val_indices, test_indices = split_data(
    adata.X,
    adata.obs[group_by],
    [0.7, 0.1, 0.2],
)
train_val_indices = np.concatenate([train_indices, val_indices])

train_dataloader, val_dataloader = MarkerMap.prepareData(
    adata,
    train_indices,
    val_indices,
    group_by,
    None, #layer, just use adata.X
    batch_size=batch_size,
)

# define the marker map, we use a loss_tradeoff of 0 to only use the supervised training loss
supervised_marker_map = MarkerMap(
    adata.X.shape[1],
    hidden_layer_size,
    z_size,
    len(adata.obs[group_by].unique()),
    k,
    loss_tradeoff=0,
)

train_model(supervised_marker_map, train_dataloader, val_dataloader)

# use only the k markers to train a simple model, in this case a random forest, then evaluate it with the test data
misclass_rate, test_rep, cm = new_model_metrics(
    adata[train_val_indices, :].X,
    adata[train_val_indices, :].obs[group_by],
    adata[test_indices, :].X,
    adata[test_indices, :].obs[group_by],
    markers = supervised_marker_map.markers().clone().cpu().detach().numpy(),
)

# print the results
print(misclass_rate)
print(test_rep['weighted avg']['f1-score'])
plot_confusion_matrix(cm, adata.obs[group_by].unique())
