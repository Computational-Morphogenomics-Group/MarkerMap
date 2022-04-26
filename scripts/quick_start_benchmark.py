import numpy as np
import scanpy as sc

from markermap.utils import RandomBaseline
from markermap.utils import MarkerMap
from markermap.utils import benchmark, parse_adata, plot_benchmarks

# Set parameters
# z_size is the dimensionality of the latent space of the VAE
z_size = 16
# hidden_layer_size is the size of the hidden layers in the VAE before and after latent space
hidden_layer_size = 64
# the max_epochs for training the MarkerMaps
max_epochs = 100

file_path = 'data/cite_seq/CITEseq.h5ad'

#get data
adata = sc.read_h5ad(file_path)
adata.obs['annotation'] = adata.obs['names']
X, y, encoder = parse_adata(adata)

# The range of k values to run the benchmark on
k_range = [10, 25, 50]

# Declare models
unsupervised_marker_map = MarkerMap.getBenchmarker(
  create_kwargs = {
    'input_size': X.shape[1],
    'hidden_layer_size': hidden_layer_size,
    'z_size': z_size,
    'num_classes': None,
    'k': k,
    'loss_tradeoff': 1.0,
  },
  train_kwargs = {
    'max_epochs': max_epochs,
  },
)

supervised_marker_map = MarkerMap.getBenchmarker(
  create_kwargs = {
    'input_size': X.shape[1],
    'hidden_layer_size': hidden_layer_size,
    'z_size': z_size,
    'num_classes': len(np.unique(y)),
    'k': k,
    'loss_tradeoff': 0,
  },
  train_kwargs = {
    'max_epochs': max_epochs,
  }
)

mixed_marker_map = MarkerMap.getBenchmarker(
  create_kwargs = {
    'input_size': X.shape[1],
    'hidden_layer_size': hidden_layer_size,
    'z_size': z_size,
    'num_classes': len(np.unique(y)),
    'k': k,
    'loss_tradeoff': 0.5,
  },
  train_kwargs = {
    'max_epochs': max_epochs,
  }
)

results, benchmark_label, benchmark_range = benchmark(
  {
    'Unsupervised Marker Map': unsupervised_marker_map,
    'Supervised Marker Map': supervised_marker_map,
    'Mixed Marker Map': mixed_marker_map,
    'Baseline': RandomBaseline.getBenchmarker(train_kwargs = { 'k': k }),
  },
  1, # num_times, how many different random train/test splits to run the models on
  X,
  y,
  benchmark='k',
  benchmark_range=k_range,
)

plot_benchmarks(results, benchmark_label, benchmark_range, mode='accuracy')
