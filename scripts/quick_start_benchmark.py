import scanpy as sc

from markermap.other_models import RandomBaseline
from markermap.vae_models import MarkerMap
from markermap.utils import benchmark, plot_benchmarks

# Set parameters
# z_size is the dimensionality of the latent space of the VAE
z_size = 16
# hidden_layer_size is the size of the hidden layers in the VAE before and after latent space
hidden_layer_size = 64
# the max_epochs for training the MarkerMaps
max_epochs = 100

file_path = 'data/cite_seq/CITEseq.h5ad'

#get data
group_by = 'annotation'
adata = sc.read_h5ad(file_path)
adata.obs[group_by] = adata.obs['names']

# The range of k values to run the benchmark on
k_range = [10, 25, 50]
k = k_range[0]  #will be unused since we are benchmarking over k_range

# Declare models
supervised_marker_map = MarkerMap.getBenchmarker(
  create_kwargs = {
    'input_size': adata.shape[1],
    'hidden_layer_size': hidden_layer_size,
    'z_size': z_size,
    'num_classes': len(adata.obs[group_by].unique()),
    'k': k,
    'loss_tradeoff': 0,
  },
  train_kwargs = {
    'max_epochs': max_epochs,
  }
)

mixed_marker_map = MarkerMap.getBenchmarker(
  create_kwargs = {
    'input_size': adata.shape[1],
    'hidden_layer_size': hidden_layer_size,
    'z_size': z_size,
    'num_classes': len(adata.obs[group_by].unique()),
    'k': k,
    'loss_tradeoff': 0.5,
  },
  train_kwargs = {
    'max_epochs': max_epochs,
  }
)

unsupervised_marker_map = MarkerMap.getBenchmarker(
  create_kwargs = {
    'input_size': adata.shape[1],
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

results, benchmark_label, benchmark_range = benchmark(
  {
    'Unsupervised Marker Map': unsupervised_marker_map,
    'Supervised Marker Map': supervised_marker_map,
    'Mixed Marker Map': mixed_marker_map,
    'Baseline': RandomBaseline.getBenchmarker(train_kwargs = { 'k': k }),
  },
  1, # num_times, how many different random train/test splits to run the models on
  adata,
  benchmark='k',
  group_by=group_by,
  benchmark_range=k_range,
)

print(results)
plot_benchmarks(results, benchmark_label, benchmark_range, mode='accuracy')
