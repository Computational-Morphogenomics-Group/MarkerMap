import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
import gc

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import scipy.stats as stats
import anndata

from markermap.utils import MarkerMap
from markermap.utils import SmashPyWrapper
from markermap.utils import (
  get_citeseq,
  get_mouse_brain,
  get_paul,
  get_zeisel,
  split_data_into_dataloaders,
  train_model,
)

#scVI sets the global seeds on import using pytorch's seed_everything which seems to
#reset numpy's seed everytime a model is run, which is pretty heinous
#we need a queue to constantly be updating the seeds
random_seeds_queue = SmashPyWrapper.getRandomSeedsQueue(length=1000)

import scvi
scvi.settings.seed = random_seeds_queue.get_nowait()

EMPTY_GROUP = -10000

def getTopVariances(arr, num_variances):
  assert num_variances <= len(arr)
  arr_copy = arr.copy()

  indices = set()
  for i in range(num_variances):
    idx = np.argmax(arr_copy)
    indices.add(idx)
    arr_copy[idx] = -1

  return indices

def l2(X1, X2):
  # (n x 1) matrix of l2 norm of each cell
  l2_cells = np.matmul(np.power(X1 - X2, 2), np.ones((X1.shape[1], 1)))
  return np.mean(np.power(l2_cells, 0.5))

def jaccard(X1, X2, num_variances):
  x_vars = np.var(X1, axis=0)
  recon_vars = np.var(X2, axis=0)

  x_top_vars = getTopVariances(x_vars, num_variances)
  recon_top_vars = getTopVariances(recon_vars, num_variances)

  return len(x_top_vars & recon_top_vars)/len(x_top_vars | recon_top_vars)

def analyzeVariance(X, recon_X, y, groups, num_variances):
  jaccard_index = EMPTY_GROUP*np.ones(len(groups))
  spearman_rho = EMPTY_GROUP*np.ones(len(groups))
  spearman_p = EMPTY_GROUP*np.ones(len(groups))

  for group in groups:
    group_indices = np.arange(len(y))[y == group]

    if (len(group_indices) == 0):
      continue

    X_group = X[group_indices, :]
    recon_X_group = recon_X[group_indices, :]

    jaccard_index[group] = jaccard(X_group, recon_X_group, num_variances)

    rho, p = stats.spearmanr(np.var(X_group, axis=0), np.var(recon_X_group, axis=0), alternative='greater')
    spearman_rho[group] = rho
    spearman_p[group] = p

  jaccard_overall = jaccard(X, recon_X, num_variances)
  spearman_rho_overall, spearman_p_overall = stats.spearmanr(
    np.var(X, axis=0),
    np.var(recon_X, axis=0),
    alternative='greater',
  )

  return jaccard_overall, jaccard_index, spearman_rho_overall, spearman_rho, spearman_p_overall, spearman_p


def getL2(X, recon_X, y, groups):
  l2_by_group = EMPTY_GROUP*np.ones(len(groups))
  for group in groups:
    group_indices = np.arange(len(y))[y == group]

    if (len(group_indices) == 0):
      continue

    X_group = X[group_indices, :]
    recon_X_group = recon_X[group_indices, :]

    l2_by_group[group] = l2(X_group, recon_X_group)

  l2_overall = l2(X, recon_X)
  return l2_overall, l2_by_group


def trainAndGetReconMarkerMap(hidden_layer_size, z_size, k, train_dataloader, val_dataloader, X_test, gpus):
  unsupervised_marker_map = MarkerMap(
    X_test.shape[1],
    hidden_layer_size,
    z_size,
    num_classes=None,
    k=k,
    loss_tradeoff=1,
  )
  train_model(
    unsupervised_marker_map,
    train_dataloader,
    val_dataloader,
    gpus=gpus,
    min_epochs=25,
    max_epochs=100,
    max_lr=0.0001,
    early_stopping_patience=4,
    lr_explore_mode='linear',
    num_lr_rates=500,
  )
  return unsupervised_marker_map.get_reconstruction(X_test)

def trainAndGetRecon_scVI(X_train, X_test):
  adata_train = anndata.AnnData(X_train)
  adata_train.layers['counts'] = adata_train.X.copy()

  scvi.model.SCVI.setup_anndata(adata_train, layer='counts')
  model = scvi.model.SCVI(adata_train)
  model.train() #this is much faster with a GPU

  adata_test = anndata.AnnData(X_test)
  adata_test.layers['counts'] = adata_test.X.copy()
  scvi.model.SCVI.setup_anndata(adata_test, layer='counts')

  return model.posterior_predictive_sample(adata_test)

def insertOrConcatenate(results, key1, key2, val):
  if np.isscalar(val):
    val = val*np.ones((1,1))
  else:
    val = val.reshape((1,len(val))) #ensure its a row vector

  if (results[key1][key2] is None):
    results[key1][key2] = val
  else:
    results[key1][key2] = np.concatenate((results[key1][key2], val), axis=0)

def handleArgs(argv):
  data_name_options = ['zeisel', 'paul', 'cite_seq', 'mouse_brain']

  parser = argparse.ArgumentParser()
  parser.add_argument('data_name', help='data set name', choices=data_name_options)
  parser.add_argument('-s', '--save_model', help='the file to save the model to', default=None)
  parser.add_argument('-r', '--runs', help='the number of runs', type=int, default=1)
  parser.add_argument('-g', '--gpus', help='how many gpus to use', type=int, default=None)
  parser.add_argument('--hidden_layer_size', help='how many hidden layers to use in the VAEs', type=int, default=256)
  parser.add_argument('-k', '--k', help='value of k', type=int, default=50)

  args = parser.parse_args()

  return args.data_name, args.save_model, args.runs, args.gpus, args.hidden_layer_size, args.k

# Main
data_name, save_model, num_times, gpus, hidden_layer_size, k = handleArgs(sys.argv)

z_size = 16
plt.rcParams['font.family'] = 'STIXGeneral'

if data_name == 'zeisel':
  X, y, encoder = get_zeisel('data/zeisel/Zeisel.h5ad')
elif data_name == 'paul':
  X, y, encoder = get_paul(
    'data/paul15/house_keeping_genes_Mouse_bone_marrow.txt',
    'data/paul15/house_keeping_genes_Mouse_HSC.txt',
  )
elif data_name == 'cite_seq':
  X, y, encoder = get_citeseq('data/cite_seq/CITEseq.h5ad')
elif data_name == 'mouse_brain':
  X, y, encoder = get_mouse_brain(
    'data/mouse_brain_broad/mouse_brain_all_cells_20200625.h5ad',
    'data/mouse_brain_broad/snRNA_annotation_astro_subtypes_refined59_20200823.csv',
    log_transform=False, #scVI requires counts, so we will normalize and log transform after.
  )

  scVI_X = X.copy()

  marker_map_adata = anndata.AnnData(X)
  sc.pp.normalize_per_cell(marker_map_adata, counts_per_cell_after=1e4)
  sc.pp.log1p(marker_map_adata)
  sc.pp.scale(marker_map_adata, max_value=10)
  X = marker_map_adata.X

result_options = [
  'l2_all',
  'l2_groups',
  'jaccard_all',
  'jaccard_groups',
  'spearman_rho_all',
  'spearman_rho_groups',
  'spearman_p_all',
  'spearman_p_groups',
]
model_options = ['marker_map', 'scVI']
results = { 'marker_map': { z: None for z in result_options}, 'scVI': { z: None for z in result_options} }

groups = np.unique(y)
for i in range(num_times):
  # we will use 80% training data, 20% validation data during the training of the marker map
  train_dataloader, val_dataloader, _, train_indices, val_indices, test_indices = split_data_into_dataloaders(
      X,
      y,
      train_size=0.7,
      val_size=0.1,
  )
  y_test = y[test_indices]

  for model in model_options:
    # get the next seed
    scvi.settings.seed = random_seeds_queue.get_nowait()

    if model == 'marker_map':
      model_X = X.copy()
      X_test = model_X[test_indices, :]

      recon_X_test = trainAndGetReconMarkerMap(
        hidden_layer_size,
        z_size,
        k,
        train_dataloader,
        val_dataloader,
        X_test,
        gpus,
      )

    if model == 'scVI':
      model_X = scVI_X.copy()
      X_test = model_X[test_indices, :]
      recon_X_test = trainAndGetRecon_scVI(
        model_X[np.concatenate([train_indices, val_indices]), :],
        X_test,
      )

    l2_all, l2_by_group = getL2(X_test, recon_X_test, y_test, groups)
    jaccard_all, jaccard_index, spearman_rho_all, spearman_rho, spearman_p_all, spearman_p = analyzeVariance(
      X_test,
      recon_X_test,
      y_test,
      groups,
      int(X_test.shape[1]*0.2),
    )

    # this is really trash code
    insertOrConcatenate(results, model, 'l2_all', l2_all)
    insertOrConcatenate(results, model, 'l2_groups', l2_by_group)
    insertOrConcatenate(results, model, 'jaccard_all', jaccard_all)
    insertOrConcatenate(results, model, 'jaccard_groups', jaccard_index)
    insertOrConcatenate(results, model, 'spearman_rho_all', spearman_rho_all)
    insertOrConcatenate(results, model, 'spearman_rho_groups', spearman_rho)
    insertOrConcatenate(results, model, 'spearman_p_all', spearman_p_all)
    insertOrConcatenate(results, model, 'spearman_p_groups', spearman_p)

    del model_X
    del recon_X_test
    gc.collect()

  if (save_model):
    np.save(save_model, results)

print(results)

