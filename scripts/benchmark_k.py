import sys
import argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder
import anndata
import pandas as pd
import gc
import scanpy as sc

from markermap.utils import SmashPyWrapper, LassoNetWrapper, RandomBaseline, RankCorrWrapper
from markermap.utils import MarkerMap, ConcreteVAE_NMSL, VAE_Gumbel_GlobalGate, VAE_l1_diag
from markermap.utils import benchmark, parse_adata, plot_benchmarks

#Consts
BASELINE = 'Baseline'
SMASH_RF = 'Smash Random Forest'
SMASH_DNN = 'Smash DNN'
L1_VAE = 'L1 VAE'
GLOBAL_GATE = 'Global Gate'
UNSUP_MM = 'Unsupervised Marker Map'
SUP_MM = 'Supervised Marker Map'
MIXED_MM = 'Mixed Marker Map'
CONCRETE_VAE = 'Concrete VAE'
LASSONET = 'LassoNet'
RANK_CORR = 'RankCorr'

def getZeisel(file_path):
  adata = sc.read_h5ad(file_path)
  adata.obs['names']=adata.obs['names0']
  adata.obs['annotation'] = adata.obs['names0']

  return parse_adata(adata)

def getPaul(housekeeping_genes_dir):
  adata = sc.datasets.paul15()
  sm = SmashPyWrapper()
  sm.data_preparation(adata)
  adata = sm.remove_general_genes(adata)
  adata = sm.remove_housekeepingenes(adata, path=[housekeeping_genes_dir + 'house_keeping_genes_Mouse_bone_marrow.txt'])
  adata = sm.remove_housekeepingenes(adata, path=[housekeeping_genes_dir + 'house_keeping_genes_Mouse_HSC.txt'])
  dict_annotation = {}

  dict_annotation['1Ery']='Ery'
  dict_annotation['2Ery']='Ery'
  dict_annotation['3Ery']='Ery'
  dict_annotation['4Ery']='Ery'
  dict_annotation['5Ery']='Ery'
  dict_annotation['6Ery']='Ery'
  dict_annotation['7MEP']='MEP'
  dict_annotation['8Mk']='Mk'
  dict_annotation['9GMP']='GMP'
  dict_annotation['10GMP']='GMP'
  dict_annotation['11DC']='DC'
  dict_annotation['12Baso']='Baso'
  dict_annotation['13Baso']='Baso'
  dict_annotation['14Mo']='Mo'
  dict_annotation['15Mo']='Mo'
  dict_annotation['16Neu']='Neu'
  dict_annotation['17Neu']='Neu'
  dict_annotation['18Eos']='Eos'
  dict_annotation['19Lymph']='Lymph'

  annotation = []
  for celltype in adata.obs['paul15_clusters'].tolist():
      annotation.append(dict_annotation[celltype])

  adata.obs['annotation'] = annotation
  adata.obs['annotation'] = adata.obs['annotation'].astype('category')

  return parse_adata(adata)

def getCiteSeq(file_path):
  adata = sc.read_h5ad(file_path)
  adata.obs['annotation'] = adata.obs['names']
  return parse_adata(adata)

def relabel_mouse_labels(label):
  if isinstance(label, str):
    return label.split('_')[0]
  else:
    return label

def remove_features_pct(adata, group_by=None, pct=0.3):
  if group_by is None:
    print("select a group_by in .obs")
    return
  if group_by not in adata.obs.columns:
    print("group_by must be in .obs")
    return


  list_keep_genes = []

  df = pd.DataFrame(data=False,
            index=adata.var.index.tolist(),
            columns=adata.obs[group_by].cat.categories)
  for g in adata.obs[group_by].cat.categories:
    reduced = adata[adata.obs[group_by]==g]
    boolean, values = sc.pp.filter_genes(reduced, min_cells = reduced.n_obs*pct, inplace=False)
    df[g] = boolean
  dfT = df.T
  for g in dfT.columns:
    if True in dfT[g].tolist():
      list_keep_genes.append(True)
    else:
      list_keep_genes.append(False)

  adata.var["general"] = list_keep_genes

  adata = adata[:, adata.var["general"]]

  return adata

def remove_features_pct_2groups(adata, group_by=None, pct1=0.9, pct2=0.5):
  if group_by is None:
    print("select a group_by in .obs")
    return
  if group_by not in adata.obs.columns:
    print("group_by must be in .obs")
    return


  list_keep_genes = []

  df = pd.DataFrame(data=False,
                      index=adata.var.index.tolist(),
                      columns=adata.obs[group_by].cat.categories)
  for g in adata.obs[group_by].cat.categories:
    reduced = adata[adata.obs[group_by]==g]
    boolean, values = sc.pp.filter_genes(reduced, min_cells = reduced.n_obs*(pct1), inplace=False)
    df[g] = boolean
  dfT = df.T
  for g in dfT.columns:
    if (sum(dfT[g].tolist())/len(dfT[g].tolist())) >= pct2:
      list_keep_genes.append(False)
    else:
      list_keep_genes.append(True)

  adata.var["general"] = list_keep_genes

  adata = adata[:, adata.var["general"]]

  return adata


def getMouseBrain(dataset_dir):
  adata_snrna_raw = anndata.read_h5ad(dataset_dir + 'mouse_brain_all_cells_20200625.h5ad')
  del adata_snrna_raw.raw
  adata_snrna_raw = adata_snrna_raw
  adata_snrna_raw.X = adata_snrna_raw.X.toarray()
  ## Cell type annotations
  labels = pd.read_csv(dataset_dir + 'snRNA_annotation_astro_subtypes_refined59_20200823.csv', index_col=0)
  labels['annotation'] = labels['annotation_1'].apply(lambda x: relabel_mouse_labels(x))
  labels = labels[['annotation']]
  labels = labels.reindex(index=adata_snrna_raw.obs_names)
  adata_snrna_raw.obs[labels.columns] = labels
  adata_snrna_raw = adata_snrna_raw[~adata_snrna_raw.obs['annotation'].isna(), :]
  adata_snrna_raw = adata_snrna_raw[adata_snrna_raw.obs["annotation"]!='Unk']
  adata_snrna_raw.obs['annotation'] = adata_snrna_raw.obs['annotation'].astype('category')

  # doing this cuz smash was a lot of data
  #https://cell2location.readthedocs.io/en/latest/notebooks/cell2location_estimating_signatures.html
  #preprocess_like_cell_location
  sc.pp.filter_cells(adata_snrna_raw, min_genes=1)
  print(adata_snrna_raw.shape)
  sc.pp.filter_genes(adata_snrna_raw, min_cells=1)
  print(adata_snrna_raw.shape)

  gc.collect()
  # calculate the mean of each gene across non-zero cells
  adata_snrna_raw.var['n_cells'] = (adata_snrna_raw.X > 0).sum(0)
  adata_snrna_raw.var['nonz_mean'] = adata_snrna_raw.X.sum(0) / adata_snrna_raw.var['n_cells']

  nonz_mean_cutoff = np.log10(1.12) # cut off for expression in non-zero cells
  cell_count_cutoff = np.log10(adata_snrna_raw.shape[0] * 0.0005) # cut off percentage for cells with higher expression
  cell_count_cutoff2 = np.log10(adata_snrna_raw.shape[0] * 0.03)# cut off percentage for cells with small expression


  adata_snrna_raw[:,(np.array(np.log10(adata_snrna_raw.var['nonz_mean']) > nonz_mean_cutoff)
          | np.array(np.log10(adata_snrna_raw.var['n_cells']) > cell_count_cutoff2))
      & np.array(np.log10(adata_snrna_raw.var['n_cells']) > cell_count_cutoff)].shape

  # select genes based on mean expression in non-zero cells
  adata_snrna_raw = adata_snrna_raw[:,(np.array(np.log10(adata_snrna_raw.var['nonz_mean']) > nonz_mean_cutoff)
          | np.array(np.log10(adata_snrna_raw.var['n_cells']) > cell_count_cutoff2))
      & np.array(np.log10(adata_snrna_raw.var['n_cells']) > cell_count_cutoff)
              & np.array(~adata_snrna_raw.var['SYMBOL'].isna())]
  gc.collect()
  adata_snrna_raw.raw = adata_snrna_raw
  adata_snrna_raw.X = adata_snrna_raw.raw.X.copy()
  del adata_snrna_raw.raw
  gc.collect()
  adata_snrna_raw = remove_features_pct(adata_snrna_raw, group_by="annotation", pct=0.3)
  gc.collect()
  adata_snrna_raw = remove_features_pct_2groups(adata_snrna_raw, group_by="annotation", pct1=0.75, pct2=0.5)

  sc.pp.normalize_per_cell(adata_snrna_raw, counts_per_cell_after=1e4)
  sc.pp.log1p(adata_snrna_raw)
  sc.pp.scale(adata_snrna_raw, max_value=10)

  return parse_adata(adata_snrna_raw)

def handleArgs(argv):
  data_name_options = ['zeisel', 'paul', 'cite_seq', 'mouse_brain']

  parser = argparse.ArgumentParser()
  parser.add_argument('data_name', help='data set name', choices=data_name_options)
  parser.add_argument('-s', '--save_file', help='the file to save the benchmark results to', default=None)
  parser.add_argument('-r', '--runs', help='the number of runs', type=int, default=1)
  parser.add_argument('-g', '--gpus', help='how many gpus to use', type=int, default=None)
  parser.add_argument('--hidden_layer_size', help='how many hidden layers to use in the VAEs', type=int, default=256)

  args = parser.parse_args()

  return args.data_name, args.save_file, args.runs, args.gpus, args.hidden_layer_size

# Main
data_name, save_file, num_times, gpus, hidden_layer_size = handleArgs(sys.argv)

z_size = 16

batch_size = 64
batch_norm = True

global_t = 3.0
k=25

k_range = [10, 25, 50, 100, 250]
label_error_range = [0.1, 0.2, 0.5, 0.75, 1]
max_epochs = 100

#pytorch lightning stuff
precision=32

if data_name == 'zeisel':
  X, y, encoder = getZeisel('data/zeisel/Zeisel.h5ad')
elif data_name == 'paul':
  X, y, encoder = getPaul('data/paul15/')
elif data_name == 'cite_seq':
  X, y, encoder = getCiteSeq('data/cite_seq/CITEseq.h5ad')
elif data_name == 'mouse_brain':
  X, y, encoder = getMouseBrain('data/mouse_brain_broad/')

# The smashpy methods set global seeds that mess with sampling. These seeds are used
# to stop those methods from using the same global seed over and over.
random_seeds_queue = SmashPyWrapper.getRandomSeedsQueue(length = len(k_range) * num_times * 5)

input_size = X.shape[1]

# Declare models
unsupervised_mm = MarkerMap.getBenchmarker(
  create_kwargs = {
    'input_size': input_size,
    'hidden_layer_size': hidden_layer_size,
    'z_size': z_size,
    'num_classes': None,
    'k': k,
    't': global_t,
    'batch_norm': batch_norm,
    'loss_tradeoff': 1.0,
  },
  train_kwargs = {
    'gpus': gpus,
    'min_epochs': 25,
    'max_epochs': max_epochs,
    'auto_lr': True,
    'max_lr': 0.0001,
    'early_stopping_patience': 4,
    'lr_explore_mode': 'linear',
    'num_lr_rates': 500,
    'precision': precision,
  },
)

supervised_mm = MarkerMap.getBenchmarker(
  create_kwargs = {
    'input_size': input_size,
    'hidden_layer_size': hidden_layer_size,
    'z_size': z_size,
    'num_classes': len(np.unique(y)),
    'k': k,
    't': global_t,
    'batch_norm': batch_norm,
    'loss_tradeoff': 0,
  },
  train_kwargs = {
    'gpus': gpus,
    'min_epochs': 25,
    'max_epochs': max_epochs,
    'auto_lr': True,
    'max_lr': 0.001,
    'lr_explore_mode': 'linear',
    'num_lr_rates': 500,
    'precision': precision,
  }
)

mixed_mm = MarkerMap.getBenchmarker(
  create_kwargs = {
    'input_size': input_size,
    'hidden_layer_size': hidden_layer_size,
    'z_size': z_size,
    'num_classes': len(np.unique(y)),
    'k': k,
    't': global_t,
    'batch_norm': batch_norm,
    'loss_tradeoff': 0.5,
  },
  train_kwargs = {
    'gpus': gpus,
    'min_epochs': 25,
    'max_epochs': max_epochs,
    'auto_lr': True,
    'max_lr': 0.0001,
    'lr_explore_mode': 'linear',
    'num_lr_rates': 500,
    'precision': precision,
  }
)

concrete_vae = ConcreteVAE_NMSL.getBenchmarker(
  create_kwargs = {
    'input_size': input_size,
    'hidden_layer_size': hidden_layer_size,
    'z_size': z_size,
    'k': k,
    't': global_t,
    'batch_norm': batch_norm,
  },
  train_kwargs = {
    'gpus': gpus,
    'min_epochs': 25,
    'max_epochs': max_epochs,
    'auto_lr': True,
    'max_lr': 0.0001,
    'lr_explore_mode': 'linear',
    'num_lr_rates': 500,
    'precision': precision,
  }
)

global_gate = VAE_Gumbel_GlobalGate.getBenchmarker(
  create_kwargs = {
    'input_size': input_size,
    'hidden_layer_size': hidden_layer_size,
    'z_size': z_size,
    'k': k,
    't': global_t,
    'temperature_decay': 0.95,
    'batch_norm': batch_norm,
  },
  train_kwargs = {
    'gpus': gpus,
    'min_epochs': 25,
    'max_epochs': max_epochs,
    'auto_lr': True,
    'max_lr': 0.00001,
    'early_stopping_patience': 10,
    'lr_explore_mode': 'linear',
    'num_lr_rates': 500,
    'precision': precision,
  }
)

smash_rf = SmashPyWrapper.getBenchmarker(
  train_kwargs = { 'restrict_top': ('global', k) },
  model='RandomForest',
  random_seeds_queue = random_seeds_queue,
)

smash_dnn = SmashPyWrapper.getBenchmarker(
  train_kwargs = { 'restrict_top': ('global', k) },
  model='DNN',
  random_seeds_queue = random_seeds_queue,
)

l1_vae = VAE_l1_diag.getBenchmarker(
  create_kwargs = {
    'input_size': input_size,
    'hidden_layer_size': hidden_layer_size,
    'z_size': z_size,
    'batch_norm': batch_norm,
  },
  train_kwargs = {
    'gpus': gpus,
    'min_epochs': 25,
    'max_epochs': max_epochs,
    'early_stopping_patience': 4,
    'precision': precision,
    'k': k,
  },
)

results, benchmark_label, benchmark_range = benchmark(
  {
    UNSUP_MM: unsupervised_mm,
    SUP_MM: supervised_mm,
    MIXED_MM: mixed_mm,
    BASELINE: RandomBaseline.getBenchmarker(train_kwargs = { 'k': k }),
    LASSONET: LassoNetWrapper.getBenchmarker(train_kwargs = { 'k': k }),
    CONCRETE_VAE: concrete_vae,
    GLOBAL_GATE: global_gate,
    SMASH_RF: smash_rf,
    SMASH_DNN: smash_dnn,
    L1_VAE: l1_vae,
    RANK_CORR: RankCorrWrapper.getBenchmarker(train_kwargs = { 'k': k, 'lamb': 20 }),
  },
  num_times,
  X,
  y,
  save_file=save_file,
  benchmark='k',
  benchmark_range=k_range,
)

plot_benchmarks(results, benchmark_label, benchmark_range, mode='accuracy', show_stdev=True)
