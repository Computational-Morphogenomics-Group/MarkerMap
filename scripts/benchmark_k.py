import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder

sys.path.insert(1, './src/')
from utils import *

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

def getZeisel(file_path):
  adata = sc.read_h5ad(file_path)
  X = adata.X.copy()
  adata.obs['names']=adata.obs['names0']
  adata.obs['annotation'] = adata.obs['names0']
  labels = adata.obs['names0'].values
  encoder = LabelEncoder()
  encoder.fit(labels)
  y = encoder.transform(labels)
  return X, y, encoder

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

  X = adata.X.copy()
  labels = adata.obs['annotation'].values
  encoder = LabelEncoder()
  encoder.fit(labels)
  y = encoder.transform(labels)
  return X, y, encoder

def getCiteSeq(file_path):
  adata = sc.read_h5ad(file_path)
  X = adata.X.copy()
  adata.obs['annotation'] = adata.obs['names']
  labels = adata.obs['names'].values
  encoder = LabelEncoder()
  encoder.fit(labels)
  y = encoder.transform(labels)
  return X, y, encoder


# Main
if len(sys.argv) != 2:
  raise Exception('usage: benchmark_k.py data_set_name')

data_name = sys.argv[1]

data_name_options = { 'zeisel', 'paul', 'cite_seq' }
if data_name not in data_name_options:
  raise Exception(f'usage: possible data sets to pick are {data_name_options}')

z_size = 16
hidden_layer_size = 256

batch_size = 64
batch_norm = True

global_t = 3.0
k=25

k_range = [10, 25, 50, 100, 250]
label_error_range = [0.1, 0.2, 0.5, 0.75, 1]
num_times = 1
max_epochs = 100

#pytorch lightning stuff
gpus = None
precision=32

if data_name == 'zeisel':
  X, y, encoder = getZeisel('data/zeisel/Zeisel.h5ad')
elif data_name == 'paul':
  X, y, encoder = getPaul('data/paul15/')
elif data_name == 'cite_seq':
  X, y, encoder = getCiteSeq('data/cite_seq/CITEseq.h5ad')

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
    'num_classes': len(encoder.classes_),
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
    'max_lr': 0.0001,
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
    'num_classes': len(encoder.classes_),
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

misclass_rates, benchmark_label, benchmark_range = benchmark(
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
  },
  num_times,
  X,
  y,
  save_path='checkpoints/',
  benchmark='label_error',
  benchmark_range=label_error_range,
)

plot_benchmarks(misclass_rates, benchmark_label, benchmark_range, mode='accuracy', show_stdev=True)
