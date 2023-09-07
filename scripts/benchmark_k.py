import sys
import argparse
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


from markermap.other_models import SmashPyWrapper, LassoNetWrapper, RandomBaseline, RankCorrWrapper, ScanpyRankGenes, COSGWrapper
from markermap.vae_models import MarkerMap, ConcreteVAE_NMSL, VAE_Gumbel_GlobalGate, VAE_l1_diag
from markermap.utils import benchmark, plot_benchmarks, get_citeseq, get_mouse_brain, get_paul, get_zeisel

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
SCANPY = 'Scanpy Rank Genes'
COSG = 'COSG'

def handleArgs(argv):
  data_name_options = ['zeisel', 'zeisel_big', 'paul', 'cite_seq', 'mouse_brain', 'mouse_brain_big']
  eval_model_options = ['random_forest', 'k_nearest_neighbors']
  benchmark_options = ['k', 'label_error', 'label_error_markers_only']

  parser = argparse.ArgumentParser()
  parser.add_argument('data_name', help='data set name', choices=data_name_options)
  parser.add_argument('-s', '--save_file', help='the file to save the benchmark results to', default=None)
  parser.add_argument('-r', '--runs', help='the number of runs', type=int, default=1)
  parser.add_argument('-g', '--gpus', help='how many gpus to use', type=int, default=None)
  parser.add_argument('--hidden_layer_size', help='how many hidden layers to use in the VAEs', type=int, default=256)
  parser.add_argument(
    '--eval_model',
    help='what simple classifier to use to evaluate the models',
    choices=eval_model_options,
    default='random_forest',
  )
  parser.add_argument('-b', '--benchmark', help='what we are benchmarking over', default='k', choices=benchmark_options)
  parser.add_argument('--data_dir', help='directory where the input data is located', type=str, default='data/')

  args = parser.parse_args()

  return (
    args.data_name, 
    args.save_file, 
    args.runs, 
    args.gpus, 
    args.hidden_layer_size, 
    args.eval_model, 
    args.benchmark,
    args.data_dir,
  )

# Main
data_name, save_file, num_times, gpus, hidden_layer_size, eval_model, benchmark_mode, data_dir = handleArgs(sys.argv)

if eval_model == 'random_forest':
  eval_model = RandomForestClassifier()
elif eval_model == 'k_nearest_neighbors':
  eval_model = KNeighborsClassifier()

z_size = 16

batch_size = 64
batch_norm = True

global_t = 3.0
k=50

if benchmark_mode == 'k':
  benchmark_range = [10, 25, 50, 100, 250]
elif benchmark_mode == 'label_error' or benchmark_mode == 'label_error_markers_only':
  benchmark_range = [0.1, 0.2, 0.5, 0.75, 1]

max_epochs = 100

#pytorch lightning stuff
precision=32

if data_name == 'zeisel':
  X, y, encoder = get_zeisel(data_dir + 'zeisel/Zeisel.h5ad')
elif data_name == 'zeisel_big':
  X, y, encoder = get_zeisel(data_dir + 'zeisel/Zeisel.h5ad', 'names1')
elif data_name == 'paul':
  X, y, encoder = get_paul(
    data_dir + 'paul15/house_keeping_genes_Mouse_bone_marrow.txt',
    data_dir + 'paul15/house_keeping_genes_Mouse_HSC.txt',
  )
elif data_name == 'cite_seq':
  X, y, encoder = get_citeseq(data_dir + 'cite_seq/CITEseq.h5ad')
elif data_name == 'mouse_brain':
  X, y, encoder = get_mouse_brain(
    data_dir + 'mouse_brain_broad/mouse_brain_all_cells_20200625.h5ad',
    data_dir + 'mouse_brain_broad/snRNA_annotation_astro_subtypes_refined59_20200823.csv',
  )
elif data_name =='mouse_brain_big':
  X, y, encoder = get_mouse_brain(
    data_dir + 'mouse_brain_broad/mouse_brain_all_cells_20200625.h5ad',
    data_dir + 'mouse_brain_broad/snRNA_annotation_astro_subtypes_refined59_20200823.csv',
    relabel=False,
  )

num_classes = len(np.unique(y))

# The smashpy methods set global seeds that mess with sampling. These seeds are used
# to stop those methods from using the same global seed over and over.
random_seeds_queue = SmashPyWrapper.getRandomSeedsQueue(length = len(benchmark_range) * num_times * 5)

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
    'num_classes': num_classes,
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
    'num_classes': num_classes,
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

results, benchmark_mode, benchmark_range = benchmark(
  {
    UNSUP_MM: unsupervised_mm,
    SUP_MM: supervised_mm,
    MIXED_MM: mixed_mm,
    BASELINE: RandomBaseline.getBenchmarker(train_kwargs = { 'k': k }),
    LASSONET: LassoNetWrapper.getBenchmarker(train_kwargs = { 'k': k }),
    CONCRETE_VAE: concrete_vae,
    GLOBAL_GATE: global_gate,
    SMASH_RF: smash_rf,
    # SMASH_DNN: smash_dnn,
    # L1_VAE: l1_vae,
    RANK_CORR: RankCorrWrapper.getBenchmarker(train_kwargs = { 'k': k, 'lamb': 20 }),
    SCANPY: ScanpyRankGenes.getBenchmarker(train_kwargs = { 'k': k, 'num_classes': num_classes }),
    SCANPY + ' overestim_var': ScanpyRankGenes.getBenchmarker(train_kwargs = { 'k': k, 'num_classes': num_classes, 'method': 't-test_overestim_var' }),
    SCANPY + ' wilcoxon': ScanpyRankGenes.getBenchmarker(train_kwargs = { 'k': k, 'num_classes': num_classes, 'method': 'wilcoxon' }),
    SCANPY + ' wilcoxon tie': ScanpyRankGenes.getBenchmarker(train_kwargs = { 'k': k, 'num_classes': num_classes, 'method': 'wilcoxon', 'tie_correct': True }),
    SCANPY + ' logreg': ScanpyRankGenes.getBenchmarker(train_kwargs = { 'k': k, 'num_classes': num_classes, 'method': 'logreg' }),
    COSG: COSGWrapper.getBenchmarker(train_kwargs = { 'k': k, 'num_classes': num_classes }),
  },
  num_times,
  X,
  y,
  save_file=save_file,
  benchmark=benchmark_mode,
  benchmark_range=benchmark_range,
  eval_model=eval_model,
  min_groups=[2,0,1],
)

plot_benchmarks(results, benchmark_mode, benchmark_range, mode='accuracy', show_stdev=True, print_vals=True)
