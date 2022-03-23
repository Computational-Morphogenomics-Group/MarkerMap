import sys
import multiprocessing
import numpy as np
from sklearn.preprocessing import LabelEncoder
import smashpy
from lassonet import LassoNetClassifier
import matplotlib.pyplot as plt

# Pytorch imports
import torch

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

z_size = 16
hidden_layer_size = 256

batch_size = 64
batch_norm = True

global_t = 3.0

k_range = [10, 25, 50, 100, 250]
num_times = 1
max_epochs = 100

#pytorch lightning stuff
gpus = None
tpu_cores = None
precision=32

# The smashpy methods set global seeds that mess with sampling. These seeds are used
# to stop those methods from using the same global seed over and over.
possible_seeds = np.random.randint(low=1, high = 1000000, size = 400)
seed_index = 0
num_workers = multiprocessing.cpu_count()

# pre-process the data
adata = sc.read_h5ad('data/zeisel/Zeisel.h5ad')
X = adata.X.copy()
adata.obs['names']=adata.obs['names0']
adata.obs['annotation'] = adata.obs['names0']
labels = adata.obs['names0'].values
encoder = LabelEncoder()
encoder.fit(labels)
y = encoder.transform(labels)
input_size = X.shape[1]


# accuracies = {
#   BASELINE : [],
#   SMASH_RF : [],
#   SMASH_DNN : [],
#   L1_VAE : [],
#   GLOBAL_GATE : [],
#   UNSUP_MM : [],
#   SUP_MM : [],
#   MIXED_MM : [],
#   CONCRETE_VAE : [],
# }
# for i in range(num_times):
#   #Split data
#   # The smashpy methods set global seeds that mess with sampling. These seeds are used
#   # to stop those methods from using the same global seed over and over.
#   np.random.seed(possible_seeds[seed_index])
#   seed_index += 1
#   train_dataloader, val_dataloader, test_dataloader, train_indices, val_indices, test_indices = split_data_into_dataloaders(
#     X,
#     y,
#     0.7,
#     0.1,
#     batch_size=batch_size,
#     # num_workers=num_workers,
#   )

#   X_train = X[np.concatenate([train_indices, val_indices]), :]
#   y_train = y[np.concatenate([train_indices, val_indices])]
#   X_test = X[test_indices,:]
#   y_test = y[test_indices]

#   for k in k_range:

#     #Smash Random Forest
#     train_X_smash = adata[np.concatenate([train_indices, val_indices]), :]
#     sm = smashpy.smashpy()
#     clf = sm.ensemble_learning(
#       train_X_smash,
#       group_by="annotation",
#       classifier="RandomForest",
#       balance=True,
#       verbose=False, #doesn't do shit lmao
#       save=False,
#     )
#     selectedGenes, selectedGenes_dict = sm.gini_importance(
#       train_X_smash,
#       clf,
#       group_by="annotation",
#       verbose=False, #also doesn't do shit
#       restrict_top=("global", k),
#     )

#     # The smashpy methods set global seeds that mess with sampling. These seeds are used
#     # to stop those methods from using the same global seed over and over.
#     np.random.seed(possible_seeds[seed_index])
#     seed_index += 1

#     # since this selects k per class, need select randomly from each classes. Huh? -Wilson
#     smash_markers = adata.var.index.get_indexer(selectedGenes)
#     smash_rf_results = new_model_metrics(X_train, y_train, X_test, y_test, markers = smash_markers)
#     print("Smash Random Forest accuracy: ", smash_rf_results[0])
#     accuracies[SMASH_RF].append(1-smash_rf_results[0])

#     # Smash DNN
#     sm.DNN(train_X_smash, group_by="annotation", model=None, balance=True, verbose=False, save=False)
#     with plt.ion(): #this outputs a plot, we don't want to stop computation
#       selectedGenes, selectedGenes_dict = sm.run_shap(
#         train_X_smash,
#         group_by="annotation",
#         model=None,
#         verbose=False,
#         pct=0.1,
#         restrict_top=("global", k),
#       )

#     # The smashpy methods set global seeds that mess with sampling. These seeds are used
#     # to stop those methods from using the same global seed over and over.
#     np.random.seed(possible_seeds[seed_index])
#     seed_index += 1

#     # since this selects k per class, need select randomly from each classes
#     smash_markers = adata.var.index.get_indexer(selectedGenes)
#     smash_dnn_results = new_model_metrics(X_train, y_train, X_test, y_test, markers = smash_markers)
#     print("Smash DNN error: ", smash_dnn_results[0])
#     accuracies[SMASH_DNN].append(1-smash_dnn_results[0])

#     # Rank Corr, skipping for now

#     #L1 VAE
#     feature_std = torch.tensor(X).std(dim = 0)
#     model = VAE_l1_diag(input_size, hidden_layer_size, z_size, batch_norm = batch_norm)
#     train_model(
#       model,
#       train_dataloader,
#       val_dataloader,
#       gpus=gpus,
#       tpu_cores = tpu_cores,
#       min_epochs = 25,
#       max_epochs = max_epochs,
#       auto_lr = True,
#       early_stopping_patience = 4,
#       precision = precision,
#     )
#     l1_markers = model.markers(feature_std = feature_std.to(model.device), k = k).clone().cpu().detach().numpy()
#     l1_vae_results = new_model_metrics(X_train, y_train, X_test, y_test, markers = l1_markers)
#     print("L1 VAE error: ", l1_vae_results[0])
#     accuracies[L1_VAE].append(1-l1_vae_results[0])

unsupervised_mm = MarkerMap.getBenchmarker(
  create_kwargs = {
    'input_size': input_size,
    'hidden_layer_size': hidden_layer_size,
    'z_size': z_size,
    'num_classes': None,
    'k': 25,
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
    'k': 25,
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
    'k': 25,
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
    'k': 25,
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
    'k': 25,
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

misclass_rates, benchmark_label, benchmark_range = benchmark(
  {
    # UNSUP_MM: unsupervised_mm,
    # SUP_MM: supervised_mm,
    # MIXED_MM: mixed_mm,
    BASELINE: RandomBaseline.getBenchmarker(),
    # LASSONET: LassoNetWrapper.getBenchmarker(),
    CONCRETE_VAE: concrete_vae,
    # GLOBAL_GATE: global_gate,
  },
  num_times,
  X,
  y,
  benchmark='k',
  k_range=k_range,
)

plot_benchmarks(misclass_rates, benchmark_label, benchmark_range, show_stdev=True)



