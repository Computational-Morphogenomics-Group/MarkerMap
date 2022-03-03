import sys
import multiprocessing
import numpy as np
from sklearn.preprocessing import LabelEncoder
import smashpy
from lassonet import LassoNetClassifier
import matplotlib.pyplot as plt

# Pytorch imports
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

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

z_size = 16
hidden_layer_size = 256

batch_size = 64
batch_norm = True

global_t = 3.0

k_range = [10, 25, 50, 100, 250]
num_times = 5
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



def printBenchmarkComparison(results):
  markers = ['.','o','v','^','<','>','8','s','p','P','*','h','H','+','x','X','D','d','|','_','1','2','3','4',',']
  print(results)
  fig1, ax1 = plt.subplots()
  i = 0
  for label, result in results.items():
    ax1.plot(k_range, result, label=label, marker=markers[i])
    i += 1
  ax1.set_title('Accuracy Benchmark, Average over 3 Runs')
  ax1.set_xlabel('K Markers Selected')
  ax1.set_ylabel('Accuracy')
  ax1.legend()

  plt.show()


accuracies = {
  BASELINE : [],
  SMASH_RF : [],
  SMASH_DNN : [],
  L1_VAE : [],
  GLOBAL_GATE : [],
  UNSUP_MM : [],
  SUP_MM : [],
  MIXED_MM : [],
  CONCRETE_VAE : [],
}
for i in range(num_times):
  #Split data
  # The smashpy methods set global seeds that mess with sampling. These seeds are used
  # to stop those methods from using the same global seed over and over.
  np.random.seed(possible_seeds[seed_index])
  seed_index += 1
  train_dataloader, val_dataloader, test_dataloader, train_indices, val_indices, test_indices = split_data_into_dataloaders(
    X,
    y,
    0.7,
    0.1,
    batch_size=batch_size,
    # num_workers=num_workers,
  )

  X_train = X[np.concatenate([train_indices, val_indices]), :]
  y_train = y[np.concatenate([train_indices, val_indices])]
  X_test = X[test_indices,:]
  y_test = y[test_indices]

  for k in k_range:
    #Baseline using all markers
    all_markers = np.arange(input_size)
    baseline_results = new_model_metrics(X_train, y_train, X_test, y_test, markers = all_markers)
    print("Baseline error: ", baseline_results[0])
    accuracies[BASELINE].append(1-baseline_results[0])

    #Smash Random Forest
    train_X_smash = adata[np.concatenate([train_indices, val_indices]), :]
    sm = smashpy.smashpy()
    clf = sm.ensemble_learning(
      train_X_smash,
      group_by="annotation",
      classifier="RandomForest",
      balance=True,
      verbose=False, #doesn't do shit lmao
      save=False,
    )
    selectedGenes, selectedGenes_dict = sm.gini_importance(
      train_X_smash,
      clf,
      group_by="annotation",
      verbose=False, #also doesn't do shit
      restrict_top=("global", k),
    )

    # The smashpy methods set global seeds that mess with sampling. These seeds are used
    # to stop those methods from using the same global seed over and over.
    np.random.seed(possible_seeds[seed_index])
    seed_index += 1

    # since this selects k per class, need select randomly from each classes. Huh? -Wilson
    smash_markers = adata.var.index.get_indexer(selectedGenes)
    smash_rf_results = new_model_metrics(X_train, y_train, X_test, y_test, markers = smash_markers)
    print("Smash Random Forest accuracy: ", smash_rf_results[0])
    accuracies[SMASH_RF].append(1-smash_rf_results[0])

    # Smash DNN
    sm.DNN(train_X_smash, group_by="annotation", model=None, balance=True, verbose=False, save=False)
    with plt.ion(): #this outputs a plot, we don't want to stop computation
      selectedGenes, selectedGenes_dict = sm.run_shap(
        train_X_smash,
        group_by="annotation",
        model=None,
        verbose=False,
        pct=0.1,
        restrict_top=("global", k),
      )

    # The smashpy methods set global seeds that mess with sampling. These seeds are used
    # to stop those methods from using the same global seed over and over.
    np.random.seed(possible_seeds[seed_index])
    seed_index += 1

    # since this selects k per class, need select randomly from each classes
    smash_markers = adata.var.index.get_indexer(selectedGenes)
    smash_dnn_results = new_model_metrics(X_train, y_train, X_test, y_test, markers = smash_markers)
    print("Smash DNN error: ", smash_dnn_results[0])
    accuracies[SMASH_DNN].append(1-smash_dnn_results[0])

    # Rank Corr, skipping for now

    #L1 VAE
    feature_std = torch.tensor(X).std(dim = 0)
    model = VAE_l1_diag(input_size, hidden_layer_size, z_size, batch_norm = batch_norm)
    train_model(
      model,
      train_dataloader,
      val_dataloader,
      gpus=gpus,
      tpu_cores = tpu_cores,
      min_epochs = 25,
      max_epochs = max_epochs,
      auto_lr = True,
      early_stopping_patience = 4,
      precision = precision,
    )
    l1_markers = model.markers(feature_std = feature_std.to(model.device), k = k).clone().cpu().detach().numpy()
    l1_vae_results = new_model_metrics(X_train, y_train, X_test, y_test, markers = l1_markers)
    print("L1 VAE error: ", l1_vae_results[0])
    accuracies[L1_VAE].append(1-l1_vae_results[0])

    #Global Gate
    model = VAE_Gumbel_GlobalGate(
      input_size,
      hidden_layer_size,
      z_size,
      k = k,
      t = global_t,
      bias = True,
      temperature_decay=0.95,
      batch_norm = batch_norm,
    )
    train_model(
      model,
      train_dataloader,
      val_dataloader,
      gpus = gpus,
      tpu_cores = tpu_cores,
      min_epochs = 25,
      max_epochs = max_epochs,
      auto_lr = True,
      max_lr = 0.00001,
      early_stopping_patience =  10,
      lr_explore_mode = 'linear',
      num_lr_rates = 500,
      precision = precision,
    )
    globalgate_markers = model.markers().clone().cpu().detach().numpy()
    global_gate_results = new_model_metrics(X_train, y_train, X_test, y_test, markers = globalgate_markers)
    print("Global Gate error: ", global_gate_results[0])
    accuracies[GLOBAL_GATE].append(1-global_gate_results[0])

    #MarkerMap unsupervised
    model = MarkerMap(
      input_size,
      hidden_layer_size,
      z_size,
      k = k,
      t = global_t,
      bias = True,
      temperature_decay=0.95,
      alpha = 0.95,
      batch_norm = batch_norm,
      loss_tradeoff = 1.0,
      num_classes = None,
    )
    train_model(
      model,
      train_dataloader,
      val_dataloader,
      gpus=gpus,
      min_epochs = 25,
      max_epochs = max_epochs,
      auto_lr = True,
      max_lr = 0.0001,
      early_stopping_patience = 4,
      lr_explore_mode = 'linear',
      num_lr_rates = 500,
      precision = precision,
    )
    unsupervised_markers = model.markers().clone().cpu().detach().numpy()
    unsupervised_mm_results = new_model_metrics(X_train, y_train, X_test, y_test, markers = unsupervised_markers)
    print("Unsupervised Marker Map results: ", unsupervised_mm_results[0])
    accuracies[UNSUP_MM].append(1-unsupervised_mm_results[0])

    #MarkerMap supervised
    model = MarkerMap(
      input_size,
      hidden_layer_size,
      z_size,
      num_classes = len(encoder.classes_),
      k = k,
      t = global_t,
      bias = True,
      temperature_decay=0.95,
      alpha = 0.95,
      batch_norm = batch_norm,
      loss_tradeoff = 0,
    )
    train_model(
      model,
      train_dataloader,
      val_dataloader,
      gpus=gpus,
      tpu_cores = tpu_cores,
      min_epochs = 25,
      max_epochs = max_epochs,
      auto_lr = True,
      early_stopping_patience = 3,
      precision = precision,
      lr_explore_mode = 'linear',
      num_lr_rates=500,
    )
    markers = model.markers().clone().cpu().detach().numpy()
    supervised_mm_results = new_model_metrics(X_train, y_train, X_test, y_test, markers = markers)
    print("Supervised Marker Map results: ", supervised_mm_results[0])
    accuracies[SUP_MM + '_' + str(num_times)].append(1-supervised_mm_results[0])

    #Mixed MarkerMap
    model = MarkerMap(
      input_size,
      hidden_layer_size,
      z_size,
      num_classes = len(encoder.classes_),
      k = k,
      t = global_t,
      bias = True,
      temperature_decay=0.95,
      alpha = 0.95,
      batch_norm = batch_norm,
      loss_tradeoff = 0.5,
    )
    train_model(
      model,
      train_dataloader,
      val_dataloader,
      gpus=gpus,
      tpu_cores = tpu_cores,
      min_epochs = 25,
      max_epochs = max_epochs,
      auto_lr = True,
      early_stopping_patience = 3,
      precision = precision,
      lr_explore_mode = 'linear',
      num_lr_rates=500,
    )
    markers = model.markers().clone().cpu().detach().numpy()
    mixed_mm_results = new_model_metrics(X_train, y_train, X_test, y_test, markers = markers)
    print("Mixed Marker Map results: ", mixed_mm_results[0])
    accuracies[MIXED_MM].append(1-mixed_mm_results[0])

    #Concrete VAE
    model = ConcreteVAE_NMSL(
      input_size,
      hidden_layer_size,
      z_size,
      k = k,
      t = global_t,
      bias = True,
      temperature_decay = 0.9,
      batch_norm = batch_norm,
    )
    train_model(
      model,
      train_dataloader,
      val_dataloader,
      gpus=gpus,
      min_epochs = 25,
      max_epochs = max_epochs,
      auto_lr = True,
      max_lr = 0.0001,
      early_stopping_patience = 3,
      lr_explore_mode = 'linear',
      num_lr_rates = 500,
      precision = precision
    )
    concrete_vae_markers = model.markers().clone().cpu().detach().numpy()
    concrete_vae_results = new_model_metrics(X_train, y_train, X_test, y_test, markers = concrete_vae_markers)
    print("Concrete VAE results: ", concrete_vae_results[0])
    accuracies[CONCRETE_VAE].append(1-concrete_vae_results[0])

    #LassoNet
    model = LassoNetClassifier()
    model.path(X[train_indices, :], y[train_indices], X_val = X[val_indices,:], y_val = y[val_indices])
    lasso_net_markers = torch.argsort(model.feature_importances_, descending = True).cpu().numpy()[:k]
    lasso_net_results = new_model_metrics(X_train, y_train, X_test, y_test, markers = lasso_net_markers)
    print("Lasso Net results: ", lasso_net_results[0])


printBenchmarkComparison(accuracies)


# res = {
#   'Baseline' : [0.0532445923460898, 0.04991680532445919, 0.04991680532445919, 0.04991680532445919, 0.04991680532445919],
#   'Smash Random Forest' : [0.1780366056572379, 0.06489184692179706, 0.048252911813643884, 0.05490848585690511, 0.056572379367720416],
#   'L1 VAE' : [0.5291181364392679, 0.21297836938435943, 0.1780366056572379, 0.1314475873544093, 0.08985024958402665],
#   'Global Gate' : [0.43261231281198, 0.26955074875207985, 0.21297836938435943, 0.1530782029950083, 0.09484193011647257],
#   'Unsupervised Marker Map' : [0.32445923460898507, 0.2579034941763727, 0.10981697171381033, 0.16805324459234605, 0.08485856905158073],
#   'Supervised Marker Map' : [0.1281198003327787, 0.08153078202995012, 0.05490848585690511, 0.0532445923460898, 0.0515806988352745],
#   'Mixed Marker Map' : [0.21797004991680535, 0.11148086522462564, 0.08985024958402665, 0.05823627287853572, 0.05990016638935103],
#   'Concrete VAE' : [0.4575707154742097, 0.2861896838602329, 0.1580698835274542, 0.16805324459234605, 0.09317803660565727],
# }


# res = {
#   'Baseline' : [0.04159733777038266, 0.04492512479201327, 0.04492512479201327],
#   'Smash Random Forest' : [0.11813643926788686, 0.04159733777038266, 0.038269550748752046],
#   'Smash DNN' : [0.21131447587354413, 0.151414309484193, 0.08985024958402665],
#   'L1 VAE' : [0.4143094841930116, 0.19800332778702168, 0.2063227953410982],
#   'Global Gate' : [0.4575707154742097, 0.2778702163061564, 0.19800332778702168],
#   'Unsupervised Marker Map' : [0.2778702163061564, 0.329450915141431, 0.16472545757071544],
#   'Supervised Marker Map' : [0.11813643926788686, 0.04991680532445919, 0.04658901830282858],
#   'Mixed Marker Map' :[0.19301164725457576, 0.13976705490848584, 0.09650582362728788],
#   'Concrete VAE' : [0.31780366056572384, 0.29950083194675536, 0.20965058236272882],
# }




#3 times, k_range = [10,25,50,100,250]
# res = {
#   'Baseline': [0.9650582362728786, 0.9584026622296173, 0.9683860232945092, 0.9633943427620633, 0.961730449251248, 0.9450915141430949, 0.9450915141430949, 0.9484193011647255, 0.9484193011647255, 0.9484193011647255, 0.9534109816971714, 0.9550748752079867, 0.961730449251248, 0.9534109816971714, 0.9517470881863561],
#   'Smash Random Forest': [0.9217970049916805, 0.9650582362728786, 0.9584026622296173, 0.9667221297836939, 0.9667221297836939, 0.8985024958402662, 0.9334442595673876, 0.940099833610649, 0.9467554076539102, 0.9500831946755408, 0.8851913477537438, 0.9351081530782029, 0.9650582362728786, 0.9517470881863561, 0.956738768718802],
#   'Smash DNN': [0.6206322795341098, 0.7820299500831946, 0.8569051580698835, 0.9151414309484193, 0.9550748752079867, 0.8668885191347754, 0.9018302828618968, 0.9168053244592346, 0.9351081530782029, 0.9467554076539102, 0.6655574043261231, 0.8569051580698835, 0.9201331114808652, 0.9267886855241264, 0.9517470881863561],
#   'L1 VAE': [0.5357737104825291, 0.64891846921797, 0.7770382695507487, 0.8668885191347754, 0.8985024958402662, 0.6372712146422629, 0.7354409317803661, 0.7537437603993344, 0.8469217970049917, 0.9151414309484193, 0.7021630615640599, 0.8153078202995009, 0.7737104825291181, 0.8419301164725458, 0.9151414309484193],
#   'Global Gate': [0.5740432612312812, 0.6722129783693843, 0.78369384359401, 0.8569051580698835, 0.9267886855241264, 0.5291181364392679, 0.7071547420965059, 0.8336106489184693, 0.8352745424292846, 0.913477537437604, 0.4825291181364393, 0.7720465890183028, 0.8202995008319468, 0.8868552412645591, 0.9034941763727121],
#   'Unsupervised Marker Map': [0.7653910149750416, 0.826955074875208, 0.8286189683860233, 0.8718801996672213, 0.9467554076539102, 0.5008319467554077, 0.8236272878535774, 0.8818635607321131, 0.8718801996672213, 0.9351081530782029, 0.740432612312812, 0.7237936772046589, 0.8718801996672213, 0.9201331114808652, 0.9384359400998337],
#   'Supervised Marker Map': [0.8835274542429284, 0.956738768718802, 0.961730449251248, 0.9633943427620633, 0.961730449251248, 0.8951747088186356, 0.9018302828618968, 0.9351081530782029, 0.9534109816971714, 0.9434276206322796, 0.9251247920133111, 0.9251247920133111, 0.9550748752079867, 0.9534109816971714, 0.9650582362728786],
#   'Mixed Marker Map': [0.8386023294509152, 0.8785357737104825, 0.8202995008319468, 0.9484193011647255, 0.9384359400998337, 0.762063227953411, 0.8801996672212978, 0.9168053244592346, 0.9384359400998337, 0.9118136439267887, 0.7903494176372712, 0.9068219633943427, 0.9284525790349417, 0.9484193011647255, 0.9517470881863561],
#   'Concrete VAE': [0.7254575707154742, 0.7687188019966722, 0.7637271214642263, 0.8103161397670549, 0.9034941763727121, 0.5574043261231281, 0.6622296173044925, 0.7038269550748752, 0.8735440931780366, 0.9217970049916805, 0.6472545757071547, 0.6339434276206323, 0.8286189683860233, 0.9068219633943427, 0.9151414309484193]
# }

# res = {
#   SUP_MM + '_1' : [0.8835274542429284, 0.956738768718802, 0.961730449251248, 0.9633943427620633, 0.961730449251248],
#   SUP_MM + '_2' : [0.8951747088186356, 0.9018302828618968, 0.9351081530782029, 0.9534109816971714, 0.9434276206322796],
#   SUP_MM + '_3' : [0.9251247920133111, 0.9251247920133111, 0.9550748752079867, 0.9534109816971714, 0.9650582362728786],
# }

# for key, val in res.items():
#   res[key] = np.array(val).reshape((3,5)).mean(axis=0)


# printBenchmarkComparison(res)

{'Baseline': [0.956738768718802, 0.956738768718802, 0.9550748752079867, 0.956738768718802, 0.9600665557404326], 'Smash Random Forest': [0.9001663893510815, 0.9434276206322796, 0.9467554076539102, 0.9534109816971714, 0.961730449251248], 'Smash DNN': [0.7154742096505824, 0.8352745424292846, 0.9168053244592346, 0.9550748752079867, 0.9633943427620633], 'L1 VAE': [0.56738768718802, 0.7138103161397671, 0.7870216306156406, 0.8652246256239601, 0.930116472545757], 'Global Gate': [0.6056572379367721, 0.6988352745424293, 0.7653910149750416, 0.8718801996672213, 0.9234608985024958], 'Unsupervised Marker Map': [0.6672212978369384, 0.8202995008319468, 0.7920133111480865, 0.8169717138103162, 0.9467554076539102], 'Supervised Marker Map': [0.8585690515806988, 0.9417637271214643, 0.9484193011647255, 0.9584026622296173, 0.961730449251248], 'Mixed Marker Map': [0.8186356073211315, 0.9168053244592346, 0.8019966722129783, 0.8801996672212978, 0.9683860232945092], 'Concrete VAE': [0.6921797004991681, 0.6522462562396006, 0.7703826955074875, 0.8519134775374376, 0.9118136439267887]}



