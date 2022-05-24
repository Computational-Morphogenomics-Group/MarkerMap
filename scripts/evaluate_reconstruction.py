import sys
import argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder
import anndata
import pandas as pd
import gc
import scanpy as sc
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

from pydiffmap import diffusion_map

from markermap.utils import MarkerMap
from markermap.utils import (
  load_model,
  new_model_metrics,
  parse_adata,
  plot_confusion_matrix,
  split_data_into_dataloaders,
  split_data_into_dataloaders_no_test,
  train_model,
  train_save_model,
)

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




def testPCAProjectionDifference(n):
  rng = np.random.default_rng()
  X_real = rng.normal(0, 1, (n, 3))

  pca = PCA(n_components=2)
  pca.fit(X_real)

  X = rng.normal(0, 1, (n,2))
  X_orig = pca.inverse_transform(X)

  pca2 = PCA(n_components=3)
  pca2.fit(X_orig)
  print(pca2.explained_variance_ratio_)

  plot3d(X_orig, 3)


def plot3d(X, n_cols, skip_eigenvecs = {}, colors = None, show=True):
    assert n_cols <= X.shape[1]

    for i in range(n_cols):
        if i in skip_eigenvecs:
            continue

        for j in range(i+1, n_cols):
            if j in skip_eigenvecs:
                continue

            for k in range(j+1, n_cols):
                if k in skip_eigenvecs:
                    continue

                fig1, ax1 = plt.subplots(subplot_kw={'projection':'3d'})
                if colors is not None:
                    ax1.scatter(X[:,i], X[:,j], X[:,k], c=colors, cmap='Spectral')
                else:
                    ax1.scatter(X[:,i], X[:,j], X[:,k])

                ax1.set_xlabel(f'Eigenvec {i}')
                ax1.set_ylabel(f'Eigenvec {j}')
                ax1.set_zlabel(f'Eigenvec {k}')

    plt.tight_layout()

    if show:
        plt.show()

def plot2d(X, n_cols, skip_eigenvecs = {}, colors = None, show=True):
    assert n_cols <= X.shape[1]

    for i in range(n_cols):
        if i in skip_eigenvecs:
            continue

        for j in range(i+1, n_cols):
            if j in skip_eigenvecs:
                continue

            fig1, ax1 = plt.subplots()
            if colors is not None:
                for color in np.unique(colors):
                  color_X = X[colors == color, :]
                  label = ('original' if color == 'blue' else 'recon')
                  ax1.scatter(color_X[:,i], color_X[:,j], c=colors[colors == color], label=label, cmap='Spectral')
            else:
                ax1.scatter(X[:,i], X[:,j])

            # used for finding the eigenvalue outliers, prob a better way of doing this?
            # for k in range(X.shape[0]):
            #     ax1.annotate(k, (X[k,i], X[k,j]))

            ax1.set_xlabel(f'Eigenvec {i}')
            ax1.set_ylabel(f'Eigenvec {j}')
            ax1.legend()

    plt.tight_layout()

    if show:
        plt.show()


def runDiffMapAndPlotPairs(X, epsilon, k, group_indices, recon_X, max_plots=7):
    n_evecs = len(np.unique(y))

    mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs = n_evecs, epsilon=epsilon, alpha=1, k=k)
    mydmap.fit(X)

    # blue is original, red is reconstructed
    coords = np.concatenate((mydmap.dmap[group_indices, :], mydmap.transform(recon_X[group_indices, :])))
    colors = np.concatenate((['blue' for x in range(len(group_indices))], ['red' for x in range(len(group_indices))]))

    print(coords.shape)

    # plot the diffusion embeddings
    plot2d(coords, n_cols=np.min([n_evecs, max_plots]), colors=colors, show=True)


def runPCAAndPlotPairs(X, group_indices, recon_X, max_plots=7):

    # X = X[group_indices, :]
    # recon_X = recon_X[group_indices, :]

    n_evecs = np.min([X.shape[1], X.shape[0]])

    pca = PCA(n_components=n_evecs)
    pca.fit(X)

    # blue is original, red is reconstructed
    coords = np.concatenate((pca.transform(X[group_indices, :]), pca.transform(recon_X[group_indices, :])))
    colors = np.concatenate((['blue' for x in range(len(group_indices))], ['red' for x in range(len(group_indices))]))

    print(coords.shape)

    # plot the diffusion embeddings
    plot2d(coords, n_cols=np.min([n_evecs, max_plots]), colors=colors, show=False)


    total_explained_variance = 0
    eig_index = 0
    for i in range(len(pca.explained_variance_ratio_)):
      total_explained_variance += pca.explained_variance_ratio_[i]
      eig_index = i
      if (total_explained_variance > 0.90):
        break

    print(eig_index)
    print(total_explained_variance)

    fig1, ax1 = plt.subplots()
    ax1.plot(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, marker='o')
    ax1.set_title('PCA Eigenvalue')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Eigenvalue')
    plt.show()

def compareVariances(X, y, recon_X):

  for group in np.unique(y):
    group_indices = np.arange(len(y))[y == group]

    X_group = X[group_indices, :]
    recon_X_group = recon_X[group_indices, :]

    x_vars = np.var(X_group, axis=0)
    recon_vars = np.var(recon_X_group, axis=0)

    percent_data = (len(group_indices) / X.shape[0])

    fig1, ax1 = plt.subplots()
    ax1.hist(x_vars, bins=30, label='Var(X)')
    ax1.hist(recon_vars, bins=30, label='Var(reconstructed_X)', alpha=0.75)
    ax1.set_title('Class {group} ({percent:.2%}) Variance Differences Histogram'.format(group=group, percent=percent_data))
    ax1.set_xlabel('Var(X), Var(reconstructed_X)')
    ax1.set_ylabel('Count')
    ax1.legend()

  plt.tight_layout()
  plt.show()


def validateDiscriminator(X, y, recon_X):
  _, _, _, train_indices, _, test_indices = split_data_into_dataloaders(
      X,
      y,
      train_size=0.8,
      val_size=0,
  )

  split = int(np.floor(len(train_indices)/2))
  X_train = X[train_indices[:split], :]
  recon_X_train = X[train_indices[split:], :]

  merged_X_train = np.concatenate((X_train, recon_X_train))
  merged_y_train = np.concatenate((np.zeros(X_train.shape[0]), np.ones(recon_X_train.shape[0])))

  split = int(np.floor(len(test_indices)/2))
  X_test = X[test_indices[:split], :]
  recon_X_test = X[test_indices[split:], :]

  merged_X_test = np.concatenate((X_test, recon_X_test))
  merged_y_test = np.concatenate((np.zeros(X_test.shape[0]), np.ones(recon_X_test.shape[0])))

  print(int(np.sqrt(merged_X_train.shape[0])))

  knn_misclass, _, knn_cm = new_model_metrics(
    merged_X_train,
    merged_y_train,
    merged_X_test,
    merged_y_test,
    model=KNeighborsClassifier(n_neighbors=int(np.sqrt(merged_X_train.shape[0]))),
  )

  rf_misclass, _, rf_cm = new_model_metrics(
    merged_X_train,
    merged_y_train,
    merged_X_test,
    merged_y_test,
  )

  nn_misclass, _, nn_cm = new_model_metrics(
    merged_X_train,
    merged_y_train,
    merged_X_test,
    merged_y_test,
    model = MLPClassifier(hidden_layer_sizes=(20,), max_iter=100),
  )

  print('knn:', knn_misclass) #knn just predicts everything is original
  print('rf: ', rf_misclass) #perfectly predicts everything
  print('nn: ', nn_misclass) #close,

  plot_confusion_matrix(knn_cm, ['original', 'recon'])
  plot_confusion_matrix(rf_cm, ['original', 'recon'])
  plot_confusion_matrix(nn_cm, ['original', 'recon'])

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

if data_name == 'zeisel':
  X, y, encoder = getZeisel('data/zeisel/Zeisel.h5ad')
elif data_name == 'paul':
  X, y, encoder = getPaul('data/paul15/')
elif data_name == 'cite_seq':
  X, y, encoder = getCiteSeq('data/cite_seq/CITEseq.h5ad')
elif data_name == 'mouse_brain':
  X, y, encoder = getMouseBrain('data/mouse_brain_broad/')

# # we will use 80% training data, 20% vaidation data during the training of the marker map
# train_dataloader, val_dataloader, train_indices, val_indices = split_data_into_dataloaders_no_test(
#     X,
#     y,
#     train_size=0.8,
# )

# input_size = X.shape[1]
# unsupervised_marker_map = MarkerMap(input_size, hidden_layer_size, z_size, num_classes=None, k=k, loss_tradeoff=1)

# train_save_model(unsupervised_marker_map, train_dataloader, val_dataloader, save_model, 50, 600)

unsupervised_marker_map = load_model(MarkerMap, save_model)

recon_X = unsupervised_marker_map.get_reconstruction(X)


# group_indices = np.arange(len(y))[y == 0]
# print(len(group_indices))
# runPCAAndPlotPairs(X, group_indices, recon_X, max_plots=4)
# runDiffMapAndPlotPairs(X, 800, 10, group_indices, recon_X)

# compareVariances(X, y, recon_X)

validateDiscriminator(X, y, recon_X)
# testPCAProjectionDifference(100)
