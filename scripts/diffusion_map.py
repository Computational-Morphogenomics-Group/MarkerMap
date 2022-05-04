import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import math

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from torch import nn

from pydiffmap import diffusion_map, visualization
from markermap.utils import new_model_metrics, parse_adata, RandomBaseline, SmashPyWrapper, split_data_into_dataloaders, form_block


def getSwissRoll():
    # set parameters
    length_phi = 15   #length of swiss roll in angular direction
    length_Z = 15     #length of swiss roll in z direction
    sigma = 0.1       #noise strength
    m = 10000         #number of samples

    # create dataset
    phi = length_phi*np.random.rand(m)
    xi = np.random.rand(m)
    Z = length_Z*np.random.rand(m)
    X = 1./6*(phi + sigma*xi)*np.sin(phi)
    Y = 1./6*(phi + sigma*xi)*np.cos(phi)

    swiss_roll = np.array([X, Y, Z]).transpose()

    # check that we have the right shape
    print(swiss_roll.shape)

    # initialize Diffusion map object.
    neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}
    n_evecs = 4

    mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs=n_evecs, k=100, epsilon=0.0156, alpha=1.0, neighbor_params=neighbor_params)
    # fit to data and return the diffusion map.
    mydmap.fit(swiss_roll)
    print(mydmap.epsilon_fitted)

    # plot the diffusion embeddings
    plot2d(mydmap.dmap, n_cols=n_evecs, colors=mydmap.dmap[:,0], show=False)

    # plot the data in the original dimensions
    plot3d(mydmap.data, n_cols=3, colors=mydmap.dmap[:,0], show=True)

    return swiss_roll


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
                ax1.scatter(X[:,i], X[:,j], c=colors, cmap='Spectral')
            else:
                ax1.scatter(X[:,i], X[:,j])

            # used for finding the eigenvalue outliers, prob a better way of doing this?
            # for k in range(X.shape[0]):
            #     ax1.annotate(k, (X[k,i], X[k,j]))

            ax1.set_xlabel(f'Eigenvec {i}')
            ax1.set_ylabel(f'Eigenvec {j}')

    plt.tight_layout()

    if show:
        plt.show()


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


def getZeiselData():
    adata = sc.read_h5ad('data/zeisel/Zeisel.h5ad')
    adata.obs['annotation'] = adata.obs['names0']
    X, y, encoder = parse_adata(adata)
    print(X.shape)

    # epsilon = 256 # this is what BGH finds
    epsilon = X.shape[1]/5
    k = 10

    return X, y, epsilon, k

def getCiteSeqData():
    adata = sc.read_h5ad('data/cite_seq/CITEseq.h5ad')
    adata.obs['annotation'] = adata.obs['names']
    X, y, encoder = parse_adata(adata)

    # remove some outliers
    outliers = [1066, 779, 8458]
    X = np.delete(X, outliers, axis=0)
    y = np.delete(y, outliers)

    # epsilon = 64 # given by BGH
    epsilon = X.shape[1]/5
    k = 100

    return X, y, epsilon, k


def getPaulData():
    adata = sc.datasets.paul15()
    sm = SmashPyWrapper()
    sm.data_preparation(adata)
    adata = sm.remove_general_genes(adata)
    adata = sm.remove_housekeepingenes(adata, path=['data/paul15/house_keeping_genes_Mouse_bone_marrow.txt'])
    adata = sm.remove_housekeepingenes(adata, path=['data/paul15/house_keeping_genes_Mouse_HSC.txt'])
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

    X,y, _ = parse_adata(adata)
    print(X.shape)

    # epsilon = X.shape[1]/5
    epsilon = 'bgh'
    k = 100

    return X, y, epsilon, k


def runDiffMapAndPlotPairs(X, y, epsilon, k, max_plots=7, eval_models = None):
    _, _, _, train_indices, _, test_indices = split_data_into_dataloaders(X,y, 0.8, 0) # train 80%, val 0%, test 20%
    n_evecs = len(np.unique(y))

    if not isinstance(eval_models, list):
        eval_models = [eval_models]

    for eval_model in eval_models:
        print(type(eval_model))

        # Diffusion map eval
        mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs = n_evecs, epsilon=epsilon, alpha=1, k=k)
        mydmap.fit(X)
        misclass_rate, _, _ = new_model_metrics(
            mydmap.dmap[train_indices, :],
            y[train_indices],
            mydmap.dmap[test_indices, :],
            y[test_indices],
            model = eval_model,
        )
        print(misclass_rate)

        # All original features baseline
        baseline_misclass, _, _ = new_model_metrics(
            X[train_indices, :],
            y[train_indices],
            X[test_indices, :],
            y[test_indices],
            model = eval_model,
        )
        print(baseline_misclass)

        # PCA baseline
        pca = PCA(n_components=n_evecs)
        X_pca = pca.fit_transform(X)
        pca_misclass, _, _ = new_model_metrics(
            X_pca[train_indices, :],
            y[train_indices],
            X_pca[test_indices, :],
            y[test_indices],
            model = eval_model,
        )
        print(pca_misclass)

    # plot the diffusion embeddings
    plot2d(mydmap.dmap, n_cols=np.min([n_evecs, max_plots]), colors=y, show=True)


def findDiffMapKValue(X, y, epsilon, k_range):
    _, _, _, train_indices, _, test_indices = split_data_into_dataloaders(X,y, 0.8, 0) # train 80%, val 0%, test 20%
    # test for different values of k
    n_evecs = len(np.unique(y))

    misclass_rates = { 'diff_map': [], 'pca': [] }
    for k in k_range:
        print('k:', k)

        mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs = n_evecs, epsilon=epsilon, alpha=1, k=k)
        mydmap.fit(X)

        misclass_rate, _, _ = new_model_metrics(
            mydmap.dmap[train_indices, :],
            y[train_indices],
            mydmap.dmap[test_indices, :],
            y[test_indices],
            # markers=n_evecs,
        )
        misclass_rates['diff_map'].append(misclass_rate)

        # PCA baseline
        pca = PCA(n_components=np.min([k,X.shape[1]]))
        X_pca = pca.fit_transform(X)
        pca_misclass, _, _ = new_model_metrics(
            X_pca[train_indices, :],
            y[train_indices],
            X_pca[test_indices, :],
            y[test_indices],
        )
        misclass_rates['pca'].append(pca_misclass)

        # mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs = n_evecs, epsilon=epsilon, alpha=0.5, k=k)
        # mydmap.fit(X)

        # misclass_rate, _, _ = new_model_metrics(
        #     mydmap.dmap[train_indices, :],
        #     y[train_indices],
        #     mydmap.dmap[test_indices, :],
        #     y[test_indices],
        #     # markers=n_evecs,
        # )
        # misclass_rates['alpha2'].append(misclass_rate)

    baseline_misclass, _, _ = new_model_metrics(X[train_indices, :], y[train_indices], X[test_indices, :], y[test_indices])

    fig1, ax1 = plt.subplots()
    ax1.scatter(k_range, misclass_rates['diff_map'], label='diff_map')
    ax1.scatter(k_range, misclass_rates['pca'], label='pca')
    ax1.hlines(baseline_misclass, k_range[0], k_range[-1])
    ax1.legend()
    plt.tight_layout()
    plt.show()

def makeNeuralNetworkClassifier(input_size, hidden_size, output_size):
    classifier = nn.Sequential(
        *form_block(input_size, hidden_size, bias = bias, batch_norm = True),
        nn.Linear(hidden_size, output_size, bias = bias),
    )




# MAIN
# eval_model = None
eval_model = [KNeighborsClassifier(), RandomForestClassifier(), MLPClassifier(max_iter=1000)]

# Zeisel
k_range = [5,10,15,20,25,30,50,100]
X, y, epsilon, k = getZeiselData()

# # Cite-Seq
# k_range = [20,30,50,100,200,500,1000,3000,5000,8000]
# X, y, epsilon, k = getCiteSeqData()

# Paul
# k_range = [15,20,25,30,50,100,200,300,400,500]
# X, y, epsilon, k = getPaulData()

runDiffMapAndPlotPairs(X, y, epsilon, k, max_plots=2, eval_models=eval_model)

# findDiffMapKValue(X, y, epsilon, k_range)


