import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import math
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from torch import nn

from pydiffmap import diffusion_map, visualization
from markermap.utils import new_model_metrics, parse_adata, RandomBaseline, SmashPyWrapper, split_data_into_dataloaders, form_block


def getMattSet():
    X = np.random.multivariate_normal([0,0],[[0.75,0],[0,0.75]],300)
    y = []
    while len(y) < 300:
        k = np.random.multivariate_normal([0,0],[[20,0],[0,20]]).tolist()
        if k[0]**2 + k[1]**2 <= 7 or k[0]**2 >= 10:
            continue
        else:
            y.append(k)
    y = np.array(y)

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

    # plot the diffusion embeddings
    # plot2d(mydmap.dmap, n_cols=n_evecs, colors=mydmap.dmap[:,0], show=False)

    # plot the data in the original dimensions
    # plot3d(mydmap.data, n_cols=3, colors=mydmap.dmap[:,0], show=True)

    return swiss_roll, mydmap.dmap[:,0], 0.0156, 100

    # return swiss_roll


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

            # fig1.savefig(f'/Users/wilson/Documents/jhu/spring_2022/highD_approx/final_project/citeseq_evec_pairs/citeseq_evecs_{i}_{j}')

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
    # epsilon = X.shape[1]/5
    epsilon = 128
    k = 10

    return X, y, epsilon, k

def getCiteSeqData():
    adata = sc.read_h5ad('data/cite_seq/CITEseq.h5ad')
    adata.obs['annotation'] = adata.obs['names']
    X, y, encoder = parse_adata(adata)
    print(X.shape)

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


def getSkreePlots(X, y, epsilon, k, n_evecs):
    mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs = n_evecs, epsilon=epsilon, alpha=1, k=k)
    mydmap.fit(X)

    pca = PCA(n_components=np.min([n_evecs, X.shape[1]]))
    X_pca = pca.fit_transform(X)

    diff_map_evals = np.sqrt(-1. / mydmap.evals)

    fig1, ax1 = plt.subplots()
    ax1.plot(range(n_evecs), diff_map_evals[:n_evecs], marker='o')
    ax1.set_title('Diffusion Map Eigenvalues')

    print(pca.explained_variance_)

    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(pca.explained_variance_)), pca.explained_variance_, marker='o')
    ax2.set_title('PCA Eigenvalue')
    plt.show()



def evaluateDiffMapClassification(X, y, epsilon, k, eval_models, num_times=1):
    n_evecs = len(np.unique(y))

    print(X.shape)
    print(y.shape)
    print(epsilon)
    print(k)
    print(n_evecs)

    mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs = n_evecs, epsilon=epsilon, alpha=1, k=k)
    mydmap.fit(X)

    print('here')

    pca = PCA(n_components=n_evecs)
    X_pca = pca.fit_transform(X)

    results = { key: np.zeros((3,num_times)) for key in eval_models.keys() }
    for i in range(num_times):
        _, _, _, train_indices, _, test_indices = split_data_into_dataloaders(X,y, 0.8, 0) # train 80%, val 0%, test 20%
        print(i)

        for model_label, eval_model in eval_models.items():

            # Diffusion map eval
            misclass_rate, _, _ = new_model_metrics(
                mydmap.dmap[train_indices, :],
                y[train_indices],
                mydmap.dmap[test_indices, :],
                y[test_indices],
                model = eval_model,
            )
            results[model_label][0,i] = misclass_rate

            # All original features baseline
            baseline_misclass, _, _ = new_model_metrics(
                X[train_indices, :],
                y[train_indices],
                X[test_indices, :],
                y[test_indices],
                model = eval_model,
            )
            results[model_label][1,i] = baseline_misclass

            # PCA baseline
            pca_misclass, _, _ = new_model_metrics(
                X_pca[train_indices, :],
                y[train_indices],
                X_pca[test_indices, :],
                y[test_indices],
                model = eval_model,
            )
            results[model_label][2,i] = pca_misclass

    for k,v in results.items():
        print(k, np.mean(results[k], axis=1))


def runDiffMapAndPlotPairs(X, y, epsilon, k, max_plots=7):
    n_evecs = len(np.unique(y))

    mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs = n_evecs, epsilon=epsilon, alpha=1, k=k)
    mydmap.fit(X)

    # plot the diffusion embeddings
    plot2d(mydmap.dmap, n_cols=np.min([n_evecs, max_plots]), colors=y, show=True)


def findDiffMapKValue(X, y, epsilon, k_range, display_time=False):
    _, _, _, train_indices, _, test_indices = split_data_into_dataloaders(X,y, 0.8, 0) # train 80%, val 0%, test 20%
    # test for different values of k
    n_evecs = len(np.unique(y))

    misclass_rates = []
    runtime = []
    for k in k_range:
        print('k:', k)

        mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs = n_evecs, epsilon=epsilon, alpha=1, k=k)
        start_time = time.time()
        mydmap.fit(X)
        end_time = time.time() - start_time
        runtime.append(end_time)

        misclass_rate, _, _ = new_model_metrics(
            mydmap.dmap[train_indices, :],
            y[train_indices],
            mydmap.dmap[test_indices, :],
            y[test_indices],
            # markers=n_evecs,
        )
        misclass_rates.append(misclass_rate)

    baseline_misclass, _, _ = new_model_metrics(X[train_indices, :], y[train_indices], X[test_indices, :], y[test_indices])

    # PCA baseline
    pca = PCA(n_components=n_evecs)
    X_pca = pca.fit_transform(X)
    pca_misclass, _, _ = new_model_metrics(
        X_pca[train_indices, :],
        y[train_indices],
        X_pca[test_indices, :],
        y[test_indices],
    )

    fig1, ax1 = plt.subplots()
    ax1.plot(k_range, misclass_rates, label='diff_map', marker='o')
    ax1.hlines(baseline_misclass, k_range[0], k_range[-1], label='baseline', colors=['red'])
    ax1.hlines(pca_misclass, k_range[0], k_range[-1], label='pca', colors=['orange'])
    ax1.legend()

    if display_time:
        fig2, ax2 = plt.subplots()
        ax2.plot(k_range, runtime, label='runtime', marker='o')
        ax2.legend()

    plt.tight_layout()
    plt.show()


def findDiffMapEpsilon(X, y, epsilon_range, k, num_times=1, display_time=False):
    # test for different values of k
    n_evecs = len(np.unique(y))

    runtime = []
    misclass_rates = []

    #First, train all dmaps over the epsilon range
    dmaps = {}
    for epsilon in epsilon_range:
        print('epsilon:', epsilon)

        mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs = n_evecs, epsilon=epsilon, alpha=1, k=k)
        start_time = time.time()
        mydmap.fit(X)
        end_time = time.time() - start_time
        runtime.append(end_time)
        dmaps[epsilon] = mydmap

    # PCA baseline
    pca = PCA(n_components=n_evecs)
    X_pca = pca.fit_transform(X)

    baseline_rates = []
    pca_rates = []
    diff_map_rates = np.zeros((num_times, len(epsilon_range)))
    for i in range(num_times):
        _, _, _, train_indices, _, test_indices = split_data_into_dataloaders(X,y, 0.8, 0) # train 80%, val 0%, test 20%

        for idx, epsilon in enumerate(epsilon_range):
            misclass_rate, _, _ = new_model_metrics(
                dmaps[epsilon].dmap[train_indices, :],
                y[train_indices],
                dmaps[epsilon].dmap[test_indices, :],
                y[test_indices],
                # markers=n_evecs,
            )
            diff_map_rates[i,idx] = misclass_rate

        baseline_misclass, _, _ = new_model_metrics(X[train_indices, :], y[train_indices], X[test_indices, :], y[test_indices])
        baseline_rates.append(baseline_misclass)

        pca_misclass, _, _ = new_model_metrics(
            X_pca[train_indices, :],
            y[train_indices],
            X_pca[test_indices, :],
            y[test_indices],
        )
        pca_rates.append(pca_misclass)

    print(diff_map_rates.shape)
    print(diff_map_rates)

    fig1, ax1 = plt.subplots()
    ax1.plot(epsilon_range, np.mean(diff_map_rates, axis=0), label='diff_map', marker='o')
    ax1.hlines(np.mean(baseline_rates), epsilon_range[0], epsilon_range[-1], label='baseline', colors=['red'])
    ax1.hlines(np.mean(pca_rates), epsilon_range[0], epsilon_range[-1], label='pca', colors=['orange'])
    ax1.set_xlabel('Epsilon')
    ax1.set_ylabel('Misclass Rate')
    ax1.legend()

    if display_time:
        fig2, ax2 = plt.subplots()
        ax2.plot(epsilon_range, runtime, label='runtime', marker='o')
        ax2.set_xlabel('Epsilon')
        ax2.set_ylabel('Time')
        ax2.legend()

    plt.tight_layout()
    plt.show()


# MAIN
# Zeisel
k_range = [5,10,20,30,50,100,200,500,1000,2000,3000]
epsilon_range = [128,200,400,800,1000,1500,2000,4000]
X, y, epsilon, k = getZeiselData()

# # Cite-Seq
# k_range = [5,10,20,30,50,100,200,500,1000,3000,5000,8000]
# epsilon_range = [32,64,80,100,150,200,300,500,750,1000]
# X, y, epsilon, k = getCiteSeqData()

# Paul
# k_range = [15,20,25,30,50,100,200,300,400,500]
# X, y, epsilon, k = getPaulData()

# Swiss Roll
# X, y, epsilon, k = getSwissRoll()

eval_models = {
    'knn': KNeighborsClassifier(),
    'rf': RandomForestClassifier(),
    'neural network': MLPClassifier(max_iter=1000),
}

# runDiffMapAndPlotPairs(X, y, epsilon, k, max_plots=7)

# evaluateDiffMapClassification(X, y, epsilon, k, num_times=1, eval_models=eval_models)

# findDiffMapKValue(X, y, epsilon, k_range, display_time=True)
findDiffMapEpsilon(X, y, epsilon_range, k, num_times=5)

# getSkreePlots(X, y, epsilon, k, n_evecs=100)


