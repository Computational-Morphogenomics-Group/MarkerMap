import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

from sklearn.preprocessing import LabelEncoder
from pydiffmap import diffusion_map, visualization


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

def getSwissRoll():
    # set parameters
    length_phi = 15   #length of swiss roll in angular direction
    length_Z = 15     #length of swiss roll in z direction
    sigma = 0.1       #noise strength
    m = 3000         #number of samples

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

    mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs=n_evecs, k=40, epsilon='bgh', alpha=1.0, neighbor_params=neighbor_params)
    # fit to data and return the diffusion map.
    mydmap.fit(swiss_roll)
    print(mydmap.epsilon_fitted)

    # plot the diffusion embeddings
    plot2d(mydmap.dmap, n_cols=n_evecs, colors=mydmap.dmap[:,0], show=False)

    # plot the data in the original dimensions
    plot3d(mydmap.data, n_cols=3, colors=mydmap.dmap[:,0], show=False)
    plot3d(mydmap.data, n_cols=3, colors=mydmap.dmap[:,1], show=False)
    plot3d(mydmap.data, n_cols=3, colors=mydmap.dmap[:,2], show=False)
    plot3d(mydmap.data, n_cols=3, colors=mydmap.dmap[:,3], show=False)


    plt.show()

    # visualization.embedding_plot(mydmap, scatter_kwargs = {'c': dmap[:,0], 'cmap': 'Spectral'}, show=False)
    # visualization.data_plot(mydmap, dim=3, scatter_kwargs = {'cmap': 'Spectral'}, show=False)

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

            ax1.set_xlabel(f'Eigenvec {i}')
            ax1.set_ylabel(f'Eigenvec {j}')

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

    if show:
        plt.show()


# MAIN
n_evecs = 5

# X,y, _ = getZeisel('data/zeisel/Zeisel.h5ad')
X = getSwissRoll()

exit()

dmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs = n_evecs, epsilon='bgh', alpha=0.5, k=10)

dmap_x = dmap.fit(X)

# skip_eigenvecs = { 1, 5, 8 } #zeisel
# skip_eigenvecs = { 0, 1, 2 } #swiss roll

plot2d(dmap_x, n_evecs)

# print(dmap_x.shape)

# # dmap_x = dmap_x[:, [0,2,3,4,6]]

# # visualization.embedding_plot(dmap, dim=3, scatter_kwargs={'c': y})
# #0,3,4,6 seem good for this
# print(np.unique(y))


# for i in range(n_evecs):
#     if i in skip_eigenvecs:
#         continue

#     for j in range(i+1, n_evecs):
#         if j in skip_eigenvecs:
#             continue

#         fig1, ax1 = plt.subplots()
#         print(i,j)
#         # fig1, ax1 = plt.subplots(subplot_kw={'projection':'3d'})
#         # ax1.scatter(dmap_x[:,i], dmap_x[:,j], dmap_x[:,k], c=y)
#         ax1.scatter(dmap_x[:,i], dmap_x[:,j], c=y)
#         ax1.set_xlabel(f'Eigenvec {i}')
#         ax1.set_ylabel(f'Eigenvec {j}')
#         # ax1.set_zlabel(f'Eigenvec {k}')


# plt.show()

