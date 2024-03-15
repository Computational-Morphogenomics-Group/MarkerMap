from functools import partial
import numpy as np
import pandas as pd
import itertools as it

import anndata
import torch
import scanpy as sc

from lassonet import LassoNetClassifier
from RankCorr.rocks import Rocks
import cosg
import persist

class BenchmarkableModel():
    @classmethod
    def getBenchmarker(cls, create_kwargs={}, train_kwargs={}):
        """
        Returns a function used by the the benchmarker to intialize and train model, then return markers
        args:
            create_kwargs (dict): ALL args used by the model constructor as a keyword arg dictionary
            train_args (dict): ALL args used by the train model step as a keyword arg dictionary
        """
        return partial(cls.benchmarkerFunctional, create_kwargs, train_kwargs)

    @classmethod
    def prepareData(cls, adata, train_indices, val_indices, group_by=None, layer=None):
        """
        Splits adata into train data and validation data based on provided indices. 
        args:
            adata (AnnData object): the data, including input and labels
            train_indices (array-like): the indices to be used as the training set
            val_indices (array-like): the indices to be used as the validation set
            group_by (string): key for where the label data is in .obs, or None to use X data
            layer (string): key for where the input data is in .layers, or None to use X data
        """
        if layer is None:
            X_train = adata[train_indices,:].X
            X_val = adata[val_indices,:].X
        else:
            X_train = adata[train_indices,:].layers[layer]
            X_val = adata[val_indices,:].layers[layer]

        if group_by is None:
            y_train = adata[train_indices,:].X
            y_val = adata[val_indices,:].X
        else:
            y_train = adata[train_indices,:].obs[group_by].cat.codes
            y_val = adata[val_indices,:].obs[group_by].cat.codes
    
        return X_train, y_train, X_val, y_val, adata

class RandomBaseline(BenchmarkableModel):
    """
    Model that just returns a random set of indices, used as a baseline for benchmarking purposes
    """

    @classmethod
    def benchmarkerFunctional(
        cls,
        create_kwargs,
        train_kwargs,
        adata,
        group_by,
        batch_size,
        train_indices,
        val_indices,
        k=None,
    ):
        """
        Class function that initializes, trains, and returns markers for the provided data with the specific params
        args:
            cls (string): The current, derived class name, used for calling derived class functions
            create_kwargs (dict): ALL args used by the model constructor as a keyword arg dictionary
            train_args (dict): ALL args used by the train model step as a keyword arg dictionary
            adata (AnnData object): input and label data
            group_by (string): string key for adata.obs[group_by] where the output labels live
            batch_size (int): batch size for models that use batches
            train_indices (array-like): the indices to be used as the training set
            val_indices (array-like): the indices to be used as the validation set
            k (int): k value for the model, the number of markers to select
        returns:
            (np.array) the selected k markers
        """
        if k is None:
            k = {**create_kwargs, **train_kwargs }['k']

        return np.random.permutation(range(adata.shape[1]))[:k]


class LassoNetWrapper(LassoNetClassifier, BenchmarkableModel):
    """
    Thin wrapper on the LassoNetClassifier that implements the BenchmarkableModel functionality
    """

    @classmethod
    def benchmarkerFunctional(
        cls,
        create_kwargs,
        train_kwargs,
        adata,
        group_by,
        batch_size,
        train_indices,
        val_indices,
        k=None,
    ):
        """
        Class function that initializes, trains, and returns markers for the provided data with the specific params
        args:
            cls (string): The current, derived class name, used for calling derived class functions
            create_kwargs (dict): ALL args used by the model constructor as a keyword arg dictionary
            train_args (dict): ALL args used by the train model step as a keyword arg dictionary
            adata (AnnData object): input and label data
            group_by (string): string key for adata.obs[group_by] where the output labels live
            batch_size (int): batch size for models that use batches
            train_indices (array-like): the indices to be used as the training set
            val_indices (array-like): the indices to be used as the validation set
            k (int): k value for the model, the number of markers to select
        returns:
            (np.array) the selected k markers
        """
        if not k:
            k = train_kwargs['k']

        X_train, y_train, X_val, y_val, adata = cls.prepareData(adata, train_indices, val_indices, group_by)

        model = LassoNetClassifier(**create_kwargs)
        model.path(X_train, y_train, X_val = X_val, y_val = y_val)
        return torch.argsort(model.feature_importances_, descending = True).cpu().numpy()[:k]


class RankCorrWrapper(Rocks, BenchmarkableModel):
    """
    Thin wrapper on the RankCorr package Rocks class that also implements the Benchmarkable Model functionality
    """

    @classmethod
    def prepareData(cls, adata, train_indices, group_by):
        """
        Splits adata into input data and label data for the train provided indices. 
        args:
            adata (AnnData object): the data, including input and labels
            train_indices (array-like): the indices to be used as the training set
            group_by (string): key for where the label data is in .obs
        """
        return adata[train_indices,:].X, adata[train_indices,:].obs[group_by].cat.codes.to_numpy()

    @classmethod
    def benchmarkerFunctional(
        cls,
        create_kwargs,
        train_kwargs,
        adata,
        group_by,
        batch_size,
        train_indices,
        val_indices,
        k=None,
    ):
        """
        Class function that initializes, trains, and returns markers for the provided data with the specific params
        args:
            cls (string): The current, derived class name, used for calling derived class functions
            create_kwargs (dict): ALL args used by the model constructor as a keyword arg dictionary
            train_args (dict): ALL args used by the train model step as a keyword arg dictionary
            adata (AnnData object): input and label data
            group_by (string): string key for adata.obs[group_by] where the output labels live
            batch_size (int): batch size for models that use batches
            train_indices (array-like): the indices to be used as the training set
            val_indices (array-like): the indices to be used as the validation set
            k (int): k value for the model, the number of markers to select
        returns:
            (np.array) the selected k markers
        """
        if not k:
            k = train_kwargs['k']

        if 'k' in train_kwargs:
            train_kwargs = { **train_kwargs } #copy train_kwargs so later iterations have 'k'
            train_kwargs.pop('k')

        X_train, y_train = cls.prepareData(adata, np.concatenate([train_indices, val_indices]), group_by)
        model = cls(X_train, y_train, **create_kwargs)
        markers = model.CSrankMarkers(**train_kwargs)

        if len(markers) < k:
            print(
                f'RankCorrWrapper::benchmarkerFunctional: Tried to find {k} markers, only found {len(markers)},'
                ' increase lamb parameter in train_kwargs.',
            )
        if len(markers) > k:
            markers = markers[:k]

        return markers


# Below this are AnnDataModel and models that inherit from AnnDataModel
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class AnnDataModel(BenchmarkableModel):
    """
    BenchmarkableModel that expect the data as an AnnData object. 
    """

    @classmethod
    def prepareData(cls, adata, train_indices, val_indices):
        """
        Restricts adata to one block with all the train and val indices.
        args:
            adata (AnnData object): the data, including input and labels
            train_indices (array-like): the indices to be used as the training set
            val_indices (array-like): the indices to be used as the validation set
        """
        return adata[np.concatenate([train_indices, val_indices]),:]

class ScanpyRankGenes(AnnDataModel):

    @classmethod
    def prepareData(cls, adata, train_indices, val_indices, group_by):
        """
        Reduces adata to those with train_indices and val_indices
        Splits adata into train data and validation data based on provided indices. 
        args:
            adata (AnnData object): the data, including input and labels
            train_indices (array-like): the indices to be used as the training set
            val_indices (array-like): the indices to be used as the validation set
            group_by (string): key for where the label data is in .obs
        """
        adata.obs[group_by + '_codes'] = pd.Categorical(adata.obs[group_by].cat.codes)
        return super().prepareData(adata, train_indices, val_indices)

    @classmethod
    def benchmarkerFunctional(
        cls,
        create_kwargs,
        train_kwargs,
        adata,
        group_by,
        batch_size,
        train_indices,
        val_indices,
        k=None,
    ):
        """
        Class function that initializes, trains, and returns markers for the provided data with the specific params
        args:
            cls (string): The current, derived class name, used for calling derived class functions
            create_kwargs (dict): ALL args used by the model constructor as a keyword arg dictionary
            train_args (dict): ALL args used by the train model step as a keyword arg dictionary
            adata (AnnData object): input and label data
            group_by (string): string key for adata.obs[group_by] where the output labels live
            batch_size (int): batch size for models that use batches
            train_indices (array-like): the indices to be used as the training set
            val_indices (array-like): the indices to be used as the validation set
            k (int): k value for the model, the number of markers to select
        returns:
            (np.array) the selected k markers
        """
        method = train_kwargs['method'] if 'method' in train_kwargs else 't-test'
        tie_correct = train_kwargs['tie_correct'] if 'tie_correct' in train_kwargs else False

        if k is None:
            k = train_kwargs['k']

        adata = cls.prepareData(adata, train_indices, val_indices, group_by)

        # First, run rank_genes_groups with a high enough n_genes that we find >= k unique genes
        unique_names = np.array([])
        multiplier = 2
        while len(unique_names) < k: #unlikely this will run more than once, but just in case
            n_genes = (k // train_kwargs['num_classes']) * multiplier

            adata_with_markers = sc.tl.rank_genes_groups(
                adata, 
                group_by + '_codes', 
                n_genes=n_genes, 
                method=method,
                tie_correct=tie_correct,
                copy=True,
            )
            names = list(it.chain(*adata_with_markers.uns['rank_genes_groups']['names']))
            unique_names = np.unique(np.array([adata_with_markers.to_df().columns.get_loc(name) for name in names], dtype=int))
            multiplier *= 2

        # add the genes by row until it would put us over budget k, at which point add ones with
        # lowest pvals
        genes = np.array([], dtype=int)
        i = 0
        while len(genes) < k:
            gene_row_names = list(adata_with_markers.uns['rank_genes_groups']['names'][i])
            gene_row = np.array([adata_with_markers.to_df().columns.get_loc(name) for name in gene_row_names], dtype=int)
            genes_added = np.unique(np.concatenate([genes, gene_row]))
            if len(genes_added) <= k:
                genes = genes_added
            else:
                for idx in np.argsort(np.array(list(adata_with_markers.uns['rank_genes_groups']['pvals'][i]))):
                    if len(genes) == k:
                        break

                    if gene_row[idx] not in genes:
                        genes = np.append(genes, gene_row[idx])

            i += 1

        assert len(genes) == k

        return genes
    
class ScanpyHVGs(AnnDataModel):
    # This is the unsupervised method in scanpy that PERSIST tests against, highly_variable_genes

    @classmethod
    def prepareData(cls, adata, train_indices, val_indices, group_by, layer='log'):
        """
        Reduces adata to those with train_indices and val_indices
        Splits adata into train data and validation data based on provided indices. 
        args:
            adata (AnnData object): the data, including input and labels
            train_indices (array-like): the indices to be used as the training set
            val_indices (array-like): the indices to be used as the validation set
            group_by (string): key for where the label data is in .obs
        """
        adata = super().prepareData(adata, train_indices, val_indices) # select the indices of train and val
        return anndata.AnnData(X=adata.layers[layer].copy()) # use the data in the layer

    @classmethod
    def benchmarkerFunctional(
        cls,
        create_kwargs,
        train_kwargs,
        adata,
        group_by,
        batch_size,
        train_indices,
        val_indices,
        k=None,
    ):
        """
        Class function that initializes, trains, and returns markers for the provided data with the specific params
        args:
            cls (string): The current, derived class name, used for calling derived class functions
            create_kwargs (dict): ALL args used by the model constructor as a keyword arg dictionary
            train_args (dict): ALL args used by the train model step as a keyword arg dictionary
            adata (AnnData object): input and label data
            group_by (string): string key for adata.obs[group_by] where the output labels live
            batch_size (int): batch size for models that use batches
            train_indices (array-like): the indices to be used as the training set
            val_indices (array-like): the indices to be used as the validation set
            k (int): k value for the model, the number of markers to select
        returns:
            (np.array) the selected k markers
        """
        if k is None:
            k = train_kwargs['k']

        adata_train = cls.prepareData(adata, train_indices, val_indices, group_by, layer='log')

        annotations = sc.pp.highly_variable_genes(adata_train, n_top_genes=k, inplace=False)
        markers = np.where(annotations['highly_variable'])[0]

        assert len(markers) == k

        return markers
    
class COSGWrapper(AnnDataModel):

    @classmethod
    def benchmarkerFunctional(
        cls,
        create_kwargs,
        train_kwargs,
        adata,
        group_by,
        batch_size,
        train_indices,
        val_indices,
        k=None,
    ):
        """
        Class function that initializes, trains, and returns markers for the provided data with the specific params
        args:
            cls (string): The current, derived class name, used for calling derived class functions
            create_kwargs (dict): ALL args used by the model constructor as a keyword arg dictionary
            train_args (dict): ALL args used by the train model step as a keyword arg dictionary
            adata (AnnData object): input and label data
            group_by (string): string key for adata.obs[group_by] where the output labels live
            batch_size (int): batch size for models that use batches
            train_indices (array-like): the indices to be used as the training set
            val_indices (array-like): the indices to be used as the validation set
            k (int): k value for the model, the number of markers to select
        returns:
            (np.array) the selected k markers
        """

        if k is None:
            k = train_kwargs['k']

        train_adata = cls.prepareData(adata, train_indices, val_indices)

        # First, run rank_genes_groups with a high enough n_genes that we find >= k unique genes
        unique_names = np.array([])
        multiplier = 2
        while len(unique_names) < k: #unlikely this will run more than once, but just in case
            n_genes = (k // train_kwargs['num_classes']) * multiplier

            cosg.cosg(
                train_adata, 
                group_by, 
                n_genes_user=n_genes, 
                key_added='cosg',
            )
            names = list(it.chain(*train_adata.uns['cosg']['names']))
            unique_names = np.unique(np.array([train_adata.to_df().columns.get_loc(name) for name in names], dtype=int))
            multiplier *= 2

        # add the genes by row until it would put us over budget k, at which point add ones with highest scores
        genes = np.array([], dtype=int)
        i = 0
        while len(genes) < k:
            gene_row_names = list(train_adata.uns['cosg']['names'][i])
            gene_row = np.array([train_adata.to_df().columns.get_loc(name) for name in gene_row_names], dtype=int)
            genes_added = np.unique(np.concatenate([genes, gene_row]))
            if len(genes_added) <= k:
                genes = genes_added
            else:
                for idx in np.argsort(-np.array(list(train_adata.uns['cosg']['scores'][i]))): # higher scores are more relevant
                    if len(genes) == k:
                        break

                    if gene_row[idx] not in genes:
                        genes = np.append(genes, gene_row[idx])

            i += 1

        assert len(genes) == k

        return genes
    
class PersistWrapper(BenchmarkableModel):

    @classmethod
    def prepareData(cls, adata, train_indices, val_indices, group_by, layer='bin'):
        """
        Add the binarized counts data to adata, then splits adata into train data and validation 
        data based on provided indices. 
        args:
            adata (AnnData object): the data, including input and labels
            train_indices (array-like): the indices to be used as the training set
            val_indices (array-like): the indices to be used as the validation set
            group_by (string): key for where the label data is in .obs, or None to use X data
        """
        try:
            adata.layers['bin'] = (adata.layers['counts'] > 0).astype(np.float32)
        except KeyError:
            raise Exception(
                'PersistWrapper::prepareData requires "counts" layer data to binarize.',
            )

        return super(PersistWrapper, cls).prepareData(adata, train_indices, val_indices, group_by, layer=layer)

    @classmethod
    def benchmarkerFunctional(
        cls,
        create_kwargs,
        train_kwargs,
        adata,
        group_by,
        batch_size,
        train_indices,
        val_indices,
        k=None,
    ):
        """
        Class function that initializes, trains, and returns markers for the provided data with the specific params
        args:
            cls (string): The current, derived class name, used for calling derived class functions
            create_kwargs (dict): ALL args used by the model constructor as a keyword arg dictionary
            train_args (dict): ALL args used by the train model step as a keyword arg dictionary
            adata (AnnData object): input and label data
            group_by (string): string key for adata.obs[group_by] where the output labels live
            batch_size (int): batch size for models that use batches
            train_indices (array-like): the indices to be used as the training set
            val_indices (array-like): the indices to be used as the validation set
            k (int): k value for the model, the number of markers to select
        returns:
            (np.array) the selected k markers
        """
        if create_kwargs['supervised']:
            loss_fn = torch.nn.CrossEntropyLoss()
        else: #unsupervised
            group_by = None # use normalized and log transformed data as the reconstruction target
            loss_fn = persist.HurdleLoss()

        layer = 'bin'
        if 'use_bin' in create_kwargs:
            layer = 'bin' if create_kwargs['use_bin'] else None

        X_train, y_train, X_val, y_val, adata = cls.prepareData(adata, train_indices, val_indices, group_by, layer)

        if k is None:
            k = train_kwargs['k']
        
        # Initialize the dataset for PERSIST
        train_dataset = persist.ExpressionDataset(X_train, y_train)
        val_dataset = persist.ExpressionDataset(X_val, y_val)

        # Use GPU device if available -- we highly recommend using a GPU!
        device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')

        # Set up the PERSIST selector
        selector = persist.PERSIST(train_dataset, val_dataset, loss_fn=loss_fn, device=device)

        # Coarse removal of genes. This can fail, so if it fails to find after 10 tries, just do the select step
        # We also use the max allowed tolerance to hopeful help with this step
        target = k * 10
        
        if train_kwargs['eliminate_step'] and target < (0.5*X_train.shape[1]): 
            max_trials = train_kwargs['max_trials'] if 'max_trials' in train_kwargs else 10 # method default
            try:
                selector.eliminate(
                    target=k*10, 
                    max_nepochs=train_kwargs['eliminate_nepochs'], 
                    tol=0.49, 
                    max_trials=max_trials,
                )
            except ValueError:
                pass

        markers, _ = selector.select(num_genes=k, max_nepochs=250)
        return markers
