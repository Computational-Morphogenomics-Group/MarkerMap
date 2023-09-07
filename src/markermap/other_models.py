import os
import contextlib
import queue
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it

import anndata
import torch
import scanpy as sc
from sklearn.ensemble import RandomForestClassifier

from lassonet import LassoNetClassifier
from smashpy import smashpy
from RankCorr.rocks import Rocks
import cosg


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

    def prepareData(X, y, train_indices, val_indices):
        """
        Sorts X and y data into train and val sets based on the provided indices
        args:
            X (np.array): input data, counts of various proteins
            y (np.array): output data, what type of cell it is
            train_indices (array-like): the indices to be used as the training set
            val_indices (array-like): the indices to be used as the validation set
        """
        X_train = X[train_indices, :]
        y_train = y[train_indices]
        X_val = X[val_indices, :]
        y_val = y[val_indices]

        return X_train, y_train, X_val, y_val


class RandomBaseline(BenchmarkableModel):
    """
    Model that just returns a random set of indices, used as a baseline for benchmarking purposes
    """

    @classmethod
    def benchmarkerFunctional(
        cls,
        create_kwargs,
        train_kwargs,
        X,
        y,
        train_indices,
        val_indices,
        train_dataloader,
        val_dataloader,
        **kwargs,
    ):
        """
        Class function that initializes, trains, and returns markers for the provided data with the specific params
        args:
            create_kwargs (dict): ALL args used by the model constructor as a keyword arg dictionary
            train_args (dict): ALL args used by the train model step as a keyword arg dictionary
            train_data ():
            val_data ():
            train_dataloader (pytorch dataloader): dataloader for training data set
            val_dataloader (pytorch dataloader): dataloader for validation data set
            k (int): k value for the model, the number of markers to select
        """
        all_kwargs = {**create_kwargs, **train_kwargs, **kwargs}
        return np.random.permutation(range(X.shape[1]))[:all_kwargs['k']]


class LassoNetWrapper(LassoNetClassifier, BenchmarkableModel):
    """
    Thin wrapper on the LassoNetClassifier that implements the BenchmarkableModel functionality
    """

    @classmethod
    def benchmarkerFunctional(
        cls,
        create_kwargs,
        train_kwargs,
        X,
        y,
        train_indices,
        val_indices,
        train_dataloader,
        val_dataloader,
        k=None,
    ):
        """
        Class function that initializes, trains, and returns markers for the provided data with the specific params
        args:
            cls (string): The current, derived class name, used for calling derived class functions
            create_kwargs (dict): ALL args used by the model constructor as a keyword arg dictionary
            train_args (dict): ALL args used by the train model step as a keyword arg dictionary
            X (np.array): the full set of training data input X
            y (np.array): the full set of training data output y
            train_indices (array-like): the indices to be used as the training set
            val_indices (array-like): the indices to be used as the validation set
            train_dataloader (pytorch dataloader): dataloader for training data set
            val_dataloader (pytorch dataloader): dataloader for validation data set
            k (int): k value for the model, the number of markers to select
        """
        if not k:
            k = train_kwargs['k']

        X_train, y_train, X_val, y_val = cls.prepareData(X, y, train_indices, val_indices)

        model = LassoNetClassifier(**create_kwargs)
        model.path(X_train, y_train, X_val = X_val, y_val = y_val)
        return torch.argsort(model.feature_importances_, descending = True).cpu().numpy()[:k]


class RankCorrWrapper(Rocks, BenchmarkableModel):
    """
    Thin wrapper on the RankCorr package Rocks class that also implements the Benchmarkable Model functionality
    """

    @classmethod
    def benchmarkerFunctional(
        cls,
        create_kwargs,
        train_kwargs,
        X,
        y,
        train_indices,
        val_indices,
        train_dataloader,
        val_dataloader,
        k=None,
    ):
        """
        Class function that initializes, trains, and returns markers for the provided data with the specific params
        args:
            create_kwargs (dict): ALL args used by the model constructor as a keyword arg dictionary
            train_args (dict): ALL args used by the train model step as a keyword arg dictionary
            X (np.array): the full set of training data input X
            y (np.array): the full set of training data output y
            train_indices (array-like): the indices to be used as the training set
            val_indices (array-like): the indices to be used as the validation set
            train_dataloader (pytorch dataloader): dataloader for training data set
            val_dataloader (pytorch dataloader): dataloader for validation data set
            k (int): k value for the model, the number of markers to select
        """
        if not k:
            k = train_kwargs['k']

        if 'k' in train_kwargs:
            train_kwargs = { **train_kwargs } #copy train_kwargs so later iterations have 'k'
            train_kwargs.pop('k')

        X_train, y_train, _, _ = cls.prepareData(X, y, np.concatenate([train_indices, val_indices]), [])
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

    def prepareData(X, y, train_indices, val_indices, group_by):
        """
        Since SmashPy requires data structured as AnnData, recreate it from X and y
        args:
            X (np.array): input data, counts of various proteins
            y (np.array): output data, what type of cell it is
            train_indices (array-like): the indices to be used as the training set
            val_indices (array-like): the indices to be used as the validation set
            group_by (string): the obs ouput the smashpy looks to
        """
        train_val_indices = np.concatenate([train_indices, val_indices])

        # this line will emit a warning, "Transforming to str index" from AnnData, I am having trouble making it
        # go away. See: https://github.com/theislab/anndata/issues/311
        aData = anndata.AnnData(X=pd.DataFrame(X).iloc[train_val_indices, :])
        y_series = pd.Series(y, dtype='string').astype('category').iloc[train_val_indices]
        # some index hackery is required here to get the index types to match
        y_series.index = y_series.index.astype('string').astype('object')

        aData.obs[group_by] = y_series
        return aData

class SmashPyWrapper(smashpy, AnnDataModel):
    """
    Thin wrapper on SmashPy that implements the BenchmarkableModel functionality, as well as fixes a few issues
    with SmashPy
    """

    def ensemble_learning(self, adata, group_by=None, classifier=None, test_size=0.2, balance=True, verbose=True, save=True):
        """
        Redo Smashpy's ensemble_learning function so that it uses all the data for training since we do our
        testing separately. This also helps us avoid issues with not having class representatives since we 
        can control that in our own split_data.
        """
        if group_by is None:
            print("select a group_by in .obs")
            return
        if group_by not in adata.obs.columns:
            print("group_by must be in .obs")
            return 
        
        if classifier not in ["RandomForest"]:
            print("classifier must be 'RandomForest'")
            return 
        
        data = adata.X

        myDict = {}
        for idx, c in enumerate(adata.obs[group_by].cat.categories):
            myDict[c] = idx

        labels = []
        for l in adata.obs[group_by].tolist():
            labels.append(myDict[l])

        labels = np.array(labels)

        X = data
        y = labels

        weight = []
        n = len(np.unique(y))
        for k in range(n):
            if balance:
                w = len(y)/float(n*len(y[y==k]))
                weight.append(w)
            else:
                weight.append(1)
            class_weight = dict(zip(range(n), weight))

        print("Running with (Weighted) Random Forest")
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight)
        clf.fit(X, y)

        return clf
            
    def DNN(self, *args, **kwargs):
        """
        SmashPy has a bug where verbose does not suppress the print statements in this function, so we do it here
        """
        if ('verbose' not in kwargs) or (kwargs['verbose'] == True):
            return super().DNN(*args, **kwargs)
        else:
            # https://stackoverflow.com/a/46129367
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                return super().DNN(*args, **kwargs)

    def run_shap(self, *args, **kwargs):
        """
        SmashPy has a bug where verbose does not suppress the show plots in this function which stops execution
        """
        if ('verbose' not in kwargs) or (kwargs['verbose'] == True):
            return super().run_shap(*args, **kwargs)
        else:
            # put plots in interactive mode so that they do not block execution
            with plt.ion():
                return super().run_shap(*args, **kwargs)

    def getRandomSeedsQueue(length = 400):
        """
        SmashPy sets the numpy random seed to 42, so we generate a queue and pass it to the benchmarker to ensure we
        aren't always using the same seed
        args:
            length (int): length of the queue, should be at least num_times * benchmark_range * # smashpy models
        returns:
            (SimpleQueue): of random seeds between 1 and 1000000
        """
        random_seeds_queue = queue.SimpleQueue()
        for seed in np.random.randint(low=1, high = 1000000, size = length):
          random_seeds_queue.put(seed)

        return random_seeds_queue

    @classmethod
    def getBenchmarker(cls, random_seeds_queue=None, model=None, create_kwargs={}, train_kwargs={}):
        """
        Returns a function used by the the benchmarker to intialize and train model, then return markers
        args:
            random_seeds_queue (queue): A queue filled with random seeds, at least one for every time SmashPy is run
            model (string): type of smashpy model, either None, 'RandomForest', or 'DNN'. Defaults to 'RandomForest'
            create_kwargs (dict): ALL args used by the model constructor as a keyword arg dictionary
            train_args (dict): ALL args used by the train model step as a keyword arg dictionary
        """

        if not random_seeds_queue:
            raise Exception(
                'SmashPyWrapper::getBenchmarker: SmashPy modifies the numpy random seeds, so a queue of random seeds must be passed as random_seeds_queue',
            )
        model_options = { None, 'RandomForest', 'DNN' }
        if model not in model_options:
            raise Exception(f'SmashPyWrapper::getBenchmarker: model must be one of {model_options}')

        return partial(
            cls.benchmarkerFunctional,
            create_kwargs,
            train_kwargs,
            random_seeds_queue=random_seeds_queue,
            model=model,
        )

    @classmethod
    def benchmarkerFunctional(
        cls,
        create_kwargs,
        train_kwargs,
        X,
        y,
        train_indices,
        val_indices,
        train_dataloader,
        val_dataloader,
        k=None,
        random_seeds_queue=None,
        model='RandomForest',
    ):
        """
        Class function that initializes, trains, and returns markers for the provided data with the specific params
        args:
            cls (string): The current, derived class name, used for calling derived class functions
            create_kwargs (dict): ALL args used by the model constructor as a keyword arg dictionary
            train_args (dict): ALL args used by the train model step as a keyword arg dictionary
            X (np.array): the full set of training data input X
            y (np.array): the full set of training data output y
            train_indices (array-like): the indices to be used as the training set
            val_indices (array-like): the indices to be used as the validation set
            train_dataloader (pytorch dataloader): dataloader for training data set
            val_dataloader (pytorch dataloader): dataloader for validation data set
            k (int): k value for the model, the number of markers to select
        returns:
            (np.array) the selected k markers
        """
        create_kwargs = {
            'group_by': 'annotation',
            'verbose': False, # has a bug, need to further control printing
            'save': False,
            **create_kwargs,
        }

        train_kwargs = {
            'group_by': 'annotation',
            'verbose': False,
            **train_kwargs,
        }
        assert create_kwargs['group_by'] == train_kwargs['group_by']

        create_kwargs['adata'] = cls.prepareData(X, y, train_indices, val_indices, create_kwargs['group_by'])

        if k:
            train_kwargs['restrict_top'] = ('global', k)

        sm = cls() # this prints "Initializing...", currently not blocking

        selectedGenes = []
        if model == 'RandomForest':
            clf = sm.ensemble_learning(**{ 'classifier': model, **create_kwargs })
            selectedGenes, _ = sm.gini_importance(create_kwargs['adata'], clf, **train_kwargs)
        elif model == 'DNN':
            sm.DNN(**create_kwargs)
            selectedGenes, _ = sm.run_shap(create_kwargs['adata'], **{ 'pct': 0.1, **train_kwargs })

        # Move to the next random seed
        seed = random_seeds_queue.get_nowait()
        np.random.seed(seed)

        return create_kwargs['adata'].var.index.get_indexer(selectedGenes)


class ScanpyRankGenes(AnnDataModel):

    @classmethod
    def benchmarkerFunctional(
        cls,
        create_kwargs,
        train_kwargs,
        X,
        y,
        train_indices,
        val_indices,
        train_dataloader,
        val_dataloader,
        k=None,
    ):
        """
        Class function that initializes, trains, and returns markers for the provided data with the specific params
        args:
            cls (string): The current, derived class name, used for calling derived class functions
            create_kwargs (dict): ALL args used by the model constructor as a keyword arg dictionary
            train_args (dict): ALL args used by the train model step as a keyword arg dictionary
            X (np.array): the full set of training data input X
            y (np.array): the full set of training data output y
            train_indices (array-like): the indices to be used as the training set
            val_indices (array-like): the indices to be used as the validation set
            train_dataloader (pytorch dataloader): dataloader for training data set
            val_dataloader (pytorch dataloader): dataloader for validation data set
            k (int): k value for the model, the number of markers to select
        returns:
            (np.array) the selected k markers
        """
        group_by = 'annotation'
        method = train_kwargs['method'] if 'method' in train_kwargs else 't-test'
        tie_correct = train_kwargs['tie_correct'] if 'tie_correct' in train_kwargs else False

        if k is None:
            k = train_kwargs['k']

        adata = cls.prepareData(X, y, train_indices, val_indices, group_by)

        # First, run rank_genes_groups with a high enough n_genes that we find >= k unique genes
        unique_names = np.array([])
        multiplier = 2
        while len(unique_names) < k: #unlikely this will run more than once, but just in case
            n_genes = (k // train_kwargs['num_classes']) * multiplier

            adata_with_markers = sc.tl.rank_genes_groups(
                adata, 
                group_by, 
                n_genes=n_genes, 
                method=method,
                tie_correct=tie_correct,
                copy=True,
            )
            names = np.array(list(it.chain(*adata_with_markers.uns['rank_genes_groups']['names'])), dtype=int)
            unique_names = np.unique(names)
            multiplier *= 2

        # add the genes by row until it would put us over budget k, at which point add ones with
        # lowest pvals
        genes = np.array([], dtype=int)
        i = 0
        while len(genes) < k:
            gene_row = np.array(list(adata_with_markers.uns['rank_genes_groups']['names'][i]), dtype=int)
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
    
class COSGWrapper(AnnDataModel):

    @classmethod
    def benchmarkerFunctional(
        cls,
        create_kwargs,
        train_kwargs,
        X,
        y,
        train_indices,
        val_indices,
        train_dataloader,
        val_dataloader,
        k=None,
    ):
        """
        Class function that initializes, trains, and returns markers for the provided data with the specific params
        args:
            cls (string): The current, derived class name, used for calling derived class functions
            create_kwargs (dict): ALL args used by the model constructor as a keyword arg dictionary
            train_args (dict): ALL args used by the train model step as a keyword arg dictionary
            X (np.array): the full set of training data input X
            y (np.array): the full set of training data output y
            train_indices (array-like): the indices to be used as the training set
            val_indices (array-like): the indices to be used as the validation set
            train_dataloader (pytorch dataloader): dataloader for training data set
            val_dataloader (pytorch dataloader): dataloader for validation data set
            k (int): k value for the model, the number of markers to select
        returns:
            (np.array) the selected k markers
        """
        group_by = 'annotation'

        if k is None:
            k = train_kwargs['k']

        adata = cls.prepareData(X, y, train_indices, val_indices, group_by)

        # First, run rank_genes_groups with a high enough n_genes that we find >= k unique genes
        unique_names = np.array([])
        multiplier = 2
        while len(unique_names) < k: #unlikely this will run more than once, but just in case
            n_genes = (k // train_kwargs['num_classes']) * multiplier

            cosg.cosg(
                adata, 
                group_by, 
                n_genes_user=n_genes, 
                key_added='cosg',
            )
            names = np.array(list(it.chain(*adata.uns['cosg']['names'])), dtype=int)
            unique_names = np.unique(names)
            multiplier *= 2

        # add the genes by row until it would put us over budget k, at which point add ones with highest scores
        genes = np.array([], dtype=int)
        i = 0
        while len(genes) < k:
            gene_row = np.array(list(adata.uns['cosg']['names'][i]), dtype=int)
            genes_added = np.unique(np.concatenate([genes, gene_row]))
            if len(genes_added) <= k:
                genes = genes_added
            else:
                for idx in np.argsort(-np.array(list(adata.uns['cosg']['scores'][i]))): # higher scores are more relevant
                    if len(genes) == k:
                        break

                    if gene_row[idx] not in genes:
                        genes = np.append(genes, gene_row[idx])

            i += 1

        assert len(genes) == k

        return genes