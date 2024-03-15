import os
import contextlib
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from smashpy import smashpy

from markermap.other_models import AnnDataModel

# Smashpy is out of date and difficult to install because it explicitly requires an older version of 
# tensorflow. If you are able to install it, you can import this and use the SmashpyWrapper, but 
# otherwise you can run all the other models without issue.

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
        adata,
        group_by,
        batch_size,
        train_indices,
        val_indices,
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
            adata (AnnData object): input and label data
            group_by (string): string key for adata.obs[group_by] where the output labels live
            batch_size (int): batch size for models that use batches
            train_indices (array-like): the indices to be used as the training set
            val_indices (array-like): the indices to be used as the validation set
            k (int): k value for the model, the number of markers to select
        returns:
            (np.array) the selected k markers
        """
        create_kwargs = {
            'group_by': group_by,
            'verbose': False, # has a bug, need to further control printing
            'save': False,
            **create_kwargs,
        }

        train_kwargs = {
            'group_by': group_by,
            'verbose': False,
            **train_kwargs,
        }
        assert create_kwargs['group_by'] == train_kwargs['group_by']

        create_kwargs['adata'] = cls.prepareData(adata, train_indices, val_indices)

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
