import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F




import os
import contextlib
import queue
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import math


from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report


import umap
import seaborn as sns
import matplotlib.pyplot as plt


import scanpy as sc
import anndata

from lassonet import LassoNetClassifier
from smashpy import smashpy

import logging
from functools import partial


# rounding up lowest float32 on my system
EPSILON = 1e-40
MIN_TEMP = 0.0001



def form_block(in_size, out_size, batch_norm = True, bias = True):
    layers = []
    layers.append(nn.Linear(in_size, out_size, bias = bias))
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_size))
    layers.append(nn.LeakyReLU())
    return layers


def make_encoder(input_size, hidden_layer_size, z_size, bias = True, batch_norm = True):

    main_enc = nn.Sequential(
            *form_block(input_size, hidden_layer_size, bias = bias, batch_norm = batch_norm),
            *form_block(hidden_layer_size, hidden_layer_size, bias = bias, batch_norm = batch_norm),
            *form_block(hidden_layer_size, hidden_layer_size, bias = bias, batch_norm = False)
            )

    enc_mean = nn.Linear(hidden_layer_size, z_size, bias = bias)
    enc_logvar = nn.Linear(hidden_layer_size, z_size, bias = bias)

    return main_enc, enc_mean, enc_logvar


def make_bernoulli_decoder(output_size, hidden_size, z_size, bias = True, batch_norm = True):

    main_dec = nn.Sequential(
            *form_block(Z_size, hidden_size, bias = bias, batch_norm = batch_norm),
            nn.Linear(1*hidden_size, output_size, bias = bias),
            nn.Sigmoid()
        )


    return main_dec

def make_gaussian_decoder(output_size, hidden_size, z_size, bias = True, batch_norm = True):

    main_dec = nn.Sequential(
            *form_block(z_size, hidden_size, bias = bias, batch_norm = batch_norm),
            nn.Linear(1*hidden_size, output_size, bias = bias),
        )

    dec_logvar = nn.Sequential(
            *form_block(z_size, hidden_size, bias = bias, batch_norm = batch_norm),
            nn.Linear(hidden_size, output_size, bias = bias)
            )
    
    return main_dec, dec_logvar

class ExperimentIndices:
    def __init__(self, train_indices, val_indices, test_indices):
        pass



class GumbelClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_layer_size, z_size, num_classes, k, batch_norm = True, t = 2, temperature_decay = 0.9, method = 'mean', alpha = 0.99, bias = True, lr = 0.000001,
            min_temp = MIN_TEMP):
        super(GumbelClassifier, self).__init__()
        self.save_hyperparameters()
        assert temperature_decay > 0
        assert temperature_decay <= 1


        self.encoder = nn.Sequential(
            *form_block(input_size, hidden_layer_size, bias = bias, batch_norm = batch_norm),
            *form_block(hidden_layer_size, hidden_layer_size, bias = bias, batch_norm = batch_norm),
            *form_block(hidden_layer_size, hidden_layer_size, bias = bias, batch_norm = False),
            nn.Linear(hidden_layer_size, z_size, bias = True),
            nn.LeakyReLU()
        )


        self.decoder = main_dec = nn.Sequential(
            *form_block(z_size, hidden_layer_size, bias = bias, batch_norm = batch_norm),
            nn.Linear(1*hidden_layer_size, num_classes, bias = bias),
            nn.LogSoftmax(dim = 1)
        )
        
        self.weight_creator = nn.Sequential(
            *form_block(input_size, hidden_layer_size, batch_norm = batch_norm),
            nn.Dropout(),
            *form_block(hidden_layer_size, hidden_layer_size, batch_norm = batch_norm),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, input_size)
        )

        self.method = method
        self.k = k
        self.batch_norm = batch_norm
        self.register_buffer('t', torch.as_tensor(1.0 * t))
        self.min_temp = min_temp
        self.temperature_decay = temperature_decay
        self.lr = lr
        self.bias = bias
        self.num_classes = num_classes

        assert alpha < 1
        assert alpha > 0

        # flat prior for the features
        # need the view because of the way we encode
        self.register_buffer('logit_enc', torch.zeros(input_size).view(1, -1))

        self.alpha = alpha
        self.loss_function = nn.NLLLoss()
        
    # training_phase determined by training_step
    def encode(self, x, training_phase=False):
        if training_phase:
            w = self.weight_creator(x)

            if self.method == 'mean':
                pre_enc = w.mean(dim = 0).view(1, -1)
            elif self.method == 'median':
                pre_enc = w.median(dim = 0)[0].view(1, -1)
            else:
                raise Exception("Invalid aggregation method inside batch of Non instancewise Gumbel")

            self.logit_enc = (self.alpha) * self.logit_enc.detach() + (1-self.alpha) * pre_enc
            
            gumbel = training_phase
            subset_indices = sample_subset(self.logit_enc, self.k, self.t, gumbel = gumbel, device = self.device)

            x = x * subset_indices
        else:
            mask = torch.zeros_like(x)
            mask.index_fill_(1, self.markers(), 1)

            x = x * mask

        h1 = self.encoder(x)
        # en
        return h1

    def forward(self, x, training_phase = False):
        h = self.encode(x, training_phase = training_phase)
        log_probs = self.decoder(h)

        return log_probs

    def training_step(self, batch, batch_idx):
        x, y = batch
        log_probs = self.forward(x, training_phase = True)
        loss = self.loss_function(log_probs, y)
        if torch.isnan(loss).any():
            raise Exception("nan loss during training")
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, training_step_outputs):
        self.t = max(torch.as_tensor(self.min_temp, device = self.device), self.t * self.temperature_decay)


        loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.log("epoch_avg_train_loss", loss)
        return None

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            log_probs = self.forward(x, training_phase = False)
            loss = self.loss_function(log_probs, y)
            acc = (y == log_probs.max(dim=1)[1]).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)

    def top_logits(self):
        with torch.no_grad():
            w = self.logit_enc.clone().view(-1)
            top_k_logits = torch.topk(w, k = self.k, sorted = True)[1]
            enc_top_logits = torch.nn.functional.one_hot(top_k_logits, num_classes = self.hparams.input_size).sum(dim = 0)
            
            #subsets = sample_subset(w, model.k,model.t,True)
            subsets = sample_subset(w, self.k, self.t, device = self.device, gumbel = False)
            #max_idx = torch.argmax(subsets, 1, keepdim=True)
            #one_hot = Tensor(subsets.shape)
            #one_hot.zero_()
            #one_hot.scatter_(1, max_idx, 1)
        
        return enc_top_logits, subsets

    def markers(self):
        logits = self.top_logits()
        inds_running_state = torch.argsort(logits[0], descending = True)[:self.k]

        return inds_running_state

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


class SmashPyWrapper(smashpy, BenchmarkableModel):
    """
    Thin wrapper on SmashPy that implements the BenchmarkableModel functionality, as well as fixes a few issues
    with SmashPy
    """

    def ensemble_learning(self, *args, **kwargs):
        """
        SmashPy has a bug where verbose does not suppress the print statements in this function, so we do it here
        """
        if ('verbose' not in kwargs) or (kwargs['verbose'] == True):
            return super().ensemble_learning(*args, **kwargs)
        else:
            # https://stackoverflow.com/a/46129367
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                return super().ensemble_learning(*args, **kwargs)

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
            raise Exception(f'SmashPyWrapper::getBenchmarker: model must be one of {mode_options}')

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


class VAE(pl.LightningModule, BenchmarkableModel):
    def __init__(self, input_size, hidden_layer_size, z_size, output_size = None, bias = True, batch_norm = True, lr = 0.000001, kl_beta = 0.1):
        super(VAE, self).__init__()
        self.save_hyperparameters()

        if output_size is None:
            output_size = input_size

        self.encoder, self.enc_mean, self.enc_logvar = make_encoder(input_size,
                hidden_layer_size, z_size, bias = bias, batch_norm = batch_norm)

        self.decoder, self.dec_logvar = make_gaussian_decoder(output_size, hidden_layer_size, z_size, bias = bias, batch_norm = batch_norm)

        self.lr = lr
        self.kl_beta = kl_beta
        self.batch_norm = batch_norm

    def encode(self, x):
        h1 = self.encoder(x)
        return self.enc_mean(h1), self.enc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):    
        return self.decoder(z)
        
    def forward(self, x):
        mu_latent, logvar_latent = self.encode(x)
        z = self.reparameterize(mu_latent, logvar_latent)
        mu_x = self.decode(z)
        logvar_x = self.dec_logvar(z)

        return mu_x, logvar_x, mu_latent, logvar_latent

    def training_step(self, batch, batch_idx):
        x, y = batch
        mu_x, logvar_x, mu_latent, logvar_latent = self(x)
        loss = loss_function_per_autoencoder(x, mu_x, logvar_x, mu_latent, logvar_latent, kl_beta = self.kl_beta) 
        if torch.isnan(loss).any():
            raise Exception("nan loss during training")
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            mu_x, logvar_x, mu_latent, logvar_latent = self(x)
            loss = loss_function_per_autoencoder(x, mu_x, logvar_x, mu_latent, logvar_latent, kl_beta = self.kl_beta) 
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)



class VAE_l1_diag(VAE):
    def __init__(self, input_size, hidden_layer_size, z_size, bias = True, batch_norm = True, lr = 0.000001, kl_beta = 0.1, l1_lambda = 1):
        super(VAE_l1_diag, self).__init__(input_size, hidden_layer_size , z_size, bias = bias, batch_norm = batch_norm)
        assert l1_lambda > 0
        self.save_hyperparameters()
        self.l1_lambda = l1_lambda
        
        # using .to(device) even against pytorch lightning recs 
        # because cannot instantiate normal with it
        self.diag = nn.Parameter(torch.normal(torch.zeros(input_size, device = self.device), 
                                 torch.ones(input_size, device = self.device)).requires_grad_(True))

    def training_step(self, batch, batch_idx):
        x, y = batch
        mu_x, logvar_x, mu_latent, logvar_latent = self(x)
        loss = loss_function_per_autoencoder(x, mu_x, logvar_x, mu_latent, logvar_latent, kl_beta = self.kl_beta) + self.l1_lambda * torch.sum(self.diag ** 2)
        if torch.isnan(loss).any():
            raise Exception("nan loss during training")
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            mu_x, logvar_x, mu_latent, logvar_latent = self(x)
            loss = loss_function_per_autoencoder(x, mu_x, logvar_x, mu_latent, logvar_latent, kl_beta = self.kl_beta) + self.l1_lambda * torch.sum(self.diag ** 2)
        self.log('val_loss', loss)
        return loss

    # feature standard deviations
    def markers(self, feature_std, k):
        assert self.diag.shape[0] == feature_std.shape[0]

        return torch.argsort(self.diag.abs()/feature_std, descending = True)[:k]
        
        
    def encode(self, x):
        xprime = x * self.diag
        h = self.encoder(xprime)
        return self.enc_mean(h), self.enc_logvar(h)

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

        if 'k' in train_kwargs:
            train_kwargs = { **train_kwargs } #copy train_kwargs so later iterations have 'k'
            train_kwargs.pop('k')

        feature_std = torch.tensor(X).std(dim = 0)
        model = cls(**create_kwargs)
        train_model(model, train_dataloader, val_dataloader, **train_kwargs)
        return model.markers(feature_std = feature_std.to(model.device), k = k).clone().cpu().detach().numpy()

def gumbel_keys(w, EPSILON):
    """
    Sample some gumbels, adapted from
    https://github.com/ermongroup/subsets/blob/master/subsets/sample_subsets.py
    Args:
        w (Tensor): Weights for each element, interpreted as log probabilities
        epsilon (float): min difference for float equalities
    """
    uniform = (1.0 - EPSILON) * torch.rand_like(w) + EPSILON
    z = -torch.log(-torch.log(uniform))
    w = w + z
    return w


def continuous_topk(w, k, t, device, separate=False, EPSILON = EPSILON):
    """
    Continuous relaxation of discrete variables, equations 3, 4, and 5
    Args:
        w (Tensor): Float Tensor of weights for each element. In gumbel mode
            these are interpreted as log probabilities
        k (int): number of elements in the subset sample
        t (float): temperature of the softmax
        separate (bool): defaults to false, swap to true for debugging
        epsilon (float): min difference for float equalities
    """

    # https://github.com/ermongroup/subsets/blob/master/subsets/sample_subsets.py 
    if separate:
        khot_list = []
        onehot_approx = torch.zeros_like(w, dtype = torch.float32, device = device)
        for i in range(k):
            max_mask = (1 - onehot_approx) < EPSILON
            khot_mask = 1 - onehot_approx
            khot_mask[max_mask] = EPSILON
            w = w + torch.log(khot_mask)
            onehot_approx = F.softmax(w/t, dim = -1)
            khot_list.append(onehot_approx)
        return torch.stack(khot_list)
    # https://github.com/ermongroup/subsets/blob/master/subsets/knn/sorting_operator.py
    else:
        relaxed_k = torch.zeros_like(w, dtype = torch.float32, device = device)
        onehot_approx = torch.zeros_like(w, dtype = torch.float32, device = device)
        for i in range(k):
            max_mask = (1 - onehot_approx) < EPSILON
            khot_mask = 1 - onehot_approx
            khot_mask[max_mask] = EPSILON
            w = w + torch.log(khot_mask)
            onehot_approx = F.softmax(w/t, dim = -1)
            relaxed_k = relaxed_k + onehot_approx
        return relaxed_k


def sample_subset(w, k, t, device, separate = False, gumbel = True, EPSILON = EPSILON):
    """
    Sample k elements using the continuous relaxation of discrete variables.
    A good default value of t is 0.0001, but let VAE gumbel constructor decide that.
    Adapted from: https://github.com/ermongroup/subsets/blob/master/subsets/sample_subsets.py
    Args:
        w (Tensor): Float Tensor of weights for each element. In gumbel mode
            these are interpreted as log probabilities
        k (int): number of elements in the subset sample
        t (float): temperature of the softmax
        separate (bool): defaults to false, swap to true for debugging
    """
    assert EPSILON > 0
    if gumbel:
        w = gumbel_keys(w, EPSILON)
    return continuous_topk(w, k, t, device, separate = separate, EPSILON = EPSILON)



# L1 VAE model we are loading
class VAE_Gumbel(VAE):
    def __init__(self, input_size, hidden_layer_size, z_size, k, t = 2, temperature_decay = 0.9, bias = True, batch_norm = True, lr = 0.000001, kl_beta = 0.1, min_temp = MIN_TEMP):
        super(VAE_Gumbel, self).__init__(input_size, hidden_layer_size, z_size, bias = bias, batch_norm = batch_norm, lr = lr, kl_beta = kl_beta)
        self.save_hyperparameters()
        assert temperature_decay > 0
        assert temperature_decay <= 1
        
        self.k = k
        self.register_buffer('t', torch.as_tensor(1.0 * t))
        self.min_temp = min_temp
        self.temperature_decay = temperature_decay
        
        # end with more positive to make logit debugging easier
        
        # should probably add weight clipping to these gradients because you 
        # do not want the final output (initial logits) of this to be too big or too small
        # (values between -1 and 10 for first output seem fine)
        self.weight_creator = nn.Sequential(
            *form_block(input_size, hidden_layer_size, batch_norm = batch_norm),
            nn.Dropout(),
            *form_block(hidden_layer_size, hidden_layer_size, batch_norm = batch_norm),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, input_size)#,
            #nn.LeakyReLU()
        )
        
    def encode(self, x, training_phase = False):

        w = self.weight_creator(x)
        if training_phase:
            self.subset_indices = sample_subset(w, self.k, self.t, gumbel = training_phase, device = self.device)
            x = x * self.subset_indices
        else:
            markers = torch.argsort(w, dim = 1)[:, :self.k]
            mask = torch.zeros_like(x)
            # for each of the markers   
            mask[torch.arange(markers.shape[0]).view(-1, 1).repeat(1, markers.shape[1]).flatten(), markers.flatten()]=1
            x = x * mask

        h1 = self.encoder(x)
        return self.enc_mean(h1), self.enc_logvar(h1)

    def forward(self, x, training_phase = False):
        mu_latent, logvar_latent = self.encode(x, training_phase = training_phase)
        z = self.reparameterize(mu_latent, logvar_latent)
        mu_x = self.decode(z)
        logvar_x = self.dec_logvar(z)

        return mu_x, logvar_x, mu_latent, logvar_latent

    def training_step(self, batch, batch_idx):
        x, y = batch
        mu_x, logvar_x, mu_latent, logvar_latent = self.forward(x, training_phase = True)
        loss = loss_function_per_autoencoder(x, mu_x, logvar_x, mu_latent, logvar_latent, kl_beta = self.kl_beta) 
        if torch.isnan(loss).any():
            raise Exception("nan loss during training")
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, training_step_outputs):
        self.t = max(torch.as_tensor(self.min_temp, device = self.device), self.t * self.temperature_decay)

        loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.log("epoch_avg_train_loss", loss)
        return None

    
# Not Instance_Wise Gumbel
class VAE_Gumbel_NInsta(VAE_Gumbel):
    def __init__(self, input_size, hidden_layer_size, z_size, k, t = 0.01, temperature_decay = 0.99, method = 'mean', bias = True, batch_norm = True, lr = 0.000001, kl_beta = 0.1):
        super(VAE_Gumbel_NInsta, self).__init__(input_size, hidden_layer_size, z_size, k=k, t=t, temperature_decay = temperature_decay, 
                bias = bias, batch_norm = batch_norm, lr = lr, kl_beta = kl_beta)
        self.save_hyperparameters()
        self.method = method


    def encode(self, x, training_phase = False, hard_subset = True):
        if training_phase and hard_subset:
            raise Exception("Cannot non-differentiable subset selection during training")


        w0 = self.weight_creator(x)

        if self.method == 'mean':
            w = w0.mean(dim = 0).view(1, -1)
        elif self.method == 'median':
            w = w0.median(dim = 0)[0].view(1, -1)
        else:
            raise Exception("Invalid aggregation method inside batch of Non instancewise Gumbel")

        if not hard_subset:
            self.subset_indices = sample_subset(w, self.k, self.t, gumbel = training_phase, device = self.device)
            x = x * self.subset_indices
        else: 
            markers = torch.argsort(w.flatten(), descending = True)[:self.k]
            mask = torch.zeros_like(x)
            mask.index_fill_(1, markers, 1)
            x = x * mask

        h1 = self.encoder(x)
        return self.enc_mean(h1), self.enc_logvar(h1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        mu_x, logvar_x, mu_latent, logvar_latent = self.forward(x, training_phase = True, hard_subset = False)
        loss = loss_function_per_autoencoder(x, mu_x, logvar_x, mu_latent, logvar_latent, kl_beta = self.kl_beta) 
        if torch.isnan(loss).any():
            raise Exception("nan loss during training")
        self.log('train_loss', loss)
        return loss



# idea of having a Non Instance Wise Gumbel that also has a state to keep consistency across batches
# probably some repetititon of code, but the issue is this class stuff, this is python 3 tho so it can be put into a good wrapper
# that doesn't duplicate code
class VAE_Gumbel_GlobalGate(VAE):
    # alpha is for  the exponential average
    def __init__(self, input_size, hidden_layer_size, z_size, k, t = 0.01, temperature_decay = 0.99, bias = True, batch_norm = True, lr = 0.000001, kl_beta = 0.1):
        super(VAE_Gumbel_GlobalGate, self).__init__(input_size, hidden_layer_size, z_size, bias = bias, batch_norm = batch_norm, lr = lr, kl_beta = kl_beta)
        self.save_hyperparameters()
        
        self.k = k
        self.register_buffer('t', torch.as_tensor(1.0 * t))
        self.temperature_decay = temperature_decay

        self.logit_enc = nn.Parameter(torch.normal(torch.zeros(input_size, device = self.device), torch.ones(input_size, device = self.device)).view(1, -1).requires_grad_(True))

        self.burned_in = False

    def encode(self, x, training_phase = False):
        if training_phase:
            subset_indices = sample_subset(self.logit_enc, self.k, self.t, gumbel = training_phase, device = self.device)

            x = x * subset_indices
        else:
            mask = torch.zeros_like(x)
            mask.index_fill_(1, self.markers(), 1)
            x = x * mask

        h1 = self.encoder(x)
        # en
        return self.enc_mean(h1), self.enc_logvar(h1)

    def forward(self, x, training_phase = False):
        mu_latent, logvar_latent = self.encode(x, training_phase = training_phase)
        z = self.reparameterize(mu_latent, logvar_latent)
        mu_x = self.decode(z)
        logvar_x = self.dec_logvar(z)

        return mu_x, logvar_x, mu_latent, logvar_latent

    def training_step(self, batch, batch_idx):
        x, y = batch
        mu_x, logvar_x, mu_latent, logvar_latent = self.forward(x, training_phase = True)
        loss = loss_function_per_autoencoder(x, mu_x, logvar_x, mu_latent, logvar_latent, kl_beta = self.kl_beta) 
        if torch.isnan(loss).any():
            raise Exception("nan loss during training")
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, training_step_outputs):
        self.t = max(torch.as_tensor(0.001, device = self.device), self.t * self.temperature_decay)

        loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.log("epoch_avg_train_loss", loss)
        return None

    def top_logits(self):
        with torch.no_grad():
            w = self.logit_enc.clone().view(-1)
            top_k_logits = torch.topk(w, k = self.k, sorted = True)[1]
            enc_top_logits = torch.nn.functional.one_hot(top_k_logits, num_classes = self.hparams.input_size).sum(dim = 0)

            #subsets = sample_subset(w, model.k,model.t,True)
            subsets = sample_subset(w, self.k, self.t, gumbel = False, device = self.device)
            #max_idx = torch.argmax(subsets, 1, keepdim=True)
            #one_hot = Tensor(subsets.shape)
            #one_hot.zero_()
            #one_hot.scatter_(1, max_idx, 1)

        return enc_top_logits, subsets

    def markers(self):
        logits = self.top_logits()
        inds_global_gate = torch.argsort(logits[0], descending = True)[:self.k]

        return inds_global_gate

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
        model = cls(**{**create_kwargs, 'k': k}) if k else cls(**create_kwargs)
        train_model(model, train_dataloader, val_dataloader, **train_kwargs)
        return model.markers().clone().cpu().detach().numpy()


    
# idea of having a Non Instance Wise Gumbel that also has a state to keep consistency across batches
# probably some repetititon of code, but the issue is this class stuff, this is python 3 tho so it can be put into a good wrapper
# that doesn't duplicate code
class VAE_Gumbel_RunningState(VAE_Gumbel):
    # alpha is for  the exponential average
    def __init__(self, input_size, hidden_layer_size, z_size, k, t = 0.01, temperature_decay = 0.99, method = 'mean', alpha = 0.9, bias = True, batch_norm = True, lr = 0.000001, kl_beta = 0.1):
        super(VAE_Gumbel_RunningState, self).__init__(input_size, hidden_layer_size, z_size, k = k, t = t, temperature_decay = temperature_decay,
                bias = bias, batch_norm = batch_norm, lr = lr, kl_beta = kl_beta)
        self.save_hyperparameters()
        self.method = method

        assert alpha < 1
        assert alpha > 0

        # flat prior for the features
        # need the view because of the way we encode
        self.register_buffer('logit_enc', torch.zeros(input_size).view(1, -1))

        self.alpha = alpha
        
    # training_phase determined by training_step
    def encode(self, x, training_phase=False):
        if training_phase:
            w = self.weight_creator(x)

            if self.method == 'mean':
                pre_enc = w.mean(dim = 0).view(1, -1)
            elif self.method == 'median':
                pre_enc = w.median(dim = 0)[0].view(1, -1)
            else:
                raise Exception("Invalid aggregation method inside batch of Non instancewise Gumbel")

            self.logit_enc = (self.alpha) * self.logit_enc.detach() + (1-self.alpha) * pre_enc
            

            gumbel = training_phase
            subset_indices = sample_subset(self.logit_enc, self.k, self.t, gumbel = gumbel, device = self.device)
            x = x * subset_indices
        else:
            mask = torch.zeros_like(x)
            mask.index_fill_(1, self.markers(), 1)
            x = x * mask


        h1 = self.encoder(x)
        # en
        return self.enc_mean(h1), self.enc_logvar(h1) 

    def top_logits(self):
        with torch.no_grad():
            w = self.logit_enc.clone().view(-1)
            top_k_logits = torch.topk(w, k = self.k, sorted = True)[1]
            enc_top_logits = torch.nn.functional.one_hot(top_k_logits, num_classes = self.hparams.input_size).sum(dim = 0)
            
            #subsets = sample_subset(w, model.k,model.t,True)
            subsets = sample_subset(w, self.k, self.t, gumbel = False, device = self.device)
            #max_idx = torch.argmax(subsets, 1, keepdim=True)
            #one_hot = Tensor(subsets.shape)
            #one_hot.zero_()
            #one_hot.scatter_(1, max_idx, 1)
        
        return enc_top_logits, subsets

    def markers(self):
        logits = self.top_logits()
        inds_running_state = torch.argsort(logits[0], descending = True)[:self.k]

        return inds_running_state

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
        model = cls(**{**create_kwargs, 'k': k}) if k else cls(**create_kwargs)
        train_model(model, train_dataloader, val_dataloader, **train_kwargs)
        return model.markers().clone().cpu().detach().numpy()


# not doing multiple inheritance because GumbelClassifier is repeating itself
class MarkerMap(VAE_Gumbel_RunningState):

    def __init__(self, input_size, hidden_layer_size, z_size, num_classes, k, 
            t = 3.0, temperature_decay = 0.95, method = 'mean', alpha = 0.95, 
            bias = True, batch_norm = True, lr = 0.000001, kl_beta = 0.1, loss_tradeoff = 0.5):

        
        assert loss_tradeoff <= 1
        assert loss_tradeoff >= 0
        if num_classes is None:
            assert loss_tradeoff == 1

        super(MarkerMap, self).__init__(input_size = input_size, hidden_layer_size = hidden_layer_size, 
                z_size = z_size, k = k, t = t, temperature_decay = temperature_decay, 
                method = method, alpha = alpha, batch_norm = batch_norm, bias = bias, lr = lr, kl_beta = kl_beta)

        self.save_hyperparameters()
        self.num_classes = num_classes
        self.register_buffer('loss_tradeoff', torch.as_tensor(1.0 * loss_tradeoff))

        if num_classes is None:
            self.classification_decoder = None
            self.classification_loss = None
        else:
            self.classification_decoder = nn.Sequential(
                    *form_block(z_size, hidden_layer_size, bias = bias, batch_norm = batch_norm),
                    nn.Linear(1*hidden_layer_size, num_classes, bias = bias),
                    nn.LogSoftmax(dim = 1)
                    )
            self.classification_loss = nn.NLLLoss(reduction = 'sum')


    def decode(self, mu_z, z):
        if self.loss_tradeoff != 0:
            mu_x = self.decoder(z)
        else:
            mu_x = None
        if self.loss_tradeoff != 1:
            log_probs = self.classification_decoder(mu_z)
        else: 
            log_probs = None
        return mu_x, log_probs


    def forward(self, x, training_phase = False):
        mu_latent, logvar_latent = self.encode(x, training_phase = training_phase)
        z = self.reparameterize(mu_latent, logvar_latent)
        # mu_latent is used for the classifier and z for the VAE
        mu_x, log_probs = self.decode(mu_latent, z)

        if self.loss_tradeoff != 0:
            logvar_x = self.dec_logvar(z)
        else:
            logvar_x = None

        return mu_x, logvar_x, log_probs, mu_latent, logvar_latent

    def training_step(self, batch, batch_idx):
        x, y = batch

        mu_x, logvar_x, log_probs, mu_latent, logvar_latent = self.forward(x, training_phase = True)
        if self.loss_tradeoff != 0:
            loss_recon = loss_function_per_autoencoder(x, mu_x, logvar_x, mu_latent, logvar_latent, kl_beta = self.kl_beta)
        else:
            loss_recon = 0

        if self.loss_tradeoff != 1: 
            loss_classification = self.classification_loss(log_probs, y) / x.size()[0]
        else:
           loss_classification = 0

        loss = self.loss_tradeoff * loss_recon + (1-self.loss_tradeoff) * loss_classification

        if torch.isnan(loss).any():
            raise Exception("nan loss during training")
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            mu_x, logvar_x, log_probs, mu_latent, logvar_latent = self.forward(x, training_phase = False)

            # if not purely supervised
            if self.loss_tradeoff != 0:
                loss_recon = loss_function_per_autoencoder(x, mu_x, logvar_x, mu_latent, logvar_latent, kl_beta = self.kl_beta)
            else:
                loss_recon = 0

            # if not purely unsupervised
            if self.loss_tradeoff != 1:
                loss_classification = self.classification_loss(log_probs, y) / x.size()[0]
                acc = (y == log_probs.max(dim=1)[1]).float().mean()
                self.log('val_acc', acc)
            else:
                loss_classification = 0

            loss = self.loss_tradeoff * loss_recon + (1-self.loss_tradeoff) * loss_classification

        self.log('val_loss', loss)
        return loss


    # k = no_narkers
    # assuming Y is numerical labels and X is already pre-processed
    # train_percentage == 1 for no validation set. Validation set is used for early stopping
    def map(self, X, Y, train_ratio = 0.8, gpus=None, min_epochs=10, max_epochs=100, verbose = False):
        if Y is None:
            # set to pure unsupervised
            self.loss_tradeoff = 1

        # Generate Dataloaders
        # and then train and fit model
        assert train_ratio >= 0
        assert train_ratio <= 1
        train_dataloader, val_dataloader, train_indices, val_indices = split_data_into_dataloaders_no_test(X, Y, train_size = train_ratio)
        train_model(self, train_dataloader, val_dataloader, gpus = gpus, min_epochs = min_epochs, max_epochs = max_epochs)
        markers = self.markers()
        # train additional model 
        
        if markers is not None and Y is not None:
            train_x = X[:, markers]
            train_y = Y
            classifier = RandomForestClassifier()
            classifier.fit(train_x, train_y)
        else: 
            classifier = None


        extras = {'train_indices': train_indices, 'val_indices': val_indices}
        # train random forest to be the classifier
        return markers, classifier, extras

    # uses hard subsetting
    # returns log probs
    def predict_logprob(self, X):
        assert self.loss_tradeoff != 1
        assert self.num_classes is not None
        log_probs = self.forward(X)[2]
        return log_probs

    # uses hard subsetting
    def predict_class(self, X):
        X = torch.Tensor(X)
        X.to(self.device)
        with torch.no_grad():
            log_probs = self.predict_logprob(X)
        return log_probs.max(dim=1)[1].cpu().numpy()


# NMSL is Not My Selection Layer
# Implementing reference paper
class ConcreteVAE_NMSL(VAE):
    def __init__(self, input_size, hidden_layer_size, z_size, k, t = 0.01, temperature_decay = 0.99, bias = True, batch_norm = True, lr = 0.000001, kl_beta = 0.1, min_temp = MIN_TEMP):
        # k because encoder actually uses k features as its input because of how concrete VAE picks it out
        super(ConcreteVAE_NMSL, self).__init__(k, hidden_layer_size, z_size, output_size = input_size, bias = bias, batch_norm = batch_norm, lr = lr, kl_beta = kl_beta)
        self.save_hyperparameters()
        assert temperature_decay > 0
        assert temperature_decay <= 1
        
        self.k = k
        self.register_buffer('t', torch.as_tensor(1.0 * t))
        self.min_temp = min_temp
        self.temperature_decay = temperature_decay

        self.logit_enc = nn.Parameter(torch.normal(torch.zeros(input_size*k, device = self.device), torch.ones(input_size*k, device = self.device)).view(k, -1).requires_grad_(True))

    def encode(self, x, training_phase = False):
        if training_phase:
            w = gumbel_keys(self.logit_enc, EPSILON = torch.finfo(torch.float32).eps)
            w = torch.softmax(w/self.t, dim = -1)
            # safe here because we do not use it in computation, only reference
            self.subset_indices = w.clone().detach()
            x = x.mm(w.transpose(0, 1))
        else:
            x = x[:, self.markers()]

        h1 = self.encoder(x)
        # en
        return self.enc_mean(h1), self.enc_logvar(h1)

    def forward(self, x, training_phase = False):
        mu_latent, logvar_latent = self.encode(x, training_phase = training_phase)
        z = self.reparameterize(mu_latent, logvar_latent)
        mu_x = self.decode(z)
        logvar_x = self.dec_logvar(z)

        return mu_x, logvar_x, mu_latent, logvar_latent

    def training_step(self, batch, batch_idx):
        x, y = batch
        mu_x, logvar_x, mu_latent, logvar_latent = self.forward(x, training_phase = True)
        loss = loss_function_per_autoencoder(x, mu_x, logvar_x, mu_latent, logvar_latent, kl_beta = self.kl_beta) 
        if torch.isnan(loss).any():
            raise Exception("nan loss during training")
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, training_step_outputs):
        self.t = max(torch.as_tensor(self.min_temp, device = self.device), self.t * self.temperature_decay)


        loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.log("epoch_avg_train_loss", loss)
        return None

    def top_logits(self):
        with torch.no_grad():

            w = gumbel_keys(self.logit_enc, EPSILON = torch.finfo(torch.float32).eps)
            w = torch.softmax(w/self.t, dim = -1)
            subset_indices = w.clone().detach()

            #max_idx = torch.argmax(subset_indices, 1, keepdim=True)
            #one_hot = Tensor(subset_indices.shape)
            #one_hot.zero_()
            #one_hot.scatter_(1, max_idx, 1)

            all_subsets = subset_indices.sum(dim = 0)

            inds = torch.argsort(subset_indices.sum(dim = 0), descending = True)[:self.k]
            all_logits = torch.nn.functional.one_hot(inds, num_classes = self.hparams.input_size).sum(dim = 0)
        
        return all_logits, all_subsets

    def markers(self):
        logits = self.top_logits()
        inds_concrete = torch.argsort(logits[1], descending = True)[:self.k]

        return inds_concrete

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
        model = cls(**{**create_kwargs, 'k': k}) if k else cls(**create_kwargs)
        train_model(model, train_dataloader, val_dataloader, **train_kwargs)
        return model.markers().clone().cpu().detach().numpy()

def loss_function_per_autoencoder(x, recon_x, logvar_x, mu_latent, logvar_latent, kl_beta = 0.1):
    # loss_rec = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # loss_rec = F.mse_loss(recon_x, x, reduction='sum')
    batch_size = x.size()[0]
    loss_rec = -torch.sum(
            (-0.5 * np.log(2.0 * np.pi))
            + (-0.5 * logvar_x)
            + ((-0.5 / torch.exp(logvar_x)) * (x - recon_x) ** 2.0)
            )

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar_latent - mu_latent.pow(2) - logvar_latent.exp())
    loss = (loss_rec + kl_beta * KLD) / batch_size

    return loss

# KLD of D(P_1||P_2) where P_i are Gaussians, assuming diagonal
def kld_joint_autoencoders(mu_1, mu_2, logvar_1, logvar_2):
    # equation 6 of Tutorial on Variational Autoencoders by Carl Doersch
    # https://arxiv.org/pdf/1606.05908.pdf
    mu_12 = mu_1 - mu_2
    kld = 0.5 * (-1 - (logvar_1 - logvar_2) + mu_12.pow(2) / logvar_2.exp() + torch.exp(logvar_1 - logvar_2))
    #print(kld.shape)
    kld = torch.sum(kld, dim = 1)
    
    return kld.sum()

# for joint
def loss_function_joint(x, ae_1, ae_2):
    # assuming that both autoencoders return recon_x, mu, and logvar
    # try to make ae_1 the vanilla vae
    # ae_2 should be the L1 penalty VAE
    mu_x_1, logvar_x_1, mu_latent_1, logvar_latent_1 = ae_1(x)
    mu_x_2, logvar_x_2, mu_latent_2, logvar_latent_2 = ae_2(x)
    
    loss_vae_1 = loss_function_per_autoencoder(x, mu_x_1, logvar_x_1, mu_latent_1, logvar_latent_1)
    loss_vae_2 = loss_function_per_autoencoder(x, mu_x_2, logvar_x_2, mu_latent_2, logvar_latent_2)
    joint_kld_loss = kld_joint_autoencoders(mu_latent_1, mu_latent_2, logvar_latent_1, logvar_latent_1)
    #print("Losses")
    #print(loss_vae_1)
    #print(loss_vae_2)
    #print(joint_kld_loss)
    return loss_vae_1, loss_vae_2, joint_kld_loss


def train_l1(df, model, optimizer, epoch, batch_size):
    model.train()
    train_loss = 0
    permutations = torch.randperm(df.shape[0])
    for i in range(math.ceil(len(df)/batch_size)):
        batch_ind = permutations[i * batch_size : (i+1) * batch_size]
        batch_data = df[batch_ind, :]
        
        optimizer.zero_grad()
        mu_x, mu_latent, logvar_latent = model(batch_data)
        loss = loss_function_per_autoencoder(batch_data, mu_x, mu_latent, logvar_latent)
        loss += 100 * torch.norm(model.diag, p = 1)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        with torch.no_grad():
            model.diag.data /= torch.norm(model.diag.data, p = 2)
        
        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (i+1) * len(batch_data), len(df),
                100. * (i+1) * len(batch_data) / len(df),
                loss.item() / len(batch_data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(df)))
    



def train_model(model, train_dataloader, val_dataloader, gpus = None, tpu_cores = None, min_epochs = 50, 
        max_epochs = 600, auto_lr = True, max_lr = 0.001, lr_explore_mode = 'exponential', num_lr_rates = 100, early_stopping_patience=3, precision = 32, verbose = False):

    assert max_epochs > min_epochs

    if not verbose: 
        logging.getLogger("lightning").setLevel(logging.ERROR)
        pl_loggers = [ logging.getLogger(name) for name in logging.root.manager.loggerDict if 'lightning' in name ]

        for logger in pl_loggers:
            logger.setLevel(logging.ERROR)

    early_stopping_callback = EarlyStopping(monitor='val_loss', mode = 'min', patience = early_stopping_patience)
    trainer = pl.Trainer(gpus = gpus, tpu_cores = tpu_cores, min_epochs = min_epochs, max_epochs = max_epochs, 
            auto_lr_find=auto_lr, callbacks=[early_stopping_callback], precision = precision, logger = verbose,
            # turn off some summaries
            enable_model_summary=verbose, enable_progress_bar = verbose)
    if auto_lr:
        # for some reason plural val_dataloaders
        lr_finder = trainer.tuner.lr_find(model, train_dataloaders = train_dataloader, val_dataloaders = val_dataloader, max_lr = max_lr, mode = lr_explore_mode, num_training = num_lr_rates)
    
    
        if verbose:
            fig = lr_finder.plot(suggest=True)
            fig.show()

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        if verbose:
            print("New Learning Rate {}".format(new_lr))
    
        # update hparams of the model
        model.hparams.lr = new_lr
        model.lr = new_lr

    model.train()
    trainer.fit(model, train_dataloader, val_dataloader)
    return trainer

def save_model(trainer, base_path):
    # make directory
    if not os.path.exists(os.path.dirname(base_path)):
        try:
            os.makedirs(os.path.dirname(base_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise Exception("COULD NOT MAKE PATH")
    trainer.save_checkpoint(base_path, weights_only = True)



def train_save_model(model, train_data, val_data, base_path, min_epochs, max_epochs, auto_lr = True, max_lr = 0.001, lr_explore_mode = 'exponential', early_stopping_patience=3, num_lr_rates = 100,
        gpus = None, tpu_cores = None, precision = 32, verbose = False):
    trainer = train_model(model, train_data, val_data, gpus=gpus, tpu_cores=tpu_cores, 
            min_epochs = min_epochs, max_epochs = max_epochs, auto_lr = auto_lr, max_lr = max_lr, lr_explore_mode = lr_explore_mode, 
            early_stopping_patience=early_stopping_patience, num_lr_rates = 100, precision = precision, verbose = verbose)
    save_model(trainer, base_path)
    return trainer

def load_model(module_class, checkpoint_path):
    model = module_class.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model





####### Metrics



# balanced accuracy per k
# accuracy per k
# return both train and test
# with markers and without
def metrics_model(train_data, train_labels, test_data, test_labels, markers, model, k = None, recon = True):
    # if model is none don't do a confusion matrix for the model with markers

    classifier_orig = RandomForestClassifier(n_jobs = -1)
    classifier_orig_markers = RandomForestClassifier(n_jobs = -1)

    classifier_orig.fit(train_data.cpu(), train_labels)
    classifier_orig_markers.fit(train_data[:,markers].cpu(), train_labels)
    

    if recon:
        classifier_recon = RandomForestClassifier(n_jobs = -1)
        classifier_recon_markers = RandomForestClassifier(n_jobs = -1)

    with torch.no_grad():
        if recon:
            train_data_recon = model(train_data)[0].cpu()
            classifier_recon.fit(train_data_recon, train_labels)
            classifier_recon_markers.fit(train_data_recon[:, markers], train_labels)

        bac_orig = balanced_accuracy_score(test_labels, classifier_orig.predict(test_data.cpu()))
        bac_orig_markers = balanced_accuracy_score(test_labels, classifier_orig_markers.predict(test_data[:, markers].cpu()))

        if recon:
            bac_recon = balanced_accuracy_score(test_labels, classifier_recon.predict(model(test_data)[0].cpu()))
            bac_recon_markers = balanced_accuracy_score(test_labels, classifier_recon_markers.predict(model(test_data)[0][:,markers].cpu()))
        else:
            bac_recon = 'Skipped'
            bac_recon_markers = 'Skipped'

        accuracy_orig = accuracy_score(test_labels, classifier_orig.predict(test_data.cpu()))
        accuracy_orig_markers = accuracy_score(test_labels, classifier_orig_markers.predict(test_data[:, markers].cpu()))
        if recon:
            accuracy_recon = accuracy_score(test_labels, classifier_recon.predict(model(test_data)[0].cpu()))
            accuracy_recon_markers = accuracy_score(test_labels, classifier_recon_markers.predict(model(test_data)[0][:,markers].cpu()))
            cos_angle_no_markers = average_cosine_angle(test_data, model(test_data)[0]).item()
            cos_angle_markers = average_cosine_angle(test_data[:, markers], model(test_data)[0][:, markers]).item()

        else:
            accuracy_recon =  'Skipped'
            accuracy_recon_markers = 'Skipped'
            cos_angle_no_markers = 'Skipped'
            cos_angle_markers = 'Skipped'

    return {'k': k, 
            'BAC Original Data': bac_orig, 'BAC Original Data Markers': bac_orig_markers, 'BAC Recon Data': bac_recon, 'BAC Recon Data Markers': bac_recon_markers,
            'AC Original Data': accuracy_orig, 'AC Original Data Markers': accuracy_orig_markers, 'AC Recon Data': accuracy_recon, 'AC Recon Data Markers': accuracy_recon_markers,
            'Cosine Angle Between Data and Reconstruction (No Markers)': cos_angle_no_markers,
            'Cosine Angle Beteween Marked Data and Marked Reconstruction Data': cos_angle_markers
            }

def confusion_matrix_orig_recon(train_data, train_labels, test_data, test_labels, markers, model):
    # if model is none don't do a confusion matrix for the model with markers
    train_labels = zeisel_label_encoder.transform(train_labels)
    test_labels = zeisel_label_encoder.transform(test_labels)

    classifier_orig = RandomForestClassifier(n_jobs = -1)
    classifier_orig_markers = RandomForestClassifier(n_jobs = -1)

    classifier_orig.fit(train_data.cpu(), train_labels)
    classifier_orig_markers.fit(train_data[:,markers].cpu(), train_labels)
    

    classifier_recon = RandomForestClassifier(n_jobs = -1)
    classifier_recon_markers = RandomForestClassifier(n_jobs = -1)

    with torch.no_grad():
        train_data_recon = model(train_data)[0].cpu()
        classifier_recon.fit(train_data_recon, train_labels)
        classifier_recon_markers.fit(train_data_recon[:, markers], train_labels)


        cm_orig = confusion_matrix(test_labels, classifier_orig.predict(test_data.cpu()))
        cm_orig_markers = confusion_matrix(test_labels, classifier_orig_markers.predict(test_data[:, markers].cpu()))
        cm_recon = confusion_matrix(test_labels, classifier_recon.predict(model(test_data)[0].cpu()))
        cm_recon_markers = confusion_matrix(test_labels, classifier_recon_markers.predict(model(test_data)[0][:,markers].cpu()))

        accuracy_orig = accuracy_score(test_labels, classifier_orig.predict(test_data.cpu()))
        accuracy_orig_markers = accuracy_score(test_labels, classifier_orig_markers.predict(test_data[:, markers].cpu()))
        accuracy_recon = accuracy_score(test_labels, classifier_recon.predict(model(test_data)[0].cpu()))
        accuracy_recon_markers = accuracy_score(test_labels, classifier_recon_markers.predict(model(test_data)[0][:,markers].cpu()))


    print("Note: Makers here are significant for the classification. Markers are used to select which features of the (possibly Reconstructed) Data go into classifier")
    print("Confusion Matrix of Original Data")
    print(cm_orig)
    print("Accuracy {}".format(accuracy_orig))


    print("Confusion Matrix of Original Data Selected by Markers.")
    print(cm_orig_markers)
    print("Accuracy {}".format(accuracy_orig_markers))

    print("Confusion Matrix of Reconstructed Data")
    print(cm_recon)
    print("Accuracy {}".format(accuracy_recon))

    print("Confusion Matrix of Reconstructed Data by Markers")
    print(cm_recon_markers)
    print("Accuracy {}".format(accuracy_recon_markers))


def new_model_metrics(train_x, train_y, test_x, test_y, markers = None, model = None):
    """
    Trains and tests a specified model (or RandomForest, if none specified) with a subset of the dimensions
    specified by the indices in the markers array. Returns the error rate, a testing report, and a confusion
    matrix of the results.
    Args:
        train_x: (numpy array) the training data input
        train_y: (numpy array) the training data labels
        test_x: (numpy array) testing data input
        test_y: (numpy array) testing data labels
        markers: (numpy array) marker indices, a subset of the column indices of train_x/test_x, defaults to all
        model: model to train and test on, defaults to RandomForest
    """
    if markers is not None:
        train_x = train_x[:, markers]
        test_x = test_x[:, markers]

    if model is None:
        model = RandomForestClassifier()
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    train_rep = classification_report(train_y, model.predict(train_x), output_dict=True)
    test_rep = classification_report(test_y, pred_y, output_dict=True)
    cm = confusion_matrix(test_y, pred_y)
    if cm is None:
        raise Exception("Some error in generating confusion matrix")
    misclass_rate = 1 - accuracy_score(test_y, pred_y)
    return misclass_rate, test_rep, cm


def visualize_save_embedding(X, y, encoder, title, path, markers = None):
    if markers is not None:
        X = X[:, markers]
    num_classes = len(encoder.classes_)
    #print('embed')
    embedding = umap.UMAP(n_neighbors=10, min_dist= 0.05).fit_transform(X)
    
    
    fig, ax = plt.subplots(1, figsize=(12, 8.5))
    
    #print('scatter')
    plt.scatter(*embedding.T, c = y)
    # plt.setp(ax, xticks=[], yticks=[])
    
    #print('color')
    cbar = plt.colorbar(ticks=np.arange(num_classes))#, boundaries = np.arange(num_classes) - 0.5)
    cbar.ax.set_yticklabels(encoder.classes_)
    
    #print('save')
    plt.title(title)
    plt.savefig(path)
    plt.close(fig)

def model_variances(path, tries):
    misclass_arr = []
    weight_f1_arr = []
    for tryy in range(1, tries + 1):
        results = np.load(path.format(tryy), allow_pickle = True)
        misclass_arr.append(results[0])
        weight_f1_arr.append(results[1]['weighted avg']['f1-score'])
    return np.mean(misclass_arr), np.mean(weight_f1_arr), np.std(misclass_arr), np.std(weight_f1_arr)

def mislabel_points(y, mislabel_percent, eligible_indices=None):
    assert mislabel_percent <= 1.0
    assert mislabel_percent >= 0.0

    if eligible_indices is None:
        eligible_indices = np.array(range(len(y)))

    assert np.max(eligible_indices) < len(y)

    num_mislabelled = int(mislabel_percent*len(eligible_indices))
    y_unique = np.unique(y)
    #sample the new wrong labels uniformly from the possible unique labels
    mislabels = y_unique[np.random.randint(0, len(y_unique), num_mislabelled)]

    #sample the indices of y without replacement, we will replace those indices with the new labels
    mislabelled_indices = np.random.permutation(eligible_indices)[:num_mislabelled]
    y_err = y.copy()
    y_err[mislabelled_indices] = mislabels

    return y_err

def benchmark(
    models,
    num_times,
    X,
    y,
    benchmark,
    train_size = 0.7,
    val_size = 0.1,
    batch_size = 64,
    save_path=None,
    benchmark_range=None,
):
    """
    Benchmark a collection of models by a benchmark param on data X,y. If save_path is specified, results are saved
    when a complete benchmark_range is complete
    args:
        models (dict): maps model labels to a function that runs the model on the data and returns markers
            use model.getBenchmarker() to automatically generate those functions
        num_times (int): number of random data splits to run the model on
        X (array): Input data
        y (vector): Output labels
        benchmark (string): type of benchmarking to do, must be one of {'k', 'label_error'}
        train_size (float): 0 to 1, fraction of data for train set, defaults to 0.7
        val_size (float): 0 to 1, fraction of data for validation set, defaults to 0.1
        batch_size (int): defaults to 64
        save_path (string): if not None, folder to save results to, defaults to None
        benchmark_range (array): values that the benchmark ranges over, defaults to none
    returns:
        (dict): maps model labels to an np.array (num_times x benchmark_levels) of misclass rates
        (string): benchmark
        (array-like): benchmark_range
    """
    benchmark_options = { 'k', 'label_error' }
    if benchmark not in benchmark_options:
        raise Exception(f'benchmark: Possible choices of benchmark are {benchmark_options}')

    if not benchmark_range:
        raise Exception(f'benchmark: For benchmark {benchmark}, please provide a range')

    results = {}
    for i in range(num_times):
        train_dataloader, val_dataloader, test_dataloader, train_indices, val_indices, test_indices = split_data_into_dataloaders(
            X,
            y,
            train_size,
            val_size,
            batch_size=batch_size,
            # num_workers=num_workers,
        )

        X_test = X[test_indices,:]
        y_test = y[test_indices]

        for model_label, model_functional in models.items():

            benchmark_results = []
            for val in benchmark_range:
                if benchmark == 'k':
                    markers = model_functional(
                        X,
                        y,
                        train_indices,
                        val_indices,
                        train_dataloader,
                        val_dataloader,
                        k=val,
                    )
                    # TODO: incorporate test_rep, cm
                    model_misclass, _, _ = new_model_metrics(
                        X[np.concatenate([train_indices, val_indices]), :],
                        y[np.concatenate([train_indices, val_indices])],
                        X_test,
                        y_test,
                        markers = markers,
                    )
                elif benchmark == 'label_error':
                    y_err = mislabel_points(y, val, np.concatenate([train_indices, val_indices]))
                    markers = model_functional(X, y_err, train_indices, val_indices, train_dataloader, val_dataloader)

                    # TODO: incorporate test_rep, cm
                    model_misclass, _, _ = new_model_metrics(
                        X[np.concatenate([train_indices, val_indices]), :],
                        y_err[np.concatenate([train_indices, val_indices])],
                        X_test,
                        y_test,
                        markers = markers,
                    )

                benchmark_results.append(model_misclass)

            benchmark_results_ndarray = np.array(benchmark_results).reshape((1,len(benchmark_results)))
            if model_label not in results:
                results[model_label] = benchmark_results_ndarray
            else:
                results[model_label] = np.append(results[model_label], benchmark_results_ndarray, axis=0)

            if save_path:
                np.save(f'{save_path}benchmark_{benchmark}_{num_times}', results)

    return results, benchmark, benchmark_range

#######


### graph

def graph_umap_embedding(data, labels, title, encoder):
    num_classes = len(encoder.classes_)
    embedding = umap.UMAP(n_neighbors=10, min_dist= 0.05).fit_transform(data)
    
    
    fig, ax = plt.subplots(1, figsize=(12, 8.5))
    
    plt.scatter(*embedding.T, c = encoder.transform(labels))
    # plt.setp(ax, xticks=[], yticks=[])
    
    cbar = plt.colorbar(ticks=np.arange(num_classes))#, boundaries = np.arange(num_classes) - 0.5)
    cbar.ax.set_yticklabels(encoder.classes_)
    
    plt.title(title)


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          save_path = None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #plt.imshow(cm, interpolation='nearest', cmap=cmap)
    sns.heatmap(cm, annot=True,cmap=cmap)
    plt.title(title)
    #plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks+0.5, target_names, rotation=45)
        plt.yticks(tick_marks + 0.5, target_names, rotation = 45)

    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.2f}; misclass={:0.2f}'.format(accuracy, misclass))
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_benchmarks(results, benchmark_label, benchmark_range, mode='misclass', show_stdev=False):
    """
    Plot benchmark results of multiple models over the values that you are benchmarking on
    args:
        results (dict): maps model label to np.array of the misclassifications with shape (num_runs x benchmark range)
        benchmark label (string): what you are benchmarking over, will be the x_label
        benchmark_range (array): values that you are benchmarking over
        mode (string): one of {'misclass', 'accuracy'}, defaults to 'misclass'
        show_stdev (bool): whether to show fill_between range of 1 stdev over the num_runs, defaults to false
    """
    mode_options = {'misclass', 'accuracy'}
    if mode not in mode_options:
        raise Exception(f'plot_benchmarks: Possible choices of mode are {mode_options}')

    markers = ['.','o','v','^','<','>','8','s','p','P','*','h','H','+','x','X','D','d','|','_','1','2','3','4',',']
    fig1, ax1 = plt.subplots()
    i = 0
    num_runs = 1
    for label, result in results.items():
        if mode == 'accuracy':
            result = np.ones(result.shape) - result

        num_runs = result.shape[0]
        mean_result = result.mean(axis=0)

        #only show standard deviation if there we multiple runs
        if show_stdev and result.shape[0] > 1:
            stdev = result.std(axis=0)
            ax1.fill_between(benchmark_range, mean_result - stdev, mean_result + stdev, alpha=0.2)

        #plot the results for this model against the benchmarked range
        ax1.plot(benchmark_range, mean_result, label=label, marker=markers[i])
        i = (i+1) % len(markers)

    ax1.set_title(f'{mode.capitalize()} Benchmark, over {num_runs} runs')
    ax1.set_xlabel(benchmark_label)
    ax1.set_ylabel(mode.capitalize())
    ax1.legend()

    plt.show()

###
    

def quick_model_summary(model, train_data, test_data, threshold, batch_size):
    input_size = train_data.shape[1]
    with torch.no_grad():
        train_pred = model(train_data[0:batch_size, :])[0]
        train_pred[train_pred < threshold] = 0 

        test_pred = model(test_data[0:batch_size,:])[0]
        test_pred[test_pred < threshold] = 0 
        
    print("Per Neuron Loss Train")
    print(F.binary_cross_entropy(train_pred, train_data[0:batch_size, :], reduction='mean'))
    print("Per Neuron Loss Test")
    print(F.binary_cross_entropy(test_pred, test_data[0:batch_size, :], reduction='mean'))
    
    print("# Non Sparse in Pred test")
    print(torch.sum(test_pred[0,:] != 0))
    print("# Non Sparse in Orig test")
    print(torch.sum(test_data[0,:] != 0))

# X can be a numpy array 
def process_data(X, Y, filter_data = False):
    adata = sc.AnnData(X)

    if filter_data:
        adata = adata[adata.obs.n_genes_by_counts < 2500, :]
        adata = adata[adata.obs.pct_counts_mt < 5, :]


    adata.layers["counts"] = np.asarray(adata.X)

    # normilise and save the data into the layer
    sc.pp.normalize_total(adata, counts_per_cell_after=1e4)
    adata.layers["norm"] = np.asarray(adata.X).copy()

    # logarithmise and save the data into the layer
    sc.pp.log1p(adata)
    adata.layers["log"] = np.asarray(adata.X.copy())
    # save in adata.raw.X normilise and logarithm data
    adata.raw = adata.copy()

    sc.pp.scale(adata, max_value=10)
    adata.layers["scale"] = np.asarray(adata.X.copy())

    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)

    return np.assarray(aData.X.copy()), Y

def split_data_into_dataloaders_no_test(X, Y, train_size, batch_size = 64, num_workers = 0, seed = None):
    """
    Split X and Y into training set (fraction train_size) and the rest into a validation
    set. This assumes that you have already set aside a test set.
    Args:
        X (array): Input data
        Y (vector): Output labels
        train_size (float): 0 to 1, fraction of data for train set, remainder is validation set
        batch_size (int): defaults to 64
        num_workers (int): number of cores for multi-threading, defaults to 0 for no multi-threading
        seed (int): defaults to none, set to reproduce experiments with same train/val split
    """
    if Y is not None:
        assert len(X) == len(Y)
    assert batch_size > 1
    
    assert train_size > 0
    assert train_size <= 1
    
    if seed is not None:
        np.random.seed(seed)
        
    slices = np.random.permutation(np.arange(X.shape[0]))
    train_end = int(train_size* len(X))

    train_indices = slices[:train_end]
    val_indices = slices[train_end:] 
    
    train_x = X[train_indices, :]
    val_x = X[val_indices, :]
    
    train_x = torch.Tensor(train_x)
    val_x = torch.Tensor(val_x)

    if Y is not None:
        train_y = Y[train_indices]
        val_y = Y[val_indices]
        train_y = torch.LongTensor(train_y)
        val_y = torch.LongTensor(val_y)
        train_dataloader = DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=batch_size, shuffle = True, num_workers = num_workers)
        val_dataloader = DataLoader(torch.utils.data.TensorDataset(val_x, val_y), batch_size=batch_size, shuffle = False, num_workers = num_workers)
    else:
        # X = Y in unsupervised
        train_dataloader = DataLoader(torch.utils.data.TensorDataset(train_x, train_x), batch_size=batch_size, shuffle = True, num_workers = num_workers)
        val_dataloader = DataLoader(torch.utils.data.TensorDataset(val_x, val_x), batch_size=batch_size, shuffle = False, num_workers = num_workers)
        
        
    if train_size == 1:
        val_dataloader = None
        val_indices = None
        
    return train_dataloader, val_dataloader, train_indices, val_indices

#TODO: extract functionality of this and " "_no_test to a helper function, code reuse
def split_data_into_dataloaders(X, y, train_size, val_size, batch_size = 64, num_workers = 0, seed = None):
    """
    Split X and Y into training set (fraction train_size), validation set (fraction val_size)
    and the rest into a test set. train_size + val_size must be less than 1.
    Args:
        X (array): Input data
        y (vector): Output labels
        train_size (float): 0 to 1, fraction of data for train set
        val_size (float): 0 to 1, fraction of data for validation set
        batch_size (int): defaults to 64
        num_workers (int): number of cores for multi-threading, defaults to 0 for no multi-threading
        seed (int): defaults to none, set to reproduce experiments with same train/val split
    """
    assert train_size + val_size < 1
    assert len(X) == len(y)
    assert batch_size > 1
    
    if seed is not None:
        np.random.seed(seed)

    test_size = 1 - train_size - val_size

    slices = np.random.permutation(np.arange(X.shape[0]))
    train_end = int(train_size* len(X))
    val_end = int((train_size + val_size)*len(X))

    train_indices = slices[:train_end]
    val_indices = slices[train_end:val_end] 
    test_indices = slices[val_end:]
    
    train_x = X[train_indices, :]
    val_x = X[val_indices, :]
    test_x = X[test_indices, :]
    
    train_y = y[train_indices]
    val_y = y[val_indices]
    test_y = y[test_indices]

    train_x = torch.Tensor(train_x)
    val_x = torch.Tensor(val_x)
    test_x = torch.Tensor(test_x)

    train_y = torch.LongTensor(train_y)
    val_y = torch.LongTensor(val_y)
    test_y = torch.LongTensor(test_y)

    train_dataloader = DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=batch_size, shuffle = True, num_workers = num_workers)
    val_dataloader = DataLoader(torch.utils.data.TensorDataset(val_x, val_y), batch_size=batch_size, shuffle = False, num_workers = num_workers)
    test_dataloader = DataLoader(torch.utils.data.TensorDataset(test_x, test_y), batch_size=batch_size, shuffle = False, num_workers = num_workers)

    return train_dataloader, val_dataloader, test_dataloader, train_indices, val_indices, test_indices


def generate_synthetic_data_with_noise(N, z_size, n_classes, D, D_noise = None, seed = None):
    if not D_noise:
        D_noise = D

    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    class_logit = torch.rand(n_classes)
    with torch.no_grad():
        class_priors = torch.nn.functional.softmax(class_logit, dim  = -1).numpy()
    
    class_samp = np.random.choice(a = np.arange(n_classes), size = N, p = class_priors)
    
    data_mapper = nn.Sequential(
        nn.Linear(z_size, 2 * z_size, bias=False),
        nn.Tanh(),
        nn.Linear(2 * z_size, D, bias = True),
        nn.LeakyReLU()
        )
    
    data_mapper.requires_grad_(False)
    stackss  = []
    ys = []
    class_numbers = np.bincount(class_samp)
    for i in range(n_classes):
        n_i = class_numbers[i]
        mean = np.random.normal(0, 1)
        var = np.random.uniform(0.5, 1.5)
        latent_data = torch.normal(mean, var, size = (n_i, z_size))
        stackss.append(latent_data)
        ys.append(i * torch.ones(n_i))
    
    X = torch.cat(stackss)
    Y = torch.cat(ys)
    
    data_mapper.requires_grad_(False)
    X.requires_grad_(False)
    X = data_mapper(X)
    
    
    noise_features = torch.empty(N * D_noise).normal_(mean=0,std=0.3).reshape(N, D_noise)
    X = torch.cat([X, noise_features], dim = 1)
    
    X = X.numpy()
    Y = Y.numpy()

    return X, Y
