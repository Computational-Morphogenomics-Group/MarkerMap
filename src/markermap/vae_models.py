import math
import logging
import os
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from sklearn.ensemble import RandomForestClassifier

import markermap.other_models as other_models
import markermap.utils as utils


# rounding up lowest float32 on my system
EPSILON = 1e-40
MIN_TEMP = 0.0001


def form_block(in_size, out_size, batch_norm = True, bias = True):
    """
    Constructs a fully connected layer with bias, batch norm, and then leaky relu activation function
    args:
        in_size (int): layer input size
        out_size (int): layer output size
        batch_norm (bool): use the batch norm in the layers, defaults to True
        bias (bool): add a bias to the layers, defaults to True
    returns (array): the layers specified
    """
    layers = []
    layers.append(nn.Linear(in_size, out_size, bias = bias))
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_size))
    layers.append(nn.LeakyReLU())
    return layers


def make_encoder(input_size, hidden_layer_size, z_size, bias = True, batch_norm = True):
    """
    Construct encoder with 2 hidden layer used in VAE.
    args:
        input_size (int): Length of the input vector
        hidden_size (int): Size of the hidden layers
        z_size (int): size of encoded layer, latent size
        bias (bool): add a bias to the layers, defaults to True
        batch_norm (bool): use the batch norm in the layers, defaults to True
    returns: torch.nn.Sequential that encodes the input, the output layer for the mean, the output layer for the logvar
    """
    main_enc = nn.Sequential(
            *form_block(input_size, hidden_layer_size, bias = bias, batch_norm = batch_norm),
            *form_block(hidden_layer_size, hidden_layer_size, bias = bias, batch_norm = batch_norm),
            *form_block(hidden_layer_size, hidden_layer_size, bias = bias, batch_norm = False)
            )

    enc_mean = nn.Linear(hidden_layer_size, z_size, bias = bias)
    enc_logvar = nn.Linear(hidden_layer_size, z_size, bias = bias)

    return main_enc, enc_mean, enc_logvar


def make_bernoulli_decoder(output_size, hidden_size, z_size, bias = True, batch_norm = True):
    """
    Construct bernoulli decoder with 1 hidden layer used in VAE. See Appendix C.1: https://arxiv.org/pdf/1312.6114.pdf
    args:
        output_size (int): Size of the reconstructed output of the VAE, probably the same as the input size
        hidden_size (int): Size of the hidden layer
        z_size (int): size of encoded layer, latent size
        bias (bool): add a bias to the layers, defaults to True
        batch_norm (bool): use the batch norm in the layers, defaults to True
    returns: torch.nn.Sequential that decodes the encoded representation
    """
    return nn.Sequential(
        *form_block(z_size, hidden_size, bias = bias, batch_norm = batch_norm),
        nn.Linear(1*hidden_size, output_size, bias = bias),
        nn.Sigmoid()
    )

def make_gaussian_decoder(output_size, hidden_size, z_size, bias = True, batch_norm = True):
    """
    Construct gaussian decoder with 1 hidden layer used in VAE. See Appendix C.2: https://arxiv.org/pdf/1312.6114.pdf
    args:
        output_size (int): Size of the reconstructed output of the VAE, probably the same as the input size
        hidden_size (int): Size of the hidden layer
        z_size (int): size of encoded layer, latent size
        bias (bool): add a bias to the layers, defaults to True
        batch_norm (bool): use the batch norm in the layers, defaults to True
    returns: torch.nn.Sequential that decodes the encoded representation
    """
    return nn.Sequential(
        *form_block(z_size, hidden_size, bias = bias, batch_norm = batch_norm),
        nn.Linear(hidden_size, output_size, bias = bias),
    )

class VAE(pl.LightningModule, other_models.BenchmarkableModel):
    def __init__(
        self,
        input_size,
        hidden_layer_size,
        z_size,
        output_size = None,
        bias = True,
        batch_norm = True,
        lr = 0.000001,
        kl_beta = 0.1,
        decoder = None,
        dec_logvar = None,
    ):
        super(VAE, self).__init__()
        self.save_hyperparameters()

        if output_size is None:
            output_size = input_size

        self.encoder, self.enc_mean, self.enc_logvar = make_encoder(input_size,
                hidden_layer_size, z_size, bias = bias, batch_norm = batch_norm)

        if (decoder is not None and dec_logvar is None) or (decoder is None and dec_logvar is not None):
            print(
                'VAE::__init__: WARNING! If decoder is specified, dec_logvar should also be specified, and vice versa'
            )

        if decoder is None:
            decoder = make_gaussian_decoder(
                output_size,
                hidden_layer_size,
                z_size,
                bias = bias,
                batch_norm = batch_norm,
            )

        if dec_logvar is None:
            dec_logvar = make_gaussian_decoder(
                output_size,
                hidden_layer_size,
                z_size,
                bias = bias,
                batch_norm = batch_norm,
            )

        self.decoder = decoder
        self.dec_logvar = dec_logvar

        self.lr = lr
        self.kl_beta = kl_beta
        self.batch_norm = batch_norm
        self.training_step_outputs = []

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
        return { 
            'optimizer': torch.optim.Adam(self.parameters(), lr = self.lr),
        }
    
    @classmethod 
    def prepareData(cls, adata, train_indices, val_indices, group_by, layer, batch_size):
        X_train, y_train, X_val, y_val, adata = super().prepareData(
            adata, 
            train_indices, 
            val_indices, 
            group_by, 
            layer,
        )

        train_dataloader = utils.get_dataloader(X_train, y_train, batch_size, shuffle=True)
        val_dataloader = utils.get_dataloader(X_val, y_val, batch_size, shuffle=False)
        return train_dataloader, val_dataloader

class VAE_l1_diag(VAE):
    def __init__(
        self,
        input_size,
        hidden_layer_size,
        z_size,
        bias = True,
        batch_norm = True,
        lr = 0.000001,
        kl_beta = 0.1,
        l1_lambda = 1,
        decoder = None,
        dec_logvar = None,
    ):
        super(VAE_l1_diag, self).__init__(
            input_size,
            hidden_layer_size,
            z_size,
            bias = bias,
            batch_norm = batch_norm,
            decoder = decoder,
            dec_logvar = dec_logvar,
        )
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
        train_dataloader, val_dataloader = cls.prepareData(
            adata, 
            train_indices, 
            val_indices, 
            group_by, 
            None, #layer, just use adata.X
            batch_size=batch_size,
        )
        if not k:
            k = train_kwargs['k']

        if 'k' in train_kwargs:
            train_kwargs = { **train_kwargs } #copy train_kwargs so later iterations have 'k'
            train_kwargs.pop('k')

        feature_std = torch.tensor(adata.X).std(dim = 0)
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
    def __init__(
        self,
        input_size,
        hidden_layer_size,
        z_size,
        k,
        t = 2,
        temperature_decay = 0.9,
        bias = True,
        batch_norm = True,
        lr = 0.000001,
        kl_beta = 0.1,
        min_temp = MIN_TEMP,
        decoder = None,
        dec_logvar = None,
    ):
        super(VAE_Gumbel, self).__init__(
            input_size,
            hidden_layer_size,
            z_size,
            bias = bias,
            batch_norm = batch_norm,
            lr = lr,
            kl_beta = kl_beta,
            decoder = decoder,
            dec_logvar = dec_logvar,
        )
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
        self.training_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        self.t = max(torch.as_tensor(self.min_temp, device = self.device), self.t * self.temperature_decay)

        loss = torch.stack([x for x in self.training_step_outputs]).mean()
        self.training_step_outputs = []
        self.log("epoch_avg_train_loss", loss)
        return None


# idea of having a Non Instance Wise Gumbel that also has a state to keep consistency across batches
# probably some repetititon of code, but the issue is this class stuff, this is python 3 tho so it can be put into a good wrapper
# that doesn't duplicate code
class VAE_Gumbel_GlobalGate(VAE):
    # alpha is for  the exponential average
    def __init__(
        self,
        input_size,
        hidden_layer_size,
        z_size,
        k,
        t = 0.01,
        temperature_decay = 0.99,
        bias = True,
        batch_norm = True,
        lr = 0.000001,
        kl_beta = 0.1,
        decoder = None,
        dec_logvar = None,
    ):
        super(VAE_Gumbel_GlobalGate, self).__init__(
            input_size,
            hidden_layer_size,
            z_size,
            bias = bias,
            batch_norm = batch_norm,
            lr = lr,
            kl_beta = kl_beta,
            decoder = decoder,
            dec_logvar = dec_logvar,
        )
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
        self.training_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        self.t = max(torch.as_tensor(0.001, device = self.device), self.t * self.temperature_decay)

        loss = torch.stack([x for x in self.training_step_outputs]).mean()
        self.training_step_outputs = []
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
        train_dataloader, val_dataloader = cls.prepareData(
            adata, 
            train_indices, 
            val_indices, 
            group_by, 
            None, #layer, just use adata.X
            batch_size=batch_size,
        )
        model = cls(**{**create_kwargs, 'k': k}) if k else cls(**create_kwargs)
        train_model(model, train_dataloader, val_dataloader, **train_kwargs)
        return model.markers().clone().cpu().detach().numpy()



# idea of having a Non Instance Wise Gumbel that also has a state to keep consistency across batches
# probably some repetititon of code, but the issue is this class stuff, this is python 3 tho so it can be put into a good wrapper
# that doesn't duplicate code
class VAE_Gumbel_RunningState(VAE_Gumbel):
    # alpha is for  the exponential average
    def __init__(
        self,
        input_size,
        hidden_layer_size,
        z_size,
        k,
        t = 0.01,
        temperature_decay = 0.99,
        method = 'mean',
        alpha = 0.9,
        bias = True,
        batch_norm = True,
        lr = 0.000001,
        kl_beta = 0.1,
        decoder = None,
        dec_logvar = None,
    ):
        super(VAE_Gumbel_RunningState, self).__init__(
            input_size,
            hidden_layer_size,
            z_size,
            k = k,
            t = t,
            temperature_decay = temperature_decay,
            bias = bias,
            batch_norm = batch_norm,
            lr = lr,
            kl_beta = kl_beta,
            decoder = decoder,
            dec_logvar = dec_logvar,
        )
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
        train_dataloader, val_dataloader = cls.prepareData(
            adata, 
            train_indices, 
            val_indices, 
            group_by, 
            None, #layer, just use adata.X
            batch_size=batch_size,
        )
        model = cls(**{**create_kwargs, 'k': k}) if k else cls(**create_kwargs)
        train_model(model, train_dataloader, val_dataloader, **train_kwargs)
        return model.markers().clone().cpu().detach().numpy()


# not doing multiple inheritance because GumbelClassifier is repeating itself
class MarkerMap(VAE_Gumbel_RunningState):

    def __init__(
        self,
        input_size,
        hidden_layer_size,
        z_size,
        num_classes,
        k,
        t = 3.0,
        temperature_decay = 0.95,
        method = 'mean',
        alpha = 0.95,
        bias = True,
        batch_norm = True,
        lr = 0.000001,
        kl_beta = 0.1,
        decoder = None,
        dec_logvar = None,
        loss_tradeoff = 0.5,
    ):
        assert loss_tradeoff <= 1
        assert loss_tradeoff >= 0
        if num_classes is None:
            assert loss_tradeoff == 1

        super(MarkerMap, self).__init__(
            input_size = input_size,
            hidden_layer_size = hidden_layer_size,
            z_size = z_size,
            k = k,
            t = t,
            temperature_decay = temperature_decay,
            method = method,
            alpha = alpha,
            batch_norm = batch_norm,
            bias = bias,
            lr = lr,
            kl_beta = kl_beta,
            decoder = decoder,
            dec_logvar = dec_logvar,
        )

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
        self.training_step_outputs.append(loss)
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
        train_dataloader, val_dataloader, train_indices, val_indices = utils.split_data_into_dataloaders_no_test(X, Y, train_size = train_ratio)
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

    def get_reconstruction(self, X):
        assert self.loss_tradeoff != 0

        X = torch.Tensor(X)
        X.to(self.device)
        with torch.no_grad():
            mu_x = self.forward(X)[0].cpu().numpy()
        return mu_x


# NMSL is Not My Selection Layer
# Implementing reference paper
class ConcreteVAE_NMSL(VAE):
    def __init__(
        self,
        input_size,
        hidden_layer_size,
        z_size,
        k,
        t = 0.01,
        temperature_decay = 0.99,
        bias = True,
        batch_norm = True,
        lr = 0.000001,
        kl_beta = 0.1,
        min_temp = MIN_TEMP,
        decoder = None,
        dec_logvar = None,
    ):
        # k because encoder actually uses k features as its input because of how concrete VAE picks it out
        super(ConcreteVAE_NMSL, self).__init__(
            k,
            hidden_layer_size,
            z_size,
            output_size = input_size,
            bias = bias,
            batch_norm = batch_norm,
            lr = lr,
            kl_beta = kl_beta,
            decoder = decoder,
            dec_logvar = dec_logvar,
        )
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
        self.training_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        self.t = max(torch.as_tensor(self.min_temp, device = self.device), self.t * self.temperature_decay)


        loss = torch.stack([x for x in self.training_step_outputs]).mean()
        self.training_step_outputs = []
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
        train_dataloader, val_dataloader = cls.prepareData(
            adata, 
            train_indices, 
            val_indices, 
            group_by, 
            None, #layer, just use adata.X
            batch_size=batch_size,
        )
        model = cls(**{**create_kwargs, 'k': k}) if k else cls(**create_kwargs)
        train_model(model, train_dataloader, val_dataloader, **train_kwargs)
        return model.markers().clone().cpu().detach().numpy()


### Model Trainers/Helpers

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

    callbacks = None
    #val_loss early stopping breaks when there is no validation set
    if not (val_dataloader is None or len(val_dataloader) == 0):
        early_stopping_callback = EarlyStopping(
            monitor='val_loss', 
            mode = 'min', 
            patience = early_stopping_patience,
            check_on_train_epoch_end=False, # only check on val_epoch_end
        )
        callbacks = [early_stopping_callback]

    if gpus is not None:
        accelerator = 'gpu'
        devices = gpus
    elif tpu_cores is not None:
        accelerator = 'tpu'
        devices = tpu_cores
    else:
        accelerator = 'auto'
        devices = 'auto'

    if auto_lr:
        lrfinder = pl.callbacks.LearningRateFinder(
            max_lr=max_lr, 
            mode=lr_explore_mode, 
            num_training_steps=num_lr_rates,
            early_stop_threshold=None,
        )
        if callbacks is None:
            callbacks = [lrfinder]
        else:
            callbacks.append(lrfinder)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        min_epochs = min_epochs, 
        max_epochs = max_epochs,
        callbacks=callbacks, 
        precision = precision, 
        logger = verbose,
        enable_model_summary=verbose, # turn off some summaries
        enable_progress_bar = verbose, 
        enable_checkpointing=False,
    )

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
