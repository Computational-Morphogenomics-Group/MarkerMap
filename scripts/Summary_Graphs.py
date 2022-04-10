#!/usr/bin/env python
# coding: utf-8

# Let's just get a quick sparsity overview of the methods so far.

# In[1]:


import torch
from torch.utils.data import DataLoader

from torchvision import datasets
import torchvision.transforms as transforms


from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

import numpy as np

from torchvision.utils import save_image

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import math

import gc

from utils import *


# In[2]:


import os
from os import listdir


# In[3]:


#BASE_PATH_DATA = '../data/'
BASE_PATH_DATA = '/scratch/ns3429/sparse-subset/data/'


# In[4]:


n_epochs = 25
batch_size = 64
lr = 0.0001
b1 = 0.9
b2 = 0.999


z_size = 100
hidden_size = 500


# from running
# EPSILON = np.finfo(tf.float32.as_numpy_dtype).tiny
#EPSILON = 1.1754944e-38
EPSILON = 1e-10


# In[5]:


cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

device = torch.device("cuda:0" if cuda else "cpu")
#device = 'cpu'


# In[6]:


print("Device")
print(device)


# In[7]:


np.random.seed(100)


# In[8]:


import scipy.io as sio


# In[9]:


a = sio.loadmat(BASE_PATH_DATA + 'zeisel/zeisel_data.mat')
data= a['zeisel_data'].T
N,d=data.shape


#load labels (first level of the hierarchy) from file
a = sio.loadmat(BASE_PATH_DATA + 'zeisel/zeisel_labels1.mat')
l_aux = a['zeisel_labels1']
l_0=[l_aux[i][0] for i in range(l_aux.shape[0])]
#load labels (second level of the hierarchy) from file
a = sio.loadmat(BASE_PATH_DATA + 'zeisel/zeisel_labels2.mat')
l_aux = a['zeisel_labels2']
l_1=[l_aux[i][0] for i in range(l_aux.shape[0])]
#construct an array with hierarchy labels
labels=np.array([l_0, l_1])

# load names from file 
a = sio.loadmat(BASE_PATH_DATA + 'zeisel/zeisel_names.mat')
names=np.array([a['zeisel_names'][i][0][0] for i in range(N)])


# In[10]:


for i in range(d):
    #data[i,:]=data[i,:]/np.linalg.norm(data[i,:])
    #mi = np.mean(data[:,i])
    #std = np.std(data[:,i])
    #data[:,i] = (data[:,i] - mi) / std
    ma = np.max(data[:,i])
    mi = np.min(data[:,i])
    data[:, i] = (data[:, i] - mi) / (ma - mi)


# In[11]:


data[data!=0].min()


# In[12]:


input_size = d


# In[13]:


slices = np.random.permutation(np.arange(data.shape[0]))
upto = int(.8 * len(data))

train_data = data[slices[:upto]]
test_data = data[slices[upto:]]

train_data = Tensor(train_data).to(device)
test_data = Tensor(test_data).to(device)


# In[14]:


train_labels = names[slices[:upto]]
test_labels = names[slices[upto:]]


# In[15]:


print(train_data.std(dim = 0).mean())
print(test_data.std(dim = 0).mean())


# Does L1 work if we normalize after every step?

# In[37]:


model_l1_diag = VAE_l1_diag(input_size, hidden_size, z_size)

model_l1_diag.to(device)
model_l1_optimizer = torch.optim.Adam(model_l1_diag.parameters(), 
                                            lr=lr,
                                            betas = (b1,b2))


# In[38]:


for epoch in range(1, n_epochs + 1):
        train_l1(train_data, model_l1_diag, model_l1_optimizer, epoch, batch_size)
        test(test_data, model_l1_diag, epoch, batch_size)


# In[39]:


bins = [10**(-i) for i in range(10)]
bins.reverse()
bins += [10]
print(np.histogram(model_l1_diag.diag.abs().clone().detach().cpu().numpy(), bins = bins))


# In[49]:


quick_model_summary(model_l1_diag, train_data, test_data, 0.15, batch_size)


# In[41]:


model_l1_diag(test_data[0:64])[0].std(dim = 0)


# In[42]:


test_data[0:64].std(dim = 0)


# Let's see latent representations real quick

# In[43]:


test_latent = model_l1_diag(test_data)[1]


# In[44]:


test_latent = test_latent.clone().detach().cpu().numpy()


# In[46]:


test_latent_clusters = TSNE(n_components=2, perplexity=30).fit_transform(test_latent)


# In[47]:


fig, ax = plt.subplots(figsize=(10, 5))
scatter_x = test_latent_clusters[:,0]
scatter_y = test_latent_clusters[:,1]
for g in np.unique(test_labels):
    ix = np.where(test_labels == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], label = g, s = 100)
ax.legend(loc=(1.05, 0.5))
plt.tight_layout()
plt.show()


# **First try Pretrained VAE and then gumble trick with it**
# 
# **Then try joint training VAE and Gumbel Model**

# # Pretrain VAE First

# In[21]:


pretrain_vae = VAE(input_size, hidden_size, z_size)

pretrain_vae.to(device)
pretrain_vae_optimizer = torch.optim.Adam(pretrain_vae.parameters(), 
                                            lr=lr,
                                            betas = (b1,b2))


# In[22]:


for epoch in range(1, n_epochs + 1):
        train(train_data, pretrain_vae, pretrain_vae_optimizer, epoch, batch_size)
        test(test_data, pretrain_vae, epoch, batch_size)


# In[23]:


quick_model_summary(pretrain_vae, train_data, test_data, 0.15, batch_size)


# In[24]:


pretrain_vae(test_data[0:64])[0]


# In[25]:


for p in pretrain_vae.parameters():
    p.requires_grad = False


# In[26]:


pretrain_vae.requires_grad_(False)


# ## Train Gumbel with the Pre-Trained VAE

# In[27]:


vae_gumbel_with_pre = VAE_Gumbel(input_size, hidden_size, z_size, k = 50)
vae_gumbel_with_pre.to(device)
vae_gumbel_with_pre_optimizer = torch.optim.Adam(vae_gumbel_with_pre.parameters(), 
                                                lr=lr, 
                                                betas = (b1,b2))


# In[28]:


for epoch in range(1, n_epochs + 1):
    train_pre_trained(train_data, vae_gumbel_with_pre, vae_gumbel_with_pre_optimizer,
                      epoch, pretrain_vae, batch_size)
    test(test_data, vae_gumbel_with_pre, epoch, batch_size)


# In[ ]:


quick_model_summary(vae_gumbel_with_pre, train_data, test_data, 0.15, batch_size)


# # Joint Training

# In[ ]:


joint_vanilla_vae = VAE(input_size, hidden_size, z_size)
joint_vanilla_vae.to(device)

joint_vae_gumbel = VAE_Gumbel(input_size, hidden_size, z_size, k = 50)
joint_vae_gumbel.to(device)


joint_optimizer = torch.optim.Adam(list(joint_vanilla_vae.parameters()) + list(joint_vae_gumbel.parameters()), 
                                                lr=lr, 
                                                betas = (b1,b2))


# In[ ]:


for epoch in range(1, n_epochs + 1):
    train_joint(train_data, joint_vanilla_vae, joint_vae_gumbel, joint_optimizer, epoch, batch_size)
    test_joint(test_data, joint_vanilla_vae, joint_vae_gumbel, epoch, batch_size)


# In[ ]:


quick_model_summary(joint_vae_gumbel, train_data, test_data, 0.15, batch_size)


# In[ ]:


del joint_vanilla_vae


# ### Let's actually Graph this.
# 
# ### Try it out at Gumbel sparsity of k = 10, 25, 50, 100, 250
# 
# ### Graph Test MSE Loss
# 
# ## Graph the mean activations at k = 50

# In[30]:


def graph_activations(test_data, model, title, fname):
    preds, _, _ = model(test_data)
    
    preds[preds < 0.09] = 0
    pred_activations = preds.mean(dim = 0)
    
    test_activations = test_data.mean(dim = 0)
    
    x = np.arange(input_size) + 1
    
    fig = plt.figure()
    plt.plot(x, pred_activations.clone().detach().cpu().numpy(), label = 'Average Predictions')
    plt.plot(x, test_activations.clone().detach().cpu().numpy(), label = 'Average Test Data')
    
    plt.title(title)
    plt.ylim([-0.1, 1.1])
    plt.xlabel("Feature Index")
    plt.ylabel("Average Activation of Feature")
    
    plt.legend()
    plt.savefig(fname)


# In[31]:


def graph_sparsity(test_data, model, title, fname):
    preds, _, _ = model(test_data)
    
    preds[preds < 0.15] = 0
    preds[preds >= 0.15] = 1
    
    pred_count = preds.sum(dim = 0) / len(test_data)
    
    test_count = test_data.sum(dim = 0) / len(test_data)
    
    x = np.arange(input_size) + 1
    
    fig = plt.figure()
    plt.plot(x, pred_count.clone().detach().cpu().numpy(), label = 'Count NonZero Predictions')
    plt.plot(x, test_count.clone().detach().cpu().numpy(), label = 'Count NonZero Test Data')
    
    plt.title(title)
    plt.ylim([-0.1, 1.1])
    plt.xlabel("Feature Index")
    plt.ylabel("Proportion of Test Set Feature Was not Sparse")
    
    plt.legend()
    plt.savefig(fname)


# In[32]:


graph_activations(test_data, model_l1_diag, 'VAE L1 Preds vs Test Data Means', 
                  BASE_PATH_DATA + 'vae_l1_activations.png')
graph_sparsity(test_data, model_l1_diag, 'VAE L1 Preds vs Test Data Sparsity', 
                  BASE_PATH_DATA + 'vae_l1_sparsity.png')

del model_l1_diag


# In[ ]:


graph_activations(test_data, joint_vae_gumbel, 'Joint Gumbel vs Test Means', 
                  BASE_PATH_DATA + 'joint_gumbel_activations.png')
graph_sparsity(test_data, joint_vae_gumbel, 'Joint Gumbel vs Test Sparsity', 
                  BASE_PATH_DATA + 'joint_gumbel_sparsity.png')

del joint_vae_gumbel


# In[ ]:


graph_activations(test_data, vae_gumbel_with_pre, 'Gumbel Matching Pretrained VAE vs Test Means', 
                  BASE_PATH_DATA + 'pretrained_gumbel_activations.png')
graph_sparsity(test_data, vae_gumbel_with_pre, 'Gumbel Matching Pretrained VAE vs Test Sparsity', 
                  BASE_PATH_DATA + 'pretrained_gumbel_sparsity.png')

del vae_gumbel_with_pre


# In[ ]:


k_all = [5, 10, 25, 50, 75, 100, 150]#, 250, 500, 1000, 2000, 3000]
n_trials = 10


# In[ ]:


losses_pre = []
losses_joint = []


# In[ ]:


for k in k_all:
    current_k_pre_losses = []
    current_k_joint_losses = []
    for trial_i in range(n_trials):
        print("RUNNING for K {} Trial {}".format(k, trial_i), flush=True)
        vae_gumbel_with_pre = VAE_Gumbel(input_size, hidden_size, z_size, k = k)
        vae_gumbel_with_pre.to(device)
        vae_gumbel_with_pre_optimizer = torch.optim.Adam(vae_gumbel_with_pre.parameters(), 
                                                        lr=lr, 
                                                        betas = (b1,b2))
    
        joint_vanilla_vae = VAE(input_size, hidden_size, z_size)
        joint_vanilla_vae.to(device)

        joint_vae_gumbel = VAE_Gumbel(input_size, hidden_size, z_size, k = k)
        joint_vae_gumbel.to(device)


        joint_optimizer = torch.optim.Adam(list(joint_vanilla_vae.parameters()) + 
                                           list(joint_vae_gumbel.parameters()),
                                                lr=lr, 
                                                betas = (b1,b2))
    
        for epoch in (1, n_epochs + 1):
            train_pre_trained(train_data, vae_gumbel_with_pre, vae_gumbel_with_pre_optimizer, 
                              epoch, pretrain_vae, batch_size)
            train_joint(train_data, joint_vanilla_vae, joint_vae_gumbel, joint_optimizer, epoch, batch_size)
    
        test_loss_pre = 0
        test_loss_joint = 0
        
        inds = np.arange(test_data.shape[0])
        with torch.no_grad():
            for i in range(math.ceil(len(test_data)/batch_size)):
                batch_ind = inds[i * batch_size : (i+1) * batch_size]
                batch_data = test_data[batch_ind, :]
                
                test_pred_pre = vae_gumbel_with_pre(batch_data)[0]
                test_pred_joint = joint_vae_gumbel(batch_data)[0]
                
                test_pred_pre[test_pred_pre < 0.09] = 0
                test_pred_joint[test_pred_joint < 0.09] = 0
                
                test_loss_pre += F.binary_cross_entropy(test_pred_pre, batch_data, reduction='mean')
                test_loss_joint += F.binary_cross_entropy(test_pred_joint, batch_data, reduction='mean')
                
                del batch_data
            
        #test_loss_pre /= len(test_data)
        #test_loss_joint /= len(test_data)
        current_k_pre_losses.append(test_loss_pre.cpu().item())
        current_k_joint_losses.append(test_loss_joint.cpu().item())
        
        # for freeing memory faster
        del vae_gumbel_with_pre
        del vae_gumbel_with_pre_optimizer
        del joint_vanilla_vae
        del joint_vae_gumbel
        del joint_optimizer

        torch.cuda.empty_cache()
        
    
    losses_pre.append(np.mean(current_k_pre_losses))
    losses_joint.append(np.mean(current_k_joint_losses))
    
    
    
fig = plt.figure()
plt.plot(k_all, losses_pre, label = 'Average BCE Losses with Gumbel Matching Pretrained')
plt.plot(k_all, losses_joint, label = 'Average BCE Losses with Gumbel Joint Training')

plt.title("Effect on Sparsity on BCE Loss")
plt.xlabel('Sparsity Level (Number of Non-Zero Features)')
plt.ylabel('Per Neuron Average BCE Loss')
plt.legend()

plt.savefig(BASE_PATH_DATA + 'comparing_across_sparsity.png')

