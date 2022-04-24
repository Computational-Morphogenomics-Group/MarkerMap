# MarkerMap

MarkerMap is a generative model for selecting the most informative gene markers by projecting cells into a shared, interpretable embedding without sacrificing accuracy.

## Installation

### MacOS
- Clone the repository `git clone https://github.com/Computational-Morphogenomics-Group/MarkerMap.git`
- Navigate to the MarkerMap directory `cd MarkerMap`
- Locally install the package `pip install -e .` (may have to use pip3 if your system has both python2 and python3 installed)
- You might have to install libomp with homebrew, `brew install libomp`

### Windows
- Coming soon!

## Quick Start

### Simple Example
Copy the code from here, or take a look at `scripts/quick_start.py` for a python script or `notebooks/quick_start.ipynb` for a Jupyter Notebook.

#### Imports
Data is handled with numpy and with scanpy which is used for many computational biology datasets.

```
import numpy as np
import scanpy as sc

from markermap.utils import MarkerMap
from markermap.utils import (
    new_model_metrics,
    parse_adata,
    plot_confusion_matrix,
    split_data_into_dataloaders,
    train_model,
)
```

#### Set Parameters
Define some parameters that we will use when creating the MarkerMap.
* z_size is the dimension of the latent space in the variational auto-encoder. We always use 16
* hidden_layer_size is the dimension of the hidden layers in the auto-encoder that come before and after the latent space layer. This is dependent on the data, a good rule of thumb is ~10% of the dimension of the input data. For the CITE-seq data which has 500 columns, we will use 64
* k is the number of markers to extract
* Set the file_path to wherever your data is

```
z_size = 16
hidden_layer_size = 64
k=50

file_path = 'data/cite_seq/CITEseq.h5ad'
```

#### Data
Set file_path to wherever your data is located. We then read in the data using scanpy and break it into X and y using the parse_data function. The text labels in adata.obs['annotation'] will be converted to number labels so that MarkerMap can use them properly.

We then split the data into training, validation, and test sets with a 70%, 10%, 20% split. MarkerMap uses a validation set during the training process.

```
file_path = '../data/cite_seq/CITEseq.h5ad'

adata = sc.read_h5ad(file_path)
adata.obs['annotation'] = adata.obs['names']
X, y, encoder = parse_adata(adata)

train_dataloader, val_dataloader, _, train_indices, val_indices, test_indices = split_data_into_dataloaders(
    X,
    y,
    train_size=0.7,
    val_size=0.1,
)
X.shape
```

#### Define and Train the Model
Now it is time to define the MarkerMap. There are many hyperparameters than can be tuned here, but the most important are k and the loss_tradeoff. The k parameter may require some domain knowledge, but it is fairly easy to benchmark for different levels of k, as we will see in the later examples. Loss_tradeoff is also important, see the paper for a further discussion. In general, we have 3 levels, 0 (supervised only), 0.5 (mixed supervised-unsupervised) and 1 (unsupervised only). This step may take a couple of minutes.

```
supervised_marker_map = MarkerMap(X.shape[1], hidden_layer_size, z_size, len(np.unique(y)), k, loss_tradeoff=0)
train_model(supervised_marker_map, train_dataloader, val_dataloader)
```

#### Evaluate the model
Finally, we test the model. The new_model_metrics function trains a simple model such as a RandomForestClassifer on the training data restricted to the k markers, and then evaluates it on the testing data. We then print the misclassification rate, the f1-score, and plot a confusion matrix.

```
misclass_rate, test_rep, cm = new_model_metrics(
    X[np.concatenate([train_indices, val_indices]), :],
    y[np.concatenate([train_indices, val_indices])],
    X[test_indices, :],
    y[test_indices],
    markers = supervised_marker_map.markers().clone().cpu().detach().numpy(),
)

print(misclass_rate)
print(test_rep['weighted avg']['f1-score'])
plot_confusion_matrix(cm, encoder.classes_)
```

## Features

- The MarkerMap package provides functionality to easily benchmark different marker selection methods to evaluate performance under a number of metrics. Each model has a `getBenchmarker` function which takes model constructor parameters and trainer parameters and returns a model function. The `benchmark` function then takes all these model functions, a dataset, and the desired type of benchmarking and runs all the models to easily compare performance. See `scripts/benchmark_k` for examples.
- Types of benchmarking:
  - k: The number of markers to select from
  - label_error: Given a range of percentages, pick that percent of points in the training + validation set and set their label to a random label form among the existing labels.

## For Developers

- If you are going to be developing this package, also install the following: `pip install pre-commit pytest`
- In the root directory, run `pre-commit install`. You should see a line like `pre-commit installed at .git/hooks/pre-commit`. Now when you commit to your local branch, it will run `jupyter nbconvert --clean-output` on all the local jupyter notebooks on that branch. This ensures that only clean notebooks are uploaded to the github.
- To run tests, simply run pytest: `pytest`.

## License
- This project is licensed under the terms of the MIT License.
