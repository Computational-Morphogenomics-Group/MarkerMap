# MarkerMap

<p align="left">
  <img src="marker_map.png" width="320" title="logo">
</p>

MarkerMap is a generative model for selecting the most informative gene markers by projecting cells into a shared, interpretable embedding without sacrificing accuracy.

## Table of Contents
1. [Installation](#installation)
	1. [MacOs](#macos)
	2. [Windows](#windows)
2. [Quick Start](#quick-start)
	1. [Simple Example](#simple-example)
	2. [Benchmark Example](#benchmark-example)
3. [Features](#features)
4. [For Developers](#for-developers)
	1. [Adding Your Model](#adding-your-model)
5. [License](#license)

## Installation

### MacOS
- The easiest way to install is with pip from https://pypi.org/project/markermap/
- Simply do `pip install markermap`
- To use the persist package, you will need to install it following the instructions: https://github.com/iancovert/persist/ 
- Note: If using on Google Colab, you may have to restart the runtime after installing because `scanpy` installs a newer version of matplotlib than the default one.
- Note: Smashpy specifies an older version of tensorflow (==2.5.0), so this creates installation problems. Thus we do not require installing Smashpy to use the package, but it will need to be installed if you want to compare that method.

### Windows
- Coming soon!

## Quick Start <a name="quick-start"/>

### Simple Example <a name="simple-example"/>
Copy the code from here, or take a look at `scripts/quick_start.py` for a python script or `notebooks/quick_start.ipynb` for a Jupyter Notebook.

#### Imports
Data is handled with numpy and with scanpy which is used for many computational biology datasets.

```
import numpy as np
import scanpy as sc

from markermap.vae_models import MarkerMap, train_model
from markermap.utils import (
    new_model_metrics,
    parse_adata,
    plot_confusion_matrix,
    split_data_into_dataloaders,
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

### Benchmark Example <a name="benchmark-example"/>
Now we will do an example where we use the benchmarking tools of the package. Follows the steps from the Simple Example through the data section, then pick up here. Alternatively, checkout out `scripts/quick_start_benchmark.py` for a python script or `notebooks/quick_start_benchmark.ipynb` for a Jupyter Notebook.

#### Define the Models
Now it is time to define all the models that we are benchmarking. For this tutorial, we will benchmark the three versions of MarkerMap: Supervised, Mixed, and Unsupervised. Each model in this repository comes with a function `getBenchmarker` where we specify all the parameters used for constructing the model and all the parameters used for training the model. The benchmark function will then run and evaluate them all. For this tutorial we will also specify a train argument, `max_epochs` which limits the number of epochs during training.

```
supervised_marker_map = MarkerMap.getBenchmarker(
  create_kwargs = {
    'input_size': X.shape[1],
    'hidden_layer_size': hidden_layer_size,
    'z_size': z_size,
    'num_classes': len(np.unique(y)),
    'k': k_range[0], # because we are benchmarking over k, this will get replaced by the benchmark function
    'loss_tradeoff': 0,
  },
  train_kwargs = {
    'max_epochs': max_epochs,
  }
)

mixed_marker_map = MarkerMap.getBenchmarker(
  create_kwargs = {
    'input_size': X.shape[1],
    'hidden_layer_size': hidden_layer_size,
    'z_size': z_size,
    'num_classes': len(np.unique(y)),
    'k': k_range[0],
    'loss_tradeoff': 0.5,
  },
  train_kwargs = {
    'max_epochs': max_epochs,
  }
)

unsupervised_marker_map = MarkerMap.getBenchmarker(
  create_kwargs = {
    'input_size': X.shape[1],
    'hidden_layer_size': hidden_layer_size,
    'z_size': z_size,
    'num_classes': None, #since it is unsupervised, we can just say that the number of classes is None
    'k': k_range[0],
    'loss_tradeoff': 1.0,
  },
  train_kwargs = {
    'max_epochs': max_epochs,
  },
)
```

#### Run the Benchmark
Finally, we run the benchmark by passing in all the models as a dictionary, the number of times to run the model, the data and labels, the type of benchmark, and the range of values we are benchmarking over. We will set the range of k values as `[10, 25, 50]`, but you may want to go higher in practice. Note that we also add the RandomBaseline model. This model selects k markers at random, and it is always a good idea to include this one because it performs better than might be expected. It is also very fast, so there is little downside.

The benchmark function splits the data, runs each model with the specified k, then trains a simple model on just the k markers and evaluates how they perform on a test set that was not used to train the marker selection model or the simple evaluation model. The results reported have accuracy and f1 score, and we can visualize them in a plot with the plot_benchmarks function.

This function will train each MarkerMap 3 times for a total of 9 runs, so it may take over 10 minutes depending on your hardware. Feel free to comment out some of the models.

```
k_range = [10, 25, 50]

results, benchmark_label, benchmark_range = benchmark(
  {
    'Unsupervised Marker Map': unsupervised_marker_map,
    'Supervised Marker Map': supervised_marker_map,
    'Mixed Marker Map': mixed_marker_map,
    'Baseline': RandomBaseline.getBenchmarker(train_kwargs = { 'k': k_range[0] }),
  },
  1, # num_times, how many different random train/test splits to run the models on
  X,
  y,
  benchmark='k',
  benchmark_range=k_range,
)

plot_benchmarks(results, benchmark_label, benchmark_range, mode='accuracy')
```

## Features

- The MarkerMap package provides functionality to easily benchmark different marker selection methods to evaluate performance under a number of metrics. Each model has a `getBenchmarker` function which takes model constructor parameters and trainer parameters and returns a model function. The `benchmark` function then takes all these model functions, a dataset, and the desired type of benchmarking and runs all the models to easily compare performance. See `scripts/benchmark_k` for examples.
- Types of benchmarking:
  - k: The number of markers to select from
  - label_error: Given a range of percentages, pick that percent of points in the training + validation set and set their label to a random label form among the existing labels.
- To load the data, you can make use of the following functions: `get_citeseq`, `get_mouse_brain`, `get_paul`, and `get_zeisel`. Note that both `get_mouse_brain` and `get_paul` do some pre-processing, including removing outliers and normalizing the data in the case of Mouse Brain.

## For Developers <a name="for-developers"/>

- You will want to set up this library as an editable install.
  - Clone the repository `git clone https://github.com/Computational-Morphogenomics-Group/MarkerMap.git`
  - Navigate to the MarkerMap directory `cd MarkerMap`
  - Locally install the package `pip install -e .` (may have to use pip3 if your system has both python2 and python3 installed)
- If you are going to be developing this package, also install the following: `pip install pre-commit pytest`
- In the root directory, run `pre-commit install`. You should see a line like `pre-commit installed at .git/hooks/pre-commit`. Now when you commit to your local branch, it will run `jupyter nbconvert --clean-output` on all the local jupyter notebooks on that branch. This ensures that only clean notebooks are uploaded to the github.
- To run tests, simply run pytest: `pytest`.

### Adding Your Model <a name="adding-your-model"/>
- The MarkerMap project is built to allow for easy comparison between different types of marker selection algorithms. If you would like to benchmark your method against the others, the process is straightforward:
  - Suppose your method is called Acme. Create a class `AcmeWrapper` in other_models.py that extends Acme and `BenchmarkableModel` from other_models.py.
  - In `AcmeWrapper`, implement `benchmarkerFunctional`, check out the other methods in that function for a full list of arguments. The most important two are `create_kwargs` and `train_kwargs` which are dictionaries of arguments. The `create_kwargs` is arguments that when be called when initializes the class, i.e. `Acme(**train_kwargs)`, and the `train_kwargs` are arguments that would be called when training the model. The `benchmarkerFunctional` method should return an array of the indices of the markers.
  - Voila! Now you can define the model with `AcmeWrapper.getBenchmarker(create_kwargs, train_kwargs)`. This will return a model functional that you pass to the `benchmark` function. See quick_start_benchmark.py for an example.
- If you would like to add your super duper model to the repository, create a branch and submit a Pull Request.

## License
- This project is licensed under the terms of the MIT License.
