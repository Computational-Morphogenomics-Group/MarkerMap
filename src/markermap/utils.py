import gc
import numpy as np
import pandas as pd
from collections import defaultdict
import time
import queue


import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, classification_report

import umap
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
import anndata

import markermap.other_models as other_models
import markermap.vae_models as vae_models

class ExperimentIndices:
    def __init__(self, train_indices, val_indices, test_indices):
        pass

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

def recon_model_metrics(train_x, test_x, markers, model=None):
    if model is None:
        model = LinearRegression()

    markers_train_x = train_x[:, markers]
    markers_test_x = test_x[:, markers]
    reg = model.fit(markers_train_x, train_x)
    test_recon_x = reg.predict(markers_test_x)

    l2_loss = np.mean((test_x - test_recon_x)**2) 
    l1_loss = np.mean(np.abs(test_x - test_recon_x))
    # divide by num cells and num genes so we can compare across datasets

    return l2_loss, l1_loss

def model_variances(path, tries):
    misclass_arr = []
    weight_f1_arr = []
    for tryy in range(1, tries + 1):
        results = np.load(path.format(tryy), allow_pickle = True)
        misclass_arr.append(results[0])
        weight_f1_arr.append(results[1]['weighted avg']['f1-score'])
    return np.mean(misclass_arr), np.mean(weight_f1_arr), np.std(misclass_arr), np.std(weight_f1_arr)

def mislabel_points(adata, group_by, mislabel_percent, eligible_indices=None):
    assert mislabel_percent <= 1.0
    assert mislabel_percent >= 0.0

    y = adata.obs[group_by]

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

    adata.obs['mislabelled_annotation'] = y_err #will this even work

    return adata

def benchmark(
    models,
    num_times,
    adata,
    benchmark,
    group_by = 'annotation',
    train_size = 0.7,
    val_size = 0.1,
    batch_size = 64,
    save_file=None,
    benchmark_range=None,
    eval_type='classify',
    eval_model=None,
    min_groups=None,
):
    """
    Benchmark a collection of models by a benchmark param on data X,y. If save_file is specified, results are saved
    when a complete benchmark_range is complete
    args:
        models (dict): maps model labels to a function that runs the model on the data and returns markers
            use model.getBenchmarker() to automatically generate those functions
        num_times (int): number of random data splits to run the model on
        adata (AnnData object): input and label data
        benchmark (string): type of benchmarking to do, must be one of {'k', 'label_error', 'label_error_markers_only'}
        group_by (string): string key for adata.obs[group_by] where the output labels live, defaults to annotation
        train_size (float): 0 to 1, fraction of data for train set, defaults to 0.7
        val_size (float): 0 to 1, fraction of data for validation set, defaults to 0.1
        batch_size (int): defaults to 64
        save_file (string): if not None, file to save results to, defaults to None
        benchmark_range (array): values that the benchmark ranges over, defaults to none
        eval_type (one of 'classify' or 'reconstruct'): whether to evaluate by classifying cells into their
            group labels, or reconstruct the cells from k to their full expression. Defaults to 'classify'
        eval_model (model): simple model to evaluate the markers. Defaults to None, which will then use
            RandomForestClassifier for classification or LinearRegression for reconstruction.
        min_groups (None or list of ints): parameter passed to split_data
    returns:
        (dict): maps 'misclass' and 'f1' to a dict of model labels to an np.array (num_times x benchmark_levels)
        (string): benchmark
        (array-like): benchmark_range
    """
    benchmark_options = { 'k', 'label_error', 'label_error_markers_only' }
    if benchmark not in benchmark_options:
        raise Exception(f'benchmark: Possible choices of benchmark are {benchmark_options}')

    if not benchmark_range:
        raise Exception(f'benchmark: For benchmark {benchmark}, please provide a range')
    
    eval_options = { 'classify', 'reconstruct' }
    if eval_type not in eval_options:
        raise Exception(f'benchmark: Possible choices of eval_type are {eval_type}')

    if (benchmark == 'label_error') and (eval_type == 'reconstruct'):
        print(f'WARNING benchmark: benchmark={benchmark} with eval_type={eval_type} doesn\'t have label error')

    if eval_type == 'classify':
        results = { 'misclass': {}, 'f1': {}, 'time': {} }
    else: # reconstruct
        results = { 'l2': {}, 'l1': {}, 'time': {} }

    for i in range(num_times):
        train_indices, val_indices, test_indices = split_data(
            adata.X,
            adata.obs[group_by],
            [train_size, val_size, 1 - train_size - val_size],
            min_groups=min_groups,
        )
        train_val_indices = np.concatenate([train_indices, val_indices])

        adata_test = adata[test_indices,:]

        for model_label, model_functional in models.items():
            print(i, model_label)

            model_results = defaultdict(lambda: [])
            for val in benchmark_range:
                if benchmark == 'k':
                    model_group_by = group_by
                    classifier_train_group_by = group_by
                elif benchmark == 'label_error':
                    adata = mislabel_points(adata, group_by, val, train_val_indices)
                    model_group_by = 'mislabelled_annotation'
                    classifier_train_group_by = 'mislabelled_annotation'
                elif benchmark == 'label_error_markers_only':
                    adata = mislabel_points(adata, group_by, val, train_val_indices)
                    model_group_by = 'mislabelled_annotation'
                    classifier_train_group_by = group_by

                start_time = time.time()
                markers = model_functional(
                    adata,
                    model_group_by,
                    batch_size,
                    train_indices,
                    val_indices,
                    k=val if benchmark == 'k' else None,
                )
                model_results['time'].append(time.time() - start_time)

                if (eval_type == 'classify'):
                    model_misclass, test_rep, _ = new_model_metrics(
                        adata[train_val_indices, :].X,
                        adata[train_val_indices, :].obs[classifier_train_group_by],
                        adata_test.X,
                        adata_test.obs[group_by],
                        markers = markers,
                        model=eval_model,
                    )
                    model_results['misclass'].append(model_misclass)
                    model_results['f1'].append(test_rep['weighted avg']['f1-score'])
                elif (eval_type == 'reconstruct'):
                    l2_error, l1_error = recon_model_metrics(
                        adata[train_val_indices, :].X,
                        adata_test.X,
                        markers,
                        eval_model,
                    )
                    model_results['l2'].append(l2_error)
                    model_results['l1'].append(l1_error)

            for key,val in model_results.items():
                val_ndarray = np.array(val).reshape((1,len(val)))
                current_res = results[key][model_label] if model_label in results[key] else np.array([]).reshape((0,len(benchmark_range)))
                results[key][model_label] = np.append(current_res, val_ndarray, axis=0)
            
            if save_file:
                np.save(save_file, results)

    return results, benchmark, benchmark_range

#######


### graph

def plot_umap_embedding(X, y, encoder, title, path = None, markers = None, close_fig = False, **umap_kwargs):
    """
    Plot the umap embedding of the data, color and label based on y. Call matplotlib.pylot.show() after to display plot.
    args:
        X (np.array): The data that we are finding an embedding for, (n,d)
        y (np.array): The labels of the data, (n,)
        encoder (LabelEncoder): encoder for the labels
        title (str): Title of the plot
        path (str): If provided, save the figure as the file path, defaults to None
        markers (np.array): indices of the markers, reduce X to only those markers
        close_fig (bool): Whether the figure should be closed, useful in Jupyter Notebooks, defaults to False
        umap_kwargs (dict): dictionary of arguments you would pass to umap, like n_neighbors and min_dist. If those
            values are not passed, they will default to 10 and 0.05 respectively.
    """
    if markers is not None:
        X = X[:, markers]
    num_classes = len(encoder.classes_)
    umap_kwargs = { 'n_neighbors': 10, 'min_dist': 0.05, **umap_kwargs }

    embedding = umap.UMAP(**umap_kwargs).fit_transform(X)

    fig, ax = plt.subplots(1, figsize=(8, 8))

    groups = np.unique(y)
    cmap = plt.cm.viridis
    norm = plt.Normalize(np.min(groups), np.max(groups))

    for group in groups:
        ax.scatter(
            *embedding[group == y, :].T,
            s=10,
            color = cmap(norm(group)),
            label=encoder.inverse_transform([group])[0],
        )

    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    if path is not None:
        plt.savefig(path, bbox_inches='tight')

    if close_fig:
        plt.close(fig)


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
        cm = np.floor(100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) / 100
    # sns.heatmap(cm, annot=True,cmap=cmap)
    sns.heatmap(cm, annot=True,cmap=cmap,cbar=False, square=True)
    plt.title(title, fontsize=20)

    plt.xticks([])
    plt.yticks([])

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks+0.5, target_names, rotation=45)
        plt.yticks(tick_marks + 0.5, target_names, rotation = 45)

    
    plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label\naccuracy={:0.2f}; misclass={:0.2f}'.format(accuracy, misclass))
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def get_display_label(label):
    UNSUP_MM = 'Unsupervised Marker Map'
    SUP_MM = 'Supervised Marker Map'
    MIXED_MM = 'Mixed Marker Map'
    BASELINE = 'Baseline'
    LASSONET = 'LassoNet'
    CONCRETE_VAE = 'Concrete VAE'
    GLOBAL_GATE = 'Global Gate'
    SMASH_RF = 'Smash Random Forest'
    RANK_CORR = 'RankCorr'
    SCANPY_T_TEST = 'Scanpy Rank Genes'
    SCANPY_OVERESTIM_VAR ='Scanpy Rank Genes overestim_var'
    SCANPY_WILCOXON = 'Scanpy Rank Genes wilcoxon'
    SCANPY_WILCOXON_TIE ='Scanpy Rank Genes wilcoxon tie'
    SCANPY_HVG = 'Scanpy Highly Variable Genes'
    COSG = 'COSG'
    UNSUP_PERSIST = 'Unsupervised PERSIST'
    SUP_PERSIST = 'Supervised PERSIST'

    # has the renaming
    models = {
        UNSUP_MM: 'MarkerMap unsupervised',
        SUP_MM: 'MarkerMap supervised',
        MIXED_MM: 'MarkerMap joint',
        BASELINE: 'Random Markers',
        LASSONET: LASSONET,
        CONCRETE_VAE: CONCRETE_VAE,
        GLOBAL_GATE: 'Global-Gumbel VAE',
        SMASH_RF: 'SMaSH',
        RANK_CORR: RANK_CORR,
        SCANPY_T_TEST: 'Scanpy t-test',
        SCANPY_OVERESTIM_VAR: 'Scanpy overestim_var',
        SCANPY_WILCOXON: 'Scanpy Wilcoxon',
        SCANPY_WILCOXON_TIE: 'Scanpy Wilcoxon Tie',
        SCANPY_HVG: SCANPY_HVG,
        COSG: COSG,
        UNSUP_PERSIST: 'PERSIST unsupervised',
        SUP_PERSIST: 'PERSIST supervised',
    }
    return models[label]


def plot_benchmarks(
    results, 
    benchmark_label, 
    benchmark_range, 
    mode='misclass', 
    show_stdev=False, 
    print_vals=False,
    save_file=None,
    title='',
    show_legend=True,
    groups=None,
):
    """
    Plot benchmark results of multiple models over the values that you are benchmarking on
    args:
        results (dict): maps model label to np.array of the misclassifications with shape (num_runs x benchmark range)
        benchmark label (string): what you are benchmarking over, will be the x_label
        benchmark_range (array): values that you are benchmarking over
        mode (string): one of {'misclass', 'accuracy', 'f1'}, defaults to 'misclass'
        show_stdev (bool): whether to show fill_between range of 1 stdev over the num_runs, defaults to false
        print_vals (bool): print the vals that are displayed in the plot
    """
    mode_options = {'misclass', 'accuracy', 'f1', 'l2', 'l1'}
    if mode not in mode_options:
        raise Exception(f'plot_benchmarks: Possible choices of mode are {mode_options}')

    markers = ['.','o','v','^','<','>','8','s','p','P','*','h','H','+','x','X','D','d','|','_','1','2','3','4',',']
    _, ax1 = plt.subplots()
    i = 0
    num_runs = 1

    if mode == 'accuracy':
        results = results['misclass']
    else:
        results = results[mode]

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
        ax1.plot(benchmark_range, mean_result, label=get_display_label(label), marker=markers[i])
        i = (i+1) % len(markers)

        if print_vals:
            print(f'{label}: {mean_result}')

    xlabels_map = {
        'k': 'Number of Markers Selected',
        'label_error': 'Fraction of Mislabeled Training Points',
        'label_error_markers_only': 'Fraction of Mislabeled Training Points',
    }

    ax1.set_title(title, fontsize=20)
    ax1.set_xlabel(xlabels_map[benchmark_label], fontsize=20)
    ax1.set_ylabel(mode.capitalize(), fontsize=20)
    ax1.set_xticks(benchmark_range)
    if show_legend:
        ax1.legend()
    plt.tight_layout()

    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight')

    plt.show()

def plot_benchmarks_aligned(
    results, 
    benchmark_label, 
    benchmark_range, 
    mode='misclass', 
    show_stdev=False, 
    print_vals=False,
    save_file=None,
    title='',
    show_legend=True,
    groups=None,
):
    """
    Plot benchmark results of multiple models over the values that you are benchmarking on
    args:
        results (dict): maps model label to np.array of the misclassifications with shape (num_runs x benchmark range)
        benchmark label (string): what you are benchmarking over, will be the x_label
        benchmark_range (array): values that you are benchmarking over
        mode (string): one of {'misclass', 'accuracy', 'f1'}, defaults to 'misclass'
        show_stdev (bool): whether to show fill_between range of 1 stdev over the num_runs, defaults to false
        print_vals (bool): print the vals that are displayed in the plot
    """
    mode_options = {'misclass', 'accuracy', 'f1', 'l2', 'l1'}
    if mode not in mode_options:
        raise Exception(f'plot_benchmarks: Possible choices of mode are {mode_options}')

    markers = ['.','o','v','^','<','>','8','s','p','P','*','h','H','+','x','X','D','d','|','_','1','2','3','4',',']
    i = 0

    if mode == 'accuracy':
        results = results['misclass']
    else:
        results = results[mode]

    _, axes = plt.subplots(ncols=len(groups), nrows=1, sharey=True, figsize=(len(groups)*6.4,4.8))
    for j, group in enumerate(groups):
        if len(groups) == 1:
            ax1 = axes 
        else:
            ax1 = axes[j]

        for label, result in results.items():
            if label not in group:
                continue

            if mode == 'accuracy':
                result = np.ones(result.shape) - result

            mean_result = result.mean(axis=0)

            #only show standard deviation if there we multiple runs
            if show_stdev and result.shape[0] > 1:
                stdev = result.std(axis=0)
                ax1.fill_between(benchmark_range, mean_result - stdev, mean_result + stdev, alpha=0.2)

            #plot the results for this model against the benchmarked range
            ax1.plot(benchmark_range, mean_result, label=get_display_label(label), marker=markers[i])
            i = (i+1) % len(markers)

            if print_vals:
                print(f'{label}: {mean_result}')

        xlabels_map = {
            'k': 'Number of Markers Selected',
            'label_error': 'Fraction of Mislabeled Training Points',
            'label_error_markers_only': 'Fraction of Mislabeled Training Points',
        }

        ax1.set_title(title, fontsize=20)
        ax1.set_xlabel(xlabels_map[benchmark_label], fontsize=20)
        ax1.set_xticks(benchmark_range)
        if show_legend:
            ax1.legend()
        if j == 0:
            ax1.set_ylabel(mode.capitalize(), fontsize=20)

    plt.tight_layout()

    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight')

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


## Data Loading and Preprocessing Helper Functions

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

def get_ssv4(file_path, housekeeping_genes_paths, names_key='cell_types_98'):
    """
        Some code taken from PERSIST: https://github.com/iancovert/persist/blob/main/notebooks/demo.ipynb
        args:
            file_path (str): file path to the v1_raw
            names_key (str): could be 'cell_types_98', 'cell_types_50', or 'cell_types_25'
    """

    # Load dataframes
    df_raw = pd.read_csv(file_path, header=0, index_col=0, delimiter=',')

    # Arrange raw and CPM data
    cell_types = pd.Categorical(df_raw[names_key])
    raw = df_raw.iloc[:, :-3].astype(np.float32) # last 3 cols are the cell_types

    adata = anndata.AnnData(X=raw)
    adata.obs['annotation'] = cell_types

    # Remove classes that have fewer than 4 elements because many of our methods require more items per class
    small_classes = None
    for label in adata.obs['annotation'].unique():
        if np.sum(adata.obs['annotation'] == label) < 4:
            if small_classes is None:
                small_classes = (adata.obs['annotation'] == label)
            else:
                small_classes = (small_classes | (adata.obs['annotation'] == label))

    adata = adata[~small_classes, :]

    adata = remove_general_genes(adata)
    adata = remove_housekeeping_genes(adata, housekeeping_genes_paths)

    # these don't do anything for this data set, but leave them in for clarity
    sc.pp.filter_cells(adata, min_genes=1)
    sc.pp.filter_genes(adata, min_cells=1)

    # these do have an affect
    adata = remove_features_pct(adata, group_by="annotation", pct=0.3)
    adata = remove_features_pct_2groups(adata, group_by="annotation", pct1=0.75, pct2=0.5)

    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    adata.layers['norm'] = adata.X.copy()
    sc.pp.log1p(adata)
    adata.layers['log'] = adata.X.copy()
    sc.pp.scale(adata, max_value=10)
    adata.layers['scale'] = adata.X.copy()

    return adata

def get_zeisel(file_path, names_key='names0'):
    """
    Get the zeisel data located a file path
    args:
        file_path (string): location of zeisel.h5ad file
        names_key (string): key of Zeisel adata for the classes, defaults to names0 for broad classes
    returns: X data array, y transformed labels, encoder that transformed the labels
    """
    adata = sc.read_h5ad(file_path)
    adata.obs['annotation'] = adata.obs[names_key]
    adata = adata[adata.obs['annotation'] != '(none)'] # if its names1, there are some unknown cells

    std = np.expand_dims(adata.var['std'].to_numpy(), axis=0)
    mean = np.expand_dims(adata.var['mean'].to_numpy(), axis=0)
    X = adata.X.copy()

    X = (X * std) + mean
    X[X < 1e-5] = 0.0 # there appears to be a cut off where these are numerically 0

    # Warning! This is not actually counts! This will only work for the purposes of binarizing the counts
    # as in the persist model. We can't totally recover it because of the clipping in scale.
    adata.layers['counts'] = X
    adata.layers['log'] = X.copy()

    return adata

def log_and_normalize(X, recon_X=None):
  """
  Perform the log(X + 1) transform, then mean center and unit variance the ENTIRE matrix.
  If recon_X is provided, that also gets mean centered and scaled according to the mean and std of X
  """
  log_X = np.log(X + np.ones(X.shape))

  X_mean = np.mean(log_X)
  X_std = np.std(log_X)

  if recon_X is None:
    return (log_X - X_mean) / X_std
  else:
    log_recon_X = np.log(recon_X + np.ones(recon_X.shape))
    return ((log_X - X_mean) / X_std, (log_recon_X - X_mean) / X_std)
  
def remove_general_genes(adata):
    general_genes = (
        adata.var_names.str.startswith("Mt") |
        adata.var_names.str.startswith("Rps") | 
        adata.var_names.str.startswith("Rpl") | 
        adata.var_names.str.startswith("Hsp") | 
        adata.var_names.str.startswith("Hla")
    )
    return adata[:,~general_genes]

def remove_housekeeping_genes(adata, housekeeping_genes_paths):
    # remove housekeeping genes
    hkg = []
    for file_name in housekeeping_genes_paths:
        hkg.append(np.loadtxt(file_name, dtype="str"))

    hkg = np.concatenate(hkg).ravel()

    adata.var["general"] = [(gene not in hkg) for gene in adata.var.index.tolist()]
    return adata[:, adata.var["general"]]

def get_paul(housekeeping_genes_paths, smashpy_preprocess=True):
    """
    Get the paul data from scanpy and remove the housekeeping genes from the two provided files
    args:
        housekeeping_bone_marrow_path (string): location of housekeeping genes bone marrow file
        house_keeping_HSC_path (string): location of housekeeping genes HSC file
    returns: X data array, y transformed labels, encoder that transformed the labels
    """
    adata = sc.datasets.paul15()
    if (smashpy_preprocess):
        adata.layers['counts'] = adata.X.copy()
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
        adata.layers['norm'] = adata.X.copy()
        sc.pp.log1p(adata)
        adata.layers['log'] = adata.X.copy()
        sc.pp.scale(adata, max_value=10)
        adata.layers['scale'] = adata.X.copy()
    else:
        # these layers are set in data_preparation and used in remove_general_genes and remove_housekeepingenes
        adata.layers['counts'] = adata.X
        adata.layers['norm'] = adata.X
        adata.layers['log'] = adata.X
        adata.layers['scale'] = adata.X

    adata = remove_general_genes(adata)
    adata = remove_housekeeping_genes(adata, housekeeping_genes_paths)

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

    adata.obs['annotation'] = pd.Categorical(annotation)
    adata.obs['annotation'] = adata.obs['annotation'].astype('category')

    return adata

def get_citeseq(file_path):
    """
    Get the CITE_seq data located a file path
    args:
        file_path (string): location of CITEseq.h5ad file
    returns: X data array, y transformed labels, encoder that transformed the labels
    """
    adata = sc.read_h5ad(file_path)
    adata.obs['annotation'] = adata.obs['names'].astype('category')

    std = np.expand_dims(adata.var['std'].to_numpy(), axis=0)
    mean = np.expand_dims(adata.var['mean'].to_numpy(), axis=0)
    X = adata.X.copy()

    X = (X * std) + mean
    X[X < 1e-5] = 0.0 # there appears to be a cut off where these are numerically 0

    # Warning! This is not actually counts! This will only work for the purposes of binarizing the counts
    # as in the persist model. We can't totally recover it because of the clipping in scale.
    adata.layers['counts'] = X
    adata.layers['log'] = X.copy()

    return adata

def relabel_mouse_labels(label):
    if isinstance(label, str):
        return label.split('_')[0]
    else:
        return label

def remove_features_pct(adata, group_by=None, pct=0.3):
    if group_by is None:
        print("select a group_by in .obs")
        return
    if group_by not in adata.obs.columns:
        print("group_by must be in .obs")
        return


    list_keep_genes = []

    df = pd.DataFrame(data=False, index=adata.var.index.tolist(), columns=adata.obs[group_by].cat.categories)
    for g in adata.obs[group_by].cat.categories:
        reduced = adata[adata.obs[group_by]==g]
        boolean, values = sc.pp.filter_genes(reduced, min_cells = reduced.n_obs*pct, inplace=False)
        df[g] = boolean
    dfT = df.T
    for g in dfT.columns:
        if True in dfT[g].tolist():
            list_keep_genes.append(True)
        else:
            list_keep_genes.append(False)

    adata.var["general"] = list_keep_genes

    adata = adata[:, adata.var["general"]]

    return adata

def remove_features_pct_2groups(adata, group_by=None, pct1=0.9, pct2=0.5):
    if group_by is None:
        print("select a group_by in .obs")
        return
    if group_by not in adata.obs.columns:
        print("group_by must be in .obs")
        return


    list_keep_genes = []

    df = pd.DataFrame(data=False, index=adata.var.index.tolist(), columns=adata.obs[group_by].cat.categories)
    for g in adata.obs[group_by].cat.categories:
        reduced = adata[adata.obs[group_by]==g]
        boolean, values = sc.pp.filter_genes(reduced, min_cells = reduced.n_obs*(pct1), inplace=False)
        df[g] = boolean
    dfT = df.T
    for g in dfT.columns:
        if (sum(dfT[g].tolist())/len(dfT[g].tolist())) >= pct2:
            list_keep_genes.append(False)
        else:
            list_keep_genes.append(True)

    adata.var["general"] = list_keep_genes
    adata = adata[:, adata.var["general"]]
    return adata

def get_mouse_brain(mouse_brain_path, mouse_brain_labels_path, smashpy_preprocess=True, relabel=True):
    """
    Get the mouse brain data and remove outliers and perform normalization. Some of the decisions in this function are
    judgement calls, so users should inspect and make their own decisions.
    args:
        mouse_brain_path (string): location of mouse_brain_all_cells.h5ad file
        mouse_brain_labels_path (string): location of cells annotations for mouse_brain data
        smashpy_preprocess (bool): whether to perform the smashpy preprocessing, defaults to True
        relabel (bool): if true, relabel to the broad cell classes, otherwise use the specific cell classes.
            defaults to True
    returns: X data array, y transformed labels, encoder that transformed the labels
    """
    adata_snrna_raw = anndata.read_h5ad(mouse_brain_path)
    del adata_snrna_raw.raw
    adata_snrna_raw.X = adata_snrna_raw.X.toarray()
    adata_snrna_raw.layers['counts'] = adata_snrna_raw.X.copy() # save the counts in a layer
    ## Cell type annotations
    labels = pd.read_csv(mouse_brain_labels_path, index_col=0)
    if relabel:
        labels['annotation'] = labels['annotation_1'].apply(lambda x: relabel_mouse_labels(x))
    else:
        labels['annotation'] = labels['annotation_1']

    labels = labels[['annotation']]
    labels = labels.reindex(index=adata_snrna_raw.obs_names)
    adata_snrna_raw.obs[labels.columns] = labels
    adata_snrna_raw = adata_snrna_raw[~adata_snrna_raw.obs['annotation'].isna(), :]
    adata_snrna_raw = adata_snrna_raw[adata_snrna_raw.obs["annotation"]!='Unk']
    adata_snrna_raw.obs['annotation'] = adata_snrna_raw.obs['annotation'].astype('category')

    # doing this cuz smash was a lot of data
    #https://cell2location.readthedocs.io/en/latest/notebooks/cell2location_estimating_signatures.html
    #preprocess_like_cell_location
    sc.pp.filter_cells(adata_snrna_raw, min_genes=1)
    print(adata_snrna_raw.shape)
    sc.pp.filter_genes(adata_snrna_raw, min_cells=1)
    print(adata_snrna_raw.shape)

    gc.collect()
    # calculate the mean of each gene across non-zero cells
    adata_snrna_raw.var['n_cells'] = (adata_snrna_raw.X > 0).sum(0)
    adata_snrna_raw.var['nonz_mean'] = adata_snrna_raw.X.sum(0) / adata_snrna_raw.var['n_cells']

    nonz_mean_cutoff = np.log10(1.12) # cut off for expression in non-zero cells
    cell_count_cutoff = np.log10(adata_snrna_raw.shape[0] * 0.0005) # cut off percentage for cells with higher expression
    cell_count_cutoff2 = np.log10(adata_snrna_raw.shape[0] * 0.03)# cut off percentage for cells with small expression


    adata_snrna_raw[:,(np.array(np.log10(adata_snrna_raw.var['nonz_mean']) > nonz_mean_cutoff)
        | np.array(np.log10(adata_snrna_raw.var['n_cells']) > cell_count_cutoff2))
        & np.array(np.log10(adata_snrna_raw.var['n_cells']) > cell_count_cutoff)].shape

    # select genes based on mean expression in non-zero cells
    adata_snrna_raw = adata_snrna_raw[:,(np.array(np.log10(adata_snrna_raw.var['nonz_mean']) > nonz_mean_cutoff)
        | np.array(np.log10(adata_snrna_raw.var['n_cells']) > cell_count_cutoff2))
        & np.array(np.log10(adata_snrna_raw.var['n_cells']) > cell_count_cutoff)
        & np.array(~adata_snrna_raw.var['SYMBOL'].isna())]

    gc.collect()
    adata_snrna_raw.raw = adata_snrna_raw
    adata_snrna_raw.X = adata_snrna_raw.raw.X.copy()
    del adata_snrna_raw.raw
    gc.collect()
    adata_snrna_raw = remove_features_pct(adata_snrna_raw, group_by="annotation", pct=0.3)
    gc.collect()
    adata_snrna_raw = remove_features_pct_2groups(adata_snrna_raw, group_by="annotation", pct1=0.75, pct2=0.5)

    if (smashpy_preprocess):
        sc.pp.normalize_per_cell(adata_snrna_raw, counts_per_cell_after=1e4)
        sc.pp.log1p(adata_snrna_raw)
        adata_snrna_raw.layers['log'] = adata_snrna_raw.X.copy()
        sc.pp.scale(adata_snrna_raw, max_value=10)

    return adata_snrna_raw

def get_dataloader(X, y, batch_size, shuffle, num_workers=0):
    return DataLoader(
        torch.utils.data.TensorDataset(torch.Tensor(X), torch.LongTensor(y)),
        batch_size=batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
    )

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
        train_dataloader = get_dataloader(train_x, train_y, batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = get_dataloader(val_x, val_y, batch_size, shuffle=False, num_workers=num_workers)
    else:
        # X = Y in unsupervised
        train_dataloader = get_dataloader(train_x, train_x, batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = get_dataloader(val_x, val_x, batch_size, shuffle=False, num_workers=num_workers)
        
        
    if train_size == 1:
        val_dataloader = None
        val_indices = None
        
    return train_dataloader, val_dataloader, train_indices, val_indices

def split_data(
    X, 
    y, 
    size_percents, 
    seed=None, 
    min_groups=None, 
):
    """
    Split X and y into
    Args:
        X (array): Input data
        y (vector): Output labels
        size_percents (list of floats): the percent sizes of sets of data. If two values are passed then
            two dataloaders and two sets of indices are returned. The size_percents list should add up to 1,
            but the last value is not used, the last set of data will just be the remaining data.
        num_workers (int): number of cores for multi-threading, defaults to 0 for no multi-threading
        seed (int): defaults to none, set to reproduce experiments with same train/val split
        min_groups (None or list of ints): if not none, should be a list of integers for each set of data.
            That set of data must have at least that number of representatives of each group in y
    """
    assert np.sum(size_percents) <= 1
    assert len(X) == len(y)
    if min_groups is not None:
        assert len(size_percents) == len(min_groups)

    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed=seed)

    if min_groups is None:
        min_groups = [0]*len(size_percents)

    # Split data by cell type class
    groups = np.unique(y)
    y_dict = {}
    for group in groups:
        y_dict[group] = np.where(y == group)[0]

    # For each set (train, val, test), pick min_group elements from each group to be in that set
    indices_list = [[] for _ in range(len(min_groups))]
    for i,min_group in enumerate(min_groups):
        if min_group == 0:
            continue 

        for group in groups:
            assert len(y_dict[group]) >= min_group, f'min_group={min_group}, but group {group} \
                only has {len(y_dict[group])} members left'
            idxs = rng.choice(np.arange(len(y_dict[group])), size=min_group, replace=False)
            indices_list[i].extend(y_dict[group][idxs])
            y_dict[group] = np.delete(y_dict[group], idxs) # remove the idxs from selection

    # Return the remaining indices to a single list
    remaining_idxs = np.array([], dtype=int)
    for group in groups:
        remaining_idxs = np.concatenate([remaining_idxs, y_dict[group]])

    rng.shuffle(remaining_idxs) # shuffle in place the remaining indices array

    # Construct the set dataloaders and indices so that the percents match up, even with min_groups
    all_indices = []
    set_start = 0
    for i, (percent, min_group) in enumerate(zip(size_percents, min_groups)):
        if i == (len(size_percents) - 1):
            set_indices = np.concatenate([
                remaining_idxs[set_start:], 
                np.array(indices_list[i], dtype=int),
            ])
        else:
            set_len = int(percent * len(X)) - (min_group * len(groups))
            assert set_len >= 0, f'Satisfying min_group of {min_group} would require set {i} to have \
                greater than {percent} percent size'
            set_indices = np.concatenate([
                remaining_idxs[set_start:(set_start + set_len)], 
                np.array(indices_list[i], dtype=int),
            ])

        assert len(indices_list[i]) == (min_group * len(groups)), f'{len(indices_list[i])} {min_group * len(groups)}'

        all_indices.append(set_indices)

        set_start += set_len

    return all_indices

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

    train_dataloader = get_dataloader(train_x, train_y, batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = get_dataloader(val_x, val_y, batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = get_dataloader(test_x, test_y, batch_size, shuffle=False, num_workers=num_workers)

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
