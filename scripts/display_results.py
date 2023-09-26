import sys
import argparse
import numpy as np

import matplotlib.pyplot as plt

def plot_benchmarks(results, benchmark_label, benchmark_range, mode='misclass', show_stdev=False, print_vals=False):
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
        if label in {
            # 'Unsupervised Marker Map',
            # 'Supervised Marker Map',
            # 'Mixed Marker Map',
            # 'Baseline',
            # 'LassoNet',
            'Concrete VAE',
            'Global Gate',
            # 'Smash Random Forest',
            # 'RankCorr',
            # 'Scanpy Rank Genes',
            'Scanpy Rank Genes overestim_var',
            # 'Scanpy Rank Genes wilcoxon',
            'Scanpy Rank Genes wilcoxon tie',
            'COSG',
            # 'Unsupervised PERSIST',
            # 'Supervised PERSIST',
        }:
          continue

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

        if print_vals:
            print(f'{label}: {mean_result}')

    ax1.set_title(f'{mode.capitalize()} Benchmark, over {num_runs} runs')
    ax1.set_xlabel(benchmark_label)
    ax1.set_ylabel(mode.capitalize())
    ax1.legend()

    plt.show()

def handleArgs(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--infile', help='the file to load results from', default=None)
  parser.add_argument('-m', '--mode', help='the mode of plot_benchmarks', default='accuracy')
  parser.add_argument('-l', '--lite', help='show in lite mode, k_range [50]', action='store_true')

  args = parser.parse_args()

  return args.infile, args.mode, args.lite

#Main
file, mode, lite = handleArgs(sys.argv)

results = np.load(file, allow_pickle=True).item()
# benchmark_label = 'k'
# benchmark_range = [50] if lite else [10, 25, 50, 100, 250]

benchmark_label = 'label_error'
benchmark_range = [0.1, 0.2, 0.5, 0.75, 1]

plot_benchmarks(results, benchmark_label, benchmark_range, mode=mode, show_stdev=False, print_vals=True)
