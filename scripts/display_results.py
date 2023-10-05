import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from markermap.utils import plot_benchmarks, plot_benchmarks_aligned

def handleArgs(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str)
  parser.add_argument('--img_dir', type=str)
  parser.add_argument('-b', '--benchmark', help='type of benchmark', type=str, default='k')
  parser.add_argument('--eval_type', help='eval type, classify or reconstruct', type=str, default='classify')
  parser.add_argument('--eval_model', help='eval model', type=str, default='rf')
  parser.add_argument('--dataset', type=str)
  parser.add_argument('--mode', type=str, default='accuracy')
  parser.add_argument('-l', '--lite', help='show in lite mode, k_range [50]', action='store_true')
  parser.add_argument('--show_stdev', action='store_true', default=False)

  args = parser.parse_args()

  return (
    args.data_dir, 
    args.img_dir, 
    args.benchmark, 
    args.eval_type, 
    args.eval_model, 
    args.dataset, 
    args.mode,
    args.lite,
    args.show_stdev,
  )

#Main
(
  data_dir, 
  img_dir, 
  benchmark, 
  eval_type, 
  eval_model, 
  dataset, 
  mode, 
  lite,
  show_stdev,
) = handleArgs(sys.argv)

file_name = f'{data_dir}{dataset}_{benchmark}_{eval_type}_{eval_model}_r10_s1729.npy'

if dataset == 'zeisel' or dataset == 'paul':
  dataset_name = dataset.capitalize()
elif dataset == 'cite_seq':
  dataset_name = 'CITE-seq'
elif dataset == 'zeisel_big':
  dataset_name = 'Zeisel Big'

results = np.load(file_name, allow_pickle=True).item()
if benchmark == 'k':
  title = f'{dataset_name} Accuracy'
  if lite:
    benchmark_range = [50]
  else:
    benchmark_range = [10, 25, 50, 100, 250]
elif benchmark == 'label_error':
  title = f'{dataset_name} Robustness to Noise'
  benchmark_range = [0.1, 0.2, 0.5, 0.75, 1]
elif benchmark == 'label_error_markers_only':
  title = f'{dataset_name} Robustness, Markers Only'
  benchmark_range = [0.1, 0.2, 0.5, 0.75, 1]



plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 10

save_file = f'{img_dir}{dataset}_{benchmark}_{eval_type}_{eval_model}_{mode}_r10_s1729_double.pdf'
groups = [
  {
    'Unsupervised Marker Map',
    'Supervised Marker Map',
    'Baseline',
    'LassoNet',
    'Smash Random Forest',
    'Scanpy Rank Genes',
    'Scanpy Rank Genes wilcoxon',
    'Supervised PERSIST',
  },
  {
    'Mixed Marker Map',
    'Concrete VAE',
    'Global Gate',
    'RankCorr',
    'Scanpy Rank Genes overestim_var',
    'Scanpy Rank Genes wilcoxon tie',
    'COSG',
    'Unsupervised PERSIST',
  },
]

plot_benchmarks_aligned(
  results, 
  benchmark, 
  benchmark_range, 
  mode=mode, 
  show_stdev=show_stdev, 
  print_vals=False, 
  title=title,
  save_file=save_file,
  groups=groups,
)
plt.close()

save_file = f'{img_dir}{dataset}_{benchmark}_{eval_type}_{eval_model}_{mode}_r10_s1729.pdf'
groups = [
  {
    'Unsupervised Marker Map',
    'Supervised Marker Map',
    'Baseline',
    'LassoNet',
    'Smash Random Forest',
    'Scanpy Rank Genes',
    'Scanpy Rank Genes wilcoxon',
    'Supervised PERSIST',
  },
]

plot_benchmarks_aligned(
  results, 
  benchmark, 
  benchmark_range, 
  mode=mode, 
  show_stdev=show_stdev, 
  print_vals=False, 
  title=title,
  save_file=save_file,
  groups=groups,
)
plt.close()
