import sys
import argparse
import numpy as np

from markermap.utils import plot_benchmarks

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
benchmark_label = 'k'
benchmark_range = [50] if lite else [10, 25, 50, 100, 250]

# benchmark_label = 'label_error'
# benchmark_range = [0.1, 0.2, 0.5, 0.75, 1]

plot_benchmarks(results, benchmark_label, benchmark_range, mode=mode, show_stdev=False, print_vals=True)
