import sys
import argparse
import numpy as np

from markermap.utils import plot_benchmarks

def handleArgs(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--infile', help='the file to load results from', default=None)
  parser.add_argument('-m', '--mode', help='the mode of plot_benchmarks', default='accuracy')

  args = parser.parse_args()

  return args.infile, args.mode

#Main
file, mode = handleArgs(sys.argv)

results = np.load(file, allow_pickle=True).item()
benchmark_label = 'k'
benchmark_range = [10, 25, 50, 100, 250]

plot_benchmarks(results, benchmark_label, benchmark_range, mode=mode, show_stdev=True, print_vals=True)
