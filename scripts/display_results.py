import sys
import argparse
import numpy as np

from markermap.utils import plot_benchmarks

def mergeFiles():
  files = [
    'checkpoints/benchmark_label_error_25_zeisel_1.npy',
    'checkpoints/benchmark_label_error_25_zeisel_2.npy',
  ]
  datas = []
  for file in files:
    datas.append(np.load(file, allow_pickle=True).item())

  data_1 = datas[0]
  for i in range(1,len(datas)):
    data_2 = datas[i]

    for key1 in ['misclass', 'f1']:
      for key2, arr in data_2[key1].items():
        data_1[key1][key2] = np.concatenate((data_1[key1][key2], arr))

  print(data_1)

  np.save('checkpoints/benchmark_label_error_50_zeisel.npy', data_1)

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

# mergeFiles()
