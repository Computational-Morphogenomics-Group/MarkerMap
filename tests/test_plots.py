import pytest
import numpy as np
import matplotlib.pyplot as plt

from markermap.utils import plot_benchmarks

class TestPlotBenchmarks:
  def test_one_model_3_tries(self):
    misclass_rates = {
      'misclass': {
        'Unsupervised Marker Map': np.array([
          [0.43427621, 0.16139767],
          [0.40099834, 0.15141431],
          [0.41930116, 0.34442596]
        ]),
      },
    }

    with plt.ion():
      plot_benchmarks(misclass_rates, 'k', benchmark_range=[10,25])


  def test_two_models_1_try(self):
    misclass_rates = {
      'misclass': {
        'Supervised Marker Map': np.array([[0.18801997, 0.23460899]]),
        'Mixed Marker Map': np.array([[0.19966722, 0.26622296]]),
      },
    }

    with plt.ion():
      plot_benchmarks(misclass_rates, 'k', benchmark_range=[10,25])


  def test_two_models_5_tries(self):
    misclass_rates = {
      'misclass': {
        'Supervised Marker Map': np.array([
          [0.08985025, 0.078203  , 0.19467554, 0.04159734],
          [0.14475874, 0.23627288, 0.06322795, 0.05490849],
          [0.14475874, 0.11148087, 0.08153078, 0.13311148],
          [0.11480865, 0.09983361, 0.09816972, 0.05324459],
          [0.1031614 , 0.08153078, 0.06489185, 0.04159734]]),
        'Mixed Marker Map': np.array([
          [0.45257903, 0.16139767, 0.1547421 , 0.11314476],
          [0.34442596, 0.30116473, 0.16638935, 0.16472546],
          [0.52246256, 0.31780366, 0.15806988, 0.16638935],
          [0.43926789, 0.15806988, 0.21464226, 0.06655574],
          [0.44925125, 0.22462562, 0.22462562, 0.05990017]
        ]),
      },
    }

    with plt.ion():
      plot_benchmarks(misclass_rates, 'k', benchmark_range=[10, 25, 50, 100])

  def test_modes(self):
    misclass_rates = {
      'misclass': {
        'Supervised Marker Map': np.array([[0.18801997, 0.23460899]]),
        'Mixed Marker Map': np.array([[0.19966722, 0.26622296]]),
      },
      'f1': {
        'Supervised Marker Map': np.array([[0.18801997, 0.23460899]]),
        'Mixed Marker Map': np.array([[0.15, 0.201]]),
      },
    }

    with plt.ion():
      plot_benchmarks(misclass_rates, 'k', mode='misclass', benchmark_range=[10,25])
      plot_benchmarks(misclass_rates, 'k', mode='accuracy', benchmark_range=[10,25])
      plot_benchmarks(misclass_rates, 'k', mode='f1', benchmark_range=[10,25])
      with pytest.raises(Exception):
        plot_benchmarks(misclass_rates, 'k', mode='fake_mode', benchmark_range=[10,25])

  def test_mismatched_number_of_runs(self):
    misclass_rates_1 = {
      'misclass': {
        'Supervised Marker Map': np.array([
          [0.08985025, 0.078203  , 0.19467554, 0.04159734],
          [0.14475874, 0.23627288, 0.06322795, 0.05490849],
          [0.14475874, 0.11148087, 0.08153078, 0.13311148],
          [0.11480865, 0.09983361, 0.09816972, 0.05324459],
          [0.1031614 , 0.08153078, 0.06489185, 0.04159734]]),
        'Mixed Marker Map': np.array([
          [0.45257903, 0.16139767, 0.1547421 , 0.11314476],
          [0.34442596, 0.30116473, 0.16638935, 0.16472546],
        ]),
      },
    }

    misclass_rates_2 = {
      'misclass': {
        'Supervised Marker Map': np.array([
          [0.08985025, 0.078203  , 0.19467554, 0.04159734],
        ]),
        'Mixed Marker Map': np.array([
          [0.45257903, 0.16139767, 0.1547421 , 0.11314476],
          [0.34442596, 0.30116473, 0.16638935, 0.16472546],
        ]),
      },
    }

    with plt.ion():
      plot_benchmarks(misclass_rates_1, 'k', benchmark_range=[10, 25, 50, 100])
      plot_benchmarks(misclass_rates_2, 'k', benchmark_range=[10, 25, 50, 100])
