import pytest
import numpy as np

from markermap.utils import split_data_into_dataloaders, split_data

class TestDataloaders:

  def test_assertions_split_data_into_dataloaders(self):
    X = np.array([1,2,3,4,5,6,7,8]).reshape((4,2))
    y = np.array([0,1,0,1])

    #total split must be less than 1
    with pytest.raises(AssertionError):
      split_data_into_dataloaders(X,y,0.7,0.4)

    #length of X and y has to be equal
    with pytest.raises(AssertionError):
      split_data_into_dataloaders(X,y[1:],0.7,0.1)

    #pass all assertions
    split_data_into_dataloaders(X,y,0.7,0.1)

  def test_split_data(self):
    X = np.array([1,2,3,4,5,6,7,8]).reshape((4,2))
    y = np.array([0,1,0,1])

    # total split must be less than 1
    with pytest.raises(AssertionError): 
      split_data(X, y, [0.7, 0.4])

    # X and y must be the same length
    with pytest.raises(AssertionError):
      split_data(X, y[:1], [0.7, 0.1, 0.2])

    # min_groups must be same length as size_percents
    with pytest.raises(AssertionError):
      split_data(X, y, [0.7, 0.1, 0.2], min_groups=[0,0])

    # check that sizes match for even splits
    X = np.arange(20).reshape((10,2))
    y = np.concatenate([np.ones(5), np.zeros(5)])
    train_indices, test_indices = split_data(X, y, [0.7, 0.3])
    assert len(train_indices) == 7
    assert len(test_indices) == 3

    # check that sizes match for uneven splits
    X = np.arange(20).reshape((10,2))
    y = np.concatenate([np.ones(5), np.zeros(5)])
    train_indices, test_indices = split_data(X, y, [0.65, 0.35])
    assert len(train_indices) == 6
    assert len(test_indices) == 4

    # check that sizes match for uneven splits
    X = np.arange(20).reshape((10,2))
    y = np.concatenate([np.ones(5), np.zeros(5)])
    train_indices, test_indices = split_data(X, y, [0.65, 0.35])
    assert len(train_indices) == 6
    assert len(test_indices) == 4

  def test_split_data_min_groups(self):
    # check that sizes match when using min groups
    X = np.arange(20).reshape((10,2))
    y = np.concatenate([np.ones(5), np.zeros(5)])
    train_indices, test_indices = split_data(X, y, [0.7, 0.3], min_groups=[1,1])
    assert np.sum(y[train_indices] == 0) > 1 and np.sum(y[train_indices] == 1) > 1

    # check that min_groups don't make a set too large
    X = np.arange(20).reshape((10,2))
    y = np.arange(10)
    with pytest.raises(AssertionError):
      train_indices, test_indices = split_data(X, y, [0.7, 0.3], min_groups=[1,1])

    # check that min_group is achievable
    X = np.arange(20).reshape((10,2))
    y = np.concatenate([np.ones(1), np.zeros(9)])
    with pytest.raises(AssertionError):
      train_indices, test_indices = split_data(X, y, [0.7, 0.3], min_groups=[1,1])

    # Check that min_groups is calculated first to ensure that they are all satisfied
    X = np.arange(20).reshape((10,2))
    y = np.concatenate([np.ones(2), np.zeros(8)])
    train_indices, test_indices = split_data(X, y, [0.7, 0.3], min_groups=[1,1])
    assert np.sum(y[train_indices] == 0) == 6
    assert np.sum(y[train_indices] == 1) == 1
    assert np.sum(y[test_indices] == 0) == 2
    assert np.sum(y[test_indices] == 1) == 1
