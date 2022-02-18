import sys
import pytest
import numpy as np

sys.path.insert(1, './src/')
from utils import split_data_into_dataloaders

X = np.array([1,2,3,4,5,6,7,8]).reshape((4,2))
y = np.array([0,1,0,1])


class TestDataloaders:

  def test_assertions_split_data_into_dataloaders(self):
    #total split must be less than 1
    with pytest.raises(AssertionError):
      split_data_into_dataloaders(X,y,0.7,0.4)

    #length of X and y has to be equal
    with pytest.raises(AssertionError):
      split_data_into_dataloaders(X,y[1:],0.7,0.1)

    #pass all assertions
    split_data_into_dataloaders(X,y,0.7,0.1)
