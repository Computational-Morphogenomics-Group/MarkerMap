import sys
import pytest
import numpy as np

sys.path.insert(1, './src/')
from utils import mislabel_points

class TestMislabelPoints:
  y = np.array([1,1,2,1,0,3,4,5,1,2]) #length 10
  eligible_indices = np.array([0,1,2,4,6,7])

  def test_mislabel_percent(self):
    with pytest.raises(AssertionError):
      mislabel_points(self.y, -1, self.eligible_indices)

    with pytest.raises(AssertionError) as e:
      mislabel_points(self.y, 2, self.eligible_indices)

    mislabel_points(self.y, 0.5, self.eligible_indices)

  def test_eligible_indices(self):
    bad_eligible_indices = np.array([0,1,2,3,4,5,6,7,8,9,10])

    with pytest.raises(AssertionError):
      mislabel_points(self.y, 0.5, bad_eligible_indices)

    mislabel_points(self.y, 0.2)

    #test that the unchanged indices are all the same in y and y_err
    y_err = mislabel_points(self.y, 0.2, self.eligible_indices)
    unchanged_indices = np.ones(len(self.y))
    unchanged_indices[self.eligible_indices] = False
    assert ((self.y == y_err)[unchanged_indices.astype(bool)]).all()

  def test_mismatches_in_bounds(self):
    y = np.random.randint(0, 10, 1000)
    mislabel_percent = 0.2

    y_err = mislabel_points(y, mislabel_percent)

    assert len(y) == len(y_err)
    mismatches = np.sum(y != y_err)
    assert mismatches <= int(mislabel_percent * len(y))
    assert mismatches > int(mislabel_percent * len(y) * 0.5) # technically possible, but v unlikely

  def test_mislabels_from_y(self):
    y_err = mislabel_points(self.y, 0.5)
    unique_set = { x for x in np.unique(self.y) }
    assert np.array([x in unique_set for x in y_err]).all()

