import pytest
import numpy as np
import anndata

from markermap.utils import mislabel_points

class TestMislabelPoints:
  adata = anndata.AnnData(np.ones((10,10)))
  adata.obs['annotation'] = np.array([1,1,2,1,0,3,4,5,1,2]) #length 10
  eligible_indices = np.array([0,1,2,4,6,7])

  def test_mislabel_percent(self):
    with pytest.raises(AssertionError):
      mislabel_points(self.adata, 'annotation', -1, self.eligible_indices)

    with pytest.raises(AssertionError) as e:
      mislabel_points(self.adata, 'annotation', 2, self.eligible_indices)

    mislabel_points(self.adata, 'annotation', 0.5, self.eligible_indices)

  def test_eligible_indices(self):
    bad_eligible_indices = np.array([0,1,2,3,4,5,6,7,8,9,10])

    with pytest.raises(AssertionError):
      mislabel_points(self.adata, 'annotation', 0.5, bad_eligible_indices)

    mislabel_points(self.adata, 'annotation', 0.2)

    #test that the unchanged indices are all the same in y and y_err
    adata_err = mislabel_points(self.adata, 'annotation', 0.2, self.eligible_indices)
    unchanged_indices = np.ones(len(self.adata))
    unchanged_indices[self.eligible_indices] = False
    assert ((self.adata.obs['annotation'] == adata_err.obs['mislabelled_annotation'])[unchanged_indices.astype(bool)]).all()

  def test_mismatches_in_bounds(self):
    adata = anndata.AnnData(np.ones((1000,10)))
    adata.obs['annotation'] = np.random.randint(0, 10, 1000)
    mislabel_percent = 0.2

    adata_err = mislabel_points(adata, 'annotation', mislabel_percent)

    len_y = len(adata.obs['annotation'])
    assert len_y == len(adata_err.obs['mislabelled_annotation'])
    mismatches = np.sum(adata.obs['annotation'] != adata_err.obs['mislabelled_annotation'])
    assert mismatches <= int(mislabel_percent * len_y)
    assert mismatches > int(mislabel_percent * len_y * 0.5) # technically possible, but v unlikely

  def test_mislabels_from_y(self):
    adata_err = mislabel_points(self.adata, 'annotation', 0.5)
    unique_set = { x for x in np.unique(self.adata.obs['annotation']) }
    assert np.array([x in unique_set for x in adata_err.obs['mislabelled_annotation']]).all()

