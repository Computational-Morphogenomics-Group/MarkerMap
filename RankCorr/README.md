# RankCorr


A marker selection method for scRNA-seq data based on rank correlation.  See the notebook `RankCorr-example.ipynb` for a full walkthough of how to run the method; an outline is presented below.

The RankCorr method is contained in a highly modified version of the
[PicturedRocks](https://github.com/picturedrocks) data analysis package.  
The modified version is included here.  See
the PicturedRocks repository for further information and extra (new) tools!

```
from picturedRocks import Rocks
```

Required inputs for the `Rocks` class:

* `X`, an `np.ndarry` of gene counts.  Each row should contain the genetic information from a cell; the columns of `X` correspond to the genes (note that this is the transpose of some commonly used packages).
* `y`, a vector of cluster labels.  These labels must be consecutive integers starting at 0.



```
data = Rocks(X, y)

lamb = 2.0 # the sparsity parameter
markers = data.CSrankMarkers(lamb=lamb)
```


