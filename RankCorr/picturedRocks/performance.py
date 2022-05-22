# Copyright © 2017 Anna Gilbert, Alexander Vargo, Umang Varma
# 
# This file is part of PicturedRocks.
#
# PicturedRocks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# PicturedRocks is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with PicturedRocks.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import plotly.graph_objs as go
import scipy.spatial.distance
from picturedrocks import Rocks
from plotly.offline import iplot

from scipy.stats import multivariate_normal

def kfoldindices(n, k, random=False):
    basearray = np.arange(n)
    if random:
        np.random.shuffle(basearray)
    lengthfloor = n//k
    extra = n % k
    cur = 0
    while cur < n:
        thislength = lengthfloor + 1 if extra > 0 else lengthfloor
        yield basearray[cur:cur + thislength]
        cur += thislength
        extra -= 1 # this should be extra = max(extra - 1, 0),
        #            but it doesn't matter

class PerformanceReport:
    def __init__(self, y, yhat):
        self.y = y
        self.yhat = yhat
        self.N = y.shape[0]
        
        self.K = y.max() + 1
        assert np.equal(np.unique(self.y), range(self.K)).all(), \
                "Cluster labels should be 0, 1, 2, ..., K -1"

        self.clusterindices = {}
        self._genclusterindices()
        
    def _genclusterindices(self):
        """Compute and store indices for cells in each cluster."""
        for k in range(self.K):
            self.clusterindices[k] = np.nonzero(self.y == k)[0]
        self.nk = np.array([len(self.clusterindices[k]) for k in range(self.K)])
        #nk[k] is the number of entries in cluster k

    def wrong(self):
        """Returns the number of cells misclassified."""
        return np.sum((self.y.flatten() != self.yhat)*1.0)
    
    def printscore(self):
        """Print a message with the score"""
        wrong = self.wrong()
        print("{} out of {} incorrect: {:.2f}%".format(wrong, self.N, 100 *
            wrong/self.N))

    def getconfusionmatrix(self):
        """Returns the confusion matrix for the latest run"""
        K = self.K
        freq_table = np.zeros([K, K])
        for i in range(K):
            clust, clust_count = np.unique(self.yhat[self.clusterindices[i]],
                    return_counts = True)
            for j, k in enumerate(clust):
                freq_table[i,k] = clust_count[j]/self.nk[i]
        return freq_table
    
    def confusionmatrixfigure(self):
        """Compute and make a confusion matrix plotly figure"""
        freq_table = self.getconfusionmatrix()
        shape = freq_table.shape
        trace = go.Heatmap(z=freq_table, x=np.arange(shape[1]),
                y=np.arange(shape[0]), colorscale="Greys",
                reversescale=True)
        layout = go.Layout(title="Confusion Matrix",
                   xaxis=dict(title="Predicted Cluster"),
                   yaxis=dict(title="Actual Cluster", scaleanchor='x'),
                   width=450,
                   height=450,
                   margin=go.Margin(l=70, r=70, t=70, b=70, pad=0,
                       autoexpand=False),
                   annotations=[dict(text="Rows sum to 1", x=0.5, y=1,
                         xref='paper', yref='paper', xanchor='center',
                         yanchor='bottom', showarrow=False)])
        return go.Figure(data=[trace], layout=layout)
    
    def show(self):
        """Print a full report"""
        self.printscore()
        iplot(self.confusionmatrixfigure())

class FoldTester:
    def __init__(self, data):
        self.data = data
        
        self.folds = None
        self.yhat = None
        self.markers = None

        self.rocks = None
        self.rocksGen = False
    
    def makefolds(self, k=5, random=False):
        self.folds = list(kfoldindices(self.data.N, k, random))
        
    def savefolds(self, file):
        d = {"k": len(self.folds), "y": self.data.y}
        for i, f in enumerate(self.folds):
            d["fold{}".format(i)] = f
        return np.savez(file, **d)
    
    def loadfolds(self, file):
        d = np.load(file)
        k = d["k"]
        self.folds = [d["fold{}".format(i)] for i in range(k)]
        assert np.array_equal(self.data.y, d["y"]),\
                "y vector does not match."
        assert self.validatefolds(), "folds are not partition of indices"
        
    def validatefolds(self):
        counts = np.zeros(self.data.N)
        for f in self.folds:
            counts[f] += 1
        return np.alltrue(counts == 1)

    # generate and save all of the Rocks objects that we will need for cross
    # validation. We use advanced indexing and thus we DO make a copy of the
    # data, so be careful that you don't use up too much memory here.  (Note
    # the you are actually making several copies of the data - nearly one copy
    # for each fold!)
    def makerocks(self, verbose=0):

        if self.folds is None: 
            print("Error in FoldTester.makerocks: need folds.")
            return(1)
        
        if self.rocksGen: return(0)
        else: 
            self.rocks = []
            self.rocksGen = True


        k = len(self.folds)
        for f in self.folds:
            mask = np.zeros(self.data.N, dtype=bool)
            mask[f] = True
            self.rocks.append( Rocks(self.data.X[~mask], self.data.y[~mask],
                    verbose=verbose) )

        # quick consistency check since we are appending...
        if len(self.rocks) != k:
            print("Error in FoldTester.makerocks: final list of rocks objects\
                    has incorrect length.  Please re-run makerocks.")
            return(1)
        
        return(0)
        
    def selectmarkers(self, select_function, verbose=0):
        k = len(self.folds)
        self.markers = []
        for (ind, f) in enumerate(self.folds):
            mask = np.zeros(self.data.N, dtype=bool)
            mask[f] = True
            traindata = Rocks(self.data.X[~mask], self.data.y[~mask],
                    verbose=verbose) if not self.rocksGen else self.rocks[ind]
            self.markers.append(select_function(traindata))
        
    def savefoldsandmarkers(self, file):
        d = {"k": len(self.folds), "y": self.data.y}
        for i, f in enumerate(self.folds):
            d["fold{}".format(i)] = f
        for i, m in enumerate(self.markers):
            d["marker{}".format(i)] = m
        return np.savez(file, **d)
    
    def loadfoldsandmarkers(self, file):
        d = np.load(file)
        k = d["k"]
        self.folds = [d["fold{}".format(i)] for i in range(k)]
        self.markers = [d["marker{}".format(i)] for i in range(k)]
        assert np.array_equal(self.data.y, d["y"]),\
                "y vector does not match."
        assert self.validatefolds(), "folds are not partition of indices"
        
    def classify(self, classifer):
        self.yhat = np.zeros(self.data.N, dtype=int) - 1
        for i, f in enumerate(self.folds):
            mask = np.zeros(self.data.N, dtype=bool)
            mask[f] = True
            # this is *advanced indexing* and thus we make a copy of the data.
            traindata = Rocks(self.data.X[~mask,:][:,self.markers[i]],
                    self.data.y[~mask])
            c = classifer()
            c.train(traindata)
            self.yhat[f] = c.test(self.data.X[f,:][:,self.markers[i]], self.data.sparse)

class NearestCentroidClassifier:
    def __init__(self):
        self.traindata = None
        self.xkibar = None
    
    def train(self, data):
        self.traindata = data
        # this seems bad.  We definitely normalize for a second time if our data
        # was already normalized.
        #data.normalize(totalexpr=1000, log=True)
        if data.sparse:
            self.xkibar = np.array([ 
                np.squeeze(
                    np.asarray( data.X[data.clusterindices[k]].mean(axis=0) )
                ) for k in range(data.K)
                ])
        else:
            self.xkibar = np.array([data.X[data.clusterindices[k]].mean(axis=0) for
                k in range(data.K)])

    
    def test(self, Xtest, sparse):
        # and then we don't normalize Xtest again...
        if sparse:
            dxixk = scipy.spatial.distance.cdist(np.squeeze(np.asarray(Xtest.todense())), self.xkibar)
        else:
            dxixk = scipy.spatial.distance.cdist(Xtest, self.xkibar)

        return dxixk.argmin(axis=1)            

"""
def GMMClassifier:
    def __init__(self):
        self.traindata = None
        self.means = None
        self.covars = None
        self.rvs = None

    def train(self, data):
        self.traindata = data

        self.means = np.array([data.X[data.clusterindices[k]].mean(axis=0) for
            k in range(data.K)])

        self.covars = [np.cov(data.X[data.clusterindices[k]], rowvar=False, bias=True)
                for k in range(data.K)]

        self.rvs = [multivariate_normal( means[k], covars[k] ) for k in range(data.K)]

    def test(self, Xtest):
        llhoods = np.array([rv.pdf(Xtest) for mv in self.rvs])
        return np.argmax(llhoods, axis=0)
"""
