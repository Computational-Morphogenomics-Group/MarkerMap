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
import scipy.spatial.distance as scipydist
from scipy.sparse.linalg import svds
import scipy.sparse as spsp

from scipy.stats import rankdata

# ******* Helper functions for the 1 bit compressed sensing method **********

#### Correlations, as in SPA Stanardize
#
# This computes a vector. The $i$-th entry is the correlation (as defined in
# section 3.2.3 of Genzel's thesis) between feature vector corresponding to gene
# $i$ and the cluster data corresponding to the input clusters . 

# compute corrlation of columns of mat with vec Note that if a denominator is 0
# (from a standard deviation being 0), then the correlation will also be 0 (as
# vec = vecBar or the column of mat = the column of matBar) this is desired for
# this situation since this will only occur if vec = 0 or the column of mat = 0
#
# There is also a scipy function to do this.  Using it results in nans (instead
# of 0s) in all of the genes that have all 0s, so this works more nicely.  But
# it might be slower.
def corrVec(vec, mat):
    matBar = mat.mean(axis=0)
    vecBar = vec.mean()
    
    r_num = np.sum( (mat-matBar)*(vec[:,None]-vecBar), axis=0) *1.0
    r_den = vec.shape[0]*np.std(vec)*np.std(mat,axis=0)

    if (len(list(r_den[r_den == 0])) != 0): 
        r_den[r_den == 0] = r_den[r_den == 0] + 0.000000001
            
    return r_num/(r_den)

# Generate a list of values of epsilon to test.
def genEpsList(nsteps=10, minEps=10.0**(-5), maxEps=10.0**(-3)):
    stepSize = (maxEps - minEps)/(nsteps*1.0)
        
    epsList = [maxEps - mult* stepSize for mult in range(nsteps+1)]
    return epsList

# soft threshold inVec by decreasing all entries by param
def softThreshold(inVec, param):
    
    signs = np.sign(inVec)
    inVec = inVec - param * signs
    inVec[ np.invert(signs == np.sign(inVec)) ] = 0.0
    return inVec
    
def rescale(inVec, param1, param2, warnings=False):
    '''Rescales the paramter inVec so that its 2-norm is less than param2 
    and its 1-norm is less than param1
 
     Return the rescaled vector along with the norm (1 or 2)
     that is "tightest"
    '''

    if np.linalg.norm(inVec) == 0:
        print("I think the inVec is a 0.", flush=True)

    tol = 1e-8

    whichNorm = 2
    outVec = inVec * param2 / np.linalg.norm(inVec)

    norm1 = np.sum(np.abs(outVec))
    if (norm1 > param1):
        whichNorm = 1
        outVec = outVec * param1/norm1
    
    # These are probably not needed and I have never seen them. 
    if (warnings):
        assert np.linalg.norm(outVec) <= param2 + tol, \
            "ERROR: 2-norm of rescaled vector exceeds desired parameter"
        assert np.sum(np.abs(outVec)) <= param1 + tol, \
            "ERROR: 1-norm of rescaled vector exceeds desired parameter"

    return outVec, whichNorm

def pca(Xin, dim=3):
    Xbar = Xin.mean(axis=0)
    Xcent = Xin - Xbar
    print("Computing Corvariance Matrix")
    Sigma = np.cov(Xcent.T)
    print("Performing SVDs")
    pcs = svds(Sigma, dim, return_singular_vectors="u")
    Xpca = Xcent.dot(pcs[0])
    return (Xcent, Sigma, pcs, Xpca)

### A FAST RANK TRANSFORM FOR SPARSE DATA
# Both of these modify the data by default;
# they can return a ranked copy if desired.
# They also both only work on vectors with all entries larger than 0
# and are faster with many repeated values
#
# You should, in general, input the size of the sparse vector,
# since the methods cannot tell between a row vector and a column 
# vector.  It will do its best to give you the correct dimension,
# but no guarantees.
#
# Also make sure that you have run "eliminate_zeros()" on your sparse
# data!  Bad things will occur if you have entries = 0.

# note that we could also implement this with one dictionary: a function to
# lookup the index of val in the array uniqueVals (after we have sorted
# uniqueVals).  You could store the counts and thus the ranks in arrays as well
# (don't need lookup for them) Probably minimal time save for our applications,
# so I'm not bothering now.
def fastRank(sparseVec, dim=0, keepZeros=False, copy=True):

    sparseVec = sparseVec.copy() if copy else sparseVec

    # short circuit if we are given a list of zeros
    if (sparseVec.nnz == 0):
        # centered rank vector will be all 0s, zeroVal=0, std=0
        return sparseVec, 0, 0

    if (dim==0): dim = sparseVec.shape[0]
    
    # some constants:
    # number of zeros, average value, rank of the zero entreis
    nz = dim - sparseVec.nnz
    mu = (dim + 1)/2.0
    zeroVal = (nz + 1)/2.0 - mu

    # go through and count the entries in data.
    # not sure if this is faster than np.unique( return_counts=True)
    # might be better to do that for numba speedups
    # but you really need to eliminate_zeros first.
    countDict = {}
    uniqueVals = []
    
    for val in sparseVec.data:
        
        # test if it's an index
        if val in countDict:
            countDict[val] += 1
            
        else:
            uniqueVals.append(val)
            countDict[val] = 1
            
    uniqueVals = np.sort(np.array(uniqueVals))
    
    # now calculate what the ranks at each value should be
    # leave the zeros as 0 if you want to ignore them in dot products
    rankDict = {0: zeroVal if not keepZeros else 0}
    running = nz
    
    # we will also calculate the variance (to get the standard deviation)
    # not sure what this really means when we delete the 0s.
    # but I will leave things like this for now.
    var = nz * zeroVal**2
    
    for val in uniqueVals:
        
        rankDict[val] = running + (countDict[val] + 1)/2 - mu
        running += countDict[val]
        
        var += countDict[val] * rankDict[val]**2
        
    sparseVec.data = np.array([rankDict[val] for val in sparseVec.data])
    
    return sparseVec, zeroVal, np.sqrt(var/dim)

# use this if your data is only nonnegative integers and has a fairly small
# dynamic range.
def fastRankInt(sparseVec, dim=0, copy=True):

    sparseVec = sparseVec.copy() if copy else sparseVec

    # short circuit if we are given a list of zeros
    if (sparseVec.nnz == 0):
        # centered rank vector will be all 0s, zeroVal=0, std=0
        return sparseVec, 0, 0

    if (dim==0): dim = sparseVec.shape[0]
    
    # some constants:
    # number of zeros, average value, rank of the zero entreis
    nz = dim - sparseVec.nnz
    mu = (dim + 1)/2.0
    zeroVal = (nz + 1)/2.0 - mu
    
    dataMax = sparseVec.data.max()
    #print("Max is {}".format(dataMax))
    counts = np.zeros(int(dataMax)+1, dtype='int') # indices 0 to dataMax
    
    # go through and count the entries in data.
    # not sure if this is faster than np.unique( return_counts=True)
    # might be better to do that for numba speedups
    # but you really need to eliminate_zeros first.
    uniqueVals = []
    
    for val in sparseVec.data:        
        counts[val] += 1

    # now calculate what the ranks at each value should be
    rankDict = {0: zeroVal}
    running = nz
    
    # we will also calculate the variance (to get the standard deviation)
    var = nz * zeroVal**2
        
    for ind,count in enumerate(counts):
        if ind != 0 and count != 0:
            rankDict[ind] = running + (count + 1)/2 - mu
            running += count
            
            var += count * rankDict[ind]**2
        
    sparseVec.data = np.array([rankDict[val] for val in sparseVec.data])
    
    return sparseVec, zeroVal, np.sqrt(var/dim)


class Rocks:
    # A Rocks object takes in the following parameters:
    # X: the gene expression matrix. We expect a numpy array of shape (N, P)
    #    containing data for N cells and P genes (note rows are cells and
    #    columns are genes.
    # y: cluster labels for N cells. We expect a numpy array of shape (N, 1) or
    #    (N,).
    # genes: names of genes. We expect an array of P strings, containing the
    #    names of various genes
    # verbose: verbosity level for debugging; defaults to 0.
    def __init__(self, X, y, genes=None, verbose=0):
        # self.X is the expression data - in adata
        # self.y is the cluster assignment - in adata
        # self.N is the number of cells - get from adata
        # self.P is the number of genes - get from adata
        # self.K is the number of clusters - get from adata
        # self.genes contains the names of the genes - in adata
        self.verbose = verbose
        self.X = X
        self.y = y
        self.N, self.P = X.shape

        self.sparse = not isinstance(self.X, np.ndarray)
        if self.sparse: self.X.eliminate_zeros()

        self.genes = genes
        assert (genes is None or len(genes) == self.P), \
                "genes must be an array of length P or None"

        self.Xcent, self.pcs, self.Xpca = (None, None, None)
        if y.shape == (self.N,):
            self.y = y.reshape((self.N, 1))
        assert self.y.shape == (self.N, 1), \
                "y should be a matrix of shape (N,1)"
        
        self.K = self.y.max() + 1
        assert np.array_equal(np.unique(self.y), range(self.K)), \
                "Cluster labels should be 0, 1, 2, ..., K -1"

        self.clusterindices = {}
        for k in range(self.K):
            self.clusterindices[k] = np.nonzero(self.y == k)[0]
        nk = np.array([len(self.clusterindices[k]) for k in range(self.K)])
    
        # Some extra elements that will be needed for OvA
        # they will be changed in the methods below
        self.cs_currY = self.y
        self.cs_currX = self.X

        # save the vectors of consts so that we don't need to deal with
        # manipulating the data over and over.  
        # Useful when testing multiple values of the parameter.
        self.cs_OvA = np.zeros((self.K, self.P))
        self.cs_rankCorr = np.zeros((self.K, self.P))
        self.cs_generated = False
        self.cs_generatedClusters = np.zeros(self.K)
        self.cs_lambFac = None
        self.cs_alpha = None

        # whether or not we want to look at rank correlation
        self.cs_useRanks = False
        
    def _debug(self, level, message):
        if self.verbose >= level:
            print(message, flush=True)

    # markers can be a list or 1-dim np array
    def markers_to_genes(self, markers):
        try:
            return [self.genes[a] for a in markers]
        except TypeError:
            raise ValueError("Gene names not specified. Set using object.genes")

    
    def normalize(self, totalexpr="median", log=True):
        cellsize = self.X.sum(axis=1).reshape((self.N,1))
        targetsize = np.median(cellsize) if totalexpr == "median" else totalexpr

        # avoid zero_divide issues
        cellsize[cellsize == 0] = 1

        self.X = (targetsize*self.X)/cellsize
        if log:
            self.X = np.log(self.X +1)
        self.Xcent, self.pcs, self.Xpca = (None, None, None)
    
    def pca(self, dims=3):
        self.Xcent, Sigma, self.pcs, self.Xpca = pca(self.X, dims)
        self.totalvariation = np.trace(Sigma)
        # Sigma is too big to be worth storing in memory
    
    # THE 1 BIT COMPRESSED SENSING METHOD (1CS, GENZEL)
    #
    # The methods below implement only one-vs-rest (ovr) multiclass
    # classification.  All-vs-all will be added soon
    #
    # We are evaluating the selected markers by a simple nearest centroid
    # method: we are no longer looking at the margins to classify the cells.
    # There is good code written to look at the margins if you want to do this
    # later (e.g. for implementing all-vs-all).  See the older versions of
    # picturedrocks (before April 29 2018) to recover these functions.
    #
    # TODO: Some functions below might be repeats of earlier functions

    # Get the indices of cells in cluster clustind
    # Note that the index that you input should be one MORE than the actual cluster 
    # you are looking for
    def clust2vec(self, clustind=1):
        returnvec = -1.0*np.ones(self.cs_currY.shape[0])
        returnvec[[i for i,x in enumerate(self.cs_currY) if x==clustind - 1]] \
                = 1.0

        #if self.cs_useRanks: 
        #    returnvec = rankdata(returnvec)
        #    return returnvec - returnvec.mean()
        #else: 
        return returnvec

    # Make the appropriate vector for 1CS methods
    def coeffs(self, clustind=1):
        """Creates the vector of coefficients used in the optimization problem.
        The solution is a soft-thresholding of this vector.  

        The method uses the data in self.cs_currX to get to the correct vector
        of coefficients, not the data in self.X

        :param clustind: The cluster that you would like to be separating.

        """
        rankvec = rankdata(self.clust2vec(clustind))
        rankvec = rankvec - rankvec.mean()
        self.cs_rankCorr[clustind - 1] = np.sum( rankvec[0:,np.newaxis]*self.cs_currX, axis=0 )
        
        return np.sum( self.clust2vec(clustind)[0:,np.newaxis]*self.cs_currX,
                axis=0 )

    def findXform(self, alpha=0.3, c=2.0, lambFac=1.0, clustOne=1, clustTwo=None):
        """Find the transform for scaling the data in the 1-bit CS methods.

        See Genzel (2015) for more information about the parameters.

        :param alpha: The CS alpha parameter
        :param lambFac: The CS lambda paramter
        :param clustOne: The first cluster that you want to consider
        :param clustTwo: [Optional] the second cluster to consider
        (for use with finding markers between two clusters; this functionality
        is not fully implemented)
        """
        # print("Finding transform with alpha = {}, c = {}, lambFac = {}, cluster
        # = {}".format(alpha, c, lambFac, clustOne), flush=True)
       
        # restrict the data if we are looking at two clusters
        if (clustTwo):
            self.cs_currY = np.concatenate( [self.y[self.y ==clustOne],
                self.y[self.y==clustTwo]] )
            self.cs_currX = np.concatenate( [self.X[
                np.squeeze(np.asarray(self.y==clustOne)) ],
                self.X[ np.squeeze(np.asarray(self.y==clustTwo)) ]] )
        
        # find vector of correlations
        rho = corrVec( self.clust2vec(clustOne), self.cs_currX )
        # print("Correlations: {}".format(rho[0:10]), flush=True)
        
        # find scaling vector
        sigma = np.std(self.cs_currX, axis=0)

        # the only time sigma = 0 for our situation is a gene that is never
        # expressed so this doesn't hurt anything.
        sigma[sigma==0] = 0.0000001
        alphaFac = alpha**(c * ( 1-np.abs(rho)))
        scaleFac = lambFac*alphaFac + (1 - alphaFac)/sigma
        
        # find center
        xbar = self.cs_currX.mean(axis=0)
        
        return xbar, scaleFac

    # This function is very specific to the 1 bit compressed sensing algorithm and
    # doesn't need to be a class function.  I like it here though.
    #
    # Try soft thresholding with each coordinate to figure out how many
    # coordinates the optimization should actually take.  Assuming that we
    # start with the 2-norm (i.e. s >= 1 and consts != 0)
    #
    # We only report 1 norm for strict '>' in the above function; thus, we
    # never cut off too many indices (even if there is a point where the 2-norm
    # and 1 are equal
    #
    # absConsts should be the sorted norms of the consts vector
    def findNormChanges(self, consts, absConsts, s=10):

        currNorm = 2
        prevNorm = 2
        switchInd = [] 
        numSwitch = 0

        # skip the largest index so that we point along an axis 
        startInd = 1 
        while (absConsts[startInd] == absConsts[0]): startInd += 1
        ind = startInd

        # for simple tests, make sure that the index only changes once when
        # running on actual data just find the first place that the index
        # changes.
        while ind < consts.shape[0] and len(switchInd) == 0:

            val = absConsts[ind]
            curr, currNorm = rescale( softThreshold(consts, val), np.sqrt(s), 1.0 )
        
            if (currNorm != prevNorm):
                self._debug(1, "Switched from {} norm to {} norm at ind {}"\
                        .format(prevNorm, currNorm, ind))
                prevNorm = currNorm
                switchInd.append(ind)
                numSwitch = numSwitch + 1
        
            ind = ind + 1

        if (numSwitch > 1): self._debug(0, "Warning: norms switched {} times"\
                .format(numSwitch))

        return switchInd

    def findCSmarkers(self, currLamb, writeOut=True, alpha=0.0, lambFac=0.0):
        """Find markers via the 1-bit compressed sensing method following
        a "one vs all" (OvA) philosophy: for each cluster, find the markers
        that best separate the cluster from the rest of the data and return 
        the union of the markers.
        
        We use a soft-thresholding of a specific vector to get to a quick solution
        (instead of optimizing).  The first run will be slower, since we generate
        a transformed version of the data.  This information is saved, however, 
        so subsequent runs will be very fast.  This should allow you to find markers
        for many different values of the inputs quite quickly.

        :param currLamb: The sparsity parameter (controls how many features to select)
        :param writeOut: (bool) whether or not to write the selected markers to a file.
        :param alpha: the CS alpha paramter
        :param lambFac: the CS lambda paramter.
        """
        self._debug(1, "Working on lamb = {}".format(currLamb))

        marks = []

        # self.K = number of clusters
        for clust in range(1,self.K+1):

            """
            self._debug(1, "Finding markers for cluster {}".format(clust))
            # Don't think that this is needed, but just being safe
            self.cs_currX = self.X

            cent,scale = self.findXform(alpha=alpha, c=2.0, lambFac=lambFac,
                clustOne=clust)

            self._debug(3, "Transform center and scale: {}, {}".\
                format(cent, scale))

            self.cs_currX = (self.X - cent)*scale
            consts = self.coeffs(clust)

            # flipud reverses the list
            # the sorting does not appear to take very long.  I could save this
            # information if it becomes the next bottleneck, however.
            absConsts = np.flipud(np.sort(np.abs(consts)))
            switchInd = self.findNormChanges(consts, absConsts, s=currLamb)
            print("Found all norm changes.", flush=True)
            if len(switchInd) > 1: self._debug(0, "More than one norm switch:\
                            assuming the first is the best")
            
            curr, currNorm = rescale(\
                softThreshold(np.asarray(consts).flatten(),\
                    absConsts[switchInd[0]]),\
                np.sqrt(currLamb), 1.0 )

            marks.append(curr.nonzero()[0])
            """

            
            # this does exactly the same stuff as in the comments, but 
            # with some more tests for generating (and saving) the consts
            # data, dealing with ranks, etc.  You can go though and diff
            # the functions if you would like.  I did this on
            # 29 July 2018
            clustMarks = self.findCSmarkersForClust(currLamb, alpha=alpha,\
                    lambFac=lambFac, clust=clust)

            marks.append(clustMarks)
            

        # delete final cs_currX
        self.cs_currX = self.X

        # write the support genes to a data file
        if (writeOut):
            geneFile = "ovrGenes-lamb{}-lFac{}-alpha{}.dat".format(currLamb,
                    lambFac, alpha)
            gFile = open(geneFile, 'w')
            for geneList in marks:
                for gene in geneList:
                    gFile.write("{} ".format(gene))
                gFile.write("\n")

        # return the list of markers
        return self.support2list(marks)

    # assuming OvA so that the center is just the center of the data
    # (and thus we are free to throw it out).
    def findCSmarkersForClust(self, currLamb, alpha=0.0, lambFac=0.0, clust=1):

        self._debug(1, "Finding markers for cluster {}".format(clust))

        consts = self._genOvaConsts(clust, alpha, lambFac)
        # flipud reverses the list
        # the sorting does not appear to take very long.  I could save this
        # information if it becomes the next bottleneck, however.
        absConsts = np.flipud(np.sort(np.abs(consts)))
        switchInd = self.findNormChanges(consts, absConsts, s=currLamb)
        self._debug(2, "Found all norm changes.")
        if len(switchInd) > 1: self._debug(0, "More than one norm switch:\
                        assuming the first is the best")
        
        curr, currNorm = rescale(\
            softThreshold(np.asarray(consts).flatten(),\
                absConsts[switchInd[0]]),\
            np.sqrt(currLamb), 1.0 )

        return curr.nonzero()[0]
    
    # This will save you some time if you are running multiple trials for the 
    # same values of alpha and lambFac. There are limited safety features here.
    # you need to make sure to run it for all of the clusters, the order doesn't
    # matter though.
    def _genOvaConsts(self, clust, alpha, lambFac):
        """Transform the data into the proper space for marker selection
        and generate the correct consts vector in the transformed space.

        You only need to run this if you are transforming the data in a 
        specific way before finding the consts vector.
        """
        # check to make sure that alpha and lambFac have remained consistent.
        # If not, we need to generate all of the consts.
        if alpha != self.cs_alpha or lambFac != self.cs_lambFac:
            self.cs_generated = False
            self.cs_generatedClusters = np.zeros(self.K)
            self.cs_alpha = alpha
            self.cs_lambFac = lambFac

        # don't do this if you don't have to.
        if self.cs_generated: 
            return self.cs_OvA[clust-1]

        # mark that we have generated the consts for the current cluster
        self.cs_generatedClusters[clust-1] = 1.0

        # update cs_generated if we have generated everthing
        if np.sum(self.cs_generatedClusters) == self.K: self.cs_generated = True

        # generate the consts vector.
        # Don't think that this is needed, but just being safe
        self.cs_currX = self.X

        cent,scale = self.findXform(alpha=alpha, c=2.0, lambFac=lambFac,
            clustOne=clust)

        self._debug(3, "Transform center and scale: {}, {}".\
            format(cent, scale))

        self.cs_currX = (self.X - cent)*scale
        consts = self.coeffs(clust)
        
        # save the vector so that you only do this once.
        self.cs_OvA[clust-1] = consts
        return consts

    # Makes a list out of a collection of support genes
    def support2list(self, sgenes):
        import itertools
        flattened = itertools.chain.from_iterable(sgenes)
        return list(set(flattened))

    def markers_CS(self, currLamb, writeOut=False, alpha = 0.0, lambFac = 0.0,
            epsList=genEpsList()):
        
        [classes, support_genes] =  self.simpleCS(currLamb, writeOut,
                alpha=alpha, lambFac=lambFac, epsList=epsList)
        return self.support2list(support_genes)

    # This finds `eps`, the offset that you need to soft threshold after you have
    # found the index when the norm constraint switches from the 2-norm to the
    # 1-norm.  (Note that, after the norm constraint switches from 2 to 1, the
    # entries are too large - that is, you have not soft thresholded enough at that
    # point).
    # 
    # * `xVec` should be the soft thresholded version of the solution that has the
    # support of the exact solution.  
    # * `kay` should be the number of nonzero entries in xVec 
    # * `ess` is the full sparsity parameter
    def findEps(self, xVec, kay, ess):
    
        assert xVec.nonzero()[0].shape[0] == kay,\
            "Error with inputs: xVec doesn't have kay nonzero entries"
    
        xNorm = np.sum(np.abs(xVec))
        c = xNorm*xNorm - ess * np.sum( np.abs(xVec) * np.abs(xVec) )
        b = -1.0 * xNorm * 2 * (kay-ess)
        a = kay*(kay - ess)
    
        # quadratic formula
        #return [(-1.0 * b - np.sqrt( b*b - 4*a*c )) / (2*a), (-1.0 * b + np.sqrt( b*b - 4*a*c )) / (2*a)]

        # use np.roots for more precision.
        return np.roots(np.array([a,b,c]))

    # curr should be the result of a rescale(softThreshold)
    #
    # it is also important to note: if we are ever actually interested in the
    # actual W vector for the OVA process (not just for testing to see how well
    # the 1 bit CS alg works on ideal data), we will need to rescale the W
    # vector that we obtain from this.  Here is the code from the old process,
    # when we were actually looking at the margins:
    #    rescale the weight vector and save it
    #    dubs.append(np.squeeze(np.asarray(currDub))*scale)
    # where scale is the vector that is returned from findXform
    def findWST(self, curr, ess):
        tol = 1e-4

        # number of nonzero entries
        kay = curr.nonzero()[0].shape[0]

        # np.roots returns array of length 2 even if root of multiplicity 2
        eps = self.findEps(curr, kay, ess)

        curr1 = softThreshold(curr, eps[0])
        curr2 = softThreshold(curr, eps[1])

        curr1 = curr1 / np.sum(np.abs(curr1)) * np.sqrt(ess) if np.sum(np.abs(curr1)) > 0 else 0.0
        curr2 = curr2 / np.sum(np.abs(curr2)) * np.sqrt(ess) if np.sum(np.abs(curr2)) > 0 else 0.0

        n1 = np.abs(1.0-np.linalg.norm(curr1))
        n2 = np.abs(1.0-np.linalg.norm(curr2))

        if (n1 < tol and n2 < tol): _debug(0, "Warning: two very similar\
                solutions for W found.  Arbitrarily choosing one.")

        return curr1 if n1 <= n2 else curr2

    # make sure to have normalized the data (however you were planning to)
    # before runnign this.
    #
    # This puts a ranked version of X into cs_currX
    # thus, it uses double memory, but it will work for now.
    #
    # keepZeros doesn't rank the 0s in the matrix so that they will still count
    def CSuseRanks(self, useRankQ=True, keepZeros=False, onlyNonZero=False):

        # do nothing if we're changing to the current behavior
        if self.cs_useRanks == useRankQ: return

        self.cs_useRanks = useRankQ

        if self.sparse:
            # do nothing here except set the flag if needed
            return

        # we are not sparse in the following:
        if useRankQ:
            # allocate memory
            self.cs_currX =  np.zeros((self.N, self.P))

            if onlyNonZero:
                sigma = []
                xbar = []
                for gene in range(self.P):

                    currNz = np.nonzero(self.X[:,gene])[0]
                    vals = rankdata(self.X[:,gene][currNz])
                    
                    # move to rank space 
                    self.cs_currX[:,gene][currNz] = vals

                    sigma.append( np.std(vals) if vals.size > 0 else 0 )
                    xbar.append( vals.mean() if vals.size > 0 else 0 )

                sigma = np.array(sigma)
                xbar = np.array(xbar)
            else: # work on all values, including zeros

                # move to rank space
                for gene in range(self.P):
                    self.cs_currX[:,gene] = rankdata(self.X[:,gene])

                # standardize:
                # find scaling vector
                sigma = np.std(self.cs_currX, axis=0)

                # find center
                xbar = self.cs_currX.mean(axis=0)

            # always run
            # the only time sigma = 0 for our situation is a gene that is never
            # expressed so this doesn't hurt anything.
            sigma[sigma==0] = 0.0000001

            self.cs_currX = (self.cs_currX - xbar)/sigma

            if keepZeros or onlyNonZero:
                self.cs_currX[ np.where(self.X == 0) ] = 0
                
        else:
            # get rid of memory
            self.cs_currX = self.X
            self.cs_generated = False
            self.cs_generatedClusters = np.zeros(self.K)

    # currently not looking at doing any shifts or standardization
    def CSrankMarkers(self, lamb, writeOut=False, keepZeros=False, onlyNonZero=False):

        if not self.cs_useRanks: 
            self.CSuseRanks(True, keepZeros=keepZeros, onlyNonZero=onlyNonZero)

        # create vectors of coeffs if not yet generated
        self._CSrankConsts()
        
        # find the markers.
        marks = []
        for clust in range(1,self.K+1):
            self._debug(1, "Finding markers for cluster {}".format(clust))

            consts = self.cs_OvA[clust-1]

            # flipud reverses the list
            # the sorting does not appear to take very long.  I could save this
            # information if it becomes the next bottleneck, however.
            absConsts = np.flipud(np.sort(np.abs(consts)))
            switchInd = self.findNormChanges(consts, absConsts, s=lamb)
            self._debug(2, "Found all norm changes.")
            if len(switchInd) > 1: self._debug(0, "More than one norm switch:\
                            assuming the first is the best")
            
            curr, currNorm = rescale(\
                softThreshold(np.asarray(consts).flatten(),\
                    absConsts[switchInd[0]]),\
                np.sqrt(lamb), 1.0 )

            marks.append(curr.nonzero()[0])

        # write the support genes to a data file
        if (writeOut):
            geneFile = "ovrRankGenes-lamb{}.dat".format(lamb)
            gFile = open(geneFile, 'w')
            for geneList in marks:
                for gene in geneList:
                    gFile.write("{} ".format(gene))
                gFile.write("\n")

        # return the list of markers
        return self.support2list(marks)

    # This will currently double the required memory.
    # ONLY call this after running CSuseRanks as it uses whatever
    # is in cs_currX to find the coefficient vector.  
    # (otherwise, we will use the raw data)
    def _CSrankConsts(self):

        if not self.cs_generated:
            if spsp.issparse(self.X):

                if not isinstance(self.X, spsp.csc_matrix):
                    self.X = self.X.tocsc()
                    self.cs_currX = self.X
                    #self.cs_currX = self.X.tocsc().astype(int)
                    self._debug(0, "Converted to csc matrix for generating consts")
                    
                for clust in range(1, self.K+1):
                    self._debug(1, "Generating coeffs for cluster {}".format(clust))
                    self.cs_OvA[clust-1] = self.sparseCoeffs(clust-1)

                    rankvec = rankdata(self.clust2vec(clust))
                    rankvec = rankvec - rankvec.mean()
                    self.cs_rankCorr[clust - 1] = self.sparseCoeffs(
                            clust-1, highval=rankvec.max(), lowval=rankvec.min()
                        )

            else: # not sparse

                for clust in range(1, self.K+1):
                    self._debug(1, "Generating coeffs for cluster {}".format(clust))
                    self.cs_OvA[clust-1] = self.coeffs(clust)

            # we get rid of the memory at the end.
            self.cs_currX = self.X
            self.cs_generated = True


    # this runs on cs_currX and transforms cs_currX to rank space
    # (if it wasn't already transformed)
    # it doesn't rank transform the vector tau by default
    def sparseCoeffs(self, clust, highval=1, lowval=-1):

        if not isinstance(self.cs_currX, spsp.csc_matrix):
            self._debug(0,"Your matrix is the wrong type: this might take a while")

        if not self.sparse:
            self._debug(0, "ERROR: sparse method called for non-sparse data.")

        consts = []
        setindices = set(self.clusterindices[clust])

        for gene in range(self.P):
            if not gene % 1000:
                print("Gene" + str(gene))
            consts.append( 
                self.sparse_tau_dot
                ( 
                    self.cs_currX.getcol(gene), 
                    self.clusterindices[clust], 
                    highval=highval, 
                    lowval=lowval, 
                    dim=self.N,
                    setindices=setindices
                )
            )

        return np.array(consts)

    # you should, in general, input the dimension.  Things will be wrong if
    # you input a row vector (e.g. a vector with shape (1,dim)).
    # this also changes the sparse vector in place so that it is now ranked.
    #
    # This doesn't really need to be a class method, but it is very specific to 
    # the rank transform 1bcs methods.  I will probably want to look at
    # specific dot products of specific genes at some point in the future.
    # So I'm keeping it here (rather than as a separate import) for now.
    #
    # I could also make it more of a class method: "gene" instead of "sparseVec"
    # "clust" instead of "clusterindices" and add a "dim" parameter since
    # this will always be the number of cells (self.N)
    #
    # the setindices parameter is purely for optimization - since we iterate over the
    # clusters, we can precompute the set of indices for all the genes, saving 
    # actually quite a lot of set constructor time. (for the 1M mouse data set, this appears
    # to save about 4 minutes per cluster)
    def sparse_tau_dot(self, sparseVec, clusterindices, highval=1, lowval=-1, dim=0, setindices=None):

        if (dim==0): dim = sparseVec.shape[0]

        if setindices is None:
            setindices = set(clusterindices)
            
        useInts = False
        if 'int' in str(sparseVec.dtype): useInts = True
        
        nz = dim - sparseVec.nnz

        nonzeroClust = len([ ind for ind in sparseVec.indices if ind in setindices ])
        zeroClust = len(clusterindices) - nonzeroClust
        zeroNonClust = nz -zeroClust

        tau = lowval * np.ones(dim)
        tau[clusterindices] = highval
        
        if useInts:
            sparseVec, zeroVal, sig = fastRankInt(sparseVec, dim=self.N)
        else:
            sparseVec, zeroVal, sig = fastRank(sparseVec, dim=self.N, keepZeros=False)
        
        dotVal = zeroVal*( zeroClust*highval + zeroNonClust*lowval)

        dotVal += np.sum(sparseVec.data * tau[sparseVec.indices])
            
        if (sig==0): sig = 1
        return dotVal/sig


def pcafigure(celldata):
    import colorlover as cl
    import plotly.graph_objs as go
    if celldata.Xpca is None:
        celldata.pca(3)
    Xpca = celldata.Xpca
    clusterindices = celldata.clusterindices
    colscal = cl.scales['9']['qual']['Set1']
    plotdata = [go.Scatter3d(
            x=Xpca[inds,0],
            y=Xpca[inds,1],
            z=Xpca[inds,2],
            mode='markers',
            marker=dict(
                size=4,
                color=colscal[k % len(colscal)], # set color to an array/list
                #                                  of desired values
                opacity=1),
            name="Cluster {}".format(k),
            hoverinfo="name"
            )
            for k, inds in clusterindices.items()]

    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    return go.Figure(data=plotdata, layout=layout)
