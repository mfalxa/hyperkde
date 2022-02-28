"""
KDE module

Author: Mikel Falxa
"""

import numpy as np
from scipy.special import logsumexp
from scipy.spatial import KDTree
from scipy.cluster.vq import kmeans2
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import corner

class KDE:
    """ A KDE class  
    """
    
    def __init__(self, param_names, chains, bandwidth=None, adapt_scale=10, bw_adapt=True):
        """
        """
        self.param_name = param_names[0]
        self.param_names = param_names
        self.adapt_scale = adapt_scale
        if len(np.shape(chains)) == 1:
            self.ndim = 1
            self.chains = chains.flatten()
        else:
            self.ndim = np.shape(chains)[1]
            self.chains = chains
        self.n = len(chains)
        if bandwidth is None:
            bw0 = np.random.rand(self.ndim)
            if self.ndim == 1:
                bw0 = bw0[0]
            if isinstance(bw0, float):
                self.bw = bw0 * np.ones(self.n)
            else:
                self.bw = np.tile(bw0, (self.n, 1))            
        elif isinstance(bandwidth, float):
            if self.ndim == 1:
                self.bw = bandwidth * np.ones(self.n)
            else:
                self.bw = bandwidth * np.ones((self.n, self.ndim))
        
        self.k = int(self.n/(self.adapt_scale))
        if self.k > self.n:
            self.k = self.n
        if bw_adapt:
            self.sort_adapt_bandwidth(self.k)


    def coeff(self, d, k):
        """
        """
        num = 1/2 + d/4 - 2**(d/2) * (1 + d/2)
        den = d/2 - k * d * (2**(d/2) - 1)
        return num / den


    def _make_B(self, k):
        """
        """
        num = 2**(-(self.ndim + 3)/2) + k * (1/(2*(2**((self.ndim-1)/2))) - 1/(2**(1/2)))
        den = 1/(8 * (2**((self.ndim-1)/2))) - 1/(2**(3/2))
        B = (num / den) * np.ones(self.ndim)
        return B


    def _make_A(self, distances):
        """
        """
        A = np.tile(distances, (self.ndim, 1))
        A += 2*np.diag(distances)
        return A


    def adapt_bandwidth(self, k):
        """
        """
        if self.ndim == 1:
            tree = KDTree(np.reshape(np.copy(self.chains), (-1, 1)))
            k_nearest_distance = tree.query(np.reshape(self.chains, (-1, 1)), k=k+1)[0]
            factor = 3 / self._make_B(k)[0]
            self.bw = np.sqrt(factor * np.sum(k_nearest_distance**2, axis=1))
            self.chains = self.chains.flatten()
        else:
            distances = np.zeros((self.n, self.ndim))
            for d in range(self.ndim):
                tree = KDTree(np.reshape(self.chains[:, d].flatten(), (-1, 1)))
                distances[:, d] = np.sum(tree.query(tree.data, k=k+1)[0]**2, axis=1)    
            for i in range(self.n):
                self.bw[i, :] = np.sqrt(1/np.linalg.solve(self._make_A(distances[i]), self._make_B(k)))


    def sort_adapt_bandwidth(self, k):
        """
        """
        if self.ndim == 1:
            factor = 3 / self._make_B(k)[0]
            self.chains = self.chains.flatten()
            for i in range(self.n):
                k_nearest_distance = np.sort((self.chains - self.chains[i])**2)[:k]
                self.bw[i] = np.sqrt(factor * np.sum(k_nearest_distance))
        else:
            for i in range(self.n):
                distances = np.zeros(self.ndim)
                for d in range(self.ndim):
                    distances[d] = np.sum(np.sort((self.chains[:, d] - self.chains[i, d])**2)[:k])
                self.bw[i, :] = np.sqrt(1/np.linalg.solve(self._make_A(distances), self._make_B(k)))


    def draw(self, size=1, random=True):
        """
        """
        n = np.arange(self.n)
        if random:
            idx = np.random.choice(n, size=size, replace=False)
        else:
            idx = n[:size]
        if self.ndim == 1:
            x = self.chains[idx] + np.random.normal(loc=0., scale=self.bw[idx])
        else:
            x = self.chains[idx] + np.random.normal(loc=0., scale=self.bw[idx])

        return x


    def logprob(self, x):
        """
        """
        if self.ndim == 1:
            d = np.subtract(self.chains, np.repeat(np.sum(x), self.n))
            y = -0.5 * np.divide(d**2, self.bw**2) - 0.5 * self.ndim * (np.log(2*np.pi) + 2*np.log(self.bw))
            logpdf = logsumexp(y) - np.log(self.n)
        else:
            y = np.empty(self.n, dtype=float)
            d = np.subtract(self.chains, np.tile(x, (self.n, 1)))
            dbd = np.sum(np.divide(d**2, self.bw**2), axis=1)
            y = -0.5 * dbd - 0.5 * (self.ndim * np.log(2*np.pi) + 2*np.log(np.prod(self.bw, axis=1)))
            logpdf = logsumexp(y) - np.log(self.n)

        return logpdf


    def logprobs(self, x):
        """
        """
        logpdfs = np.zeros(len(x))
        for k in range(len(x)):
            logpdfs[k] = self.logprob(x[k])

        return logpdfs


    def prob(self, x):
        """
        """
        return np.exp(self.logprob(x))


    def probs(self, x):
        """
        """
        return np.exp(self.logprobs(x))


    def _density_at_point_i(self, i, data):
        """
        """
        edges_min = self.chains[i] - self.bw[i]
        edges_max = self.chains[i] + self.bw[i]
        bin_volume = np.prod(edges_max - edges_min)
        n_points = np.count_nonzero(self._mask_data(data, edges_min, edges_max))
        return n_points/bin_volume
