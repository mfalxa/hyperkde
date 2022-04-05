"""
Hyper-KDE module

Author: Mikel Falxa
"""

import numpy as np
import scipy.stats as scistats
from .kde import KDE
import matplotlib.pyplot as plt

class HyperKDE:
    """ A HyperKDE class  
    """
    def __init__(self, model_params, chains, chains_params, js_threshold, kde_bandwidth=None, bw_adapt=True, adapt_scale=10, use_kmeans=False, n_kde_max=1):
        """
        @ model_params : list of currently sampled model parameters
        @ chains : previously sampled chains to which KDE is applied
        @ chains_params : list of parameters of previously sampled chains to which KDE is applied
        @ js_threshold : Jensen-Shannon threshold value for which KDE parameters are considered correlated
        @ kde_bandwidth : set KDE bandwidth
        @ bw_adapt : adaptive bandwidth, works only with custom KDE
        @ adapt_scale : bandwidth adaptation scale
        @ n_kde_max : max random number of active sub-KDEs
        @ lib : used library to generate KDE ('scipy' or 'sklearn')
        """
        self.model_params = np.array(model_params)
        self.kde_bandwidth = kde_bandwidth
        self.bw_adapt = bw_adapt
        self.adapt_scale = adapt_scale
        self.use_kmeans = use_kmeans
        self.par_idx = [i for i in range(len(chains_params)) if chains_params[i] in list(model_params)]
        self.params = list(chains_params[self.par_idx])
        self.js_threshold = js_threshold
        self.groups_idx, self.paramlists = self._get_correlated_groups(self._get_JS_matrix(chains[:, self.par_idx]), chains_params[self.par_idx], js_threshold)
        for pl in self.paramlists:
            print(pl)
        print(len(self.groups_idx), 'groups of parameters found')
        self.kdes = self._get_distributions(chains[:, self.par_idx])
        self.nmax = len(self.kdes) if n_kde_max is None else n_kde_max
        self.weights = np.array([kde.ndim for kde in self.kdes]) / np.sum(np.array([kde.ndim for kde in self.kdes]))
        self.set_all_kdes()


    def _get_params_idx(self, chains_params):
        """ Match sub-kde parameter index to dataset parameter index
        """
        idxs = np.array([], dtype='int')
        for i in range(len(self.kdes)):
            iidxs = np.array([self.params.index(p) for p in self.kdes[i].param_names])
            idxs = np.append(idxs, iidxs)

        return idxs

    def draw(self):
        """ Draw new sample
        """
        x = np.array([])
        for idx in self.distr_idxs:
            x = np.append(x, self.kdes[idx].draw())
        return x

    def redraw_ordered_dataset(self):
        """
        """
        self.set_all_kdes()
        x = np.hstack([self.kdes[i].draw() for i in range(len(self.kdes))])


    def logprob(self, x):
        """ Get log-probability for sample x
        """
        logp = 0.
        n = 0
        for idx in self.distr_idxs:
            kde_ndim = self.kdes[idx].ndim
            logp += self.kdes[idx].logprob(x[n:n+kde_ndim])
            n += kde_ndim

        return logp


    def draw_from_random_hyp_kde(self, x):
        """Function to use as proposal if PTMCMC is used, draws samples from
        a random subset of KDEs
        """

        q = x.copy()
        lqxy = 0

        self.randomize_kdes()

        oldsample = [x[list(self.model_params).index(p)]
                    for p in self.param_names]
        newsample = self.draw()

        for p,n in zip(self.param_names, newsample):
            q[list(self.model_params).index(p)] = n

        lqxy += (self.logprob(oldsample) -
                self.logprob(newsample))

        return q, float(lqxy)


    def draw_from_hyp_kde(self, x):
        """Function to use as proposal if PTMCMC is used, draws samples from
        full set of KDEs

        """
        q = x.copy()
        lqxy = 0

        self.set_all_kdes()

        oldsample = [x[list(self.model_params).index(p)]
                    for p in self.param_names]
        newsample = self.draw()

        for p,n in zip(self.param_names, newsample):
            q[list(self.model_params).index(p)] = n

        lqxy += (self.logprob(oldsample) -
                self.logprob(newsample))

        return q, float(lqxy)


    def set_all_kdes(self):
        """ Use all sub KDEs
        """
        self.distr_idxs = np.arange(len(self.paramlists))
        self.param_names = []
        for idx in self.distr_idxs:
            self.param_names.extend(self.paramlists[idx])


    def randomize_kdes(self):
        """ Randomize subset of KDEs
        """
        size = np.random.randint(1, self.nmax+1)
        size = min(len(self.paramlists), size)
        self.distr_idxs = np.random.choice(np.arange(len(self.paramlists)), size=size,
                                      replace=False, p=self.weights)
        self.param_names = []
        for idx in self.distr_idxs:
            self.param_names.extend(self.paramlists[idx])


    def _rand_idx(self, n):
        """
        """
        i = np.arange(n)
        idx =  np.random.choice(i, size=n, replace=False)

        return idx


    def KL(self, p, q, bin_area):
        """Get Kullback-Leibler divergence for p and q distributions (2d
        histograms)

        """
        pmin = np.amin([np.amin(p[np.where(p != 0.)]), np.amin(q[np.where(q != 0.)])])
        p[np.where(p == 0.)] = pmin / 1000
        q[np.where(q == 0.)] = pmin / 1000
        kl = bin_area * np.sum(p * (np.log(p) - np.log(q)))

        return kl


    def JS(self, chains, a, b, bins=25):
        """Get Jensen-Shannon divergence for a and b parameter indexes
(orignal vs shuffled data)
        """
        
        p, e0, e1 = np.histogram2d(chains[:, a], chains[:, b], bins=bins, density=True)
        q, _, _ = np.histogram2d(chains[:, a], chains[self._rand_idx(len(chains)), b], bins=bins, density=True)
        bin_area = (e0[1] - e0[0]) * (e1[1] - e1[0])
        m = 0.5 * (p + q)
        js = 0.5 * self.KL(p, m, bin_area) + 0.5 * self.KL(q, m, bin_area)

        return js

    def _get_JS_matrix(self, chains):
        """Get Jensen-Shannon divergence (original vs shuffled data) for each
        pair of parameter

        """

        ndim = len(chains[0, :])
        js_matrix = np.zeros((ndim, ndim))
        for i in range(ndim):
            for j in range(i):
                js_matrix[i, j] = self.JS(chains, i, j)
                js_matrix[j, i] = js_matrix[i, j]
        
        return js_matrix


    def _get_correlated_groups(self, corrmatrix, params, corr_threshold):
        """
         Get correlated sample groups from previous chains
         @ groups : groups of correlated parameter index
         @ paramlists : groups of correlated parameter names
        """
        corrmat = np.copy(corrmatrix)

        for i in range(len(corrmat[0, :])):
            corrmat[i, i] = 0.
        corrmat[abs(corrmat) > corr_threshold] = 1.
        corrmat[abs(corrmat) < corr_threshold] = 0.

        groups = []
        for i in range(len(corrmat[:, 0])):
            igroup = []
            x = np.where(corrmat[i, :] == 1.)[0]
            if len(x) > 0:
                corrmat[i, x] = 0.
                corrmat[x, i] = 0.
                igroup.extend([i])
            while len(x) > 0:
                igroup.extend(x)
                ix = []
                for idx in x:
                    iidx = np.where(corrmat[idx, :] == 1.)[0]
                    if len(iidx) > 0:
                        ix.extend(iidx)
                        corrmat[idx, iidx] = 0.
                        corrmat[iidx, idx] = 0.
                x = ix
            if len(igroup) > 0:
                groups.append(list(np.unique(igroup)))

        grouped = []
        for subgroup in groups:
            grouped.extend(subgroup)

        for i in range(len(params)):
            if i not in grouped:
                groups.append([i])

        paramlists = []
        for group in groups:
            paramlists.append(list(params[group]))

        return groups, paramlists


    def _get_distributions(self, chains):
        """ Generated KDEs for each groups of parameters
        """
        distr = []
        n = 1

        for pl in self.paramlists:

            print('Building sub KDEs', str(n)+'/'+str(len(self.paramlists)), end='\r')

            if type(pl) is not list:
                pl = [pl]

            idx = [self.params.index(p) for p in pl]

            new_distr = KDE(pl, chains[:, idx], bandwidth=self.kde_bandwidth, bw_adapt=self.bw_adapt, adapt_scale=self.adapt_scale, use_kmeans=self.use_kmeans)
            distr.append(new_distr)
        
            n += 1

        return distr


    def get_KL_term(self, data):
        """ Get KL term to compute KL difference between two different KDEs
        """
        self.set_all_kdes()

        KL_term = 0

        idx = [list(self.model_params).index(p) for p in self.params]
        for i in range(len(data)):
            KL_term -= self.logprob(data[i, idx]) / len(data)

        return KL_term
