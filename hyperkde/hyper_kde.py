"""
Hyper-KDE module

Author: Mikel Falxa
"""

import numpy as np
import scipy.stats as scistats
from .kde import KDE
import matplotlib.pyplot as plt
import time

class HyperKDE:
    """ A HyperKDE class  
    """
    def __init__(self, model_params, chains, chains_params, js_threshold, adapt_scale=10, use_kmeans=False, global_bw=False, n_kde_max=None, groups_idx=None, paramlists=None):
        """
        @ model_params : list of currently sampled model parameters
        @ chains : previously sampled chains to which KDE is applied
        @ chains_params : list of parameters of previously sampled chains to which KDE is applied
        @ js_threshold : Jensen-Shannon threshold value for which KDE parameters are considered correlated
        @ adapt_scale : bandwidth adaptation scale
        @ use_kmeans : use clustering algorithm for bw adaptation
        @ global_bw : use global bandwidth instead of local bandwidth
        @ n_kde_max : max random number of active sub-KDEs
        """
        self.model_params = np.array(model_params)
        self.use_kmeans = use_kmeans
        self.global_bw = global_bw
        self.adapt_scale = adapt_scale
        self.par_idx = [i for i in range(len(chains_params)) if chains_params[i] in list(model_params)]
        self.params = list(chains_params[self.par_idx])
        self.js_threshold = js_threshold
        if groups_idx is None and paramlists is None:
            self.groups_idx, self.paramlists = self._get_correlated_groups(self._get_JS_matrix(chains[:, self.par_idx]), chains_params[self.par_idx], js_threshold)
        else:
            self.groups_idx = groups_idx
            self.paramlists = paramlists
        for pl in self.paramlists:
            print(pl)
        print(len(self.groups_idx), 'groups of parameters found')
        self.kdes = self._get_distributions(chains[:, self.par_idx])
        self.nmax = len(self.kdes) if n_kde_max is None else n_kde_max
        self.weights = np.array([kde.ndim for kde in self.kdes]) / np.sum(np.array([kde.ndim for kde in self.kdes]))
        self.param_names, self.distr_idxs = self.set_all_kdes()

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

    
    def logprobs(self, x):
        """ Get log-probabilites for samples x
        """
        logp = np.zeros(len(x))
        n = 0
        for idx in self.distr_idxs:
            kde_ndim = self.kdes[idx].ndim
            logp += self.kdes[idx].logprobs(x[:, n:n+kde_ndim])
            n += kde_ndim


    def draw_from_random_hyp_kde(self, x, **kwargs):
        """Function to use as proposal if PTMCMC is used, draws samples from
        a random subset of KDEs
        """

        q = x.copy()
        lqxy = 0

        self.param_names, self.distr_idxs = self.randomize_kdes()

        oldsample = [x[list(self.model_params).index(p)]
                    for p in self.param_names]
        newsample = self.draw()

        for p,n in zip(self.param_names, newsample):
            q[list(self.model_params).index(p)] = n

        lqxy += (self.logprob(oldsample) -
                self.logprob(newsample))

        return q, float(lqxy)


    def draw_from_hyp_kde(self, x, **kwargs):
        """Function to use as proposal if PTMCMC is used, draws samples from
        full set of KDEs

        """
        q = x.copy()
        lqxy = 0

        self.param_names, self.distr_idxs = self.set_all_kdes()

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
        distr_idxs = np.arange(len(self.paramlists))
        param_names = []
        for idx in distr_idxs:
            param_names.extend(self.paramlists[idx])

        return param_names, distr_idxs


    def randomize_kdes(self):
        """ Randomize subset of KDEs
        """
        # size = np.random.randint(1, max(2, self.nmax))
        size = self.nmax
        if size>len(self.paramlists):
            size = len(self.paramlists)
        distr_idxs = np.random.choice(np.arange(len(self.paramlists)), size=size,
                                      replace=False, p=self.weights)
        param_names = []
        for idx in distr_idxs:
            param_names.extend(self.paramlists[idx])

        return param_names, distr_idxs


    def _rand_idx(self, n):
        """ Get n shuffled indexes
        """
        i = np.arange(n)
        idx =  np.random.choice(i, size=n, replace=False)

        return idx


    def KL(self, p, q, bin_area):
        """Get Kullback-Leibler divergence for p and q distributions (2d
        histograms)

        """
        pmin = np.amin([np.amin(p[np.where(p != 0.)]), np.amin(q[np.where(q != 0.)])])
        p[np.where(p == 0.)] = pmin
        q[np.where(q == 0.)] = pmin
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


    def _get_mean_JS_matrix(self, chains, it):
        """Get Jensen-Shannon divergence (original vs shuffled data) for each
        pair of parameter

        """
        ndim = len(chains[0, :])
        mean_js_matrix = np.zeros((ndim, ndim))
        for _ in range(it):
            mean_js_matrix += self._get_JS_matrix(chains)/it
        return mean_js_matrix


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

            new_distr = KDE(pl, chains[:, idx], adapt_scale=self.adapt_scale, use_kmeans=self.use_kmeans, global_bw=self.global_bw)
            distr.append(new_distr)
        
            n += 1

        return distr

    
    def get_original_dataset(self):
        """ Get recombined dataset from all sub-KDEs
        """

        x0 = self.kdes[0].chains
        if self.kdes[0].ndim == 1:
            x0 = np.reshape(x0, (-1, 1))
        for kde in self.kdes[1:]:
            if kde.ndim == 1:
                x_kde = kde.chains
                x0 = np.hstack((x0, np.reshape(x_kde (-1, 1))))
            else:
                x_kde = kde.chains
                x0 = np.hstack((x0, x_kde))
        return x0

    
    def redraw_dataset(self):
        """ Get recombined and shuffled dataset from all sub-KDEs
        """

        x0 = self.kdes[0].chains + np.random.normal(scale=self.kdes[0].bw)
        if self.kdes[0].ndim == 1:
            x0 = np.reshape(x0, (-1, 1))
        for kde in self.kdes[1:]:
            if kde.ndim == 1:
                shuffle_idx = np.random.choice(np.arange(kde.n), size=kde.n, replace=False)
                x_kde = kde.chains + np.random.normal(scale=kde.bw)
                x_kde = x_kde[shuffle_idx]
                x0 = np.hstack((x0, np.reshape(x_kde, (-1, 1))))
            else:
                shuffle_idx = np.random.choice(np.arange(kde.n), size=kde.n, replace=False)
                x_kde = kde.chains + np.random.normal(scale=kde.bw)
                x_kde = x_kde[shuffle_idx]
                x0 = np.hstack((x0, x_kde))
        return x0


    def get_KL(self, kde):
        """ Compute KL divergence between this KDE and another
        """

        self.param_names, self.distr_idxs = self.set_all_kdes()
        kde.param_names, kde.distr_idxs = kde.set_all_kdes()

        x0 = self.redraw_dataset()
        ns = len(x0)
        param_idxs = [list(kde.param_names).index(p) for p in self.param_names]
        pmins_self = [ke._find_pmin() for ke in self.kdes]
        pmins = [ke._find_pmin() for ke in kde.kdes]
        pmin = np.log(np.amin([np.amin(pmins_self), np.amin(pmins)]))
        p0_log_p0 = np.zeros(ns)
        p0_log_p1 = np.zeros(ns)
        if len(self.param_names) > 1:
            for i in range(ns):
                p0_log_p0[i] = self.logprob(x0[i])
                p0_log_p1[i] = kde.logprob(x0[i, param_idxs])
        else:
            for i in range(ns):
                p0_log_p0[i] = self.logprob(x0[i])
                p0_log_p1[i] = kde.logprob(x0[i])
        p0_log_p0[p0_log_p0 < pmin] = pmin
        p0_log_p1[p0_log_p1 < pmin] = pmin
        kl = np.sum(p0_log_p0 - p0_log_p1)/ns
        return kl


    def get_KL_term(self, data):
        """ Get KL term to compute KL difference between two different KDEs
        """
        self.set_all_kdes()

        KL_term = 0

        idx = [list(self.model_params).index(p) for p in self.params]
        for i in range(len(data)):
            KL_term -= self.logprob(data[i, idx]) / len(data)

        return KL_term
