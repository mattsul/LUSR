import numpy as np
from sklearn.preprocessing import PolynomialFeatures as pf
import scipy
from scipy.stats import truncnorm as tn
from numpy import linalg as LA
import GPy as gpy
import copy
import itertools
from scipy.spatial.distance import cdist
import tqdm
from sklearn.metrics.pairwise import haversine_distances
import networkx as nx

class AdaptiveSampler(object):
    def __init__(self, X, K, gamma, eta, tau, beta, G=None):

        # input parameters
        self.field = X
        self.N = X.shape[0]
        self.K = K
        self.gamma = gamma
        self.eta = eta
        self.tau = tau
        self.beta = beta
        
        # Path parameters
        self.paths = {}
        self.key = -999
        
        # LUSR parameters
        self.score = -999

        # measurement variables
        self.samp_inds = []    # indices of sampled locations
        self.locs = []         # actual sampled locations
        self.meas = []
        
        # regression variables
        self.mu = tau*np.ones(self.N)
        self.sigma = np.ones(self.N) / np.sqrt(gamma)
        self.a  = np.ones(self.N) * -999
        self.Kt_inv = None

        # certain sets
        self.Ut = np.arange(self.N)
        self.Ht = []
        self.Lt = []
        self.mu_all = []
        
        # cost variables
        self.dist = 0
        self.Ut_dist = None
        self.Uprev = None
        
    def update_dist(self, ind):
        if len(self.locs) == 0:
            start_loc = self.field[0, :]
            new_dist = np.linalg.norm(start_loc - self.field[ind,:], ord=2)
        else:
            new_dist = np.linalg.norm(self.locs[-1] - self.field[ind,:], ord=2)
        self.dist += new_dist
    
    def update_prediction(self, ind, measurement, store=True):
        samp_inds = self.samp_inds.copy()
        samp_inds.append(ind)
        meas = self.meas.copy()
        meas.append(measurement)
        # update covariance inverse
        if len(meas) == 1:
            Kt_inv = np.array([[1 / (self.K[ind, ind] + self.gamma)]])
        else:
            b = self.K[samp_inds[:-1], ind][:, None]
            K22 = 1 / (self.K[ind, ind] + self.gamma - b.T @ (self.Kt_inv @ b))
            K11 = self.Kt_inv + K22 * self.Kt_inv @ np.outer(b, b) @ self.Kt_inv
            K12 = -K22 * self.Kt_inv @ b
            K21 = -K22 * b.T @ self.Kt_inv
            Kt_inv = np.vstack((np.hstack((K11, K12)), 
                                np.hstack((K21, K22))))
            
        # update mean and standard deviation of each point
        mu = np.zeros(self.N)
        sigma = np.zeros(self.N)
        meas_array = np.array(meas)
        # mean update
        Kfield = self.K[samp_inds, :]
        mu = Kfield.T @ (Kt_inv @ meas_array)
        if mu.ndim == 2:
            mu = mu[:, 0]
        # variance/std update
        explained_var = np.maximum(0, np.diag(self.K) - np.diag(Kfield.T @ (Kt_inv @ Kfield)))
        sigma = np.sqrt(explained_var)

        # choose whether to update these globally
        if store:
            self.update_dist(ind)
            self.samp_inds.append(ind)
            self.locs.append(self.field[ind, :])
            self.meas.append(measurement)
            
            self.mu = mu
            self.sigma = sigma
            self.Kt_inv = Kt_inv
            upper  = np.maximum(mu + np.power(self.beta, .5) * sigma, mu - np.power(self.beta, .5) * sigma)
            lower  = np.minimum(mu + np.power(self.beta, .5) * sigma, mu - np.power(self.beta, .5) * sigma)
            self.a = np.minimum(upper - self.tau, self.tau - lower)

            # Q = mu(x) +- B sigma(x) --- confidence interval
            mu_lb = self.mu - self.eta*self.sigma/np.sqrt(self.gamma)
            mu_ub = self.mu + self.eta*self.sigma/np.sqrt(self.gamma)
            self.Ht = np.argwhere(mu_lb > self.tau)
            self.Lt = np.argwhere(mu_ub < self.tau)
            self.Ut = np.setdiff1d(np.arange(self.N), self.Ht)
            self.Ut = np.setdiff1d(self.Ut, self.Lt)
            
        return mu, sigma
        
    def ambiguity(self, n_neighbors):
        if len(self.locs) > 1:
            curr_loc = self.locs[-1]
        else:
            curr_loc = self.field[0, :]
        Ut_dists = cdist(curr_loc[None,:], self.field[self.Ut, :])
        Ut_k     = self.Ut[np.argsort(Ut_dists)][0][:n_neighbors]
        idx = np.argmax(self.a[Ut_k])
        nextLoc = self.Ut[idx]
        return nextLoc
    
    def variance(self, n_neighbors):
        if len(self.locs) > 1:
            curr_loc = self.locs[-1]
        else:
            curr_loc = self.field[0, :]
        Ut_dists = cdist(curr_loc[None,:], self.field[self.Ut, :])
        Ut_k     = self.Ut[np.argsort(Ut_dists)][0][:n_neighbors]
        idx = np.argmax(self.sigma[Ut_k])
        nextLoc = self.Ut[idx]
        return nextLoc

    def margin(self, n_neighbors):
        margin = np.abs(self.mu - self.tau)
        if len(self.locs) > 1:
            curr_loc = self.locs[-1]
        else:
            curr_loc = self.field[0, :]
        Ut_dists = cdist(curr_loc[None,:], self.field[self.Ut, :])
        Ut_k     = self.Ut[np.argsort(Ut_dists)][0][:n_neighbors]
        idx = np.argmax(margin[Ut_k])
        nextLoc = self.Ut[idx]
        return nextLoc

    def LUSR(self, n_neighbors=100):
        # only search over k nearest neighbors in Ut
        if len(self.locs) > 1:
            curr_loc = self.locs[-1]
        else:
            curr_loc = self.field[0, :]
        Ut_dists = cdist(curr_loc[None, :], self.field[self.Ut, :])
        self.Ut_dist = Ut_dists
        self.Up = self.Ut.copy()
        Ut_k = self.Ut[np.argsort(Ut_dists)][0][:n_neighbors]
        ut_scores = np.zeros(len(Ut_k))
        for ii, ind in enumerate(Ut_k):
            # output mean, std if we obtain fake measurement
            if self.mu[ind] > self.tau:
                fake_meas = self.mu[ind] - self.eta*self.sigma[ind]/np.sqrt(self.gamma)
            else:
                fake_meas = self.mu[ind] + self.eta*self.sigma[ind]/np.sqrt(self.gamma)
            mu_i, sigma_i = self.update_prediction(ind, fake_meas, store=False)
            mu_lb = mu_i - self.eta*sigma_i/np.sqrt(self.gamma)
            mu_ub = mu_i + self.eta*sigma_i/np.sqrt(self.gamma)
            Ht = np.argwhere(mu_lb > self.tau)
            Lt = np.argwhere(mu_ub < self.tau)
            Ut = np.setdiff1d(np.arange(self.N), Ht)
            Ut = np.setdiff1d(Ut, Lt)
            # reduction in size of Ut
            ut_scores[ii] = len(self.Ut) - len(Ut)
        ut_ind = np.argmax(ut_scores)
        self.score = ut_scores[ut_ind]
        ind = Ut_k[ut_ind]
        return ind