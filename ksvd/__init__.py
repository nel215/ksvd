# coding:utf-8
import numpy as np
import scipy as sp
from sklearn.linear_model import orthogonal_mp_gram


class ApproximateKSVD(object):
    def __init__(self, n_components):
        self.components_ = None
        self.max_iter = 2
        self.tol = 1e-6
        self.n_components = n_components

    def _update_dict(self, X, D, gamma):
        for j in range(self.n_components):
            I = gamma[:, j] > 0
            if np.sum(I) == 0:
                continue

            D[j, :] = 0
            g = gamma[I, j].T
            r = X[I, :] - gamma[I, :].dot(D)
            d = r.T.dot(g)
            d /= np.linalg.norm(d)
            g = r.dot(d)
            D[j, :] = d
            gamma[I, j] = g.T
        return D, gamma

    def _initialize(self, X):
        u, s, vt = sp.sparse.linalg.svds(X, k=self.n_components)
        return np.dot(np.diag(s), vt)


    def fit(self, X):
        """
        Parameters
        ----------
        X: shape = [n_samples, n_features]
        """
        D = self._initialize(X)
        D /= np.linalg.norm(D, axis=1)[:, np.newaxis]
        for i in range(self.max_iter):
            gram = D.dot(D.T)
            Xy = D.dot(X.T)
            gamma = orthogonal_mp_gram(gram, Xy).T
            e = np.linalg.norm(X - gamma.dot(D))
            if e < self.tol:
                break
            D, gamma = self._update_dict(X, D, gamma)

        self.components_ = D
        return self

    def transform(self, X):
        gram = self.components_.dot(self.components_.T)
        Xy = self.components_.dot(X.T)
        gamma = orthogonal_mp_gram(gram, Xy).T
        return gamma
