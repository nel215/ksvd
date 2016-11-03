# coding:utf-8
import numpy as np
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

    def fit(self, X):
        D = np.random.randn(self.n_components, X.shape[1])
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
