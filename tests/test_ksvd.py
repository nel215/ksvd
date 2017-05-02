# coding:utf-8
from ksvd import ApproximateKSVD
from sklearn.decomposition import DictionaryLearning
from sklearn.utils.testing import assert_array_almost_equal
import numpy as np
import scipy as sp
from scipy.linalg import norm


def test_initialize_with_small_n_features():
    N = 500
    n_components = 128
    n_features = 64
    X = np.random.randn(N, n_features)
    dico = ApproximateKSVD(n_components=n_components)
    dico.fit(X)


def test_fit():
    np.random.seed(0)
    N = 1000
    L = 64
    n_features = 128
    B = np.array(sp.sparse.random(N, L, density=0.5).todense())
    D = np.random.randn(L, n_features)
    X = np.dot(B, D)
    dico = ApproximateKSVD(n_components=L, transform_n_nonzero_coefs=L)
    dico.fit(X)
    gamma = dico.transform(X)
    assert_array_almost_equal(X, gamma.dot(dico.components_))


def test_size():
    np.random.seed(0)
    N = 50
    L = 12
    n_features = 16
    D = np.random.randn(L, n_features)
    B = np.array(sp.sparse.random(N, L, density=0.5).todense())
    X = np.dot(B, D)
    dico1 = ApproximateKSVD(n_components=L, transform_n_nonzero_coefs=L)
    dico1.fit(X)
    gamma1 = dico1.transform(X)
    e1 = norm(X - gamma1.dot(dico1.components_))

    dico2 = DictionaryLearning(n_components=L, transform_n_nonzero_coefs=L)
    dico2.fit(X)
    gamma2 = dico2.transform(X)
    e2 = norm(X - gamma2.dot(dico2.components_))

    assert dico1.components_.shape == dico2.components_.shape
    assert gamma1.shape == gamma2.shape
    assert e1 < e2
