# coding:utf-8
from ksvd import ApproximateKSVD
from sklearn.decomposition import DictionaryLearning
from sklearn.utils.testing import assert_array_almost_equal
import numpy as np
from scipy.linalg import norm


def test_fit():
    np.random.seed(0)
    N = 1000
    L = 256
    X = np.random.randn(N, 20) + np.random.rand(N, 20)
    dico = ApproximateKSVD(n_components=L)
    dico.fit(X)
    gamma = dico.transform(X)
    assert_array_almost_equal(X, gamma.dot(dico.components_))


def test_size():
    np.random.seed(0)
    N = 100
    L = 128
    X = np.random.randn(N, 10) + np.random.rand(N, 10)
    dico1 = ApproximateKSVD(n_components=L)
    dico1.fit(X)
    gamma1 = dico1.transform(X)
    e1 = norm(X - gamma1.dot(dico1.components_))

    dico2 = DictionaryLearning(n_components=L)
    dico2.fit(X)
    gamma2 = dico2.transform(X)
    e2 = norm(X - gamma2.dot(dico2.components_))

    assert dico1.components_.shape == dico2.components_.shape
    assert gamma1.shape == gamma2.shape
    assert e1 < e2
