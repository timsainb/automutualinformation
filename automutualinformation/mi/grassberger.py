import numpy as np
import scipy
import scipy.special
from sklearn.metrics.cluster import contingency_matrix
from scipy import sparse as sp


def est_mutual_info_p(a, b):
    # contingency matrix of a * b
    contingency = contingency_matrix(a, b, sparse=True)
    nzx, nzy, Nall = sp.find(contingency)

    # entropy of a
    Na = np.ravel(contingency.sum(axis=0))  # number of A
    S_a, var_a = entropyp(Na)  # entropy with P(A) as input

    # entropy of b
    Nb = np.ravel(contingency.sum(axis=1))
    S_b, var_b = entropyp(Nb)

    # joint
    S_ab, var_ab = entropyp(Nall)

    # mutual information
    MI = S_a + S_b - S_ab

    # uncertainty and variance of MI
    MI_var = var_a + var_b + var_ab

    uncertainty = np.sqrt((MI_var) / len(a))

    return MI, uncertainty


######## Mutual Information Faster ############
def entropyp(Nall):
    N = np.sum(Nall)
    pAll = np.array([float(Ni) * scipy.special.psi(float(Ni)) for Ni in Nall])
    S_hat = np.log2(N) - 1.0 / N * np.sum(pAll)
    var = np.var(scipy.special.psi(np.array(Nall, dtype="float32")))

    return S_hat, var


def mutual_info_p(a, b):
    """Fast mutual information calculation based upon sklearn,
    but with estimation of uncertainty from Lin & Tegmark 2016
    """
    # contingency matrix of a * b
    contingency = contingency_matrix(a, b, sparse=True)
    nzx, nzy, Nall = sp.find(contingency)
    N = len(a)
    # entropy of a
    Na = np.ravel(contingency.sum(axis=0))
    S_a, var_a = entropyp(Na / np.sum(Na))

    # entropy of b
    Nb = np.ravel(contingency.sum(axis=1))
    S_b, var_b = entropyp(Nb / np.sum(Nb))

    # joint entropy
    S_ab, var_ab = entropyp(Nall / N)

    # mutual information
    MI = S_a + S_b - S_ab

    # uncertainty and variance of MI
    MI_var = var_a + var_b + var_ab

    uncertainty = np.sqrt((MI_var) / len(a))

    return MI, uncertainty
