"""Algorithms for updating the truncated singular value decomposition (SVD) of evolving matrices.
"""
import time
import numpy as np
import scipy
import scipy.sparse
from scipy.linalg import block_diag


def zhasimon_update(A, Uk, Sk, Vk, E):
    """ write ISVD code here """
    # print(f"ZS")
    s = E.shape[0]
    k = Uk.shape[1]
    V = Vk

    tmp = E.T - V @ (V.T @ E.T)
    Q, R = np.linalg.qr(tmp)
    Z = block_diag(Uk, np.eye(s))
    W = np.concatenate((V, Q), axis=1)

    Mu = np.concatenate((np.diag(Sk), np.zeros((k, s), dtype=np.float64)), axis=1)
    Md = np.concatenate((E@V, R.T), axis=1)
    M = np.concatenate((Mu, Md), axis=0)

    # Calculate SVD of M
    time_0 = time.perf_counter()
    Fk, Tk, GHk = np.linalg.svd(M, full_matrices=False)
    time_1 = time.perf_counter()

    # Truncate if necessary
    if k < len(Tk):
        Fk = Fk[:, :k]
        Tk = Tk[:k]
        GHk = GHk[:k]

    # Calculate updated values for Uk, Sk, Vk
    Uk_new = Z @ Fk
    Vk_new = W @ (GHk.T)
    return Uk_new, Tk, Vk_new


def kalantzis1_update(A, Uk, Sk, VHk, E):
    """Calculate truncated SVD update using kalantzis1 projection algorithm.

    Parameters
    ----------
    A : array, shape (m, n)
        Updated matrix

    Uk : array, shape (m, k)
        Left singular vectors from previous update

    Sk : array, shape (k,)
        Singular values from previous update

    VHk : array, shape (k, n)
        Right singular vectors from previous update

    E : array, shape (s, n)
        Appended submatrix

    Returns
    -------
    Uk_new : array, shape (m, k)
        Updated left singular vectors

    Sk_new : array, shape (k,)
        Updated singular values

    VHk_new : array, shape (k, n)
        Update right singular vectors

    References
    ----------
    V. Kalantzis, G. Kollias, S. Ubaru, A. N. Nikolakopoulos, L. Horesh, and K. L. Clarkson,
        “Projection techniquesto update the truncated SVD of evolving matrices with applications,”
        inProceedings of the 38th InternationalConference on Machine Learning,
        M. Meila and T. Zhang, Eds. PMLR, 7 2021, pp. 5236-5246.

    H. Zha and H. D. Simon, “Timely communication on updating problems in latent semantic indexing,
        ”Society for Industrial and Applied Mathematics, vol. 21, no. 2, pp. 782-791, 1999.
    """
    # Construct Z and ZH*A matrices
    s = E.shape[0]
    k = Uk.shape[1]
    Z = block_diag(Uk, np.eye(s))


    ZHA = Z.T @ A

    # Calculate SVD of ZH*A
    Fk, Tk, _ = np.linalg.svd(ZHA, full_matrices=False)
    # Truncate if necessary
    if k < len(Tk):
        Fk = Fk[:, :k]
        Tk = Tk[:k]


    # Calculate updated values for Uk, Sk, Vk
    Uk_new = Z.dot(Fk)
    Vk_new = A.T.dot(Uk_new.dot(np.diag(1 / Tk)))
    
    return Uk_new, Tk, Vk_new.T
