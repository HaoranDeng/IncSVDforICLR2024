"""EvolvingMatrix class for updating the truncated singular value decomposition (SVD) of evolving matrices.
"""

import time
import math
import numpy as np
import scipy.sparse
from .svd_update import (
    kalantzis1_update,
    zhasimon_update,
)


msg_len = 60

class EvolvingMatrix(object):
    """Evolving matrix with periodically appended rows.

    This class simulates a matrix subject to the periodic addition of new rows.
    Applications with such matrices include latent semantic indexing (LSI) and recommender systems.

    Given an initial matrix, rows from a matrix to be appended are added in sequential updates.
    With each batch update, the truncated singular value decomposition (SVD)
    of the new matrix can be calculated using a variety of methods.
    The accuracy of these methods can be evaluated by four metrics.

    Parameters
    ----------
    initial_matrix : ndarray of shape (m, n)
        Initial matrix

    append_matrix : nd array of shape (s, n)
        Entire matrix to be appended row-wise to initial matrix

    n_batches : int, default=1
        Number of batches

    k_dim : int, default=50
        Rank of truncated SVD to be calculated

    Attributes
    ----------
    m_dim : int
        Number of rows in current data matrix

    n_dim : int
        Number of columns in data matrix. Does not change in row-update case.

    s_dim : int
        Number of rows in matrix to be appended

    k_dim : int
        Desired rank for truncated SVD

    A : ndarray (m + n_appended_total, n)
        Current data matrix

    Uk, sigmak, VHk : ndarrays of shape (m, k), (k,), (n, k)
        Truncated SVD of current update calculated using update method

    runtime : float
        Total time elapsed in calculating updates so far

    append_matrix : ndarray of shape (s, n)
        Entire matrix to be appended over the course of updates

    update_matrix : ndarray of shape (u, n)
        Matrix appended in last update

    phi : int
        Current update index

    n_batches : int
        Number of batches (updates)

    n_appended : int
        Number of rows appended in last update

    n_appended_total : int
        Number of rows appended in all updates

    runtime : float
        Total runtime for updates

    freq_dir : FrequentDirections
        FrequentDirections object based on Ghashami et al. (2016)

    References
    ----------
    V. Kalantzis, G. Kollias, S. Ubaru, A. N. Nikolakopoulos, L. Horesh, and K. L. Clarkson,
        “Projection techniquesto update the truncated SVD of evolving matrices with applications,”
        inProceedings of the 38th InternationalConference on Machine Learning,
        M. Meila and T. Zhang, Eds.PMLR, 7 2021, pp. 5236-5246.

    Mina Ghashami et al. “Frequent Directions: Simple and Deterministic Matrix Sketching”.
        In: SIAM Journalon Computing45.5 (Jan. 2016), pp. 1762-1792.
    """
    def __init__(self, initial_matrix, n_batches=1, k_dim=None, name="", full_matrix=None, max_rows=None, network=None):
        self.Uall = np.zeros((max_rows, k_dim), dtype=np.float64)
        self.network = network
        self.name = name
        self.mm = np.zeros((max(max_rows, 1000000),), dtype=np.int64)

        # Initial matrix
        self.initial_matrix = initial_matrix
        (self.m_dim, self.n_dim) = np.shape(self.initial_matrix)
        print(f"{'Initial matrix of shape ':<{msg_len}}{self.initial_matrix.shape}")

        # Matrix after update (initialize to initial matrix)
        
        self.A = self.initial_matrix
        print(type(self.A))

        # Set desired rank of truncated SVD
        assert k_dim < min(self.m_dim, self.n_dim), "k must be smaller than or equal to min(m,n)."
        self.k_dim = k_dim

        # Calculate true truncated SVD of current matrix
        # Get initial truncated SVD
        # U_true, sigma_true, VH_true = get_truncated_svd(self.initial_matrix, k=self.k_dim)
        U_true, sigma_true, VH_true = scipy.sparse.linalg.svds(self.initial_matrix, self.k_dim)
        self.Uk = U_true[:, :self.k_dim]
        self.Uall[:self.Uk.shape[0]] = self.Uk
        self.cur_row = self.Uk.shape[0]
        self.sigmak = sigma_true[:self.k_dim]
        self.VHk = VH_true[: self.k_dim, :]
        self.Vk = self.VHk.T

        print(f"{'Initial Uk matrix of evolving matrix set to shape of ':<{msg_len}}{np.shape(self.Uk)}.")
        print(f"{'Initial sigmak array of evolving matrix set to shape of ':<{msg_len}}{np.shape(self.sigmak)}.")
        print(f"{'Initial VHk matrix of evolving matrix set to shape of ':<{msg_len}}{np.shape(self.VHk)}.")

        if max_rows is not None:
            self.max_rows = max_rows

        self.cond_list = []

        if name in ["isvd1", "isvd2", "isvd3"]:
            self.Ku = np.eye(k_dim, dtype=np.float64)
            self.Kv = np.eye(k_dim, dtype=np.float64)
        else:
            self.Ku, self.Kv = None, None

        # Initialize matrix to be appended
        self.n_batches = n_batches
        
        self.append_matrix = np.array([])
        self.s_dim = 0 

        # Initialize submatrix to be appended at each update
        self.update_matrix = np.array([])

        # Initialize parameters to keep track of updates
        self.phi = 0
        self.n_appended = 0
        self.n_appended_total = 0
        self.step = 0

        # Initialize total runtime
        self.runtime = 0.0
        if full_matrix is not None:
            self.full_matrix = full_matrix
            # print(self.full_matrix)

        self.tot1 = 0
        self.tot2 = 0

        self.svd_time = 0
        self.num_restart = 0


    def set_append_matrix(self, append_matrix):
        """Set entire matrix to appended over the course of updates and calculates SVD for final matrix.

        Parameters
        ----------
        append_matrix : ndarray of shape (s, n)
            Matrix to be appended
        """
        # if network is not None:



        self.append_matrix = append_matrix  # ensure data is in dense format
        self.s_dim, n_dim = self.append_matrix.shape
        self.res_s_dim = self.s_dim
        assert (
            n_dim == self.n_dim
        ), "Number of columns must be the same for initial matrix and matrix to be appended."
        print(f"{'Appending matrix set to shape of ':<{msg_len}}{np.shape(self.append_matrix)}.")


    def set_network_append_matrix(self, A_csr, A_csc, init):
        """Set entire matrix to appended over the course of updates and calculates SVD for final matrix.

        Parameters
        ----------
        append_matrix : ndarray of shape (s, n)
            Matrix to be appended
        """
        # if network is not None:

        # print(self.n_batches)
        cur = init
        self.append_rows = []
        self.append_cols = []
        for i in range(self.n_batches):
            cur_num_append = int((self.max_rows - cur) / (self.n_batches-i))
            cur_next = cur + cur_num_append
            self.append_rows.append( A_csr[cur:cur_next, :cur] )
            self.append_cols.append( A_csr[cur:cur_next, :cur_next] )
            cur = cur_next

    def evolve_network(self):
        """Evolve matrix by one update."""
        # Check if number of appended rows exceeds number of remaining rows in appendix matrix'
        # print(f"n_appended_total: {self.n_appended_total}")
        # print(f"self.res_s_dim: {self.res_s_dim}")
        # print(self.step)
        
        self.E1 = self.append_rows[self.phi]
        self.E2 = self.append_cols[self.phi]
        
        # Update counters for update
        self.phi += 1
        self.n_appended_total += self.n_appended


    def evolve(self):
        """Evolve matrix by one update."""
        # Check if number of appended rows exceeds number of remaining rows in appendix matrix'
        # print(f"n_appended_total: {self.n_appended_total}")
        # print(f"self.res_s_dim: {self.res_s_dim}")
        # print(self.step)
        
        self.step = math.floor(self.res_s_dim / self.n_batches)
        self.res_s_dim -= self.step
        self.n_batches -= 1

        self.n_appended = (self.step)

        # Append to current data matrix

        self.update_matrix = self.append_matrix[
            self.n_appended_total : self.n_appended_total + self.n_appended, :
        ]

        # self.A = scipy.sparse.vstack([self.A, self.update_matrix])
        # self.A = np.append(self.A, self.update_matrix, axis=0)
        self.m_dim = self.A.shape[0]

        # Update counters for update
        self.phi += 1
        self.n_appended_total += self.n_appended

    def update_svd_isvd1(self):
        start = time.perf_counter()

        """ write ISVD code here """
        s = self.update_matrix.shape[0]
        k = self.Uk.shape[1]
        uid = np.unique(self.update_matrix.indices)
        B = np.zeros((s, len(uid)), dtype=np.float64)
        E = self.update_matrix
        
        for i in range(len(uid)):
            self.mm[ uid[i] ] = i
        cur = 0
        for i in range(s):
            for j in range( E.indptr[i+1] - E.indptr[i] ):
                B[i, self.mm[ E.indices[cur] ]] = E.data[cur]
                cur += 1
        B = B.T

        C = (self.Kv.T @ self.Vk[uid].T @ B)
        C0 = C.copy()

        R = np.zeros((s, s), dtype=np.float64)

        for i in range(s):
            tmp = np.dot(B[:, i], B[:, i]) - np.dot(C[:, i], C[:, i])
            alpha = 0
            if abs(tmp) > 1e-5:
                alpha = np.sqrt(tmp)
                B[:, i] /= alpha
                C[:, i] /= alpha
            R[i, i] = alpha

            for j in range(i+1, s):
                beta =  np.dot(B[:, i], B[:, j]) - np.dot(C[:, i], C[:, j])
                B[:, j] -= beta * B[:, i]
                C[:, j] -= beta * C[:, i]
                R[i, j] = beta

        time_1 = time.perf_counter()

        Mu = np.concatenate((np.diag(self.sigmak), np.zeros((k, s), dtype=np.float64)), axis=1)
        Md = np.concatenate((C0.T, R.T), axis=1)
        M = np.concatenate((Mu, Md), axis=0)

        Fk, Tk, Gk = np.linalg.svd(M, full_matrices=False)
        Gk = Gk.T

        # Truncate if necessary
        if k < len(Tk):
            Fk = Fk[:, :k]
            Tk = Tk[:k]
            Gk = Gk[:, :k]
        
        time_2 = time.perf_counter()
        self.svd_time += time_2 - time_1

        self.Ku = self.Ku @ Fk[:k]
        self.Kv = self.Kv @ (Gk[:k] - C @ Gk[k:])



        time_3 = time.perf_counter()

        delta_Uk = Fk[k:] @ np.linalg.inv(self.Ku)
        self.Uk = np.append(self.Uk, delta_Uk, axis=0)
        
        self.sigmak = Tk

        delta_Vk = B @ Gk[k:] @ np.linalg.inv(self.Kv)
        self.Vk[uid] += delta_Vk
        
        self.runtime += time.perf_counter() - start

    def update_svd_isvd2(self):
        """Return truncated SVD of updated matrix using the ISVD method."""

        start = time.perf_counter()
        k = self.Uk.shape[1]
        E = self.update_matrix
        s = E.shape[0]
        l = min(10, s)
        uid = np.unique(self.update_matrix.indices)
        B = np.zeros((s, len(uid)), dtype=np.float64)
        for i in range(len(uid)):
            self.mm[ uid[i] ] = i
        cur = 0
        for i in range(s):
            for j in range( E.indptr[i+1] - E.indptr[i] ):
                B[i, self.mm[ E.indices[cur] ]] = E.data[cur]
                cur += 1
        B = B.T

        C = (self.Kv.T @ (self.Vk[uid].T) @ B)
 
        Bp = np.zeros((B.shape[0], l+1), dtype=np.float64)
        Cp = np.zeros((C.shape[0], l+1), dtype=np.float64)

        P = np.zeros((s, l+2), dtype=np.float64)
        P[:, 1] = np.random.randn(s)
        P[:, 1] = P[:, 1] / np.linalg.norm(P[:, 1])
        beta = np.zeros((l+1, ), dtype=np.float64)
        alpha = np.zeros((l+1, ), dtype=np.float64)

        for i in range(1, l+1):
            Bp[:, i] = B @ P[:, i] - beta[i-1] * Bp[:, i-1]
            Cp[:, i] = C @ P[:, i] - beta[i-1] * Cp[:, i-1]
            tmp = np.dot(Bp[:, i], Bp[:, i]) - np.dot(Cp[:, i], Cp[:, i])
            if abs(tmp) < 1e-9:
                alpha[i] = 0
            else:
                alpha[i] = np.sqrt( tmp )
                Bp[:, i] /= alpha[i]
                Cp[:, i] /= alpha[i]

           
            P[:, i+1] = B.T @ Bp[:, i] - C.T @ Cp[:, i] - alpha[i] * P[:, i]
            for j in range(1, i+1):
                P[:, i+1] -= np.dot(P[:, i+1], P[:, j]) * P[:, j]
            beta[i] = np.linalg.norm(P[:, i+1])
            if abs(beta[i]) < 1e-9:
                l = i
                break
            P[:, i+1] /= beta[i]
        
        L = np.zeros((l, l+1), dtype=np.float64)
        for i in range(l):
            L[i, i] = alpha[i]
            L[i, i+1] = beta[i]
        
        Bp = Bp[:, 1:]
        Cp = Cp[:, 1:]
        P = P[:, 1:]

        Bp = Bp[:, :l]
        Cp = Cp[:, :l]
        P = P[:, :l+1]

        Mu = np.concatenate((np.diag(self.sigmak), np.zeros((k, l), dtype=np.float64)), axis=1)
        Md = np.concatenate((C.T, P @ L.T), axis=1)
        M = np.concatenate((Mu, Md), axis=0)

        # Calculate SVD of M
        time_1 = time.perf_counter()
        Fk, Tk, GHk = np.linalg.svd(M, full_matrices=False)
        # Truncate if necessary
        if k < len(Tk):
            Fk = Fk[:, :k]
            Tk = Tk[:k]
            GHk = GHk[:k]
        Gk = GHk.T
        time_2 = time.perf_counter()
        self.svd_time += time_2 - time_1

        # Calculate updated values for Uk, Sk, Vk`
        
        self.Ku = self.Ku @ Fk[:k]
        self.Kv = self.Kv @ (Gk[:k] - Cp @ Gk[k:])

        delta_Uk = Fk[k:] @ np.linalg.inv(self.Ku)
        self.Uk = np.append(self.Uk, delta_Uk, axis=0)
        
        self.sigmak = Tk

        delta_Vk = Bp @ Gk[k:] @ np.linalg.inv(self.Kv)
        self.Vk[uid] += delta_Vk

        self.runtime += time.perf_counter() - start
        return self.Uk, self.sigmak, self.Vk


    def update_svd_isvd3(self):
        """Return truncated SVD of updated matrix using the random method."""
        start = time.perf_counter()
        E = self.update_matrix

        s = E.shape[0]
        k = self.Uk.shape[1]
        l = min(10, s)
        if l == 0:
            l = 1
        num_iter = 3

        uid = np.unique(self.update_matrix.indices)
        B = np.zeros((s, len(uid)), dtype=np.float64)
        for i in range(len(uid)):
            self.mm[ uid[i] ] = i
        cur = 0
        for i in range(s):
            for j in range( E.indptr[i+1] - E.indptr[i] ):
                B[i, self.mm[ E.indices[cur] ]] = E.data[cur]
                cur += 1
        B = B.T
        C = (self.Kv.T @ (self.Vk[uid].T) @ B)


        Q = np.zeros((s, l), dtype=np.float64)
        for i in range(l):
            Q[:, i] = np.random.randn(s)
            Q[:, i] = Q[:, i] / np.linalg.norm(Q[:, i])

        for _ in range(num_iter):
            Q, R = np.linalg.qr(Q)
            BP, CP = B @ Q, C @ Q

            R = np.zeros((l, l), dtype=np.float64)
            for i in range(l):
                for j in range(i):
                    beta = np.dot(BP[:, i], BP[:, j]) - np.dot(CP[:, i], CP[:, j])
                    R[j, i] = beta
                    BP[:, i] -= beta * BP[:, j]
                    CP[:, i] -= beta * CP[:, j]
                tmp = np.dot(BP[:, i], BP[:, i]) - np.dot(CP[:, i], CP[:, i])
                if abs(tmp) < 1e-9:
                    R[i, i] = 0
                    continue
                alpha = np.sqrt( tmp )
                R[i, i] = alpha
                BP[:, i] /= alpha
                CP[:, i] /= alpha

            # P, R = np.linalg.qr(P)
            if _ != num_iter-1:
                Q = B.T @ BP - C.T @ CP
        


        Mu = np.concatenate((np.diag(self.sigmak), np.zeros((k, l), dtype=np.float64)), axis=1)
        Md = np.concatenate((C.T, P), axis=1)

        M = np.concatenate((Mu, Md), axis=0)
        # Calculate SVD of M


        time_1 = time.perf_counter()
        
        Fk, Tk, GHk = np.linalg.svd(M, full_matrices=False)

        # Truncate if necessary
        if k < len(Tk):
            Fk = Fk[:, :k]
            Tk = Tk[:k]
            GHk = GHk[:k]
        Gk = GHk.T


        time_2 = time.perf_counter()
        self.svd_time += time_2 - time_1
        # Calculate updated values for Uk, Sk, Vk


        self.Ku = self.Ku @ Fk[:k]
        self.Kv = self.Kv @ (Gk[:k] - CP @ Gk[k:])

        delta_Uk = Fk[k:] @ np.linalg.inv(self.Ku)
        self.Uk = np.append(self.Uk, delta_Uk, axis=0)
        
        self.sigmak = Tk

        delta_Vk = BP @ Gk[k:] @ np.linalg.inv(self.Kv)
        self.Vk[uid] += delta_Vk

        self.runtime += time.perf_counter() - start
        return self.Uk, self.sigmak, self.Vk



    def update_svd_zhasimon(self):
        """Return truncated SVD of updated matrix using the Zha-Simon projection method."""
        
        '''=====Step 1====='''
        start_time_step_1 = time.perf_counter()
        E = self.update_matrix
        V = self.Vk

        s = E.shape[0]
        k = self.Uk.shape[1]

        Q, R = np.linalg.qr(E.T - V @ (V.T @ E.T))
        Z = scipy.linalg.block_diag(self.Uk, np.eye(s))
        W = np.concatenate((V, Q), axis=1)
        self.runtime_step1 += time.perf_counter() - start_time_step_1


        '''=====Step 2====='''
        start_time_step_2 = time.perf_counter()
        Mu = np.concatenate((np.diag(self.sigmak), np.zeros((k, s), dtype=np.float64)), axis=1)
        Md = np.concatenate((E@V, R.T), axis=1)
        M = np.concatenate((Mu, Md), axis=0)
        # print(M)

        # Calculate SVD of M
        # Fk, Tk, GHk = scipy.sparse.linalg.svds(M, k)
        Fk, Tk, GHk = np.linalg.svd(M, full_matrices=False)
        # print(len(Tk))

        # Truncate if necessary
        if k < len(Tk):
            Fk = Fk[:, :k]
            Tk = Tk[:k]
            GHk = GHk[:k]
        self.runtime_step2 += time.perf_counter() - start_time_step_2

        '''=====Step 3====='''
        start_time_step_3 = time.perf_counter()
        # Calculate updated values for Uk, Sk, Vk
        self.sigmak = Tk
        self.Uk = Z @ Fk
        self.Vk = W @ (GHk.T)
        self.runtime_step3 += time.perf_counter() - start_time_step_3


    def update_vecharynski(self):
        """Return truncated SVD of updated matrix using the Zha-Simon projection method."""

        '''=====Step 1====='''
        start_time_step_1 = time.perf_counter()
        E = self.update_matrix
        V = self.Vk
        s = E.shape[0]
        k = self.Uk.shape[1]
        n = V.shape[0]
        l = min(10, s)

        Q = np.zeros((n, l+1), dtype=np.float64)
        # X = E.T - V @ ((V.T) @ (E.T))

        P = np.zeros((s, l+2), dtype=np.float64)
        P[:, 1] = np.random.randn(s)
        P[:, 1] = P[:, 1] / np.linalg.norm(P[:, 1])
        beta = np.zeros((l+1, ), dtype=np.float64)
        alpha = np.zeros((l+1, ), dtype=np.float64)
        for i in range(1, l+1):
            time_1 = time.perf_counter()
            # Q[:, i] = X @ P[:, i] - beta[i-1] * Q[:, i-1]
            Q[:, i] = E.T @ P[:, i] - V @ ((V.T @ E.T) @ P[:, i]) - beta[i-1] * Q[:, i-1]
            self.runtime_tmp1 += time.perf_counter() - time_1

            alpha[i] = np.linalg.norm(Q[:, i])
            if alpha[i] == 0:
                Q[:, i] = 0
            else:
                Q[:, i] /= alpha[i]
            
            time_2 = time.perf_counter()
            # P[:, i+1] = X.T @ Q[:, i] - alpha[i] * P[:, i]
            P[:, i+1] = E @ Q[:, i] - E @ (V @ (V.T @ Q[:, i])) - alpha[i] * P[:, i]
            self.runtime_tmp2 += time.perf_counter() - time_2
            for j in range(1, i+1):
                P[:, i+1] -= np.dot(P[:, i+1], P[:, j]) * P[:, j]
            
            beta[i] = np.linalg.norm(P[:, i+1])
            if beta[i] == 0:
                P[:, i+1] = 0
                continue
            P[:, i+1] /= beta[i]

        B = np.zeros((l, l+1), dtype=np.float64)
        for i in range(l):
            B[i, i] = alpha[i]
            B[i, i+1] = beta[i]
        
        P = P[:, 1:]
        Q = Q[:, 1:]

        Z = scipy.linalg.block_diag(self.Uk, np.eye(s))
        W = np.concatenate((self.Vk, Q), axis=-1)
        self.runtime_step1 += time.perf_counter() - start_time_step_1

        '''=====Step 2====='''
        start_time_step_2 = time.perf_counter()
        Mu = np.concatenate((np.diag(self.sigmak), np.zeros((k, l), dtype=np.float64)), axis=1)
        Md = np.concatenate((E@V, P @ B.T), axis=1)
        M = np.concatenate((Mu, Md), axis=0)

        # Calculate SVD of M
        Fk, Tk, GHk = np.linalg.svd(M, full_matrices=False)
        
        # Truncate if necessary
        if k < len(Tk):
            Fk = Fk[:, :k]
            Tk = Tk[:k]
            GHk = GHk[:k]
        self.runtime_step2 += time.perf_counter() - start_time_step_2

        '''=====Step 3====='''
        start_time_step_3 = time.perf_counter()
        # Calculate updated values for Uk, Sk, Vk
        self.Uk = Z @ Fk
        self.sigmak = Tk
        self.Vk = (W @ (GHk.T))  
        self.runtime_step3 += time.perf_counter() - start_time_step_3      


    def update_random(self):
        """Return truncated SVD of updated matrix using the random method."""

        '''=====Step 1====='''
        start_time_step_1 = time.perf_counter()
        E = self.update_matrix
        V = self.Vk

        s = E.shape[0]
        k = self.Uk.shape[1]
        l = min(10, s)
        num_iter = 3
        # X = E.T - V @ ((V.T) @ (E.T))

        Q = np.zeros((s, l), dtype=np.float64)
        for i in range(l):
            Q[:, i] = np.random.randn(s)
            Q[:, i] = Q[:, i] / np.linalg.norm(Q[:, i])

        for i in range(num_iter):
            Q, R = np.linalg.qr(Q)
            # P = X @ Q
            P = E.T @ Q - V @ (((V.T) @ (E.T)) @ Q)
            P, R = np.linalg.qr(P)
            if i != num_iter-1:
                # Q = X.T @ P
                Q = E @ P - E @ (V @ (V.T @ P))

        Z = scipy.linalg.block_diag(self.Uk, np.eye(s))
        W = np.concatenate((self.Vk, P), axis=-1)
        self.runtime_step1 += time.perf_counter() - start_time_step_1
        
        '''=====Step 2====='''
        start_time_step_2 = time.perf_counter()
        Mu = np.concatenate((np.diag(self.sigmak), np.zeros((k, l), dtype=np.float64)), axis=1)
        Md = np.concatenate((E@V, Q), axis=1)
        M = np.concatenate((Mu, Md), axis=0)
        # Calculate SVD of M
        Fk, Tk, GHk = np.linalg.svd(M, full_matrices=False)
        
        # Truncate if necessary
        if k < len(Tk):
            Fk = Fk[:, :k]
            Tk = Tk[:k]
            GHk = GHk[:k]
        self.runtime_step2 += time.perf_counter() - start_time_step_2

        '''=====Step 3====='''
        start_time_step_3 = time.perf_counter()
        # Calculate updated values for Uk, Sk, Vk
        self.Uk = Z @ Fk
        self.sigmak = Tk
        self.Vk = (W @ (GHk.T))
        self.runtime_step3 += time.perf_counter() - start_time_step_3     
