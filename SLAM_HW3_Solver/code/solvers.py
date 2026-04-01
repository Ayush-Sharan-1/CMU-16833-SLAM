'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import inv, splu, spsolve, spsolve_triangular
from sparseqr import rz, permutation_vector_to_matrix, solve as qrsolve
import numpy as np
import matplotlib.pyplot as plt


def solve_default(A, b):
    from scipy.sparse.linalg import spsolve
    x = spsolve(A.T @ A, A.T @ b)
    return x, None


def solve_pinv(A, b):
    # TODO: return x s.t. Ax = b using pseudo inverse.
    N = A.shape[1]
    AtA = csc_matrix(A.T @ A)
    Atb = A.T @ b
    x = inv(AtA) @ Atb
    return x, None


def solve_lu(A, b):
    # TODO: return x, U s.t. Ax = b, and A = LU with LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    N = A.shape[1]
    AtA = csc_matrix(A.T @ A)
    Atb = A.T @ b
    lu = splu(AtA, permc_spec='NATURAL')
    x = lu.solve(Atb)
    U = lu.U
    return x, U


def solve_lu_colamd(A, b):
    # TODO: return x, U s.t. Ax = b, and Permutation_rows A Permutation_cols = LU with reordered LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    N = A.shape[1]
    AtA = csc_matrix(A.T @ A)
    Atb = A.T @ b
    lu = splu(AtA, permc_spec='COLAMD')
    x = lu.solve(Atb)
    U = lu.U
    return x, U


def solve_qr(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |Rx - d|^2 + |e|^2
    # https://github.com/huangjuite/PySPQR
    N = A.shape[1]
    z, R, E, rank = rz(A, b, permc_spec='NATURAL')
    x = spsolve_triangular(R, z, lower=False)
    return x, R


def solve_qr_colamd(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |R E^T x - d|^2 + |e|^2, with reordered QR decomposition (E is the permutation matrix).
    # https://github.com/huangjuite/PySPQR
    N = A.shape[1]
    z, R, E, rank = rz(A, b, permc_spec='COLAMD')
    y = spsolve_triangular(R, z, lower=False)
    E_mat = permutation_vector_to_matrix(E)
    x = E_mat @ y
    return x, R


def solve_lu_colamd_fbsub(A, b):
    # TODO (Bonus): return x, U s.t. Ax = b using forward/backward substitution with COLAMD reordering.
    AtA = csc_matrix(A.T @ A)
    Atb = A.T @ b
    lu = splu(AtA, permc_spec='COLAMD')
    U = lu.U

    Pb = Atb[lu.perm_r]
    y = spsolve_triangular(lu.L.tocsr(), Pb, lower=True)
    z = spsolve_triangular(lu.U.tocsr(), y, lower=False)
    inv_perm_c = np.argsort(lu.perm_c)
    x = z[inv_perm_c]

    return x, U


def solve(A, b, method='default'):
    '''
    \param A (M, N) Jacobian matrix
    \param b (M, 1) residual vector
    \return x (N, 1) state vector obtained by solving Ax = b.
    '''
    M, N = A.shape

    fn_map = {
        'default': solve_default,
        'pinv': solve_pinv,
        'lu': solve_lu,
        'qr': solve_qr,
        'lu_colamd': solve_lu_colamd,
        'qr_colamd': solve_qr_colamd,
        'lu_colamd_fbsub': solve_lu_colamd_fbsub,
    }

    return fn_map[method](A, b)
