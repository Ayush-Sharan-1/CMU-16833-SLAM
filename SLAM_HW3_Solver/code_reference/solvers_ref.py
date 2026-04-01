'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import inv, splu, spsolve, spsolve_triangular
from sparseqr import rz, permutation_vector_to_matrix, solve as qrsolve
import numpy as np
import matplotlib.pyplot as plt


def forward_subs(L_csc, b):
    """Solves L * z = b where L is a lower triangular CSC sparse matrix."""
    N = L_csc.shape[0]
    z = np.array(b, copy=True, dtype=float)
    
    indptr = L_csc.indptr
    indices = L_csc.indices
    data = L_csc.data
    
    for j in range(N):
        start, end = indptr[j], indptr[j+1]
        col_indices = indices[start:end]
        col_data = data[start:end]
        
        diag_mask = (col_indices == j)
        if np.any(diag_mask):
            diag_val = col_data[diag_mask][0]
            z[j] /= diag_val
        
        below_mask = (col_indices > j)
        z[col_indices[below_mask]] -= col_data[below_mask] * z[j]
        
    return z

def backward_subs(U_csc, z):
    """Solves U * y = z where U is an upper triangular CSC sparse matrix."""
    N = U_csc.shape[0]
    y = np.array(z, copy=True, dtype=float)
    
    indptr = U_csc.indptr
    indices = U_csc.indices
    data = U_csc.data
    
    for j in range(N-1, -1, -1):
        start, end = indptr[j], indptr[j+1]
        col_indices = indices[start:end]
        col_data = data[start:end]
        
        diag_mask = (col_indices == j)
        if np.any(diag_mask):
            diag_val = col_data[diag_mask][0]
            y[j] /= diag_val
        
        above_mask = (col_indices < j)
        y[col_indices[above_mask]] -= col_data[above_mask] * y[j]
        
    return y

def solve_bonus(A, b, invA):
    g = A.T @ b
    g_perm = np.zeros_like(g)
    g_perm[invA.perm_r] = g
    z = forward_subs(invA.L, g_perm)
    y = backward_subs(invA.U, z)
    x = y[invA.perm_c]
    return x.flatten()

def solve_default(A, b):
    from scipy.sparse.linalg import spsolve
    x = spsolve(A.T @ A, A.T @ b)
    return x, None


def solve_pinv(A, b, invA):
    # TODO: return x s.t. Ax = b using pseudo inverse.
    N = A.shape[1]
    x = inv(A.T @ A) @ A.T @ b
    return x, None


def solve_lu(A, b):
    # TODO: return x, U s.t. Ax = b, and A = LU with LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    N = A.shape[1]
    invA = splu(csc_matrix(A.T @ A, dtype=float), permc_spec="NATURAL")
    
    # BONUS START
    x = solve_bonus(A, b, invA)
    # BONUS END

    # x = invA.solve(A.T @ b)
    U = invA.U
    return x, U


def solve_lu_colamd(A, b):
    # TODO: return x, U s.t. Ax = b, and Permutation_rows A Permutation_cols = LU with reordered LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    N = A.shape[1]
    invA = splu(csc_matrix(A.T @ A, dtype=float), permc_spec="COLAMD")
    
    # BONUS START
    x = solve_bonus(A, b, invA)
    # BONUS END
    
    # x = invA.solve(A.T @ b)
    U = invA.U
    return x, U


def solve_qr(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |Rx - d|^2 + |e|^2
    # https://github.com/huangjuite/PySPQR
    N = A.shape[1]
    z, R, E, rank = rz(A, b, tolerance=0, permc_spec='NATURAL')
    x = spsolve_triangular(csc_matrix(R), z, lower=False)
    return x.flatten(), R


def solve_qr_colamd(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |R E^T x - d|^2 + |e|^2, with reordered QR decomposition (E is the permutation matrix).
    # https://github.com/huangjuite/PySPQR
    N = A.shape[1]
    z, R, E, rank = rz(A, b, tolerance=0, permc_spec='COLAMD')
    y = spsolve_triangular(csc_matrix(R), z, lower=False)
    
    P = permutation_vector_to_matrix(E)
    x = P @ y
    return x.flatten(), R


def solve(A, b, method='default'):
    r'''
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
    }

    return fn_map[method](A, b)
