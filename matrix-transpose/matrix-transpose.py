import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.asarray(A)
    n_rows,n_col = A.shape
    A_t = np.zeros((n_col,n_rows))
    for i in range (n_rows):
        for j in range(n_col):
            A_t[j,i] = A[i,j]
    return A_t
