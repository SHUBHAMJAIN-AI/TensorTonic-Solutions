import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    x = np.asarray(x)
    vector_erf = np.vectorize(math.erf)
    first_term =x/2
    second_term = 1+vector_erf(x/math.sqrt(2))
    ans = first_term*second_term
    return ans
