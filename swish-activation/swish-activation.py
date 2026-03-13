import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x= np.asarray(x)
    sigmoid = 1/(1+np.exp(-x))
    ans = x*sigmoid
    return ans