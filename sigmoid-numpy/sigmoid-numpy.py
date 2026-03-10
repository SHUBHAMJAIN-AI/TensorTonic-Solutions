import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.asarray(x)
    num = 1
    den = 1+np.exp(-x)
    ans= num/den
    return ans