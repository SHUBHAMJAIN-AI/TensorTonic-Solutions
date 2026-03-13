import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.asarray(x)
    axis=1 if x.ndim>1 else 0
    e_x = np.exp(x-np.max(x,axis=axis,keepdims =True))

    softmax = e_x/np.sum(e_x,axis=axis,keepdims=True)
    
    
    return softmax