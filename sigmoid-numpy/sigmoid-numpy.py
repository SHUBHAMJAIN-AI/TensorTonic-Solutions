import numpy as np

def sigmoid(x):
    x = np.asarray(x)
    
    # Initialize an output array of the same shape
    ans = np.zeros_like(x, dtype=float)
    
    # Mask for positive and negative values
    pos_mask = (x >= 0)
    neg_mask = ~pos_mask
    
    # Formula for x >= 0: 1 / (1 + exp(-x))
    ans[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    
    # Formula for x < 0: exp(x) / (1 + exp(x))
    ans[neg_mask] = np.exp(x[neg_mask]) / (1 + np.exp(x[neg_mask]))
    
    return ans