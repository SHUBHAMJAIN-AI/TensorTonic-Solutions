def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    import numpy as np 
    x = np.asarray(x)
    ans = np.where(x>0,np.maximum(0,x),alpha*(np.exp(x)-1))
    return ans.tolist()