import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    p = np.asarray(p)
    q= np.asarray(q)
    term1 = np.log(p/(q+eps))
    ans = np.sum(p*term1)
    return ans