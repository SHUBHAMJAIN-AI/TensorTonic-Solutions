import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    p = np.asarray(p)
    q= np.asarray(q)
    mask = p>0
    term1 = np.log(p[mask]/(q[mask]+eps))
    ans = np.sum(p[mask]*term1)
    return ans