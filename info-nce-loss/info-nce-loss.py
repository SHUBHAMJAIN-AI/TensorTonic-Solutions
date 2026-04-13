import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)
    
    # 1. Compute similarity matrix
    logits = np.matmul(Z1, Z2.T) / temperature
    
    # 2. For numerical stability, subtract the max of each row
    logits_max = np.max(logits, axis=1, keepdims=True)
    logits_stable = logits - logits_max
    
    # 3. Compute the components for the loss
    exp_logits = np.exp(logits_stable)
    
    # Positive pairs are the diagonal elements
    positives = np.diag(logits_stable)
    
    # Denominator is the sum of row-wise exponentials
    denominator = np.sum(exp_logits, axis=1)
    
    # 4. InfoNCE formula: -log(exp(pos) / sum(exp(all)))
    # Which simplifies to: - (pos - log(sum(exp(all))))
    loss = -np.mean(positives - np.log(denominator))
    
    return loss