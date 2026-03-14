import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples, n_features = X.shape
    
    # 1. Initialize weights as zeros and bias as 0.0
    w = np.zeros(n_features)
    b = 0.0
    
    # 2. Implement the training loop
    for _ in range(steps):
        # Linear model: z = Xw + b
        # Use @ for matrix-vector multiplication
        model = np.dot(X, w) + b
        p = _sigmoid(model)
        
        # 3. Compute gradients
        # Error (p - y)
        error = p - y
        grad_w = np.dot(X.T, error) / n_samples
        grad_b = np.mean(error)
        
        # 4. Update parameters
        w -= lr * grad_w
        b -= lr * grad_b
        
    return w, b
    