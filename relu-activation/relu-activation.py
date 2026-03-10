import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    ans = np.maximum(x,0)
    return ans