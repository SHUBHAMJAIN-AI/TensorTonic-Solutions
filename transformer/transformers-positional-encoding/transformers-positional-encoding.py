import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    positional_indices = np.arange(seq_length).reshape(-1,1)
    division_term = np.arange(0,d_model,2)
    division_term= np.exp(division_term*(-np.log(10000))/d_model)
    output = np.zeros((seq_length,d_model))
    output[:,0::2] = np.sin(positional_indices*division_term)
    output[:,1::2] = np.cos(positional_indices*division_term)
    
    return output