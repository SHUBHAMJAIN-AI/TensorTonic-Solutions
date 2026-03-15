import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    n = y_pred.shape[0]
    correct_prob = y_pred[np.arange(n),y_true]
    cel = -np.mean(np.log(correct_prob))
    
    return cel