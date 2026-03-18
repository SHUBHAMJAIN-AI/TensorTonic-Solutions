import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    anchor = np.asarray(anchor)
    positive = np.asarray(positive)
    negative = np.asarray(negative)
    distance_1 = np.sum((anchor - positive)**2, axis=-1)
    distance_2 = np.sum((anchor - negative)**2, axis=-1)

    loss = np.maximum(0,distance_1-distance_2+margin)
    return loss.mean()