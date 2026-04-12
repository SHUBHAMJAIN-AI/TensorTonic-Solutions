def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    
    """
    import numpy as np
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    p_t = np.where(targets==1,predictions,1-predictions)

    fce = (-1*alpha*(1-p_t)**gamma)*np.log(p_t)
    mean_loss = np.sum(fce)/len(predictions)

    return mean_loss
    