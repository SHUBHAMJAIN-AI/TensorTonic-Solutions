def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    import numpy as np
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    cosine_prodcut = np.dot (x1,x2)
    noramlized_product = np.linalg.norm(x1)*np.linalg.norm(x2)
    cosine_similarity = cosine_prodcut/noramlized_product
    loss1 = 1-cosine_similarity
    loss2 = np.maximum(0,cosine_similarity-margin)
    output = np.where(label==1,loss1,loss2)

    return output.item()

#     import numpy as np

# def cosine_embedding_loss(x1, x2, label, margin):
#     x1, x2 = np.asarray(x1), np.asarray(x2)
    
#     # Combined calculation for clarity
#     cos_sim = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    
#     if label == 1:
#         loss = 1 - cos_sim
#     else:
#         loss = np.maximum(0, cos_sim - margin)
        
#     return float(loss) # Ensuring it returns a single float
    