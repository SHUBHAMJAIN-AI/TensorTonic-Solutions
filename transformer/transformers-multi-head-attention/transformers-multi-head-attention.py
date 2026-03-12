import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    batch,seq_len,d_model = Q.shape
    d_k = d_model//num_heads
    Q_projected = np.matmul(Q,W_q)
    K_projected = np.matmul(K,W_k)
    V_projected = np.matmul(V,W_v)

    Q_heads = Q_projected.reshape(batch,seq_len,num_heads,d_k)
    K_heads = K_projected.reshape(batch,seq_len,num_heads,d_k)
    V_heads= V_projected.reshape(batch,seq_len,num_heads,d_k)

    scaling = np.sqrt(d_k)
    attn_score = np.matmul(Q_heads,K_heads.transpose(0,1,3,2))/scaling
    attn_weights = softmax(attn_score,axis=-1)
    attn_output = np.matmul(attn_weights,V_heads)
    attn_output = attn_output.transpose(0,2,1,3).reshape(batch,seq_len,d_model)
    return np.matmul(attn_output,W_o)