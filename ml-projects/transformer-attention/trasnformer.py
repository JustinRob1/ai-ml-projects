import numpy as np
# Sources: https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
# https://machinelearningmastery.com/how-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras/
# https://github.com/phohenecker/pytorch-transformer/blob/master/src/main/python/transformer/decoder.py
# https://machinelearningmastery.com/implementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras/

def initialize_parameters(input_size, output_size):
    """Initialize weight parameters with small random values."""
    return np.random.randn(output_size, input_size) * 0.1

# Task 1: Implement the Softmax Function
# The softmax function converts a vector of values to a probability distribution.
# Each element is transformed using the exponential function, making them positive,
# and then normalized so that the sum of the resulting values is 1.
def softmax(x):
    vals = np.exp(x - np.max(x, axis=-1, keepdims=True)) 
    return vals / np.sum(vals, axis=-1, keepdims=True)

# Task 2: Implement the Scaled Dot-Product Attention Mechanism
# The attention function computes a weighted sum of values V, where the weight assigned
# to each value is computed by a compatibility function of the query Q with the corresponding key K.
def scaled_dot_product_attention(Q, K, V):
    model = Q.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(model)  
    weights = softmax(scores)
    output = np.matmul(weights.T, V)
    return output, weights

# Task 3: Implement the Transformer Decoder Layer
# A transformer decoder layer consists of a self-attention mechanism, cross-attention with
# respect to the encoder outputs, and a position-wise feed-forward network.
def transformer_decoder_layer(Q, K, V, memory, params, mask=None):
    W_o = params['W_o']
    W_m_v = params['W_m_v']
    W_ff1 = params['W_ff1']
    b_ff1 = params['b_ff1']
    W_ff2 = params['W_ff2']
    b_ff2 = params['b_ff2']
    
    att_output, _ = scaled_dot_product_attention(Q, K, V)
    att_output += np.matmul(memory, W_m_v) 
    att_output = np.matmul(att_output, W_o)
    att_output = layer_norm(att_output + Q)
    ff_output = np.matmul(np.maximum(0, np.matmul(att_output, W_ff1) + b_ff1), W_ff2) + b_ff2
    ff_output = layer_norm(ff_output + att_output)

    return ff_output
    
#Layer_norm is given as:
def layer_norm(x):
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + 1e-6)
# ----------------------
# Parameters Initialization
# ----------------------

d_model = 10  # Embedding size
d_ff = 20  # Size of the feed-forward network
vocab_size = 10  # Assuming a vocab size of 10 for simplicity

# Initialize weights
params = {
    'W_q': initialize_parameters(d_model, d_model),
    'W_k': initialize_parameters(d_model, d_model),
    'W_v': initialize_parameters(d_model, d_model),
    'W_o': initialize_parameters(d_model, d_model),
    'W_m_k': initialize_parameters(d_model, d_model),
    'W_m_v': initialize_parameters(d_model, d_model),
    'W_ff1': initialize_parameters(d_ff, d_model),
    'b_ff1': np.zeros(d_ff),
    'W_ff2': initialize_parameters(d_model, d_ff),
    'b_ff2': np.zeros(d_model),
    'd_model': d_model
}

# Test Check 1: Softmax Function
def check_softmax():
    print("Checking the softmax function...")
    test_input = np.array([1.0, 2.0, 3.0])
    output = softmax(test_input)
    if np.allclose(output, np.array([0.09003057, 0.24472847, 0.66524096])):
        print("Softmax function seems to be implemented correctly.")
    else:
        print("Softmax function may be incorrect. Please check your implementation.")

# Test Check 2: Scaled Dot-Product Attention
def check_attention():
    print("Checking the attention mechanism...")
    Q = np.array([[1, 0, 0], [0, 1, 0]])
    K = V = np.array([[1, 2, 3], [4, 5, 6]])
    output, _ = scaled_dot_product_attention(Q, K, V)
    expected_output = np.array([[3.54902366, 4.54902366, 5.54902366], [3.54902366, 4.54902366, 5.54902366]])
    if np.allclose(output, expected_output):
        print("Attention mechanism seems to be implemented correctly.")
    else:
        print("Attention mechanism may be incorrect. Please check your implementation.")

# Test Check 3: Transformer Decoder Layer Functionality
def check_decoder_layer():
    print("Checking the transformer decoder layer...")
    Q = K = V = memory = np.random.randn(1, 10, d_model)
    output = transformer_decoder_layer(Q, K, V, memory, params)
    # Instead of just checking the shape, let's ensure there's a non-zero variance
    # across the output, indicating that the layer has applied some transformation.
    if output.shape == (1, 10, d_model) and np.var(output) != 0:
        print("Transformer decoder layer output shape is correct and shows variance across outputs.")
    else:
        print("There might be an issue with the transformer decoder layer. Please check your implementation.")

check_softmax()
check_attention()
check_decoder_layer()