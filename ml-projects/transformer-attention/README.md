# Transformer Attention Mechanism Demo

This project demonstrates a basic implementation of the transformer attention mechanism, a core component of modern deep learning models for natural language processing. The code provides an exploration of self-attention and the transformer decoder layer.

## Summary

- Implements the scaled dot-product attention mechanism from scratch.
- Includes a simple transformer decoder layer to illustrate how attention is used in sequence modeling.
- Contains test functions to verify the correctness of the softmax, attention, and decoder layer implementations.

## Assumptions

- The code is written for Python 3.10+ and uses only `numpy` as a dependency.

## How to Run

1. **Install dependencies** (if not already installed):

   ```sh
   pip install numpy
    ```

2. **Run the script**:
Execute the script trasnformer.py to run the included test functions and see example outputs:

   ```sh
   python transformer.py
   ```

The script will print the results of the softmax function, attention weights, and the output of the transformer decoder layer to the console.