# Convolutional Neural Networks on Decentered MNIST

## Problem Summary

This program demonstrates the use of neural networks for image classification on a modified version of the MNIST dataset. The images are "decentered" by translating digits toward the top-left or bottom-right corners, making the classification task more challenging. The goal is to compare the performance of different neural network architectures—two multilayer perceptrons (MLP1, MLP2) and a convolutional neural network (CNN)—on this dataset.

## Code Overview

The code provides implementations for three models:
- **MLP1**: A simple feedforward neural network with one hidden layer.
- **MLP2**: A feedforward neural network with two hidden layers.
- **CNN**: A convolutional neural network with two convolutional layers and two max-pooling layers, followed by fully connected layers.

The models are trained and evaluated on both top-left and bottom-right decentered MNIST datasets. The code also generates example images and prints the accuracy of each model.

### Files

- **cnn.py**  
  Main script containing model definitions (`MLP1`, `MLP2`, `CNN`), training and evaluation routines, and the entry point for running experiments.  
  - Defines and trains the neural network models.
  - Evaluates accuracy on both decentered test sets.
  - Generates a side-by-side example image (`examples.png`).

- **cnn_utils.py**  
  Utility functions for data loading and preprocessing.  
  - `get_MNIST`: Loads and decentered the MNIST dataset.
  - `show_examples`: Creates a visual comparison of decentered digits.
  - `hout`: Computes output size for convolutional layers.

## Assumptions

- The code uses PyTorch and torchvision for model definition and data loading.
- The MNIST dataset is automatically downloaded if not present.
- The models are trained on CPU by default.
- The decentered images are padded to 30x30 pixels.

## How to Run

From the `cnn-mnist` directory, run:

```sh
python3 cnn.py
```

This will train all models, evaluate their accuracy, and generate an examples.png file showing sample decentered digits.