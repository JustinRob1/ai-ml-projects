# Neural Network Classifiers on MNIST and CIFAR10

## Overview

This project implements and evaluates two types of neural network classifiers—a Feedforward Neural Network (FNN) and a Convolutional Neural Network (CNN)—on the MNIST (handwritten digits) and CIFAR10 (color images) datasets. The code demonstrates the process of building, training, and testing deep learning models using PyTorch.

## Files

- **fnn_model.py**: Implementation of the Feedforward Neural Network (FNN) model.
- **fnn_classifier.py**: Training and evaluation routines for the FNN model.
- **cnn_model.py**: Implementation of the Convolutional Neural Network (CNN) model.
- **cnn_classifier.py**: Training and evaluation routines for the CNN model.

## Summary

- The code supports both MNIST and CIFAR10 datasets.
- Training and evaluation include accuracy computation and runtime measurement.
- Hyperparameters (such as learning rate, batch size, and number of epochs) can be adjusted in the respective classifier scripts.

## Assumptions

- The datasets (MNIST, CIFAR10) are automatically downloaded if not present locally.
- The code is designed to run on both CPU and GPU (if available).
- Default hyperparameters are set for demonstration and can be modified in the scripts.

## How to Run

1. **Install Dependencies**

   Ensure you have Python 3.x and the following packages installed:
   - torch
   - torchvision
   - numpy

   You can install them using pip:
   ```sh
   pip install torch torchvision numpy
    ```

2. **Train and Evaluate FNN**

   To train and evaluate the FNN on MNIST or CIFAR10, run the following command in your terminal:
   ```sh
   python fnn_classifier.py
   ```

   You can modify the dataset and hyperparameters directly in the `fnn_classifier.py` file.

3. **Train and Evaluate CNN**
    To train and evaluate the CNN on MNIST or CIFAR10, run the following command in your terminal:
    ```sh
    python cnn_classifier.py
    ```
    
    You can modify the dataset and hyperparameters directly in the `cnn_classifier.py` file.

4. **Results**
   After running, the scripts will print the accuracy and runtime for the selected model and dataset.
