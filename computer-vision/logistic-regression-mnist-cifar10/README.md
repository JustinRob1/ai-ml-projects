# Logistic Regression on MNIST and CIFAR10

## Overview

This project implements and evaluates logistic regression models on two popular image datasets: MNIST (handwritten digits) and CIFAR10 (color images of objects). The project demonstrates basic deep learning workflows, including data loading, model training, evaluation, and hyperparameter tuning using PyTorch.

## Summary

- The code supports running logistic regression on either the MNIST or CIFAR10 dataset.
- Data is loaded and preprocessed using torchvision datasets and transforms.
- The logistic regression model is implemented as a single-layer neural network using PyTorch's `nn.Module`.
- Training and evaluation loops are provided, including accuracy computation and optional hyperparameter tuning.
- Results such as accuracy, score, and runtime are printed after execution.

## Assumptions

- The datasets (MNIST, CIFAR10) are downloaded automatically if not present in the `./data` directory.
- The code is designed to run on both CPU and GPU (if available and specified).
- Default hyperparameters are set for demonstration; these can be adjusted in the code or via command-line arguments if using `paramparse`.

## Files

- **main.py**: Main script to run experiments, parse arguments, and evaluate models.
- **model.py**: Contains the core implementation of the logistic regression model, training, and evaluation routines.


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

2. **Run the Script**  
   You can run the main script with default parameters:
   ```sh
   python main.py
   ```

   To specify different parameters, you can modify the `main.py` file.
   Example:
   ```sh
   class Args:
    dataset = "CIFAR10"
    mode = "logistic"
    gpu = 1
   ```

3. **Output**  
   After running, the script will print the accuracy, score, and runtime of the model.
