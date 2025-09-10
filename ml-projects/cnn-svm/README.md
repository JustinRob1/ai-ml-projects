# CNN and SVM for Image Classification

This project explores image classification using Convolutional Neural Networks (CNNs) and Support Vector Machines (SVMs) on the Fashion MNIST and low-resolution MNIST datasets. The code and experiments demonstrate the implementation, training, evaluation, and comparison of these two approaches.

## Summary

- **CNN Implementation**: The code in [cnn.py](cnn.py) defines and trains a CNN from scratch using Keras, with support for both standard Fashion MNIST (28x28) and a downsampled low-resolution MNIST (7x7) dataset.
- **SVM Implementation**: [svm.py](svm.py) trains SVM classifiers on both datasets, exploring the effect of the RBF kernel's gamma parameter and evaluating performance.
- **Experimentation and Visualization**: [graph.py](graph.py) orchestrates experiments and generates plots to compare training and test accuracy for both CNN and SVM models. The generated plots (PNG files) summarize model performance and the impact of hyperparameters.
- **Additional Scripts**: [q2b.py](q2b.py) provides code for plotting SVM decision boundaries on synthetic data, and [test.py](test.py) contains additional CNN training/testing code.

## Assumptions

- The code is written for Python 3.10+ and requires `numpy`, `matplotlib`, `scikit-learn`, `keras`, and `Pillow`.
- The Fashion MNIST and MNIST datasets are automatically downloaded by Keras if not present.
- All scripts assume the working directory is the project root.

## How to Run

1. **Install dependencies** (if not already installed):

   ```sh
   pip install numpy matplotlib scikit-learn keras pillow
    ```

2. **Run CNN Experiments**:
   Execute `cnn.py` to train and evaluate the CNN on both datasets. This will save model weights and accuracy plots.

   ```sh
   python cnn.py
   ```

3. **Run SVM Experiments**:
    Execute `svm.py` to train and evaluate SVM classifiers on both datasets. This will generate accuracy plots for different gamma values.
    ```sh
    python svm.py
    ```

4. **Generate Plots**:
    Run `graph.py` to create summary plots comparing CNN and SVM performance.
    
    ```sh
    python graph.py
    ```

5. **View Results**:
    Output plots (e.g., fashion_mnist_plot.png, low_res_mnist_plot.png, svm_mnist.png, svm_low_res.png) will be saved in the project directory, summarizing model performance and comparisons.

