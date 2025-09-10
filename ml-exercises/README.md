# Machine Learning Exercises

This repository contains a series of Julia [Pluto.jl](https://github.com/fonsp/Pluto.jl) notebooks. The exercsise cover foundational topics in machine learning, including univariate and multivariate regression, gradient descent, and logistic regression, with a focus on hands-on implementation and experimentation.

## Repository Structure

- `univariate_regression.jl`  
  Explore univariate regression, Gaussian distributions, and basic statistics. Includes simulation and visualization of sample statistics.

- `gradient_descent.jl`  
  Implements and compares various gradient descent algorithms (stochastic, batch, mini-batch) for linear regression, including baseline regressors.

- `multivariate_regression.jl`  
  Extends regression to the multivariate case, including polynomial feature transformations and advanced optimization strategies (SGD, AdaGrad, etc.).

- `logistic_regression.jl`  
  Implements logistic regression for binary classification, including polynomial logistic regression, cross-entropy loss, and adaptive optimizers. Also includes hypothesis testing for model comparison.

## Purpose

The purpose of this repository is to provide practical experience with core machine learning algorithms and concepts, including:

- Implementing regression and classification models from scratch
- Understanding and applying gradient-based optimization techniques
- Experimenting with feature transformations and regularization
- Evaluating models using statistical tests and visualization

## How to Run the Notebooks

1. **Install Julia**  
   Download and install Julia from [https://julialang.org/downloads/](https://julialang.org/downloads/).

2. **Install Pluto**  
   Open a Julia REPL and run:
   ```julia
   import Pkg
   Pkg.add("Pluto")
    ```

3. **Launch Pluto**
    In the Julia REPL, run:
    ```julia
    using Pluto
    Pluto.run()
    ```
This will open a web interface.

4. **Open Notebooks**  
   In the Pluto interface, open the desired `.jl` notebook files from this repository (e.g., `univariate_regression.jl`).

5. **Install Required Packages**  
   Each notebook may require specific Julia packages. You can install them using:
   ```julia
   import Pkg
   Pkg.add("PackageName")
   ```
   Replace `"PackageName"` with the actual package names used in the notebooks.

6. **Run the Notebooks**  
   Follow the instructions within each notebook to execute the code cells and explore the exercises and solutions.