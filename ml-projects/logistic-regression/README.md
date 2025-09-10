# Logistic Regression with Newton's Method

This project implements logistic regression for binary classification using Newton's method for optimization. The code and experiments demonstrate both the theory and practical application of logistic regression on a provided dataset.

## Summary

- **Logistic Regression** is a fundamental classification algorithm used to predict binary outcomes.
- **Newton's Method** is used for efficient optimization of the logistic regression cost function, leveraging second-order derivatives (the Hessian).
- The project includes:
  - An implementation of logistic regression with Newton's method ([newton.py](newton.py)).
  - A Jupyter notebook ([lr.ipynb](lr.ipynb)) for running experiments, visualizing convergence, and evaluating model performance.
  - Example feature and observation data in `.npy` format.

## Assumptions

- The dataset files `feature.npy` and `obs.npy` are present in the project directory.
- The code is written for Python 3.10+ and uses `numpy` and `matplotlib`.
- The notebook assumes a binary classification task.

## How to Run

1. **Install dependencies** (if not already installed):

   ```sh
   pip install numpy matplotlib
    ```
2. **Run the Jupyter Notebook**:
Open `lr.ipynb` in Jupyter Notebook or JupyterLab to execute the experiments and visualize results:
- Load the data
- Train the logistic regression model using Newton's method
- Visualize the optimization process and results

3. **Run the implementation directly**:
You can also import and use the functions in newton.py in your own scripts for logistic regression tasks.

## Files
- `newton.py`: Contains the implementation of logistic regression using Newton's method.
- `lr.ipynb`: Jupyter notebook for experiments and visualizations.
- `feature.npy`: Example feature data for training/testing.
- `obs.npy`: Example observation labels for training/testing.
