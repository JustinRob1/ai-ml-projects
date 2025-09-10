# Object Detection and Segmentation

## Overview

This project is an implementation of object detection and segmentation on a custom dataset. The goal is to detect and segment objects (digits 0-9) in 64x64 color images, using deep learning techniques. The project involves training, validating, and testing models for both object detection (bounding boxes) and semantic segmentation (pixel-wise classification).

## Contents

- `train.ipynb`: Main Jupyter notebook for data preparation, training, evaluation, and visualization.
- `main.py`: Script for running the full pipeline from the command line.
- `model.py`: Contains model definitions for detection and segmentation (e.g., UNet, YOLO, or custom CNNs).
- `train.py`: Training loop and utilities for model optimization.
- `utils.py`: Helper functions for data processing, visualization, and evaluation metrics.
- `best.pt`: Saved PyTorch model weights for the best-performing model.
- `unet_model.h5`: (If present) Saved Keras/TensorFlow UNet model weights.
- `valid_bboxes.npy`, `valid_seg.npy`, `valid_Y.npy`: Validation data (bounding boxes, segmentation masks, and labels).

## Assumptions

- The dataset consists of 64x64 RGB images containing a single digit (0-9) per image.
- Ground truth for object detection is provided as bounding boxes, and for segmentation as pixel masks.
- The code assumes the use of PyTorch (and optionally TensorFlow/Keras for UNet).
- All dependencies are listed in the notebook or can be installed via standard Python package managers.

## How to Run

### Using the Jupyter Notebook

1. Open `train.ipynb` in Jupyter Notebook or JupyterLab.
2. Follow the notebook cells to:
   - Prepare and visualize the dataset.
   - Train the detection and segmentation models.
   - Evaluate model performance on the validation set.
   - Visualize predictions.

### Using the Python Scripts

1. Ensure you have Python 3.8+ and PyTorch installed.
2. (Optional) Install additional dependencies:
   ```sh
   pip install numpy matplotlib pillow scikit-learn
    ```
3. Run the main script:
   ```sh
   python main.py
   ```
   This will execute the full training and evaluation pipeline.

Model Weights:
- Predicted model weights are saved in `best.pt` for PyTorch models and `unet_model.h5` for Keras/TensorFlow models.

Validation data (valid_bboxes.npy, valid_seg.npy, valid_Y.npy) are provided for evaluating model performance.
