# Vision Transformer and Contrastive Representation Learning

## Overview

This project implements a Vision Transformer (ViT) model for image classification and explores contrastive representation learning using triplet loss. 
The code is applied to the FashionMNIST dataset and demonstrates both supervised classification and metric learning for embedding representations. 
The implementation is based on PyTorch and PyTorch Metric Learning.

## Files

- **vit_model.py**: Implementation of the Vision Transformer architecture and embedding head.
- **vit.ipynb**: Jupyter notebook for training, evaluating, and experimenting with the ViT model and contrastive learning.
- **check_vit.py**: Script to test the correctness of the ViT implementation.
- **vit_embeds.pt**: Saved model weights for the trained embedding model.

## Assumptions

- The FashionMNIST dataset is automatically downloaded if not present.
- The code is designed to run on both CPU and GPU (if available).
- Required Python packages are installed (see below).

## How to Run

1. **Install Dependencies**

   Ensure you have Python 3.x and the following packages installed:
   - torch
   - torchvision
   - numpy
   - matplotlib
   - pytorch-metric-learning
   - tqdm

   You can install them using pip:
   ```sh
   pip install torch torchvision numpy matplotlib pytorch-metric-learning tqdm
   ```

2. **Run the Notebook**

   Open `vit.ipynb` in Jupyter Notebook and run the cells to train and evaluate the ViT model.

3. **Test the ViT Implementation**
   You can run the `check_vit.py` script to verify the correctness of the ViT implementation:
   ```sh
   python check_vit.py
   ```

4. **Using the Trained Model**
   The trained embedding model weights are saved in `vit_embeds.pt`. You can load this file in your own scripts to use the trained model for inference or further training.