DCGAN for Handwritten Digit Synthesis (MNIST)
This project implements a Deep Convolutional GAN (DCGAN) to generate realistic 28×28 grayscale handwritten digits using the MNIST dataset. It includes a minimal IDX data loader, a TensorFlow/Keras DCGAN architecture, a full training loop, and utilities to visualize both real and generated samples.

Key features
Custom MNIST IDX loader

Loads raw IDX image/label files without external dependencies.

Normalizes images to [-1, 1] and ensures shape (N, 28, 28, 1).

DCGAN architecture (TensorFlow/Keras)

Generator: Dense → Reshape(7×7×256) → Conv2DTranspose(128, 64) → Tanh output (28×28×1).

Discriminator: Two Conv2D blocks with LeakyReLU + Dropout → Flatten → Dense(sigmoid).

Stable training setup

Adam(learning_rate=2e-4, beta_1=0.5) for both networks.

Discriminator trained on mixed real/fake batches; generator trained via frozen discriminator.

Visualizations

Grid sampling for real MNIST digits.

Periodic snapshots of generated digits to monitor training quality.

Project structure
Data loading

MnistDataloader class reads four IDX files (train/test images and labels).

Paths are configurable; raw bytes are reshaped to (N, 28, 28) and cast to float32.

Models

Generator

Input: latent_dim=100.

Layers: Dense(7×7×256) → Reshape → Conv2DTranspose(128, stride=2) → BN → LeakyReLU → Conv2DTranspose(64, stride=2) → BN → LeakyReLU → Conv2DTranspose(1, kernel=7, tanh).

Discriminator

Input: (28, 28, 1).

Layers: Conv2D(64, stride=2) → LeakyReLU → Dropout(0.25) → Conv2D(128, stride=2) → LeakyReLU → Dropout(0.25) → Flatten → Dense(1, sigmoid).

Training

Alternates: train discriminator on real+fake; train generator via combined model with discriminator frozen.

Configurable epochs, batch size, sampling frequency.

Utilities

show_images(): display grids of real or generated digits.

Model summaries logged for quick inspection of parameter counts and shapes.

Requirements
Python 3.10+

TensorFlow 2.x / Keras

NumPy, Matplotlib

Install dependencies (example):

pip install tensorflow numpy matplotlib

Usage
Prepare MNIST IDX files

Download the four MNIST IDX files:

train-images-idx3-ubyte

train-labels-idx1-ubyte

t10k-images-idx3-ubyte

t10k-labels-idx1-ubyte

Update the file paths in the notebook where MnistDataloader is initialized.

Run the notebook

Open DCGAN-for-Handwritten-Digit-Synthesis.ipynb.

Execute cells in order:

Imports and data loader.

Dataset normalization to [-1, 1].

Visualize random training samples.

Build generator/discriminator.

Compile and start training.

Periodically inspect generated samples.
