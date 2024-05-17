# Variational Autoencoder (VAE) on MNIST Dataset

This repository contains an implementation of a Variational Autoencoder (VAE) using the MNIST dataset. The VAE is a type of generative model that learns to encode input data into a lower-dimensional latent space and then reconstruct the input data from this latent representation.

## Description

The Variational Autoencoder (VAE) is a powerful generative model that combines neural networks and probabilistic modeling. This project demonstrates how to implement a VAE using TensorFlow and Keras to encode and decode handwritten digits from the MNIST dataset. The implementation includes training the VAE and visualizing the original and reconstructed images to compare the performance.

## Files

- `vae_mnist.py`: Main Python script containing the VAE implementation.
- `README.md`: This readme file.

## Setup Instructions

To run this project, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone [https://github.com/SanaFarooq/VAE_AUTOENCODER_FROMSCRATCH.git]
    cd vae-mnist
    ```

2. **Install the required libraries:**

    Make sure you have `TensorFlow` installed. You can install the required packages using `pip`:

    ```bash
    pip install tensorflow numpy matplotlib
    ```

3. **Run the script:**

    You can run the script using Python:

    ```bash
    python vae_mnist.py
    ```

## Code Overview

### Loading and Preprocessing Data

The MNIST dataset is loaded and normalized. The images are reshaped to have a single channel.

```python
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
