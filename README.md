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
## VAE Architecture
The VAE consists of an encoder and a decoder. The encoder compresses the input data into a latent space, and the decoder reconstructs the input data from the latent space.
**Encoder:**
```python
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten

inputs = Input(shape=input_shape, name='encoder_input')
x = Flatten()(inputs)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

## **Decoder:**
```python
from tensorflow.keras.layers import Reshape

latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(128, activation='relu')(latent_inputs)
x = Dense(256, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(np.prod(input_shape), activation='sigmoid')(x)
outputs = Reshape(input_shape)(x)
## LOSS FUNCTION
The VAE loss function combines the reconstruction loss and the KL divergence loss:
```python
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
reconstruction_loss *= 28 * 28

kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5

vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
### Loading and Preprocessing Data

The MNIST dataset is loaded and normalized. The images are reshaped to have a single channel.

```python
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
## Acknowledgments
This project is based on the tutorial and documentation of TensorFlow and Keras for implementing Variational Autoencoders.
