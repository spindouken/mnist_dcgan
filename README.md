# Deep Convolutional GAN (DCGAN) - MNIST Digits

Welcome to my recreation of the classic **DCGAN for MNIST** project! This repository contains a comprehensive setup for training a Deep Convolutional Generative Adversarial Network (DCGAN) on the MNIST dataset to generate realistic handwritten digit images.

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation and Setup](#installation-and-setup)
- [Configuration](#configuration)
- [Training](#training)
- [Logging and Monitoring](#logging-and-monitoring)
- [Model Architecture](#model-architecture)
- [License](#license)

---
## Project Overview

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** for generating synthetic images of handwritten digits. Using the MNIST dataset, which consists of grayscale 28x28 images of handwritten digits, the model learns to create images that resemble real digits by training a generator and discriminator network in a competitive framework. 

**What _is_ a DCGAN?**

A **Deep Convolutional GAN (DCGAN)** is a type of Generative Adversarial Network (GAN) that uses convolutional and deconvolutional layers to produce realistic images from random noise. A DCGAN consists of two neural networks:
- The **generator** Takes random noise as input and tries to produce realistic digit images.
- The **discriminator** Judges whether an image is real (from the dataset) or fake (produced by the generator).

The two models are trained simultaneously: the generator improves by learning to "fool" the discriminator, while the discriminator becomes better at identifying real vs. fake images. This adversarial process leads to a generator capable of producing images that closely mimic the MNIST dataset.

## Directory Structure

- **configs**: Contains configuration files that specify training parameters, model settings, and logging intervals.
- **docker**: Holds Docker-related files to set up a reproducible environment for training and deployment.
- **logs**: Stores logs related to the training process, such as loss values and training metrics.
- **models**: Contains the DCGAN model definition with layers and hyperparameters.
- **utils**: Includes helper scripts for data loading, preprocessing, and image generation during training.
- **wandb**: Contains Docker and requirements files for setting up Weights & Biases tracking, if enabled.
- **train_MNIST_DCGAN.py**: The main script that orchestrates data loading, model training, logging, and image saving.


## Installation and Setup

1. Clone the repository and navigate to the project directory.
2. Use Docker for an isolated environment by building and running the Docker image.
3. Alternatively, you can install the required dependencies from `requirements.txt` and run the project locally.

## Configuration

Configuration options are specified in the `DCGANconfig.yaml` file. Key parameters include:

- **Experiment Settings**: Specify `experiment_type` (e.g., 'baseline') and `log_directory`.
- **Hyperparameters**: Set parameters such as `batch_size`, `epochs`, `noise_dim` (dimension of the generator’s input noise vector), learning rates, optimizers, and `patience` for early stopping.
- **Data Settings**: Configure data shuffling with `buffer_size` and `auto_tune_buffer`.
- **Logging**: Define how often to save generated images and the number of example images to generate.
- **Miscellaneous**: Toggle W&B integration (`use_wandb`) and set the number of example images to save at each interval (`IMGExampleCount`).

## Training

The main training script, `train_MNIST_DCGAN.py`, orchestrates the entire GAN training process using the configurations specified in `DCGANconfig.yaml`. This script performs several tasks:

1. **Load Configurations**:
   - Loads all hyperparameters, data paths, and settings from `DCGANconfig.yaml`.
   
2. **Data Loading and Preprocessing**:
   - Loads the MNIST dataset and preprocesses it, scaling pixel values to the range -1 to 1 to match the output of the generator.

3. **Model Initialization**:
   - Builds the generator and discriminator models using the `build_generator` and `build_discriminator` functions in `baselineDCGAN.py`.

4. **Training Loop**:
   - **For each epoch**:
      - **Discriminator Training**: The discriminator is trained to classify real MNIST images as “real” and generated images from the generator as “fake.”
      - **Generator Training**: The generator is updated to improve its ability to fool the discriminator, aiming for the discriminator to classify its outputs as “real.”
      - **Image Generation**: At specified intervals, sample images are generated to visualize the generator’s progress.

5. **Logging and Saving**:
   - Logs training progress, including losses for the generator and discriminator, and saves example generated images.


## Logging and Monitoring

The training process can be monitored in several ways:
- **Logs**: Training metrics and results are saved in `logs/training.log` for review.
- **Image Generation**: Intermediate images generated by the generator are saved at intervals defined in the configuration file, allowing you to track visual improvements in the generated images.
- **Weights & Biases (Optional)**: If enabled, this integration provides detailed insights into model metrics, images, and model performance across different runs.

### Generator

The **generator** takes a random noise vector (of size 100 by default) and produces a 28x28 grayscale image that resembles the MNIST digits. Here’s an in-depth look at the generator’s structure:

1. **Dense Layer**: 
   - The generator starts with a dense layer that transforms the 100-dimensional noise vector into a large, 7x7x256 tensor, which will be upsampled in the next steps.
   
2. **Batch Normalization**: 
   - This layer helps stabilize and speed up training by standardizing outputs, helping prevent internal covariate shift.

3. **LeakyReLU Activation**: 
   - Adds non-linearity, which is critical for learning complex features.

4. **Reshape Layer**: 
   - Reshapes the tensor into a 3D shape (7x7x256), preparing it for convolutional upsampling.

5. **Transposed Convolutional Layers (Conv2DTranspose)**:
   - **First Transpose Convolution**: Upsamples the tensor to 7x7x64, retaining the spatial size but reducing feature depth.
   - **Second Transpose Convolution**: Upsamples to 14x14x32.
   - **Final Transpose Convolution**: Upsamples to 28x28x1 (grayscale image with a single channel), using a `tanh` activation function to produce output values between -1 and 1, matching the normalized pixel range of the MNIST dataset.

### Discriminator

The **discriminator** is a convolutional neural network (CNN) that takes an image (either real or generated) as input and outputs a single value, where positive values indicate “real” and negative values indicate “fake.” Its structure is as follows:

1. **Convolutional Layers (Conv2D)**:
   - Each Conv2D layer applies filters to detect patterns, shapes, and textures at different levels of detail.
   - **First Convolution**: Extracts low-level features from the 28x28 image using 32 filters with a stride of 2, which reduces the spatial size of the feature map.
   
2. **LeakyReLU Activation**:
   - Adds non-linearity to each Conv2D layer, which is essential for capturing complex image details.

3. **Dropout Layers**:
   - Randomly sets 30% of input units to zero at each update during training to prevent overfitting.

4. **Flatten Layer**:
   - Converts the 3D tensor from the last Conv2D layer into a 1D vector, preparing it for the final dense output layer.

5. **Dense Output Layer**:
   - Outputs a single value, where positive indicates “real” and negative indicates “fake.”


## License

This project is licensed under the MIT License. See the LICENSE file for more details.
