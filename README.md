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

In a DCGAN setup:
- The **generator** tries to produce realistic digit images from random noise.
- The **discriminator** attempts to distinguish real images from those generated by the generator.

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

1. **Data Loading and Preprocessing**: Loads the MNIST dataset and applies transformations needed for training.
2. **Model Initialization**: Builds the generator and discriminator models, initializing them with the settings specified in the config file.
3. **Training Loop**: Implements a loop where:
   - The discriminator learns to distinguish real images from those generated by the generator.
   - The generator learns to create images that are increasingly challenging for the discriminator to classify as fake.
4. **Saving and Logging**: Logs model performance, losses, and generates sample images periodically for visual monitoring.

## Logging and Monitoring

The training process can be monitored in several ways:
- **Logs**: Training metrics and results are saved in `logs/training.log` for review.
- **Image Generation**: Intermediate images generated by the generator are saved at intervals defined in the configuration file, allowing you to track visual improvements in the generated images.
- **Weights & Biases (Optional)**: If enabled, this integration provides detailed insights into model metrics, images, and model performance across different runs.

## Model Architecture

This DCGAN model consists of two core components:

1. **Generator**: Takes in random noise and produces synthetic images that mimic the style of MNIST digits.
2. **Discriminator**: Distinguishes between real MNIST images and those produced by the generator, gradually improving through feedback from training.

The generator and discriminator are trained together in an adversarial fashion, where each network continually improves in response to the other. This leads to a generator that can eventually produce images that closely resemble those from the MNIST dataset.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
