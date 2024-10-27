# utils/load_and_preprocess.py
from tensorflow.keras.datasets import mnist

def load_and_preprocess_mnist():
    """Load and preprocess the MNIST dataset.

    Returns:
        train_images: Preprocessed training images.
    """
    (train_images, _), (_, _) = mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5

    return train_images
