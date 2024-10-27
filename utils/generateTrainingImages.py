#!/usr/bin/env python3
import matplotlib.pyplot as plt
import os
import numpy as np
import yaml
from datetime import datetime
import pytz


with open("./configs/DCGANconfig.yaml", "r") as f:
    config = yaml.safe_load(f)

USE_WANDB = config["settings"]["use_wandb"]
IMGExampleCount = config["settings"]["IMGExampleCount"]
experiment_type = config["experiment"]["experiment_type"]
    

os.makedirs("./images", exist_ok=True)

# Generate timestamp and folder name
central = pytz.timezone('America/Chicago')
utc_now = datetime.now(pytz.utc)
dt = utc_now.astimezone(central)
timestamp = dt.strftime('%Y-%m-%d_%H-%M-%S')
folder_name = f"{experiment_type}_{timestamp}"
image_save_path = f"./images/{folder_name}/"

if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)


def generateTrainingImages(model, epoch, test_input):
    predictions = model(test_input, training=False)
    predictions = (predictions + 1) / 2.0  # Rescale to [0, 1]

    grid_size = int(np.sqrt(IMGExampleCount))
    fig = plt.figure(figsize=(grid_size * 2, grid_size * 2))

    for i in range(IMGExampleCount):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")

    file_name = f"image_at_epoch_{epoch:04d}.png"
    file_path = os.path.join(image_save_path, file_name)

    plt.savefig(f"{image_save_path}image_at_epoch_{epoch:04d}.png")
    plt.close(fig)

    if USE_WANDB:
        import wandb
        wandb.log({"Generated Images": [wandb.Image(file_path)]})
