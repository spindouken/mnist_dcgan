#!/usr/bin/env python3
import tensorflow as tf
from tqdm import tqdm
import pytz
import os
import yaml
import logging
from utils.load_and_preprocess import load_and_preprocess_mnist
from utils.generateTrainingImages import generateTrainingImages
from models.baselineDCGAN import build_generator as default_build_generator, build_discriminator as default_build_discriminator


# load yaml file
with open('./configs/DCGANconfig.yaml', 'r') as f:
    config = yaml.safe_load(f)

# extract yaml configurations
BATCH_SIZE = config['hyperparameters']['batch_size']
EPOCHS = config['hyperparameters']['epochs']
NOISE_DIM = config['hyperparameters']['noise_dim']
BUFFER_SIZE = config['data']['buffer_size']
USE_WANDB = config['settings']['use_wandb']
IMGExampleCount = config['settings']['IMGExampleCount']
IMAGE_SAVE_INTERVAL = config['logging']['image_save_interval']


log_directory = config['experiment']['log_directory']
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
logging.basicConfig(
    filename=os.path.join(log_directory, 'training.log'),
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# hyperparameters and more settings from yaml
generator_learningRate = config['hyperparameters']['generator_learningRate']
discriminator_learningRate = config['hyperparameters']['discriminator_learningRate']
generator_optimizer = getattr(tf.keras.optimizers, config['hyperparameters']['generatorOptimizer'])(generator_learningRate)
discriminator_optimizer = getattr(tf.keras.optimizers, config['hyperparameters']['discriminatorOptimizer'])(discriminator_learningRate)
patience = config['hyperparameters']['patience']
experiment_type = config['experiment']['experiment_type']


# depending on the experiment type, use different model architectures
if experiment_type == 'baseline':
    build_generator = default_build_generator
    build_discriminator = default_build_discriminator
elif experiment_type == 'experiment1':
    from models.experiment1DCGAN import build_generator, build_discriminator
else:
    raise ValueError(f"Invalid experiment type: {experiment_type}")

# initialize wandb if enabled
if USE_WANDB:
    import wandb
    wandb.init(project='mnist_dcgan', name=f'dcgan_run_{experiment_type}', config=config)

# load and preprocess
train_images = load_and_preprocess_mnist()
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

generator = build_generator()
discriminator = build_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# loop for training steps
def train_step(images, step):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
   
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output), fake_output)
        
    # accuracies
    real_accuracy = tf.reduce_mean(tf.cast(tf.math.greater_equal(real_output, 0.5), tf.float32))
    fake_accuracy = tf.reduce_mean(tf.cast(tf.math.less(fake_output, 0.5), tf.float32))
    disc_accuracy = (real_accuracy + fake_accuracy) / 2.0
    gen_accuracy = tf.reduce_mean(tf.cast(tf.math.greater_equal(fake_output, 0.5), tf.float32))
    
    # gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss, gen_accuracy, disc_accuracy

# create a random noise vector for image generation
test_input = tf.random.normal([IMGExampleCount, NOISE_DIM])

# loss and accuracy metrics
cumulative_gen_loss = 0.0
cumulative_disc_loss = 0.0
cumulative_gen_accuracy = 0.0
cumulative_disc_accuracy = 0.0
num_steps = 0

# initialize tqdm with total number of epochs
pbar = tqdm(range(EPOCHS))

best_loss = float('inf')
patience_counter = 0

# training loop with tqdm and wandb
for epoch in pbar:
    cumulative_gen_loss = 0.0
    cumulative_disc_loss = 0.0
    cumulative_gen_accuracy = 0.0
    cumulative_disc_accuracy = 0.0
    num_steps = 0
    
    for step, image_batch in enumerate(train_dataset):
        # training
        gen_loss, disc_loss, gen_acc, disc_acc = train_step(image_batch, step)
        cumulative_gen_loss += gen_loss
        cumulative_disc_loss += disc_loss
        cumulative_gen_accuracy += gen_acc
        cumulative_disc_accuracy += disc_acc
        num_steps += 1

        # log metrics in wandb
        if USE_WANDB:
            wandb.log({"batch_gen_loss": gen_loss.numpy(), 
                    "batch_disc_loss": disc_loss.numpy(), 
                    "batch_gen_accuracy": gen_acc.numpy(), 
                    "batch_disc_accuracy": disc_acc.numpy()})

    avg_gen_loss = cumulative_gen_loss / num_steps
    avg_disc_loss = cumulative_disc_loss / num_steps
    avg_gen_accuracy = cumulative_gen_accuracy / num_steps
    avg_disc_accuracy = cumulative_disc_accuracy / num_steps

    # early stopping
    if avg_gen_loss < best_loss:
        best_loss = avg_gen_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter > patience:
            logging.info("Early stopping triggered.")
            print("Early stopping triggered. Training has been halted.")
            break



    logging.info(f"Experiment Type: {experiment_type}, Epoch {epoch+1}, Avg Gen Loss: {avg_gen_loss}, Avg Disc Loss: {avg_disc_loss}")
    
    # log average metrics each epoch
    if USE_WANDB:
        wandb.log({"epoch": epoch, "avg_gen_loss": avg_gen_loss, "avg_disc_loss": avg_disc_loss, "avg_gen_accuracy": avg_gen_accuracy, "avg_disc_accuracy": avg_disc_accuracy, "experiment_type": experiment_type})
    else:
        print(f"Epoch {epoch+1}, Avg Gen Loss: {avg_gen_loss}, Avg Disc Loss: {avg_disc_loss}, Avg Gen Accuracy: {avg_gen_accuracy}, Avg Disc Accuracy: {avg_disc_accuracy}")
    # image generation
    if epoch % IMAGE_SAVE_INTERVAL == 0:
        generateTrainingImages(generator, epoch, test_input)
    
    # tqdm description
    pbar.set_description(f"Epoch {epoch+1}, Avg Gen Loss: {avg_gen_loss}, Avg Disc Loss: {avg_disc_loss}")
