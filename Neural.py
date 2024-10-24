import tensorflow as tf
from tensorflow import keras
import argparse
from Engine.Discriminator import Discriminator
from Engine.Generator import Generator
from Engine.StyleGAN import StyleGAN
from Tools.LoadImage import LoadImage

# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Train a StyleGAN model with custom parameters.')
    parser.add_argument('--min_level', type=int, default=2, help='Minimum network level to start training.')
    parser.add_argument('--max_level', type=int, default=5, help='Maximum network level to train.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--steps_per_epoch', type=int, default=16, help='Number of steps per epoch.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training.')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory to save models.')
    parser.add_argument('--model_name', type=str, default='model', help='Model name for saving/loading.')
    return parser.parse_args()

# Define optimizers
generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

# Define loss functions
def discriminator_loss(real_img, fake_img):
    """
    Computes the discriminator loss as the difference between real and fake images.
    """
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss

def generator_loss(fake_img):
    """
    Computes the generator loss based on fake images.
    """
    return -tf.reduce_mean(fake_img)

# Main function for training
def main():
    args = parse_args()

    discriminator_instance = Discriminator()
    generator_instance = Generator()
    image_instance = LoadImage()
    image_training = image_instance.get_dataset_image()

    # Train over the range of levels
    for level in range(args.min_level, args.max_level):
        generator_model = generator_instance.get_generator(level)
        discriminator_model = discriminator_instance.get_discriminator(level)

        # Create StyleGAN instance and compile the model
        styleGan = StyleGAN(discriminator=discriminator_model, 
                            generator=generator_model, 
                            number_discriminator_steps=2, 
                            network_level=level)
        styleGan.compile(discriminator_optimizer=discriminator_optimizer, 
                         generator_optimizer=generator_optimizer, 
                         generator_loss=generator_loss, 
                         discriminator_loss=discriminator_loss)

        # Train the model
        styleGan.fit(image_training, 
                     batch_size=args.batch_size, 
                     steps_per_epoch=args.steps_per_epoch, 
                     epochs=args.epochs)

        # Save and load the models
        discriminator_instance.save_discriminator(args.save_dir, args.model_name)
        discriminator_instance.load_discriminator(args.save_dir, args.model_name)
        generator_instance.save_generator(args.save_dir, args.model_name)
        generator_instance.load_generator(args.save_dir, args.model_name)

        # Generate images using the trained model
        styleGan.generate_images()

if __name__ == "__main__":
    main()
