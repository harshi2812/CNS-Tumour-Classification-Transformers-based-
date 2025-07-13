import zipfile
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

zip_file_path = '/content/BTP-20250111T090356Z-001.zip'
extract_folder = '/content/BTP'

os.makedirs(extract_folder, exist_ok=True)
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

extracted_folder = os.path.join(extract_folder, 'BTP')

image_files = [f for f in os.listdir(extracted_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

images = []
for image_file in image_files:
    image_path = os.path.join(extracted_folder, image_file)
    image = Image.open(image_path)
    image = image.resize((64, 64))
    image = np.array(image) / 127.5 - 1
    
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    
    images.append(image)

images = np.array(images)

plt.imshow(images[0])
plt.axis('off')
plt.show()

def build_generator(latent_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(128 * 8 * 8, input_dim=latent_dim))
    model.add(layers.Reshape((8, 8, 128)))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.ReLU())
    
    model.add(layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.ReLU())
    
    model.add(layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.ReLU())
    
    model.add(layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh'))
    return model

def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(64, 64, 3)))
    model.add(layers.Conv2D(64, kernel_size=4, strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    
    model.add(layers.Conv2D(128, kernel_size=4, strides=2, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(0.2))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential([generator, discriminator])
    return model

def train_gan(generator, discriminator, gan, images, latent_dim, epochs=10000, batch_size=32):
    half_batch = batch_size // 2
    for epoch in range(epochs):
        idx = np.random.randint(0, images.shape[0], half_batch)
        real_images = images[idx]
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_images = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(real_images, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss[0]}")
            if epoch % 5000 == 0:
                generate_and_save_images(generator, epoch, latent_dim)

def generate_and_save_images(generator, epoch, latent_dim):
    noise = np.random.normal(0, 1, (16, latent_dim))
    generated_images = generator.predict(noise)
    
    fig, axs = plt.subplots(4, 4)
    cnt = 0
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(generated_images[cnt])
            axs[i, j].axis('off')
            cnt += 1
    plt.savefig(f"gan_generated_image_epoch_{epoch}.png")
    plt.close()

latent_dim = 100

generator = build_generator(latent_dim)
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gan.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_gan(generator, discriminator, gan, images, latent_dim, epochs=10000, batch_size=32)