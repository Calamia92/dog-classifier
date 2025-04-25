import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# --- Générateur Keras avec paramètres d'augmentation ---
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

def random_contrast(image, contrast_range=(0.5, 3.0)):
    contrast_factor = np.random.uniform(*contrast_range)
    return tf.image.adjust_contrast(image, contrast_factor)

# --- Générateur personnalisé ---
def load_and_augment_images(image_paths, labels, batch_size, target_size, processed_images_dir):
    i = 0
    while True:
        batch_paths = image_paths[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        images = []
        for path in batch_paths:
            img_path = os.path.join(processed_images_dir, path)
            if os.path.exists(img_path):
                img = tf.keras.utils.load_img(img_path, target_size=target_size)
                img_array = tf.keras.utils.img_to_array(img)
                img_array = random_contrast(img_array)
                images.append(img_array)
            else:
                print(f"⚠️ Image non trouvée : {img_path}")
        images = np.array(images)
        labels_batch = np.array(batch_labels)
        augmented = datagen.flow(images, labels_batch, batch_size=batch_size, shuffle=False)
        yield next(augmented)
        i = (i + batch_size) % len(image_paths)

# --- Affichage d'exemples ---
def display_augmented_images(generator, num_images=4):
    plt.figure(figsize=(12, 3))
    for i in range(num_images):
        img_batch, _ = next(generator)
        img = np.clip(img_batch[0], 0, 1)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.suptitle("Exemples d'images augmentées", fontsize=16)
    plt.show()
