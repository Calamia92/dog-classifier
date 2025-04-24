import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from scipy.io import loadmat

# --- Chemins ---
raw_data_dir = "data/raw"
processed_images_dir = "data/processed/Images"

# --- Lecture des fichiers .mat ---
train_mat = loadmat(os.path.join(raw_data_dir, "train_list.mat"))
test_mat = loadmat(os.path.join(raw_data_dir, "test_list.mat"))

train_list = [item[0][0] for item in train_mat["file_list"]]
test_list = [item[0][0] for item in test_mat["file_list"]]

print("Exemple d'image dans train:", train_list[0])

# --- Data augmentation ---
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
    image = tf.image.adjust_contrast(image, contrast_factor)
    return image

# --- Générateur avec augmentation ---
def load_and_augment_images(image_paths, batch_size=32, target_size=(224, 224)):
    while True:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            images = []
            for path in batch_paths:
                img_path = os.path.join(processed_images_dir, path)
                if os.path.exists(img_path):
                    img = tf.keras.utils.load_img(img_path, target_size=target_size)
                    img_array = tf.keras.utils.img_to_array(img)
                    img_array = random_contrast(img_array)
                    images.append(img_array)
                else:
                    print(f"Image non trouvée : {img_path}")
            images = np.array(images)
            augmented_images = datagen.flow(images, batch_size=batch_size, shuffle=False)
            yield next(augmented_images)

# --- Générateur d'entraînement ---
train_generator = load_and_augment_images(train_list)

# --- Affichage de quelques images ---
def display_augmented_images(generator, num_images=4):
    plt.figure(figsize=(12, 3))
    for i in range(num_images):
        img_batch = next(generator)
        img = np.clip(img_batch[0], 0, 1)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.suptitle("Exemples d'images augmentées", fontsize=16)
    plt.show()

def count_augmented_images(generator, total_images, batch_size=32):
    count = 0
    while count < total_images:
        img_batch = next(generator)
        count += img_batch.shape[0]
        print(f"{count} images traitées...", flush=True)
    print(f"{count} images augmentées au total.")

# --- Execution ---
count_augmented_images(train_generator, total_images=len(train_list))
display_augmented_images(train_generator, num_images=4)

print("Augmentation des images terminée.")
