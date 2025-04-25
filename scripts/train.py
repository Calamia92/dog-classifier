import argparse
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models import build_cnn_scratch, build_transfer_model
from scipy.io import loadmat
import pandas as pd

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['scratch', 'transfer'], required=True)
parser.add_argument('--custom_aug', action='store_true')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--train_mat', type=str, default='data/raw/train_list.mat')
parser.add_argument('--val_mat', type=str, default='data/raw/test_list.mat')
parser.add_argument('--data_dir', type=str, default='data/processed/images')
parser.add_argument('--output', type=str, default='models/model.keras')
args = parser.parse_args()

# --- Fonction pour charger les chemins des fichiers à partir des fichiers .mat ---
def load_file_list(mat_path):
    print(f"⚙️ Chargement du fichier .mat depuis : {mat_path}")
    mat = loadmat(mat_path)
    return [item[0][0] for item in mat["file_list"]]

# --- Chargement des chemins d'images à partir des fichiers .mat ---
print("🔄 Chargement des chemins d'images de l'entraînement et de la validation...")
train_list = load_file_list(args.train_mat)
val_list = load_file_list(args.val_mat)

# Fonction pour extraire les labels des chemins
def get_label_from_path(path):
    return path.split('/')[0]  # Utilise '/' pour extraire le dossier parent comme label

# --- Générateurs de données avec augmentation ---
print("🔄 Initialisation des générateurs de données avec augmentation...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20 if not args.custom_aug else 0,
    width_shift_range=0.2 if not args.custom_aug else 0,
    height_shift_range=0.2 if not args.custom_aug else 0,
    zoom_range=0.2 if not args.custom_aug else 0,
    horizontal_flip=True if not args.custom_aug else False,
    brightness_range=[0.8, 1.2] if not args.custom_aug else None,
    validation_split=0.2  # 20% pour la validation
)

val_datagen = ImageDataGenerator(rescale=1./255)

# --- Récupérer les chemins absolus des images pour train et validation ---
def get_image_paths(file_list):
    return [os.path.join(args.data_dir, path) for path in file_list]

train_paths = get_image_paths(train_list)
val_paths = get_image_paths(val_list)

train_labels = [get_label_from_path(p) for p in train_list]
val_labels = [get_label_from_path(p) for p in val_list]

train_df = pd.DataFrame({'filename': train_paths, 'class': train_labels})
val_df = pd.DataFrame({'filename': val_paths, 'class': val_labels})

# --- Générateur pour l'entraînement ---
print("🔄 Initialisation du générateur d'entraînement...")
train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,  # Pass the DataFrame
    x_col='filename',    # Column with image paths
    y_col='class',       # Column with labels
    target_size=(args.img_size, args.img_size),
    batch_size=args.batch_size,
    class_mode='categorical',
)

# Ce message est celui qui est affiché
print(f"Nombre d'images et de classes trouvées pour l'entraînement : {train_gen.samples}, {len(train_gen.class_indices)}")

# Imprime les classes détectées dans les sous-dossiers
print(f"Classes détectées : {train_gen.class_indices}")

# --- Générateur pour la validation ---
print("🔄 Initialisation du générateur de validation...")
val_gen = val_datagen.flow_from_dataframe(
    dataframe=val_df,    # Pass the DataFrame
    x_col='filename',    # Column with image paths
    y_col='class',       # Column with labels
    target_size=(args.img_size, args.img_size),
    batch_size=args.batch_size,
    class_mode='categorical',
    shuffle=False,
)

# --- Choix du modèle ---
print(f"🧠 Construction du modèle '{args.model}'...")
input_shape = (args.img_size, args.img_size, 3)
if args.model == 'scratch':
    model = build_cnn_scratch(len(train_gen.class_indices), len(val_gen.class_indices))
else:
    model = build_transfer_model(input_shape, len(train_gen.class_indices))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Checkpoint ---
os.makedirs(os.path.dirname(args.output), exist_ok=True)
checkpoint_cb = ModelCheckpoint(args.output, save_best_only=True, monitor='val_accuracy', mode='max')

# --- Entraînement ---
print(f"🚀 Démarrage de l'entraînement du modèle pour {args.epochs} epochs...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=args.epochs,
    steps_per_epoch=train_gen.samples // args.batch_size,
    validation_steps=val_gen.samples // args.batch_size,
    callbacks=[checkpoint_cb]
)

print(f"✅ Modèle sauvegardé dans {args.output}")