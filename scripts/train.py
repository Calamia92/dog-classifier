import argparse
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models import build_cnn_scratch, build_transfer_model
from scipy.io import loadmat
import pandas as pd
from tensorflow.keras.utils import to_categorical
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

# --- Chargement fichiers .mat ---
def load_file_list(mat_path):
    print(f"⚙️ Chargement du fichier .mat depuis : {mat_path}")
    mat = loadmat(mat_path)
    return [item[0][0] for item in mat["file_list"]]

train_list = load_file_list(args.train_mat)
val_list = load_file_list(args.val_mat)

# --- Extraction des labels ---
def get_label_from_path(path):
    return path.split('/')[0]

train_labels = [get_label_from_path(p) for p in train_list]
val_labels = [get_label_from_path(p) for p in val_list]

train_paths = [os.path.join(args.data_dir, path) for path in train_list]
val_paths = [os.path.join(args.data_dir, path) for path in val_list]

train_df = pd.DataFrame({'filename': train_paths, 'class': train_labels})
val_df = pd.DataFrame({'filename': val_paths, 'class': val_labels})

# --- Générateurs de données ---
if args.custom_aug:
    print("🧪 Utilisation du générateur personnalisé avec augmentation.")
    from augmentation_data import load_and_augment_images, display_augmented_images

    class_names = sorted(set(train_labels))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_to_idx)

    train_encoded = [class_to_idx[label] for label in train_labels]
    val_encoded = [class_to_idx[label] for label in val_labels]

    train_labels_cat = to_categorical(train_encoded)
    val_labels_cat = to_categorical(val_encoded)

    train_gen = load_and_augment_images(
        train_list, train_labels_cat, args.batch_size,
        target_size=(args.img_size, args.img_size),
        processed_images_dir=args.data_dir
    )

    # Générateur sans augmentation pour la validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_gen = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filename',
        y_col='class',
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=False
    )

    print("🖼️ Un exemple d'image augmentée va s'ouvrir")
    display_augmented_images(train_gen)

else:
    print("📁 Utilisation du générateur standard sans augmentation custom.")
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filename',
        y_col='class',
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical'
    )

    val_gen = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filename',
        y_col='class',
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=False
    )

    class_to_idx = train_gen.class_indices
    num_classes = len(class_to_idx)

# --- Construction du modèle ---
print(f"🧠 Construction du modèle '{args.model}'...")
input_shape = (args.img_size, args.img_size, 3)

if args.model == 'scratch':
    model = build_cnn_scratch(num_classes)
else:
    model = build_transfer_model("vgg16", input_shape, num_classes)  # Changez "vgg16" par "mobilenet" si nécessaire

# Afficher le résumé du modèle
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Checkpoint ---
os.makedirs(os.path.dirname(args.output), exist_ok=True)
checkpoint_cb = ModelCheckpoint(args.output, save_best_only=True, monitor='val_accuracy', mode='max')

# --- Entraînement ---
print(f"🚀 Démarrage de l'entraînement pour {args.epochs} epochs...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=args.epochs,
    steps_per_epoch=len(train_list) // args.batch_size,
    validation_steps=len(val_list) // args.batch_size,
    callbacks=[checkpoint_cb]
)

print(f"✅ Modèle sauvegardé dans {args.output}")
print("✅ Entraînement terminé.")
# Évaluation
test_loss, test_acc = model.evaluate(val_gen, steps=len(val_gen))
print(f"✅ Test Accuracy: {test_acc:.4f} - Test Loss: {test_loss:.4f}")
print("📦 Modèle prêt à être utilisé pour la prédiction ou l'évaluation.")
