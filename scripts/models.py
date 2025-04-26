from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2, VGG16, MobileNet
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers, models

def build_cnn_scratch(num_classes):
    # --- Définition du modèle CNN ---
    model = models.Sequential()

    # Première couche de convolution
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Deuxième couche de convolution
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Troisième couche de convolution
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Aplatir les résultats pour la couche fully connected
    model.add(layers.Flatten())

    # Couche fully connected
    model.add(layers.Dense(128, activation='relu'))

    # Couche de sortie
    model.add(layers.Dense(num_classes, activation='softmax'))  # Nombre de classes

    # --- Compilation du modèle ---
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

def build_transfer_model(base_model_name, input_shape, num_classes, unfreeze_layers=0):
    if base_model_name == "vgg16":
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    elif base_model_name == "mobilenet":
        base_model = MobileNet(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Unsupported base model. Choose 'vgg16' or 'mobilenet'.")

    # Freeze all layers initially
    base_model.trainable = False

    # Unfreeze the top N layers
    if unfreeze_layers > 0:
        for layer in base_model.layers[-unfreeze_layers:]:
            layer.trainable = True

    # Add custom head
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])

    return model
