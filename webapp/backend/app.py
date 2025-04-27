from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import json
import io

app = FastAPI()

# --- Charger le modèle ---
model = load_model("models/final_transfer_model.h5")

# --- Charger les classes ---
with open("models/class_indices.json", "r") as f:
    class_indices = json.load(f)

# Inverser le dictionnaire pour avoir {index: classe}
index_to_class = {v: k for k, v in class_indices.items()}


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Charger l'image
        contents = await file.read()  # lire le contenu du fichier envoyé
        image = load_img(io.BytesIO(contents), target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0) / 255.0

        # Faire une prédiction
        predictions = model.predict(image)[0]
        top_indices = predictions.argsort()[-3:][::-1]

        # Associer l'indice au label de classe
        top_classes = [{"class": index_to_class[i], "score": float(predictions[i])} for i in top_indices]

        return {"predictions": top_classes}

    except Exception as e:
        return {"error": str(e)}
