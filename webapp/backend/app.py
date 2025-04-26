from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

app = FastAPI()

# Charger le modèle en mémoire
model = load_model("models/best_model.h5")
class_indices = {0: "class1", 1: "class2", 2: "class3"}  # Remplacez par vos classes réelles

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Charger l'image
    image = load_img(file.file, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0

    # Effectuer la prédiction
    predictions = model.predict(image)[0]
    top_indices = predictions.argsort()[-3:][::-1]
    top_classes = [{"class": class_indices[i], "score": float(predictions[i])} for i in top_indices]

    return {"predictions": top_classes}
