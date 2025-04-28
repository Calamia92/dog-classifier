# 🐶 Dog Breed Classification with Deep Learning

Projet de classification de races de chiens utilisant un CNN from scratch et du Transfer Learning.

## 👨‍💻 Collaborateurs

- **Bouba** (Lead Developer & Data Preprocessing)
- **Théo** (Model Architect & Experimentation)
- **Hicham** (DevOps & Web Integration)

---

## 📂 Structure du projet

```
├── data/
│   ├── processed/          # Images redimensionnées
│   └── raw/                # Images brutes originales (.mat inclus)
├── models/                 # Modèles entraînés (.h5 CNN et Transfer Learning)
├── notebooks/              # Notebooks Jupyter (EDA, Modèles)
│   ├── cnn_scratch.ipynb
│   ├── transfer_learning.ipynb
│   ├── eda_images.ipynb
│   └── eda_classes.ipynb
├── presentation/           # Slides de présentation
├── scripts/                # Scripts Python (processing, training, augmentation)
│   ├── augmentation_data.py
│   ├── download_data.py
│   ├── infer.py
│   ├── models.py
│   ├── preprocess.py
│   ├── test_app.py
│   └── train.py
├── webapp/
│   ├── backend/
│   │   └── app.py          # FastAPI backend
│   └── frontend/
│       └── frontend_app.py # Streamlit frontend
├── requirements.txt        # Dépendances principales
└── README.md               # Ce fichier
```

---

## ⚙️ Installation

1. **Cloner le dépôt :**
```bash
git clone https://github.com/Calamia92/dog-classifier.git
cd dog-classifier
```

2. **Créer et activer un environnement virtuel :**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

3. **Installer les dépendances :**
```bash
pip install -r requirements.txt -r dev-requirements.txt
```

---

## 🧠 Deep Learning & Traitement d'images

Technos principales utilisées :
- TensorFlow / Keras
- NumPy / Pandas
- Matplotlib
- Pillow (PIL)
- Albumentations
- OpenCV
- FastAPI
- Streamlit

---

## 🚀 Usage

### 1. Prétraitement des images
```bash
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed --image_size 224
```

---

### 2. Entraînement du modèle

**a. CNN from scratch**
```bash
python scripts/train.py --model scratch --batch_size 32 --epochs 10
```

**b. Transfer Learning (VGG16)**
```bash
python scripts/train.py --model transfer --batch_size 32 --epochs 10
```

---

## 🌐 Lancement de l'application Web

### 1. Lancer le Backend FastAPI
Depuis la racine :
```bash
uvicorn webapp.backend.app:app --reload
```
Backend disponible sur : http://localhost:8000

### 2. Lancer le Frontend Streamlit
Depuis la racine :
```bash
streamlit run webapp/frontend/frontend_app.py
```
Frontend disponible sur : http://localhost:8501

---

## 📚 API Documentation

- **GET /** → Test serveur (Hello World)
- **POST /predict** → Envoyer une image et obtenir les 3 meilleures prédictions

Exemple réponse JSON :
```json
{
  "predictions": [
    {"class": "Labrador_retriever", "score": 0.85},
    {"class": "Golden_retriever", "score": 0.10},
    {"class": "Beagle", "score": 0.05}
  ]
}
```

---

## ✅ Tests

Pour lancer les tests unitaires :
```bash
pytest scripts/test_app.py
```

---

## 📦 Dépendances principales (`requirements.txt`)

```text
numpy
pandas
tensorflow
keras
scikit-learn
matplotlib
Pillow
opencv-python
albumentations
fastapi
uvicorn
streamlit
python-multipart
```

---

## 📝 Licence

Projet sous licence **MIT**.