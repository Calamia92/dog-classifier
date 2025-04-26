# Dog Breed Classification with Deep Learning

This repository contains code and resources for classifying dog breeds using both a custom CNN and transfer learning.

## Collaborators

- **Bouba** (Lead Developer & Data Preprocessing)
- **Théo** (Model Architect & Experimentation)
- **Hicham** (DevOps & Web Integration)


## Project Structure
```
├── data/
│   ├── raw/                # Images brutes téléchargées
│   └── processed/          # Images redimensionnées et augmentées
├── notebooks/              # Notebooks d'exploration et d'expérimentation
│   ├── eda.ipynb           # Analyse exploratoire des données
│   ├── cnn_scratch.ipynb   # Modèle CNN from scratch
│   └── transfer_learning.ipynb  # Transfer learning
├── scripts/                # Scripts Python réutilisables
│   ├── preprocess.py       # Redimensionnement, cropping, augmentation
│   ├── train.py            # Entraînement (scratch / transfer)
│   └── infer.py            # Inférence Top 3 classes
├── webapp/                 # Application Web pour démo
│   ├── backend/            # API Flask/FastAPI
│   └── frontend/           # HTML/CSS/JS
├── presentation/           # Slides de présentation
├── azure-pipelines.yml     # CI/CD Azure DevOps
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Installation

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/Calamia92/dog-classifier.git
   cd dog-classifier
   ```
2. **Créer et activer un environnement virtuel**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # ou `venv\\Scripts\\activate` sous Windows
   ```
3. **Installer les dépendances**
   Pour avoir tous les outils nécessaires, installez :
   - **Production** (libs pour exécution du code et modèles) :
     ```bash
     pip install -r requirements.txt
     ```
   - **Développement** (tests, linting, CI/CD) :
     ```bash
     pip install -r dev-requirements.txt
     ```

*Vous pouvez installer les deux si vous comptez contribuer au projet :*
```bash
pip install -r requirements.txt -r dev-requirements.txt
```
### Deep Learning & Data

Le projet s'appuie principalement sur les bibliothèques suivantes pour le traitement d'images, l'entraînement de modèles de Deep Learning et l'exploration des données :

- **TensorFlow + Keras** : Framework de Deep Learning choisi pour sa simplicité, sa documentation riche et son intégration avec des modèles pré-entraînés via `keras.applications`. Il permet à la fois de construire un CNN from scratch et de faire du transfert learning (VGG16, MobileNet, etc.).
- **NumPy / Pandas** : Pour la manipulation de données, les statistiques de base, le traitement de chemins de fichiers, et la gestion des labels.
- **Matplotlib / Seaborn** : Pour les visualisations exploratoires (distribution des classes, exemples d’images, courbes de loss/accuracy, etc.).
- **Pillow (PIL) / OpenCV** : Chargement, redimensionnement et manipulation des images (prétraitement de base).
- **Albumentations** : Utilisé pour la data augmentation avancée (rotations, flips, jitter, etc.), très rapide et flexible comparé aux outils intégrés de Keras.

Ces choix technologiques permettent une bonne balance entre performance, facilité d’implémentation, et lisibilité du code pour un projet pédagogique ou associatif.

## Dependencies

Dans le fichier `requirements.txt` et `dev-requirements.txt`, on liste les principales librairies Python nécessaires :

requirements
```text
numpy>=1.21
pandas>=1.3
tensorflow>=2.10
keras>=2.10
torch>=1.13          # si vous choisissez PyTorch pour partie du projet
scikit-learn>=1.0
matplotlib>=3.5
Pillow>=9.2
opencv-python>=4.6
albumentations>=1.3  # pour la data augmentation avancée
flask>=2.1           # pour l’API Web (ou FastAPI)
fastapi>=0.85
uvicorn>=0.18        # serveur ASGI pour FastAPI
python-dotenv>=0.21  # gestion des variables d’environnement
```

dev-requirements
```text
# Outils de dev et CI/CD
jupyter>=1.0
papermill>=2.5       # exécution de notebooks en CI
pytest>=7.1          # tests unitaires
flake8>=4.0
azure-devops>=7.1.0  # interactions avec Azure Pipelines
ipykernel            # pour exécuter les notebooks avec papermill
```

**Remarque :**  
Pour exécuter les notebooks avec papermill, il faut aussi installer le kernel Jupyter dans votre environnement virtuel :
```bash
pip install ipykernel
python -m ipykernel install --user --name=python3
```

## Usage

Avant tout, installez pillow si nécessaire :
```bash
pip install pillow
```

### 1. Prétraitement des images

Pour lancer le prétraitement et extraire/traiter les images depuis votre archive :

```bash
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed --image_size 224 --crop_type center
```

Après exécution, vos images redimensionnées se trouvent dans le dossier de sortie, typiquement :

Linux/macOS:
```bash
ls data/processed/Images/
```

Windows PowerShell:
```powershell
dir data\processed\Images\
```

Note : « Images » est le nom du dossier extrait depuis votre archive `images.tar`.  

### 2. Entraînement

#### a. CNN from scratch
```bash
python scripts/train.py --approach scratch --data_dir data/processed --epochs 10 --batch_size 32
```

#### b. Transfer Learning
```bash
python scripts/train.py --approach transfer --model vgg16 --data_dir data/processed --epochs 5 --batch_size 32
```

### 3. Inférence
```bash
python scripts/infer.py --model_path models/best_model.h5 --image_path path/to/image.jpg
```

## API REST

### Endpoint `/predict`

- **Méthode** : POST
- **Description** : Envoie une image pour obtenir les 3 meilleures prédictions.
- **Exemple de réponse** :
  ```json
  {
    "predictions": [
      {"class": "Labrador", "score": 0.85},
      {"class": "Golden Retriever", "score": 0.10},
      {"class": "Beagle", "score": 0.05}
    ]
  }
  ```

### Lancer l'API

```bash
uvicorn webapp.backend.app:app --reload
```

### Tests Unitaires

Pour exécuter les tests unitaires :

```bash
pytest scripts/test_app.py
```

## Tests
Pour lancer les tests unitaires :
```bash
pytest --maxfail=1 -q
```

## Licence

Ce projet est sous licence MIT.

---

