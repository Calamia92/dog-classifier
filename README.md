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
```
## Usage

### 1. Prétraitement des images
```bash
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed --size 224 --crop centered
```

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

## Licence

Ce projet est sous licence MIT.

---

