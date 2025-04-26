import os
import urllib.request
import tarfile

DATA_DIR = "data/raw"
URL_IMAGES = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
URL_LISTS = "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"

os.makedirs(DATA_DIR, exist_ok=True)

def download_file(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"📥 Téléchargement de {url}...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"✅ Téléchargé : {dest_path}")
    else:
        print(f"✅ Fichier déjà présent : {dest_path}")

def extract_tar(file_path, extract_to):
    print(f"📦 Extraction de {file_path}...")
    with tarfile.open(file_path) as tar:
        tar.extractall(path=extract_to)
    print(f"✅ Extraction terminée : {extract_to}")

# Télécharger les fichiers
download_file(URL_IMAGES, os.path.join(DATA_DIR, "images.tar"))
download_file(URL_LISTS, os.path.join(DATA_DIR, "lists.tar"))

# Extraire les fichiers
extract_tar(os.path.join(DATA_DIR, "images.tar"), DATA_DIR)
extract_tar(os.path.join(DATA_DIR, "lists.tar"), DATA_DIR)
