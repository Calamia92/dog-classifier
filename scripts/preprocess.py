import sys
import os
import argparse
import tarfile
import tempfile
import shutil
import random
import logging

try:
    from PIL import Image
except ImportError:
    logging.error("Module PIL introuvable. Veuillez installer pillow : pip install pillow")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def preprocess_images(input_dir, output_dir, image_size, crop_type):
    total = 0
    # Parcours récursif de tous les dossiers de classes et sous-classes
    for root, dirs, files in os.walk(input_dir):
        # Déterminer le chemin relatif pour la sortie
        rel_path = os.path.relpath(root, input_dir)
        class_out = os.path.join(output_dir, rel_path)
        os.makedirs(class_out, exist_ok=True)
        count = 0
        for fname in files:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            src_path = os.path.join(root, fname)
            dst_path = os.path.join(class_out, fname)
            try:
                img = Image.open(src_path).convert('RGB')
                # --- Crop selon type ---
                if crop_type in ('center', 'random'):
                    w, h = img.size
                    m = min(w, h)
                    if crop_type == 'center':
                        left = (w - m) // 2
                        top = (h - m) // 2
                    else:
                        left = random.randint(0, w - m)
                        top = random.randint(0, h - m)
                    img = img.crop((left, top, left + m, top + m))
                # --- Resize ---
                # Compatibilité Pillow >=10
                try:
                    resample = Image.Resampling.LANCZOS
                except AttributeError:
                    resample = Image.LANCZOS
                img = img.resize((image_size, image_size), resample)
                img.save(dst_path)
                count += 1
                total += 1
            except Exception as e:
                logging.error("Erreur traitement %s : %s", src_path, e)
        if count > 0:
            logging.info(f"{count} images traitées dans '{rel_path}'")
    logging.info(f"Total images traitées : {total}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prétraitement des images de chiens")
    parser.add_argument('--input_dir', required=True, help="Répertoire source ou .tar")
    parser.add_argument('--output_dir', required=True, help="Répertoire de sortie")
    parser.add_argument('--image_size', type=int, default=224, help="Taille carrée des images")
    parser.add_argument(
        '--crop_type',
        choices=['none', 'center', 'random'],
        default='none',
        help="Type de crop avant resize"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.info(
        f"Prétraitement: input={args.input_dir}, output={args.output_dir}, "
        f"size={args.image_size}, crop={args.crop_type}"
    )
    if not os.path.exists(args.input_dir):
        logging.error("Répertoire d'entrée introuvable : %s", args.input_dir)
        sys.exit(1)

    temp_dir = None
    input_path = args.input_dir

    # Si input_path est un dossier sans sous‑dossier mais contenant un .tar, on l'extrait
    if os.path.isdir(input_path):
        subs = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
        tars = [f for f in os.listdir(input_path) if f.lower().endswith('.tar')]
        if not subs and tars:
            tar_path = os.path.join(input_path, tars[0])
            logging.info("Extraction automatique du tar : %s", tar_path)
            temp_dir = tempfile.mkdtemp()
            with tarfile.open(tar_path) as tar:
                tar.extractall(temp_dir)
            input_path = temp_dir

    # --- Si on reçoit directement un .tar ---
    if os.path.isfile(input_path) and input_path.lower().endswith('.tar'):
        temp_dir = tempfile.mkdtemp()
        with tarfile.open(input_path) as tar:
            tar.extractall(temp_dir)
        input_path = temp_dir

    os.makedirs(args.output_dir, exist_ok=True)
    if not any(os.path.isdir(os.path.join(input_path, d)) for d in os.listdir(input_path)):
        logging.warning("Aucune sous-classe trouvée dans %s", input_path)

    preprocess_images(input_path, args.output_dir, args.image_size, args.crop_type)

    # --- Nettoyage du dossier temporaire ---
    if temp_dir:
        shutil.rmtree(temp_dir)
    logging.info("Prétraitement terminé")


if __name__ == '__main__':
    main()
