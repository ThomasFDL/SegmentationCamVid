import os
import csv
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A

class CamVidDataset(Dataset):
    def __init__(self, images_dir, masks_dir, csv_path, processor, is_train=True):
        """
        Dataset CamVid avec support de la Data Augmentation pour l'entraînement.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.processor = processor
        self.images = sorted(os.listdir(images_dir))
        self.color_to_class = self._load_color_mapping(csv_path)
        self.is_train = is_train

        # Définition du pipeline de Data Augmentation (appliqué uniquement si is_train=True)
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),                  
            A.RandomBrightnessContrast(p=0.3),         # Ajuste la luminosité
            A.ShiftScaleRotate(
                shift_limit=0.05, 
                scale_limit=0.05, 
                rotate_limit=10, 
                p=0.4, 
                border_mode=0,                         # Remplit les pixels hors de l'image
                cval=(0, 0, 0),       
                mask_cval=255         
            ), 
        ])

    def _load_color_mapping(self, csv_path):
        """
        Charge le mapping des couleurs RGB vers les indices de classes depuis un fichier CSV.
        """
        mapping = {}
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                mapping[(int(row['r']), int(row['g']), int(row['b']))] = idx
        return mapping


    def _rgb_to_class_indices(self, mask_rgb_array):
        """
        Convertit un masque RGB en indices de classes.
        """
        h, w, _ = mask_rgb_array.shape
        # 🛡️ FIX 1 : On initialise avec 255 (ignore_index) au lieu de 0 
        # pour éviter de polluer la première classe avec les pixels inconnus ou les bordures.
        class_mask = np.full((h, w), 255, dtype=np.int64)
        
        for color, class_idx in self.color_to_class.items():
            match = (mask_rgb_array == color).all(axis=-1)
            class_mask[match] = class_idx
        return class_mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        filename, extension = os.path.splitext(image_name)
        mask_name = f"{filename}_L{extension}"
        
        # Charger l'image et le masque sous forme de tableaux NumPy
        image = np.array(Image.open(os.path.join(self.images_dir, image_name)).convert("RGB"))
        mask_rgb = np.array(Image.open(os.path.join(self.masks_dir, mask_name)).convert("RGB"))
        
        # Traduction des couleurs en indices numériques (0 à 31)
        mask_indices = self._rgb_to_class_indices(mask_rgb)

        # Application des transformations si on est sur le dataset de Train
        if self.is_train:
            augmented = self.transform(image=image, mask=mask_indices)
            image = augmented['image']
            mask_indices = augmented['mask']

        # 🛡️ FIX 2 : On force explicitement Hugging Face à conserver la taille originale du masque
        # sans appliquer de réduction automatique des labels (do_reduce_labels=False)
        inputs = self.processor(
            images=image, 
            segmentation_maps=mask_indices, 
            return_tensors="pt",
            do_reduce_labels=False #Permet de conserver les labels originaux sans réduction automatique
        )
        
        # Suppression de la dimension de batch générée par le processeur (1, C, H, W) -> (C, H, W)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        return inputs