import os
import csv
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A

class CamVidDataset(Dataset):
    """
    Dataset personnalisé qui gère le chargement des paires d'images et de masques, la conversion 
    des masques de couleur RGB vers des indices de classes utilisables par les fonctions 
    de perte, et applique la data augmentation.
    """
    
    def __init__(self, images_dir, masks_dir, csv_path, processor, is_train=True):
    
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.processor = processor
        self.images = sorted(os.listdir(images_dir))
        self.color_to_class = self._load_color_mapping(csv_path)
        self.is_train = is_train

        # Pipeline d'ENTRAÎNEMENT 
        self.train_transform = A.Compose([ 
            A.Resize(height=512, width=512),
            A.HorizontalFlip(p=0.5), 
            
            # Changements de lumière 
            A.OneOf([
                A.RandomBrightnessContrast(p=1.0),
                A.ColorJitter(p=1.0),
                A.RandomShadow(p=1.0),
            ], p=0.6),
            
            # Flou ou bruit pour empêcher le surapprentissage 
            A.OneOf([
                A.GaussianBlur(p=1.0),
                A.GaussNoise(p=1.0),
            ], p=0.3),
        ])

        self.val_transform = A.Compose([
            A.Resize(height=512, width=512), 
        ])

    

    def _load_color_mapping(self, csv_path):
        """
        Crée un dictionnaire de correspondance entre les couleurs RGB et les indices de classes.
        """
        mapping = {}
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                mapping[(int(row['r']), int(row['g']), int(row['b']))] = idx
        return mapping

    def _rgb_to_class_indices(self, mask_rgb_array):
        """
        Convertit un masque RGB en indices de classes (0 à 31) en utilisant le dictionnaire de correspondance.
        """
        h, w, _ = mask_rgb_array.shape
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
        
        # Charger l'image et le masque
        image = np.array(Image.open(os.path.join(self.images_dir, image_name)).convert("RGB"))
        mask_rgb = np.array(Image.open(os.path.join(self.masks_dir, mask_name)).convert("RGB"))
        
        # Conversion des couleurs en indices (0 à 31)
        mask_indices = self._rgb_to_class_indices(mask_rgb)

        # Application de la data augmentation
        if self.is_train:
            augmented = self.train_transform(image=image, mask=mask_indices)
        else:
            augmented = self.val_transform(image=image, mask=mask_indices)
            
        image = augmented['image']
        mask_indices = augmented['mask']

        
        inputs = self.processor(
            images=image, 
            segmentation_maps=mask_indices, 
            return_tensors="pt",
            do_reduce_labels=False,      
        )
        
        # Suppression de la dimension de batch parasite (1, C, H, W) -> (C, H, W)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        return inputs