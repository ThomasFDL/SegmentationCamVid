import os
import csv
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A

class CamVidDataset(Dataset):
    """
    Dataset CamVid optimisé qui pré-charge et pré-convertit les masques en mémoire
    pour éliminer le goulot d'étranglement du CPU.
    """
    def __init__(self, images_dir, masks_dir, csv_path, processor, is_train=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.processor = processor
        self.images = sorted(os.listdir(images_dir))
        self.color_to_class = self._load_color_mapping(csv_path)
        self.is_train = is_train

        # Pré-chargement et pré-conversion des masques en RAM
        self.precomputed_masks = []
        for image_name in self.images:
            filename, extension = os.path.splitext(image_name)
            mask_name = f"{filename}_L{extension}"
            mask_rgb = np.array(Image.open(os.path.join(self.masks_dir, mask_name)).convert("RGB"))
            
            # Conversion vectorielle NumPy
            h, w, _ = mask_rgb.shape
            class_mask = np.full((h, w), 255, dtype=np.int64)
            for color, class_idx in self.color_to_class.items():
                match = (mask_rgb[:, :, 0] == color[0]) & (mask_rgb[:, :, 1] == color[1]) & (mask_rgb[:, :, 2] == color[2])
                class_mask[match] = class_idx
                
            self.precomputed_masks.append(class_mask)

        # Data Augmentation
        self.train_transform = A.Compose([ 
            A.HorizontalFlip(p=0.5), 
            A.OneOf([
                A.RandomBrightnessContrast(p=1.0),
                A.ColorJitter(p=1.0),
                A.RandomShadow(p=1.0),
            ], p=0.6),
            A.OneOf([
                A.GaussianBlur(p=1.0),
                A.GaussNoise(p=1.0),
            ], p=0.3),
        ])

    def _load_color_mapping(self, csv_path):
        mapping = {}
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx == 30:
                     mapping[(int(row['r']), int(row['g']), int(row['b']))] = 255
                else:    
                    mapping[(int(row['r']), int(row['g']), int(row['b']))] = idx
        return mapping

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image = np.array(Image.open(os.path.join(self.images_dir, image_name)).convert("RGB"))
        
        # Récupération du masque depuis la RAM
        mask_indices = self.precomputed_masks[idx]

        if self.is_train:
            augmented = self.train_transform(image=image, mask=mask_indices)
            image = augmented['image']
            mask_indices = augmented['mask']
      
        inputs = self.processor(
            images=image, 
            segmentation_maps=mask_indices, 
            return_tensors="pt",
            do_reduce_labels=False  
        )
        
        return {k: v.squeeze(0) for k, v in inputs.items()}
