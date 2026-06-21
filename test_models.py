import os
import csv
import time
import numpy as np
import torch
import coremltools as ct
from transformers import SegformerImageProcessor
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex

from src.dataset import CamVidDataset  
from src.model import get_segformer_model

# Force l'utilisation du GPU/MPS pour la partie PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def sync_device(device):
    if device.type == "mps":
        torch.mps.synchronize()

# ==========================================
# 1. CONFIGURATION ET CHEMINS
# ==========================================
NUM_CLASSES = 32
CHECKPOINT = "nvidia/mit-b3"

PATH_MODEL_PYTORCH = "./../CarSegm/model"  # Modèle original (Float32)
PATH_MODEL_COREML  = "./../CarSegm/model_segformer_fp16.mlpackage" # Modèle réduit CoreML
OUTPUT_CSV = "test_models.csv"  

PATH_TEST = "./../CarSegm/CamVid"
PATH_TO_CSV  = os.path.join(PATH_TEST, "class_dict.csv")
PATH_TEST_IMG = os.path.join(PATH_TEST, "test")
PATH_TEST_MSK = os.path.join(PATH_TEST, "test_labels")

# ==========================================
# 2. PRÉPARATION DES DONNÉES
# ==========================================
print("Chargement du jeu de données Test...")
processor = SegformerImageProcessor.from_pretrained(CHECKPOINT)

test_dataset = CamVidDataset(
    images_dir=PATH_TEST_IMG, masks_dir=PATH_TEST_MSK, csv_path=PATH_TO_CSV, processor=processor, is_train=False  
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch_size=1 obligatoire pour CoreML

# ==========================================
# 3. CHARGEMENT DES DEUX MODÈLES
# ==========================================
print("Initialisation du Modèle 1 (Original PyTorch Float32)...")
model_pytorch = get_segformer_model(checkpoint=PATH_MODEL_PYTORCH, num_classes=NUM_CLASSES)
model_pytorch.to(DEVICE)
model_pytorch.eval()

print("Initialisation du Modèle 2 (Réduit CoreML FP16)...")
# Note : Ce premier chargement va compiler le modèle pour le Neural Engine (peut prendre 1-2 min)
model_coreml = ct.models.MLModel(PATH_MODEL_COREML)

# Initialisation de la métrique globale mIoU (en ignorant la classe Void 255)
miou_metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average='macro', ignore_index=255).to(DEVICE)

# ==========================================
# 4. ÉVALUATION DU MODÈLE ORIGINAL (PYTORCH)
# ==========================================
print("\n" + "="*50)
print("⏱️ Évaluation du Modèle Original PyTorch (Float32)...")
miou_metric.reset()

sync_device(DEVICE)
start_time_1 = time.time()

with torch.no_grad():
    for batch in test_loader:
        if isinstance(batch, dict):
            images = batch.get("pixel_values").to(DEVICE)
            masks = batch.get("labels").to(DEVICE).long()
        else:
            images, masks = batch[0].to(DEVICE), batch[1].to(DEVICE).long()
            
        outputs = model_pytorch(images)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # Redimensionnement des logits à la taille du masque
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=masks.shape[1:], mode='bilinear', align_corners=False
        )
        preds = torch.argmax(upsampled_logits, dim=1)
        miou_metric.update(preds, masks)

sync_device(DEVICE)
duration_1 = time.time() - start_time_1
miou_1 = miou_metric.compute().item()
print(f"✅ Terminé | mIoU: {miou_1*100:.2f}% | Temps: {duration_1:.2f}s")

# ==========================================
# 5. ÉVALUATION DU MODÈLE RÉDUIT (COREML FP16)
# ==========================================
print("\n" + "="*50)
print("⏱️ Évaluation du Modèle Réduit CoreML (Float16)...")
miou_metric.reset()

start_time_2 = time.time()

for batch in test_loader:
    if isinstance(batch, dict):
        images_torch = batch.get("pixel_values")
        masks_torch = batch.get("labels").to(DEVICE).long()
    else:
        images_torch, masks_torch = batch[0], batch[1].to(DEVICE).long()
        
    # Extraction et conversion du tenseur d'entrée au format NumPy pour CoreML
    np_images = images_torch.cpu().numpy()
    
    # Inférence ultra-rapide exécutée sur le Neural Engine
    coreml_inputs = {"pixel_values": np_images}
    predictions = model_coreml.predict(coreml_inputs)
    
    # Récupération dynamique du tenseur de sortie des logits (forme: 1, 32, 128, 128)
    output_key = list(predictions.keys())[0]
    logits_np = predictions[output_key]
    
    # Transfert des logits NumPy vers un tenseur PyTorch pour centraliser le traitement
    logits_torch = torch.from_numpy(logits_np).to(DEVICE)
    
    # Redimensionnement des logits à la taille originale du masque (128x128 -> 512x512)
    upsampled_logits = torch.nn.functional.interpolate(
        logits_torch, size=masks_torch.shape[1:], mode='bilinear', align_corners=False
    )
    preds = torch.argmax(upsampled_logits, dim=1)
    miou_metric.update(preds, masks_torch)

duration_2 = time.time() - start_time_2
miou_2 = miou_metric.compute().item()
print(f"✅ Terminez | mIoU: {miou_2*100:.2f}% | Temps: {duration_2:.2f}s")

# ==========================================
# 6. ENREGISTREMENT DANS LE FICHIER CSV
# ==========================================
print(f"\n✍️ Enregistrement des résultats dans {OUTPUT_CSV}...")

headers = ["Metric", "Framework", "Precision", "Score_Raw", "Score_Percentage", "Inference_Time_Seconds"]

rows = [
    ["mIoU", "PyTorch (Original)", "Float32", f"{miou_1:.4f}", f"{miou_1 * 100:.2f}%", f"{duration_1:.2f}"],
    ["mIoU", "CoreML (Réduit)", "Float16", f"{miou_2:.4f}", f"{miou_2 * 100:.2f}%", f"{duration_2:.2f}"]
]

with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(headers)
    writer.writerows(rows)

print("🏆 Banc d'essai CoreML finalisé avec succès !")