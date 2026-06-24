import os
import csv
import numpy as np
import torch
import coremltools as ct
from transformers import SegformerImageProcessor
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex

from src.dataset import CamVidDataset  
from src.model import get_segformer_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def sync_device(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

NUM_CLASSES = 32
CHECKPOINT = "nvidia/mit-b3"

OUTPUT_CSV = "test_models.csv"  

PATH_TEST = "./../CarSegm/CamVid"
PATH_TO_CSV  = os.path.join(PATH_TEST, "class_dict.csv")
PATH_TEST_IMG = os.path.join(PATH_TEST, "test")
PATH_TEST_MSK = os.path.join(PATH_TEST, "test_labels")

def get_class_names(csv_path):
    class_names = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names.append(row['name'])
    return class_names

print("Lecture des noms de classes...")
class_names_list = get_class_names(PATH_TO_CSV)

if len(class_names_list) < NUM_CLASSES:
    class_names_list += [f"Class_{i:02d}" for i in range(len(class_names_list), NUM_CLASSES)]

print("Chargement du jeu de données Test...")
processor = SegformerImageProcessor.from_pretrained(CHECKPOINT)

test_dataset = CamVidDataset(
    images_dir=PATH_TEST_IMG, masks_dir=PATH_TEST_MSK, csv_path=PATH_TO_CSV, processor=processor, is_train=False  
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) 

model_pytorch = None
model_coreml = None
framework_used = ""

if os.path.exists("./../CarSegm/model_CoreML.mlpackage"):
    try:
        print("Modèle CoreML détecté. Initialisation...")
        model_coreml = ct.models.MLModel("./../CarSegm/model_CoreML.mlpackage")
        framework_used = "CoreML"
    except Exception as e:
        print(f" Erreur lors du chargement de CoreML ({e}). Bascule sur PyTorch...")
        model_coreml = None

if model_coreml is None:
    print(" Modèle CoreML absent ou défaillant. Initialisation du modèle PyTorch...")
    model_pytorch = get_segformer_model(checkpoint="./../CarSegm/model", num_classes=NUM_CLASSES)
    model_pytorch.to(DEVICE)
    model_pytorch.eval()
    framework_used = "PyTorch (Original)"

metric_macro = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average='macro', ignore_index=255).to(DEVICE)
metric_per_class = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average='none', ignore_index=255).to(DEVICE)

print(f"\n" + "="*50)
print(f"Démarrage de l'évaluation avec : {framework_used}")
metric_macro.reset()
metric_per_class.reset()

with torch.no_grad():
    for batch in test_loader:
        if isinstance(batch, dict):
            images_torch = batch.get("pixel_values")
            masks_torch = batch.get("labels").to(DEVICE).long()
        else:
            images_torch, masks_torch = batch, batch.to(DEVICE).long()
            
        if model_coreml is not None:
            np_images = images_torch.cpu().numpy()
            coreml_inputs = {"pixel_values": np_images}
            predictions = model_coreml.predict(coreml_inputs)
            output_key = list(predictions.keys())[0]
            logits_np = predictions[output_key]
            logits_torch = torch.from_numpy(logits_np).to(DEVICE)
            
        else:
            images_gpu = images_torch.to(DEVICE)
            outputs = model_pytorch(images_gpu)
            logits_torch = outputs.logits if hasattr(outputs, 'logits') else outputs

        upsampled_logits = torch.nn.functional.interpolate(
            logits_torch, size=masks_torch.shape[1:], mode='bilinear', align_corners=False
        )
        preds = torch.argmax(upsampled_logits, dim=1)
        
        metric_macro.update(preds, masks_torch)
        metric_per_class.update(preds, masks_torch)

miou_global = metric_macro.compute().item()
iou_per_class = metric_per_class.compute().cpu().numpy()

print(f" Évaluation terminée | mIoU Global: {miou_global*100:.2f}%")

print(f"\nEnregistrement des résultats épurés dans {OUTPUT_CSV}...")

headers = ["Metric", "Framework", "Score_Raw", "Score_Percentage"]

rows = [
    ["mIoU Global", framework_used, f"{miou_global:.4f}", f"{miou_global * 100:.2f}%"],
]

for class_idx in range(NUM_CLASSES):
    val_raw = iou_per_class[class_idx]
    class_name = class_names_list[class_idx] 
    
    rows.append([
        class_name, 
        framework_used, 
        f"{val_raw:.4f}", 
        f"{val_raw * 100:.2f}%"
    ])

with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(headers)
    writer.writerows(rows)

print("Banc d'essai CamVid finalisé avec succès !")