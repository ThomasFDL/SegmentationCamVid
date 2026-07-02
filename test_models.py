import os
import csv
import torch
import coremltools as ct
from transformers import SegformerImageProcessor
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex
from src.utils import evaluate_model
from src.dataset import CamVidDataset  
from src.model import get_model



def get_class_names(csv_path):
    """Récupère les noms des classes depuis le fichier CSV de configuration."""
    class_names = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names.append(row['name'])
    return class_names


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# ==========================================
# 1. CONFIGURATION ET CHEMINS DES DOSSIERS
# ==========================================
NUM_CLASSES = 32
CHECKPOINT = "nvidia/mit-b1"
OUTPUT_CSV = "test_models.csv"  
PATH_TEST = "./CamVid"
PATH_TO_CSV  = os.path.join(PATH_TEST, "class_dict.csv")
PATH_TEST_IMG = os.path.join(PATH_TEST, "test")
PATH_TEST_MSK = os.path.join(PATH_TEST, "test_labels")


# ==========================================
# 2. PRÉPARATION DES DONNÉES ET MÉTRIQUES
# ==========================================

processor = SegformerImageProcessor.from_pretrained(CHECKPOINT)
metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average='none', ignore_index=255).to(DEVICE)

test_dataset = CamVidDataset(
    images_dir=PATH_TEST_IMG, masks_dir=PATH_TEST_MSK, csv_path=PATH_TO_CSV, processor=processor, is_train=False  
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) 


# ==========================================
# 3. CHARGEMENT DU MODÈLE (PyTorch ou CoreML)
# ==========================================
choix = input("Frameworks: 1=CoreML, 2=PyTorch | Choix : ").strip()
model = None


if choix == "1" and os.path.exists("./model_CoreML.mlpackage"):
    model = ct.models.MLModel("./model_CoreML.mlpackage")
    print("Modèle CoreML chargé avec succès.")

if choix == "2" or model is None:
    model = get_model(checkpoint="./model", num_classes=NUM_CLASSES)
    model.eval()
    print("Modèle PyTorch chargé avec succès.")


# ==========================================
# 4. EXÉCUTION DE L'ÉVALUATION
# ==========================================
print("\nDémarrage de l'évaluation du modèle...")
if choix == "1":
    iou_per_class, miou_global = evaluate_model(model, test_loader, DEVICE, is_coreml=True)
    print(f" Évaluation terminée | mIoU : {miou_global*100:.2f}%")
else:
    iou_per_class, miou_global = evaluate_model(model, test_loader, DEVICE, is_coreml=False)
    print(f" Évaluation terminée | mIoU : {miou_global*100:.2f}%")


# ==========================================
# 5. ÉCRITURE DES RÉSULTATS DANS UN FICHIER CSV
#.   (CODÉE PAR UNE IA)
# ==========================================
print(f"\nEnregistrement des résultats dans {OUTPUT_CSV}...")

# Récupération et ajustement des noms de classes
class_names_list = get_class_names(PATH_TO_CSV)
if len(class_names_list) < NUM_CLASSES:
    class_names_list += [f"Class_{i:02d}" for i in range(len(class_names_list), NUM_CLASSES)]

# Structuration des lignes du fichier CSV
headers = ["Metric", "Score_Percentage"]
rows = [
    ["mIoU", f"{miou_global * 100:.2f}%"],
]

for class_idx in range(NUM_CLASSES):
    val_raw = iou_per_class[class_idx]
    class_name = class_names_list[class_idx] 
    
    rows.append([
        class_name, 
        f"{val_raw * 100:.2f}%"
    ])

# Écriture physique du fichier
with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(headers)
    writer.writerows(rows)

print("Test finalisé avec succès !")
