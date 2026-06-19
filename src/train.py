import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    SegformerImageProcessor, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
import matplotlib.pyplot as plt

from src.dataset import CamVidDataset  
from src.model import get_segformer_model
from src.utils import ComboDiceFocalLoss, compute_metrics

# 🛡️ SÉCURITÉ MULTI-GPU : On force l'utilisation du GPU numéro 0 uniquement
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==========================================
# 1. CHEMINS VERS LES DOSSIERS (TRAIN & VAL)
# ==========================================
KAGGLE_PATH = "/kaggle/input/datasets/carlolepelaars/camvid/CamVid" 
LOCAL_PATH = "./CamVid"

if os.path.exists(KAGGLE_PATH):
    BASE_PATH = KAGGLE_PATH
    print("Environnement détecté : KAGGLE GPU CLOUD")
else:
    BASE_PATH = LOCAL_PATH
    print("Environnement détecté : MAC LOCAL CPU")

PATH_TO_CSV   = os.path.join(BASE_PATH, "class_dict.csv")
PATH_TRAIN_IMG = os.path.join(BASE_PATH, "train")
PATH_TRAIN_MSK = os.path.join(BASE_PATH, "train_labels")
PATH_VAL_IMG   = os.path.join(BASE_PATH, "val")
PATH_VAL_MSK   = os.path.join(BASE_PATH, "val_labels")

CHECKPOINT = "nvidia/mit-b3"
NUM_CLASSES = 32

# ==========================================
# 2. INSTANCIATION DES DATASETS (TRAIN & VAL)
# ==========================================
processor = SegformerImageProcessor.from_pretrained(CHECKPOINT)

train_dataset = CamVidDataset(
    images_dir=PATH_TRAIN_IMG, masks_dir=PATH_TRAIN_MSK, csv_path=PATH_TO_CSV, processor=processor, is_train=True  
)
val_dataset = CamVidDataset(
    images_dir=PATH_VAL_IMG, masks_dir=PATH_VAL_MSK, csv_path=PATH_TO_CSV, processor=processor, is_train=False  
)

print(f"Images d'entraînement : {len(train_dataset)} | Images de validation : {len(val_dataset)}")

# ==========================================
# 3. INSTANCIATION DU MODÈLE VIA SRC
# ==========================================
model = get_segformer_model(checkpoint=CHECKPOINT, num_classes=NUM_CLASSES)

# ==========================================
# 4. TRAINER PERSONNALISÉ 
# ==========================================
class SegmentationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels").long()
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Redimensionnement des logits à la taille originale de l'image
        upsampled_logits = F.interpolate(
            logits, size=labels.shape[1:], mode='bilinear', align_corners=False
        )
        
        # Calcul de la nouvelle perte combinée (Dice + Focal) importée de src.utils
        loss_fn = ComboDiceFocalLoss(num_classes=NUM_CLASSES, gamma=2.0, ignore_index=255).to(logits.device)
        total_loss = loss_fn(upsampled_logits, labels)
        
        return (total_loss, outputs) if return_outputs else total_loss

# ==========================================
# 5. CONFIGURATION DU GESTIONNAIRE D'ENTRAÎNEMENT
# ==========================================
training_args = TrainingArguments(
    output_dir="./results_segformer", 
    learning_rate=6e-5, 
    num_train_epochs=200,                
    per_device_train_batch_size=4, 
    per_device_eval_batch_size=4, 
    eval_strategy="epoch",         
    save_strategy="epoch", 
    logging_steps=10, 
    remove_unused_columns=False, 
    use_cpu=False,                       
    fp16=torch.cuda.is_available(), 
    lr_scheduler_type="cosine", 
    warmup_ratio=0.1,                    
    report_to="none",                    
    
    # SÉCURITÉ ANTI-SATURATION MEMOIRE (KAGGLE)
    load_best_model_at_end=True,         
    metric_for_best_model="eval_loss", 
    greater_is_better=False,             
    save_total_limit=2,                  
)

trainer = SegmentationTrainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=val_dataset,            
    compute_metrics=compute_metrics, 
    callbacks=[EarlyStoppingCallback(early_stopping_patience=15)] 
)

# ==========================================
# 6. ENTRAÎNEMENT ET COURBES
# ==========================================
if __name__ == "__main__":
    print("Démarrage de l'entraînement...")
    trainer.train()
    
    print("Extraction et sauvegarde du meilleur modèle...")
    trainer.save_model("./mon_modele_final")
    
    print("Génération du graphique complet (Loss + Mean IoU)...")
    history = trainer.state.log_history
    train_loss = [log["loss"] for log in history if "loss" in log]
    train_steps = [log["step"] for log in history if "loss" in log]
    val_loss = [log["eval_loss"] for log in history if "eval_loss" in log]
    val_iou = [log["eval_mean_iou"] for log in history if "eval_mean_iou" in log]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # --- GRAPHIQUE 1 : LES COURBES DE LOSS ---
    ax1.plot(train_steps, train_loss, label="Train Loss", color="blue", alpha=0.6)
    if val_loss and train_steps:
        steps_per_epoch = train_steps[-1] / training_args.num_train_epochs
        val_steps = [i * steps_per_epoch for i in range(1, len(val_loss) + 1)]
        ax1.plot(val_steps, val_loss, label="Validation Loss", color="orange", marker="o")
    ax1.set_xlabel("Steps (Étapes de calcul)")
    ax1.set_ylabel("Loss (Erreur)")
    ax1.set_title("Évolution de la Perte (Loss)")
    ax1.legend()
    ax1.grid(True)
    
    # --- GRAPHIQUE 2 : LA COURBE MEAN IOU ---
    if val_iou and train_steps:
        ax2.plot(val_steps, val_iou, label="Validation Mean IoU", color="green", marker="s")
        ax2.set_xlabel("Steps (Étapes de calcul)")
        ax2.set_ylabel("Mean IoU")
        ax2.set_title("Évolution du Mean IoU")
        ax2.legend()
        ax2.grid(True)
        
    plt.tight_layout()
    plt.savefig("./training_metrics.png")
    plt.show()