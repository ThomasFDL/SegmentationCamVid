import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    SegformerImageProcessor, 
    SegformerForSemanticSegmentation,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
import evaluate
from dataset import CamVidDataset

# 🛡️ SÉCURITÉ MULTI-GPU : On force l'utilisation du GPU numéro 0 uniquement
# Cela évite les crashs de DataParallel avec SegFormer
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==========================================
# 1. CHEMINS VERS LES DOSSIERS (TRAIN & VAL)
# ==========================================
KAGGLE_PATH = "/kaggle/input/datasets/carlolepelaars/camvid/CamVid" 
LOCAL_PATH = "./CamVid"

if os.path.exists(KAGGLE_PATH):
    BASE_PATH = KAGGLE_PATH
    print("[ℹ️] Environnement détecté : KAGGLE GPU CLOUD")
else:
    BASE_PATH = LOCAL_PATH
    print("[ℹ️] Environnement détecté : MAC LOCAL CPU")

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
# 3. MODÈLE ET MÉTRIQUES
# ==========================================
model = SegformerForSemanticSegmentation.from_pretrained(
    CHECKPOINT, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
)
metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        outputs = torch.nn.functional.interpolate(
            logits_tensor, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        preds = outputs.argmax(dim=1).numpy()
        preds_clean = np.ascontiguousarray(preds)
        labels_clean = np.ascontiguousarray(labels)
        metrics = metric.compute(
            predictions=preds_clean, references=labels_clean, num_labels=NUM_CLASSES, ignore_index=255, reduce_labels=False
        )
        return {"mean_iou": metrics["mean_iou"]}

# ==========================================
# 4. COMBO ET TRAINER PERSONNALISÉ
# ==========================================
class MulticlassDiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        
        # On évite le plantage de F.one_hot en remplaçant l'index ignoré par 0 temporairement
        mask_valid = (targets != self.ignore_index)
        targets_clean = targets.clone()
        targets_clean[~mask_valid] = 0
        
        # Encodage One-Hot (N, H, W) -> (N, H, W, C) -> (N, C, H, W)
        targets_one_hot = F.one_hot(targets_clean, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # On masque les pixels non valides pour qu'ils n'impactent pas le score
        mask_valid = mask_valid.unsqueeze(1)
        probs = probs * mask_valid
        targets_one_hot = targets_one_hot * mask_valid

        # Somme sur les dimensions Spatial (H, W) et Batch (N)
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)
        
        dice_score = (2. * intersection + 1e-6) / (cardinality + 1e-6)
        return 1.0 - dice_score.mean()

class SegmentationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels").long()
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Redimensionnement des logits à la taille originale de l'image (indispensable pour SegFormer)
        upsampled_logits = F.interpolate(
            logits, size=labels.shape[1:], mode='bilinear', align_corners=False
        )
        
        # 1. Calcul de la Cross-Entropy standard
        ce_loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        ce_loss = ce_loss_fn(upsampled_logits, labels)
        
        # 2. Calcul de la Dice Loss personnalisée
        dice_loss_fn = MulticlassDiceLoss(num_classes=NUM_CLASSES, ignore_index=255).to(logits.device)
        dice_loss = dice_loss_fn(upsampled_logits, labels)
        
        # 3. Combinaison équilibrée (50% Cross-Entropy / 50% Dice)
        total_loss = 0.5 * ce_loss + 0.5 * dice_loss
        
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
    
    # 🛡️ SÉCURITÉ ANTI-SATURATION DISQUE (KAGGlE)
    load_best_model_at_end=True,         
    metric_for_best_model="eval_loss", 
    greater_is_better=False,             
    save_total_limit=2,                  # <-- LIMITE STRICTE : Garde uniquement le meilleur et le dernier checkpoint
)

# Utilisation du nouveau SegmentationTrainer au lieu du Trainer générique
trainer = SegmentationTrainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=val_dataset,            
    compute_metrics=compute_metrics, 
    callbacks=[EarlyStoppingCallback(early_stopping_patience=15)] 
)

# ==========================================
# 6. ENTRAÎNEMENT
# ==========================================
if __name__ == "__main__":
    print("Démarrage de l'entraînement supérieur sur GPU avec Combo Loss...")
    trainer.train()
    
    print("Extraction et sauvegarde du meilleur modèle...")
    trainer.save_model("./mon_modele_final")
    
    print("Génération du graphique complet (Loss + Mean IoU)...")
    import matplotlib.pyplot as plt
    
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
        ax2.set_ylabel("Score mIoU (Précision géométrique)")
        ax2.set_title("Évolution de l'Exactitude Globale (Mean IoU)")
        ax2.legend()
        ax2.grid(True)
        ax2.set_ylim(0, 1.0)
        
    plt.tight_layout() 
    
    plt.savefig("/kaggle/working/courbes_apprentissage.png")
    print("Graphique sauvegardé avec succès sous le nom 'courbes_apprentissage.png' !")