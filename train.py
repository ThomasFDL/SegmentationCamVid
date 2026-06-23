import os
import torch
from transformers import (
    SegformerImageProcessor, 
    TrainingArguments, 
    EarlyStoppingCallback
)
import matplotlib.pyplot as plt
from src.dataset import CamVidDataset  
from src.model import get_segformer_model, SegmentationTrainer
from src.utils import compute_metrics


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
# 4. CONFIGURATION DU GESTIONNAIRE D'ENTRAÎNEMENT
# ==========================================
training_args = TrainingArguments(
    output_dir="./results_segformer", 
    learning_rate=1e-4, 
    num_train_epochs=200,                
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16, 
    eval_strategy="epoch",         
    save_strategy="epoch", 
    logging_steps=10, 
    remove_unused_columns=False, 
    use_cpu=False,                       
    fp16=torch.cuda.is_available(), 
    lr_scheduler_type="cosine", 
    warmup_ratio=0.1,                    
    report_to="tensorboard",
    run_name="SegFormer_CamVid_ComboLoss",                    
    
    # SÉCURITÉ ANTI-SATURATION MEMOIRE 
    load_best_model_at_end=True,         
    metric_for_best_model="eval_mean_iou", 
    greater_is_better=True,             
    save_total_limit=2,                  
)

trainer = SegmentationTrainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=val_dataset,            
    compute_metrics=compute_metrics, 
    callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
    num_classes=NUM_CLASSES,
                
)

# ==========================================
# 5. ENTRAÎNEMENT ET COURBES
# ==========================================
if __name__ == "__main__":
    print("Démarrage de l'entraînement")
    trainer.train()
    
    print("Extraction et sauvegarde du meilleur modèle...")
    trainer.save_model("./mon_modele_final")
    
    print("Génération du graphique complet à 3 courbes séparées...")
    history = trainer.state.log_history
    train_loss = [log["loss"] for log in history if "loss" in log]
    train_steps = [log["step"] for log in history if "loss" in log]
    val_loss = [log["eval_loss"] for log in history if "eval_loss" in log]
    val_iou = [log["eval_mean_iou"] for log in history if "eval_mean_iou" in log]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    ax1.plot(train_steps, train_loss, label="Train Loss", color="blue", alpha=0.6)
    ax1.set_xlabel("Steps (Étapes de calcul)")
    ax1.set_ylabel("Loss")
    ax1.set_title("Évolution de la Perte d'Entraînement (Train Loss)")
    ax1.legend()
    ax1.grid(True)
    
    val_steps = []
    if train_steps and val_loss:
        steps_per_epoch = train_steps[-1] / training_args.num_train_epochs
        val_steps = [i * steps_per_epoch for i in range(1, len(val_loss) + 1)]

    if val_loss and val_steps:
        ax2.plot(val_steps, val_loss, label="Validation Loss", color="orange", marker="o")
    ax2.set_xlabel("Steps (Étapes de calcul)")
    ax2.set_ylabel("Loss")
    ax2.set_title("Évolution de la Perte de Validation (Validation Loss)")
    ax2.legend()
    ax2.grid(True)
    
    if val_iou and val_steps:
        ax3.plot(val_steps, val_iou, label="Validation Mean IoU", color="green", marker="s")
    ax3.set_xlabel("Steps (Étapes de calcul)")
    ax3.set_ylabel("Mean IoU")
    ax3.set_title("Évolution du Mean IoU")
    ax3.legend()
    ax3.grid(True)
        
    plt.tight_layout()
    plt.savefig("./training_metrics.png")
    plt.show()