import os
import torch.nn.functional as F
from transformers import (
    SegformerImageProcessor, 
    TrainingArguments, 
    EarlyStoppingCallback,
    TrainerCallback,
    Trainer
)
import matplotlib.pyplot as plt
from src.dataset import CamVidDataset  
from src.model import get_model
from src.utils import compute_metrics
from src.utils import combo_loss


class UnfreezeBackboneCallback(TrainerCallback):
    """
    Callback personnalisé pour déverrouiller le backbone du modèle après 
    un certain nombre d'époques, et gestion du LR.
    """
    def __init__(self, unfreeze_epoch=10, reduced_lr=1e-5):
        self.unfreeze_epoch = unfreeze_epoch
        self.reduced_lr = reduced_lr
        self.has_dropped = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        if round(state.epoch) == self.unfreeze_epoch and not self.has_dropped:
            optimizer = kwargs['optimizer']
            model = kwargs['model']
            if optimizer is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.reduced_lr
                print(f"Learning rate reduced to {self.reduced_lr} at epoch {state.epoch}.")
                self.has_dropped = True

            for param in model.segformer.parameters():
                param.requires_grad = True
            


def compute_loss(outputs, labels, num_items_in_batch=None):
    """
    Fonction de perte personnalisée pour le modèle SegFormer.
    """

    logits = outputs.get("logits")
    upsampled_logits = F.interpolate(
        logits, size=labels.shape[1:], mode='bilinear', align_corners=False
    )

    # 2. Calcul de la perte 
    total_loss = combo_loss(
        logits=upsampled_logits,
        targets=labels,
        num_classes=NUM_CLASSES,
        ignore_index=255
    )
    
    return total_loss
    
    
   
# ==========================================
# 1. CHEMINS VERS LES DOSSIERS (TRAIN & VAL)
# ==========================================
KAGGLE_PATH = "/kaggle/input/datasets/carlolepelaars/camvid/CamVid" 
LOCAL_PATH = "./CamVid"

if os.path.exists(KAGGLE_PATH):
    BASE_PATH = KAGGLE_PATH
    print("Environnement détecté : KAGGLE")
else:
    BASE_PATH = LOCAL_PATH
    print("Environnement détecté : LOCAL")

PATH_TO_CSV   = os.path.join(BASE_PATH, "class_dict.csv")
PATH_TRAIN_IMG = os.path.join(BASE_PATH, "train")
PATH_TRAIN_MSK = os.path.join(BASE_PATH, "train_labels")
PATH_VAL_IMG   = os.path.join(BASE_PATH, "val")
PATH_VAL_MSK   = os.path.join(BASE_PATH, "val_labels")

CHECKPOINT = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
NUM_CLASSES = 32

# ==========================================
# 2. INSTANCIATION DES DATASETS (TRAIN & VAL)
# ==========================================
processor = SegformerImageProcessor.from_pretrained(CHECKPOINT)
processor.size = {"height": 512, "width": 512} #Pour la gestion mémoire
train_dataset = CamVidDataset(
    images_dir=PATH_TRAIN_IMG, masks_dir=PATH_TRAIN_MSK, csv_path=PATH_TO_CSV, processor=processor, is_train=True  
)
val_dataset = CamVidDataset(
    images_dir=PATH_VAL_IMG, masks_dir=PATH_VAL_MSK, csv_path=PATH_TO_CSV, processor=processor, is_train=False  
)

# ==========================================
# 3. INSTANCIATION DU MODÈLE 
# ==========================================
model = get_model(checkpoint=CHECKPOINT, num_classes=NUM_CLASSES)


# ==========================================
# 4. CONFIGURATION DU GESTIONNAIRE D'ENTRAÎNEMENT
# ==========================================
training_args = TrainingArguments(
    output_dir="./results", 
    learning_rate=2e-4, 
    num_train_epochs=200,                
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16, 
    eval_strategy="epoch",         
    save_strategy="epoch", 
    logging_steps=10, 
    remove_unused_columns=False,                       
    fp16=True,          
    report_to="tensorboard",
    run_name="training_run",                    
    
    # SÉCURITÉ ANTI-SATURATION MEMOIRE 
    load_best_model_at_end=True,         
    metric_for_best_model="eval_mean_iou", 
    greater_is_better=True,             
    save_total_limit=2,                  
)

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=val_dataset,   
    compute_loss_func=compute_loss,         
    compute_metrics=compute_metrics, 
    callbacks=[EarlyStoppingCallback(early_stopping_patience=20),
               UnfreezeBackboneCallback(unfreeze_epoch=10, reduced_lr=1e-5)]
                
)

# ==========================================
# 5. ENTRAÎNEMENT ET ENREGISTREMENT DES COURBES
#    (CODÉE PAR UNE IA)
# ==========================================
if __name__ == "__main__":
    print("Démarrage de l'entraînement")
    trainer.train()
    
    print("Extraction et sauvegarde du meilleur modèle...")
    trainer.save_model("./model")
    
    print("Génération des graphiques d'entraînement...")
    history = trainer.state.log_history
    train_loss = [log["loss"] for log in history if "loss" in log]
    train_steps = [log["step"] for log in history if "loss" in log]
    val_loss = [log["eval_loss"] for log in history if "eval_loss" in log]
    val_iou = [log["eval_mean_iou"] for log in history if "eval_mean_iou" in log]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    ax1.plot(train_steps, train_loss, label="Train Loss", color="blue", alpha=0.6)
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.set_title("Évolution de la Perte d'Entraînement")
    ax1.legend()
    ax1.grid(True)
    
    val_steps = []
    if train_steps and val_loss:
        steps_per_epoch = train_steps[-1] / training_args.num_train_epochs
        val_steps = [i * steps_per_epoch for i in range(1, len(val_loss) + 1)]

    if val_loss and val_steps:
        ax2.plot(val_steps, val_loss, label="Validation Loss", color="orange", marker="o")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Loss")
    ax2.set_title("Évolution de la Perte de Validation")
    ax2.legend()
    ax2.grid(True)
    
    if val_iou and val_steps:
        ax3.plot(val_steps, val_iou, label="Validation Mean IoU", color="green", marker="s")
    ax3.set_xlabel("Steps ")
    ax3.set_ylabel("Mean IoU")
    ax3.set_title("Évolution du Mean IoU")
    ax3.legend()
    ax3.grid(True)
        
    plt.tight_layout()
    plt.savefig("./training_metrics.png")
    plt.show()