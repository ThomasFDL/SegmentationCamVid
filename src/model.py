from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
from transformers import Trainer
from src.utils import ComboLoss


def get_segformer_model(checkpoint="nvidia/mit-b3", num_classes = 32):
    """
    Instancie et configure le modèle SegFormer pour la segmentation sémantique.
    """
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name_or_path=checkpoint, 
        num_labels=num_classes, 
        ignore_mismatched_sizes=True
    )
    return model


class SegmentationTrainer(Trainer):
    def __init__(self, *args, num_classes=32, **kwargs):
        """
        Définition d'un Trainer personnalisé pour la segmentation sémantique avec SegFormer.
        """
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        
        # Instanciation de la fonction de perte 
        self.loss_fn = ComboLoss(
            num_classes=self.num_classes, 
            gamma=2.0, 
            ignore_index=255
        )
        self.loss_fn_device = None  # Suivi du périphérique pour le transfert automatique

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels").long()
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Redimensionnement des logits à la taille originale du masque
        upsampled_logits = F.interpolate(
            logits, size=labels.shape[1:], mode='bilinear', align_corners=False
        )
        
        # Transfert de la perte sur le bon GPU uniquement si nécessaire (changement de device)
        if self.loss_fn_device != logits.device:
            self.loss_fn = self.loss_fn.to(logits.device)
            self.loss_fn_device = logits.device
            
        # Calcul de la perte globale (Dice + Focal)
        total_loss = self.loss_fn(upsampled_logits, labels)
        
        return (total_loss, outputs) if return_outputs else total_loss