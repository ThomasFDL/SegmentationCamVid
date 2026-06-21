import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluate
from torchmetrics.classification import MulticlassJaccardIndex

metric = evaluate.load("mean_iou")

class MulticlassDiceLoss(nn.Module):
    """
    Implementation de la Dice Loss pour la segmentation multi-classes.
    """
    def __init__(self, num_classes, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        
        mask_valid = (targets != self.ignore_index)
        targets_clean = targets.clone()
        targets_clean[~mask_valid] = 0
        
        targets_one_hot = F.one_hot(targets_clean, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        mask_valid = mask_valid.unsqueeze(1)
        probs = probs * mask_valid
        targets_one_hot = targets_one_hot * mask_valid

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)
        
        dice_score = (2. * intersection + 1e-6) / (cardinality + 1e-6)
        return 1.0 - dice_score.mean()


class MulticlassFocalLoss(nn.Module):
    """
    Implementation de la Focal Loss pour la segmentation multi-classes.
    """
    def __init__(self, num_classes, gamma=2.0, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # Calcul de la Cross Entropy par pixel sans réduction immédiate
        ce_loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index, reduction='none')
        
        # Calcul de pt (la probabilité de la bonne classe pour chaque pixel)
        pt = torch.exp(-ce_loss)
        
        # Formule de la Focal Loss : (1 - pt)^gamma * ce_loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # On ne fait la moyenne que sur les pixels valides
        mask_valid = (targets != self.ignore_index) & (targets != 30)
        if mask_valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
            
        return focal_loss[mask_valid].mean()


class ComboDiceFocalLoss(nn.Module):
    """
    Combine la Dice Loss et la Focal Loss de manière équilibrée.
    """
    def __init__(self, num_classes, gamma=2.0, ignore_index=255):
        super().__init__()
        self.dice = MulticlassDiceLoss(num_classes, ignore_index)
        self.focal = MulticlassFocalLoss(num_classes, gamma, ignore_index)

    def forward(self, logits, targets):
        dice_loss = self.dice(logits, targets)
        focal_loss = self.focal(logits, targets)
        
        return 0.5 * dice_loss + 0.5 * focal_loss


def compute_metrics(eval_pred, num_classes=32):
    """
    Fonction pour calculer le mIoU à partir des prédictions et des labels.
    """
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
            predictions=preds_clean, 
            references=labels_clean, 
            num_labels=num_classes, 
            ignore_index=255, 
            reduce_labels=False
        )
        return {"mean_iou": metrics["mean_iou"]}
    



def evaluate_model(model, test_loader, num_classes=32, device=None):
    """
    Évalue le modèle sur le jeu de test et calcule le mIoU.
    """

    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    # Configuration du modèle
    model.to(device)
    model.eval()
    
    # Initialisation de la métrique IoU macro pour le mIoU
    miou_metric = MulticlassJaccardIndex(num_classes=num_classes, average='macro', ignore_index=255).to(device)
    
    print(f"Démarrage de l'évaluation")
    
    with torch.no_grad():
        for batch in test_loader:
            # 1. Extraction et sécurisation des données selon la structure du batch
            if isinstance(batch, dict):
                # Si le dataset renvoie un dictionnaire (Hugging Face / SegFormer standard)
                images = batch.get("pixel_values")
                masks = batch.get("labels")
            elif isinstance(batch, (list, tuple)):
                # Si le dataset renvoie un tuple (images, masks, éventuellement paths)
                images = batch[0]
                masks = batch[1]
            else:
                # Cas imprévu
                raise TypeError(f"Format de batch non supporté : {type(batch)}")

            # 2. Envoi sur le périphérique (GPU/CPU)
            images = images.to(device)
            masks = masks.to(device).long()
            # 3. Prédiction adaptée à SegFormer
            outputs = model(images)
            # Récupération des logits (gère les sorties brutes ou les objets complexes)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            upsampled_logits = F.interpolate(
                logits, 
                size=masks.shape[1:],  # Utilise la hauteur et largeur du masque (ex: 512, 512)
                mode='bilinear', 
                align_corners=False
            )
            # Extraction des prédictions (classe dominante par pixel)
            preds = torch.argmax(upsampled_logits, dim=1)
            # 4. Accumulation du score mIoU
            miou_metric.update(preds, masks)
            
    # Calcul final du mIoU
    final_miou = miou_metric.compute().item()
    print(f"Score mIoU Final : {final_miou * 100:.2f}%")

    return final_miou