import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluate
from torchmetrics.classification import MulticlassJaccardIndex

metric = evaluate.load("mean_iou")


class GeneralizedDiceLoss(nn.Module):
    """
    Implémentation de la Generalized Dice Loss 
    """
    def __init__(self, num_classes, ignore_index=255, epsilon=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def forward(self, logits, targets):
        # 1. Conversion des logits en probabilités via Softmax
        probs = F.softmax(logits, dim=1)  # Shape: [Batch, Classes, H, W]
        
        # 2. Masquage des pixels invalides (votre logique métier CamVid)
        mask_valid = (targets != self.ignore_index) & (targets != 30) # Shape: [Batch, H, W]
        targets_clean = targets.clone()
        targets_clean[~mask_valid] = 0
        
        # 3. Conversion des targets en One-Hot encoding
        targets_one_hot = F.one_hot(targets_clean, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float() # Shape: [Batch, Classes, H, W]
        
        # 4. Application du masque d'exclusion sur les probabilités et les targets
        mask_valid = mask_valid.unsqueeze(1) # Shape: [Batch, 1, H, W]
        probs = probs * mask_valid
        targets_one_hot = targets_one_hot * mask_valid

        # 5. Somme sur les dimensions spatiales (Batch, Hauteur, Largeur) pour chaque classe
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims) # Shape: [Classes]
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)   # Shape: [Classes]
        
        # 6. Calcul des poids de généralisation : w_l = 1 / (somme des pixels de la classe l)^2
        # On ajoute epsilon pour éviter la division par zéro sur les classes absentes du batch
        volumes = torch.sum(targets_one_hot, dim=dims)
        weights = 1.0 / (volumes ** 2 + self.epsilon)
        
        # 7. Calcul du score Dice Généralisé pondéré
        numerator = 2.0 * torch.sum(weights * intersection)
        denominator = torch.sum(weights * cardinality)
        
        generalized_dice_score = (numerator + self.epsilon) / (denominator + self.epsilon)
        
        # Retourne la perte à minimiser (0 = prédiction parfaite, 1 = aucune intersection)
        return 1.0 - generalized_dice_score


class CrossEntropyLoss(nn.Module):
    """
    Implementation de la Cross Entropy Loss.
    """
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # Calcul par pixel sans réduction automatique
        ce_loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index, reduction='none')
        
        
        mask_valid = (targets != self.ignore_index) 
        
        if mask_valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
            
        return ce_loss[mask_valid].mean()


class DiceLoss(nn.Module):
    """
    Implementation de la Dice Loss 
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


class FocalLoss(nn.Module):
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
        mask_valid = (targets != self.ignore_index) 
        if mask_valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
            
        return focal_loss[mask_valid].mean()


class ComboLoss(nn.Module):
    """
    Combine 2 fonctions de perte.
    """
    def __init__(self, num_classes, gamma=2.0, ignore_index=255):
        super().__init__()
        self.gdice = GeneralizedDiceLoss(num_classes, ignore_index)
        self.focal = FocalLoss(num_classes, gamma, ignore_index)
        

    def forward(self, logits, targets):
        dice_loss = self.gdice(logits, targets)
        #focal_loss = self.focal(logits, targets)
        ce_loss = self.ce(logits, targets)

        return 0.5 * dice_loss + 0.5 * ce_loss


def compute_metrics(eval_pred, num_classes=32):
    """
    Calcule le mIoU pendant l'entraînement.
    """
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        
        # Redimensionnement géométrique des logits à la taille réelle des masques
        outputs = torch.nn.functional.interpolate(
            logits_tensor, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        
        # Extraction de la classe dominante par pixel (Tenseur PyTorch)
        preds = outputs.argmax(dim=1)
        labels_tensor = torch.from_numpy(labels).long()
        
        metric_jaccard = MulticlassJaccardIndex(
            num_classes=num_classes, 
            average='macro', 
            ignore_index=30
        )
        
        miou = metric_jaccard(preds, labels_tensor).item()
        return {"mean_iou": miou}
    



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