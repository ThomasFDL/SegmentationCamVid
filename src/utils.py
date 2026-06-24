import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassJaccardIndex


metric = MulticlassJaccardIndex(
    num_classes=32, 
    average='macro', 
    ignore_index=255
)


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
        probs = F.softmax(logits, dim=1)  
        
        # 1. Masquage de l'ignore_index (255)
        mask_valid = (targets != self.ignore_index) 
        targets_clean = targets.clone()
        targets_clean[~mask_valid] = 0
        
        # 2. One-hot encoding
        targets_one_hot = F.one_hot(targets_clean, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float() 
        
        # Application du masque spatial sur les probabilités et les cibles
        mask_valid = mask_valid.unsqueeze(1) 
        probs = probs * mask_valid
        targets_one_hot = targets_one_hot * mask_valid

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims) 
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)   
        volumes = torch.sum(targets_one_hot, dim=dims)
        
        # 3. Identifier les classes réellement présentes dans ce batch
        classes_presentes = (volumes > 0).float()
        
        # Calcul des poids (puissance 1 pour adoucir le déséquilibre sur CamVid)
        weights = 1.0 / (volumes + self.epsilon)
        
        # On force le poids des classes absentes à 0 pour les exclure du calcul
        weights = weights * classes_presentes
        
        # 4. Somme pondérée uniquement sur les classes présentes
        numerator = 2.0 * torch.sum(weights * intersection)
        denominator = torch.sum(weights * cardinality)
        
        # Sécurité si le batch ne contient aucun pixel valide
        if denominator == 0:
            return torch.tensor(0.0, device=logits.device)
            
        generalized_dice_score = (numerator + self.epsilon) / (denominator + self.epsilon)
        
        return 1.0 - generalized_dice_score


class DiceLoss(nn.Module):
    """
    Implementation de la Dice Loss.
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
    Implémentation de la Focal Loss 
    """
    def __init__(self, num_classes, gamma=2.0, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # 1. Calcul de la Cross Entropy classique pixel par pixel (sans réduction)
        ce_loss = F.cross_entropy(
            logits, 
            targets, 
            ignore_index=self.ignore_index, 
            reduction='none'
        )
        
        # 2. Calcul de pt : la probabilité que le modèle a attribuée à la BONNE classe
        pt = torch.exp(-ce_loss)
        
        # 3. Application de la formule mathématique de la Focal Loss : (1 - pt)^gamma * CE
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # 4. Création du masque pour ne calculer la moyenne que sur les pixels valides (!= 255)
        mask_valid = (targets != self.ignore_index)
        
        # Sécurité si le batch ne contient aucun pixel valide (très rare)
        if mask_valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
            
        # 5. Retourne la moyenne de la perte uniquement sur les pixels valides
        return focal_loss[mask_valid].mean()

class CrossEntropyLoss(nn.Module):
    """
    Implementation de la Cross Entropy Loss.
    """
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index, reduction='none')
        
        
        mask_valid = (targets != self.ignore_index)
        
        if mask_valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
            
        return ce_loss[mask_valid].mean()


class ComboLoss(nn.Module):
    """
    Combine la Dice Loss et la Focal Loss.
    """
    def __init__(self, num_classes, gamma=2.0, ignore_index=255):
        super().__init__()
        self.dice = DiceLoss(num_classes, ignore_index)
        self.focal = FocalLoss(num_classes, gamma=2.0, ignore_index=ignore_index)

    def forward(self, logits, targets):
        dice_loss = self.dice(logits, targets)
        focal_loss = self.focal(logits, targets)

        return 0.5 * dice_loss + 0.5 * focal_loss


def compute_metrics(eval_pred, num_classes=32):
    """
    Calcule le mIoU.
    """
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        
        outputs = torch.nn.functional.interpolate(
            logits_tensor, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        
        preds = outputs.argmax(dim=1)
        labels_tensor = torch.from_numpy(labels).long()
        
        
        miou = metric(preds, labels_tensor).item()
        return {"mean_iou": miou}
    

def evaluate_model(model, test_loader, num_classes=32, device=None):
    """
    Évalue le modèle sur le jeu de test et calcule le mIoU global.
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    model.to(device)
    metric.to(device)
    model.eval()
    

   
    print("Démarrage de l'évaluation")
    
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                images = batch.get("pixel_values")
                masks = batch.get("labels")
            elif isinstance(batch, (list, tuple)):
                images = batch[0]
                masks = batch[1]
            else:
                raise TypeError(f"Format de batch non supporté : {type(batch)}")

            images = images.to(device)
            masks = masks.to(device).long()
            
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            upsampled_logits = F.interpolate(
                logits, size=masks.shape[1:], mode='bilinear', align_corners=False
            )
            preds = torch.argmax(upsampled_logits, dim=1)
            metric.update(preds, masks)
            
    final_miou = metric.compute().item()
    print(f"Score mIoU Final : {final_miou * 100:.2f}%")
    return final_miou


def evaluate_model_per_class(model, test_loader, num_classes=32, device=None):
    """
    Évalue le modèle sur le jeu de test et affiche l'IoU classe par classe.
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    model.to(device)
    metric.to(device)
    model.eval()
    
    
    print("Démarrage de l'évaluation détaillée...")
    
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                images = batch.get("pixel_values")
                masks = batch.get("labels")
            elif isinstance(batch, (list, tuple)):
                images = batch[0]
                masks = batch[1]
            else:
                raise TypeError(f"Format de batch non supporté : {type(batch)}")

            images = images.to(device)
            masks = masks.to(device).long()
            
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            upsampled_logits = F.interpolate(
                logits, size=masks.shape[1:], mode='bilinear', align_corners=False
            )
            preds = torch.argmax(upsampled_logits, dim=1)
            
            # Accumulation des matrices de confusion pixel par pixel
            metric.update(preds, masks)
            
    # Extraction du tenseur contenant les IoU de chaque classe
    iou_per_class = metric.compute() # Tenseur de taille [32]
    
    print("\n" + "="*40)
    print("       SCORE IoU CLASSE PAR CLASSE      ")
    print("="*40)
    
    
    for class_idx, iou_value in enumerate(iou_per_class):
        # Convertir le tenseur en valeur Python native
        iou_val = iou_value.item() * 100
        print(f"Classe {class_idx:02d} : {iou_val:.2f}%")
        
    # Calcul manuel de la moyenne macro (mIoU global) pour vérification
    final_miou = iou_per_class.mean().item()
    print("="*40)
    print(f"Score mIoU Global (Moyenne) : {final_miou * 100:.2f}%")
    print("="*40)

    return iou_per_class.cpu().numpy()