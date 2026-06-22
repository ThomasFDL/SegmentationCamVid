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
        probs = F.softmax(logits, dim=1)  
        
    
        mask_valid = (targets != self.ignore_index) 
        targets_clean = targets.clone()
        targets_clean[~mask_valid] = 0
        
        targets_one_hot = F.one_hot(targets_clean, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float() 
        
        mask_valid = mask_valid.unsqueeze(1) 
        probs = probs * mask_valid
        targets_one_hot = targets_one_hot * mask_valid

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims) 
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)   
        
        volumes = torch.sum(targets_one_hot, dim=dims)
        weights = 1.0 / (volumes ** 2 + self.epsilon)
        
        numerator = 2.0 * torch.sum(weights * intersection)
        denominator = torch.sum(weights * cardinality)
        
        generalized_dice_score = (numerator + self.epsilon) / (denominator + self.epsilon)
        return 1.0 - generalized_dice_score


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
    Combine la Generalized Dice Loss et la Cross Entropy Loss.
    """
    def __init__(self, num_classes, gamma=2.0, ignore_index=255):
        super().__init__()
        self.gdice = GeneralizedDiceLoss(num_classes, ignore_index)
        self.ce = CrossEntropyLoss(ignore_index) 

    def forward(self, logits, targets):
        dice_loss = self.gdice(logits, targets)
        ce_loss = self.ce(logits, targets) 

        return 0.5 * dice_loss + 0.5 * ce_loss


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
        
        
        metric_jaccard = MulticlassJaccardIndex(
            num_classes=num_classes, 
            average='macro', 
            ignore_index=255
        )
        
        miou = metric_jaccard(preds, labels_tensor).item()
        return {"mean_iou": miou}
    

def evaluate_model(model, test_loader, num_classes=32, device=None):
    """
    Évalue le modèle sur le jeu de test et calcule le mIoU global.
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    model.to(device)
    model.eval()
    

    miou_metric = MulticlassJaccardIndex(num_classes=num_classes, average='macro', ignore_index=255).to(device)
    
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
            miou_metric.update(preds, masks)
            
    final_miou = miou_metric.compute().item()
    print(f"Score mIoU Final : {final_miou * 100:.2f}%")
    return final_miou
