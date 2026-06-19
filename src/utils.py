import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluate

metric = evaluate.load("mean_iou")

class MulticlassDiceLoss(nn.Module):
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
        
        # Combinaison ajustable (ici 50% de chaque)
        return 0.5 * dice_loss + 0.5 * focal_loss


def compute_metrics(eval_pred, num_classes=32):
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