import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluate

# Initialisation globale de la métrique Hugging Face
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


def compute_metrics(eval_pred, num_classes=32):
    """
    Calcule le Mean IoU en redimensionnant les prédictions à la taille des masques cibles.
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