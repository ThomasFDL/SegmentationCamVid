import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassJaccardIndex

#Définition des métriques globales et par classe pour l'évaluation
metric_global = MulticlassJaccardIndex(
    num_classes=32, 
    average='macro', 
    ignore_index=255
)

metric_per_class = MulticlassJaccardIndex(
    num_classes=32, 
    average='none', 
    ignore_index=255
)


class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Calcule la Dice Loss entre les prédictions (logits) et les cibles (targets).
        """
        probs = F.softmax(logits, dim=1)
        
        # Filtrage des pixels à ignorer
        mask_valid = (targets != self.ignore_index)
        targets_clean = targets.clone()
        targets_clean[~mask_valid] = 0
        
        # Encodage des étiquettes
        targets_one_hot = F.one_hot(targets_clean, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Application du masque de validité
        mask_valid = mask_valid.unsqueeze(1)
        probs = probs * mask_valid
        targets_one_hot = targets_one_hot * mask_valid

        # Réduction sur le batch, la hauteur et la largeur
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)
        
        # Calcul du score et de la perte
        dice_score = (2. * intersection + 1e-6) / (cardinality + 1e-6)
        return 1.0 - dice_score.mean()



class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Calcule la Focal Loss entre les prédictions (logits) et les cibles (targets).
        """
        # 1. Calcul de la Cross Entropy classique pixel par pixel (sans réduction)
        ce_loss = F.cross_entropy(
            logits, 
            targets, 
            ignore_index=self.ignore_index, 
            reduction='none'
        )
        
        # 2. Calcul de pt : la probabilité que le modèle a attribuée à la BONNE classe
        pt = torch.exp(-ce_loss)
        
        # 3. Application de la formule de la Focal Loss : (1 - pt)^gamma * CE
        focal_loss_value = ((1 - pt) ** self.gamma) * ce_loss
        
        # 4. Création du masque pour ne calculer la moyenne que sur les pixels valides
        mask_valid = (targets != self.ignore_index)
           
        # 5. Retourne la moyenne de la focal loss
        return focal_loss_value[mask_valid].mean()


class ComboLoss(nn.Module):
    def __init__(self, num_classes, gamma=2.0, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.dice = DiceLoss(num_classes=self.num_classes, ignore_index=self.ignore_index)
        self.focal = FocalLoss(gamma=self.gamma, ignore_index=self.ignore_index)

    def forward(self, logits, targets):
        """
        Combine la Dice Loss et la Focal Loss 
        """
        # Appel des deux fonctions indépendantes
        d_loss = self.dice(logits, targets)
        f_loss = self.focal(logits, targets)
        
        # Combinaison linéaire des deux pertes
        return (0.5 * d_loss) + (0.5 * f_loss)


def compute_metrics(eval_pred):
    """
    Calcule le mIoU.
    """
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        
        outputs = F.interpolate(
            logits_tensor, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        
        preds = outputs.argmax(dim=1)
        labels_tensor = torch.from_numpy(labels).long()
        
        
        miou = metric_global(preds, labels_tensor).item()
        return {"mean_iou": miou}
  
      

            
def evaluate_model(model, test_loader, device="cpu", is_coreml=False):
    """
    Évalue le modèle et renvoie l'IoU par classe ainsi que le mIoU global.
    """ 
    
    # 1. Configuration et transfert du modèle sur le périphérique
    metric_global.to(device)
    metric_per_class.to(device)
    if not is_coreml:
        model.to(device)
        model.eval()
    
    # 2. Initialisation des métriques
    metric_global.reset()
    metric_per_class.reset()

    # 3. Boucle d'évaluation sur le jeu de données
    with torch.no_grad():
        for batch in test_loader:
            
            # Extraction propre des images et des masques 
            if isinstance(batch, dict):
                images = batch.get("pixel_values")
                masks = batch.get("labels").to(device).long()
            else:
                images, masks = batch, batch.to(device).long()
                
            
            # Propagation avant (Inférence PyTorch)
            if not is_coreml:
                images = images.to(device)
                output = model(images)
                logits = output.logits if hasattr(output, 'logits') else output
            else:
                images = F.interpolate(images, size=(512, 512), mode='bilinear', align_corners=False)
                np_images = images.cpu().numpy()
                inputs = {"pixel_values": np_images}
                predictions = model.predict(inputs)
                output = list(predictions.keys())[0]
                logits_np = predictions[output]
                logits = torch.from_numpy(logits_np).to(device)

   
            
            # Redimensionnement des logits à la taille originale du masque (Calcul exact)
            upsampled = F.interpolate(logits, size=masks.shape[1:], mode='bilinear', align_corners=False)
            preds = upsampled.argmax(dim=1)
            
            # Accumulation simultanée dans les deux métriques
            metric_global.update(preds, masks)
            metric_per_class.update(preds, masks)
            
    # 4. Extraction et calcul des scores finaux
    miou = metric_global.compute().item()
    ious = metric_per_class.compute()
        
    return ious.cpu().numpy(), miou