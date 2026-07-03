from transformers import SegformerForSemanticSegmentation


def get_model(checkpoint="nvidia/segformer-b1-finetuned-cityscapes-1024-1024", num_classes = 32, freeze_backbone=True):
    """
    Instancie et configure le modèle SegFormer pour la segmentation sémantique.
    """
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name_or_path=checkpoint, 
        num_labels=num_classes, 
        ignore_mismatched_sizes=True
    )
    if freeze_backbone:
        for param in model.segformer.encoder.parameters():
            param.requires_grad = False
    return model


