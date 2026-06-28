from transformers import SegformerForSemanticSegmentation


def get_model(checkpoint="nvidia/mit-b1", num_classes = 32):
    """
    Instancie et configure le modèle SegFormer pour la segmentation sémantique.
    """
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name_or_path=checkpoint, 
        num_labels=num_classes, 
        ignore_mismatched_sizes=True
    )
    return model


