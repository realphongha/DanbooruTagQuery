import timm
from torch import nn

class ImageTagger(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        try: self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        except Exception:
            self.backbone = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=pretrained, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, num_classes)
    def forward(self, images):
        features = self.backbone.forward_features(images)
        if features.ndim == 3: features = features[:, 0]
        return self.head(features)
