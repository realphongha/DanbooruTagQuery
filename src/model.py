import timm
from torch import nn


class LinearHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, tokens):
        if tokens.ndim == 3:
            tokens = tokens[:, 0]
        return self.linear(tokens)


class ImageTagger(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        self.head = LinearHead(self.backbone.num_features, num_classes)

    def forward(self, images):
        tokens = self.backbone.forward_features(images)
        return self.head(tokens)

    def extract_features(self, images):
        return self.backbone.forward_features(images)
