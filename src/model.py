import timm
import torch
from torch import nn


class LinearHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, tokens):
        if tokens.ndim == 3:
            tokens = tokens[:, 0]
        return self.linear(tokens)


class TagQueryHead(nn.Module):

    def __init__(self, embed_dim, num_classes, num_heads=8, init_queries=None):
        super().__init__()
        if init_queries is None:
            self.tag_queries = nn.Parameter(
                torch.randn(num_classes, embed_dim)
            )
        else:
            assert init_queries.shape == (num_classes, embed_dim)
            self.tag_queries = nn.Parameter(
                init_queries.clone()
            )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, tokens):
        B = tokens.size(0)
        queries = self.tag_queries.unsqueeze(0).expand(B, -1, -1)
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ):
            tag_features, _ = self.cross_attn(
                query=queries,
                key=tokens,
                value=tokens,
                need_weights=False
            )
        logits = self.classifier(tag_features)
        return logits.squeeze(-1)


class ImageTagger(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True, head_type="tag_query_head", tag_embeddings=None):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        if head_type == "linear":
            self.head = LinearHead(self.backbone.num_features, num_classes)
        elif head_type == "tag_query_head":
            self.head = TagQueryHead(self.backbone.num_features, num_classes, init_queries=tag_embeddings)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, images):
        tokens = self.backbone.forward_features(images)
        return self.head(tokens)

    def extract_features(self, images):
        return self.backbone.forward_features(images)
