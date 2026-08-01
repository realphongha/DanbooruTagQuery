from pathlib import Path

import timm
import torch
from torch import nn


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
        tag_features, _ = self.cross_attn(
            query=queries,
            key=tokens,
            value=tokens,
            need_weights=False
        )
        logits = self.classifier(tag_features)
        return logits.squeeze(-1)


class Projector(nn.Module):
    """Linear projection from student backbone dim into teacher head dim.

    Used for transfer learning when the student backbone's embed dim differs
    from the teacher checkpoint's head embed dim, so the trained head (built at
    the teacher's dim) can be reused 1:1.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, tokens):
        return self.proj(tokens)


class ImageTagger(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True, tag_embeddings=None, head_embed_dim=None):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        backbone_dim = self.backbone.num_features
        if head_embed_dim is None:
            head_embed_dim = backbone_dim
        if head_embed_dim != backbone_dim:
            self.projector = Projector(backbone_dim, head_embed_dim)
        else:
            self.projector = None
        self.head = TagQueryHead(head_embed_dim, num_classes, init_queries=tag_embeddings)

    def forward(self, images):
        tokens = self.backbone.forward_features(images)
        if self.projector is not None:
            tokens = self.projector(tokens)
        return self.head(tokens)

    def extract_features(self, images):
        tokens = self.backbone.forward_features(images)
        if self.projector is not None:
            tokens = self.projector(tokens)
        return tokens


def _transfer_module(module, old_state, prefix, verbose=True):
    module_sd = module.state_dict()
    to_load = {}
    skipped = 0
    for old_key, old_val in old_state.items():
        if not old_key.startswith(prefix):
            continue
        local_key = old_key[len(prefix):]
        if local_key not in module_sd:
            continue
        if old_val.shape == module_sd[local_key].shape:
            to_load[local_key] = old_val
        else:
            skipped += 1
            if verbose:
                print(f"    [SKIP] {prefix}{local_key} — shape mismatch")
    if to_load:
        module.load_state_dict(to_load, strict=False)
    return to_load, skipped


def transfer_weights(
    model: ImageTagger,
    new_tag_to_id: dict[str, int],
    checkpoint_path: str | Path,
    weights_key: str = "model_ema",
    verbose: bool = True,
    state: dict | None = None,
) -> dict:
    if state is None:
        state = torch.load(checkpoint_path, map_location="cpu")
    if weights_key not in state:
        raise KeyError(f"Checkpoint missing '{weights_key}'. Available: {list(state.keys())}")
    if "tag_to_id" not in state:
        raise KeyError("Checkpoint missing 'tag_to_id'")
    if not isinstance(model.head, TagQueryHead):
        raise TypeError(f"Expected TagQueryHead, got {type(model.head).__name__}")

    old_state = state[weights_key]
    old_tag_to_id = state["tag_to_id"]

    old_tags = set(old_tag_to_id.keys())
    new_tags = set(new_tag_to_id.keys())
    common_tags = sorted(old_tags & new_tags, key=lambda t: old_tag_to_id[t])
    removed_tags = sorted(old_tags - new_tags, key=lambda t: old_tag_to_id[t])
    added_tags = sorted(new_tags - old_tags, key=lambda t: new_tag_to_id[t])
    n_common = len(common_tags)
    n_removed = len(removed_tags)
    n_added = len(added_tags)
    new_coverage = n_common / len(new_tags) * 100 if new_tags else 0.0
    old_coverage = n_common / len(old_tags) * 100 if old_tags else 0.0

    old_indices = []
    new_indices = []
    for tag in common_tags:
        old_indices.append(old_tag_to_id[tag])
        new_indices.append(new_tag_to_id[tag])
    old_idx_t = torch.tensor(old_indices, dtype=torch.long)
    new_idx_t = torch.tensor(new_indices, dtype=torch.long)

    backbone_loaded, backbone_skipped = _transfer_module(model.backbone, old_state, "backbone.", verbose)
    cross_attn_loaded, cross_attn_skipped = _transfer_module(model.head.cross_attn, old_state, "head.cross_attn.", verbose)
    classifier_loaded, classifier_skipped = _transfer_module(model.head.classifier, old_state, "head.classifier.", verbose)

    if "head.tag_queries" in old_state:
        old_q = old_state["head.tag_queries"]
        if old_q.shape[1] == model.head.tag_queries.shape[1]:
            with torch.no_grad():
                model.head.tag_queries[new_idx_t] = old_q[old_idx_t].to(
                    model.head.tag_queries.device,
                    dtype=model.head.tag_queries.dtype,
                )
            query_count = len(old_idx_t)
        else:
            if verbose:
                print(f"    [SKIP] head.tag_queries — embedding dim mismatch ({old_q.shape[1]} vs {model.head.tag_queries.shape[1]})")
            query_count = 0
    else:
        query_count = 0

    projector_loaded = getattr(model, "projector", None) is not None
    if verbose:
        print(f"    Projector    : {'random init' if projector_loaded else 'n/a'}")

    if verbose:
        def _module_status(loaded, skipped):
            if not loaded and skipped == 0:
                return "not found"
            if loaded and skipped == 0:
                return f"transferred ({len(loaded)} tensors)"
            if not loaded and skipped > 0:
                return f"not transferred ({skipped} shape mismatches)"
            return f"transferred ({len(loaded)} loaded, {skipped} skipped)"

        src_name = Path(checkpoint_path).name
        print(f"\n{'=' * 50}")
        print(f"  Model Transfer Summary")
        print(f"{'=' * 50}")
        print(f"  Source        : {src_name}")
        print(f"  Weights       : {weights_key}")
        print(f"  Old tags      : {len(old_tags)}")
        print(f"  New tags      : {len(new_tags)}")
        print(f"\n  Vocabulary:")
        print(f"    Matched     : {n_common} / {len(old_tags)} old ({old_coverage:.1f}%)")
        print(f"                  {n_common} / {len(new_tags)} new ({new_coverage:.1f}%)")
        print(f"    Removed     : {n_removed}")
        if n_removed and removed_tags:
            shown = removed_tags[:20]
            print(f"      {', '.join(shown)}{' ...' if n_removed > 20 else ''}")
        print(f"    Added       : {n_added}")
        if n_added and added_tags:
            shown = added_tags[:20]
            print(f"      {', '.join(shown)}{' ...' if n_added > 20 else ''}")
        print(f"\n  Modules:")
        print(f"    Backbone     : {_module_status(backbone_loaded, backbone_skipped)}")
        print(f"    Cross-attn   : {_module_status(cross_attn_loaded, cross_attn_skipped)}")
        print(f"    Classifier   : {_module_status(classifier_loaded, classifier_skipped)}")
        print(f"    Tag queries  : {query_count} / {len(new_tags)} student tags, matched from {len(old_tags)} teacher queries")
        print(f"{'=' * 50}\n")

    return {
        "old_tag_count": len(old_tags),
        "new_tag_count": len(new_tags),
        "matched": n_common,
        "removed": n_removed,
        "added": n_added,
        "removed_tags": removed_tags,
        "added_tags": added_tags,
        "new_coverage_pct": new_coverage,
        "old_coverage_pct": old_coverage,
        "backbone_transferred": len(backbone_loaded) > 0,
        "cross_attn_transferred": len(cross_attn_loaded) > 0,
        "classifier_transferred": len(classifier_loaded) > 0,
        "query_transfer_count": query_count,
    }
