import argparse
import torch
from .config import TrainConfig
from .dataset import make_loader
from .metrics import multilabel_metrics, print_metrics
from .model import ImageTagger
from .transforms import val_transforms
from .utils import device, load_json

def evaluate(checkpoint, config=TrainConfig()):
    tag_to_id = load_json(config.tag_to_id)
    config.num_classes = len(tag_to_id)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False); dev = device(); model = ImageTagger(config.model_name, config.num_classes, pretrained=False); model.load_state_dict(state["model"]); model.to(dev).eval(); loader = make_loader(config.val_parquet, config, val_transforms(config.image_size), False); predictions = []; targets = []
    with torch.no_grad():
        for batch in loader: predictions.append(torch.sigmoid(model(batch["image"].to(dev))).cpu()); targets.append(batch["labels"])
    return multilabel_metrics(torch.cat(predictions), torch.cat(targets)), tag_to_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("checkpoint"); args = parser.parse_args(); metrics, tag_to_id = evaluate(args.checkpoint); print_metrics(metrics, tag_to_id)
