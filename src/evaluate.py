import argparse
import torch
from .config import TrainConfig
from .dataset import make_loader
from .metrics import multilabel_metrics
from .model import ImageTagger
from .transforms import val_transforms
from .utils import device

def evaluate(checkpoint, config=TrainConfig()):
    state = torch.load(checkpoint, map_location="cpu"); dev = device(); model = ImageTagger(config.model_name, config.num_classes, pretrained=False); model.load_state_dict(state["model"]); model.to(dev).eval(); loader = make_loader(config.val_parquet, config, val_transforms(config.image_size), False); predictions = []; targets = []
    with torch.no_grad():
        for batch in loader: predictions.append(torch.sigmoid(model(batch["image"].to(dev))).cpu()); targets.append(batch["labels"])
    return multilabel_metrics(torch.cat(predictions), torch.cat(targets), config.threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("checkpoint"); print(evaluate(parser.parse_args().checkpoint))
