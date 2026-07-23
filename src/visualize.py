import argparse
import math
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image
from .config import TrainConfig
from .dataset import DanbooruDataset
from .model import ImageTagger
from .transforms import val_transforms

def visualize(checkpoint, output="runs/validation_examples.png", count=12):
    config = TrainConfig(); state = torch.load(checkpoint, map_location="cpu"); model = ImageTagger(config.model_name, config.num_classes, pretrained=False); model.load_state_dict(state["model"]); model.eval(); inverse = {v: k for k, v in state["tag_to_id"].items()}; raw = DanbooruDataset(config.val_parquet, config.tag_to_id); indices = random.sample(range(len(raw)), min(count, len(raw))); columns = 3; figure, axes = plt.subplots(math.ceil(len(indices) / columns), columns, squeeze=False, figsize=(15, 5 * math.ceil(len(indices) / columns)))
    for axis, index in zip(axes.flat, indices):
        row = raw.frame.iloc[index]; image = Image.open(row.image_path).convert("RGB"); tensor = val_transforms(config.image_size)(image).unsqueeze(0)
        with torch.no_grad(): scores = torch.sigmoid(model(tensor))[0]
        predicted = [inverse[i] for i in scores.argsort(descending=True)[:5]]; axis.imshow(image); axis.axis("off"); axis.set_title(f"GT: {', '.join(row.tags)}\nPred: {', '.join(predicted)}", fontsize=8)
    figure.tight_layout(); figure.savefig(output, dpi=150); plt.close(figure)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("checkpoint"); parser.add_argument("--output", default="runs/validation_examples.png"); visualize(parser.parse_args().checkpoint, parser.parse_args().output)
