import argparse
import math
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from .config import TrainConfig
from .dataset import DanbooruDataset
from .model import ImageTagger
from .transforms import val_transforms


def visualize(checkpoint, output="runs/validation_examples.png", count=12):
    config = TrainConfig()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config.num_classes = len(state["tag_to_id"])
    ckpt_cfg = state.get("config", {})
    config.model_name = ckpt_cfg.get("model_name", config.model_name)
    proj = ckpt_cfg.get("projector") or ""
    config.projector = str(proj)
    proj_dims = config.projector_dims()
    model = ImageTagger(
        config.model_name, config.num_classes, pretrained=False,
        head_embed_dim=proj_dims[1] if proj_dims else None,
    )
    model.load_state_dict(state["model"])
    model.eval()
    inverse = {v: k for k, v in state["tag_to_id"].items()}
    raw = DanbooruDataset(config.val_parquet, config.tag_to_id)
    indices = random.sample(range(len(raw)), min(count, len(raw)))
    columns = 3
    rows = math.ceil(len(indices) / columns)
    figure, axes = plt.subplots(rows, columns, squeeze=False, figsize=(15, 8 * rows))
    for axis, index in zip(axes.flat, indices):
        row = raw.frame.iloc[index]
        image = Image.open(row.image_path).convert("RGB")
        tensor = val_transforms(config.image_size)(image).unsqueeze(0)
        with torch.no_grad():
            scores = torch.sigmoid(model(tensor))[0]
        predicted = [inverse[int(i)] for i in scores.argsort(descending=True)[:5]]
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(f"GT: {', '.join(row.tags)}\nPred: {', '.join(predicted)}", fontsize=7)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default="runs/validation_examples.png")
    args = parser.parse_args()
    visualize(args.checkpoint, args.output)
