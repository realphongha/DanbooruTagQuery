import argparse
import torch
from pathlib import Path
from PIL import Image
from .config import TrainConfig
from .model import ImageTagger
from .transforms import val_transforms

def load_ignored_tags(path="data/ignored_tags.txt"):
    p = Path(path)
    if not p.exists(): return set()
    return {line.strip() for line in p.read_text().splitlines() if line.strip() and not line.startswith("#")}

def predict(image_path, checkpoint, top_k=10, ignored_tags=None):
    config = TrainConfig()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config.num_classes = len(state["tag_to_id"])
    model = ImageTagger(config.model_name, config.num_classes, pretrained=False, head_type=config.head_type)
    model.load_state_dict(state["model"])
    model.eval()
    with Image.open(image_path) as image:
        tensor = val_transforms(config.image_size)(image.convert("RGB")).unsqueeze(0)
    scores = torch.sigmoid(model(tensor))[0]
    inverse = {value: key for key, value in state["tag_to_id"].items()}
    results = sorted(((inverse[index], float(scores[index])) for index in scores.argsort(descending=True)), key=lambda x: x[1], reverse=True)
    if ignored_tags:
        results = [(tag, score) for tag, score in results if tag not in ignored_tags]
    return results[:top_k]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("checkpoint")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ignore-file", default="data/ignored_tags.txt")
    args = parser.parse_args()
    ignored = load_ignored_tags(args.ignore_file)
    print("\n".join(f"{tag:<24} {score:.4f}" for tag, score in predict(args.image, args.checkpoint, args.top_k, ignored)))
