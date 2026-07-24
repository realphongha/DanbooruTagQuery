import argparse
import torch
from PIL import Image
from .config import TrainConfig
from .model import ImageTagger
from .transforms import val_transforms

def predict(image_path, checkpoint, top_k=10):
    config = TrainConfig(); state = torch.load(checkpoint, map_location="cpu"); model = ImageTagger(config.model_name, config.num_classes, pretrained=False, head_type=config.head_type); model.load_state_dict(state["model"]); model.eval()
    with Image.open(image_path) as image: tensor = val_transforms(config.image_size)(image.convert("RGB")).unsqueeze(0)
    scores = torch.sigmoid(model(tensor))[0]; inverse = {value: key for key, value in state["tag_to_id"].items()}
    return [(inverse[index], float(scores[index])) for index in scores.argsort(descending=True)[:top_k]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("image"); parser.add_argument("checkpoint"); parser.add_argument("--top-k", type=int, default=10); print("\n".join(f"{tag:<24} {score:.4f}" for tag, score in predict(parser.parse_args().image, parser.parse_args().checkpoint, parser.parse_args().top_k)))
