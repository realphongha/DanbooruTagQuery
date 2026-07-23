import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from .utils import load_json

class DanbooruDataset(Dataset):
    def __init__(self, parquet_path, tag_to_id_path, transform=None):
        self.frame = pd.read_parquet(parquet_path); self.tag_to_id = load_json(tag_to_id_path); self.transform = transform
        if "image_path" not in self.frame: raise ValueError(f"{parquet_path} has no image_path column; run src.download_images first")
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as image: image = image.convert("RGB")
        if self.transform: image = self.transform(image)
        labels = torch.zeros(len(self.tag_to_id), dtype=torch.float32)
        for tag in row.tags:
            if tag in self.tag_to_id: labels[self.tag_to_id[tag]] = 1.0
        return {"image": image, "labels": labels}

def make_loader(parquet_path, config, transform, shuffle):
    dataset = DanbooruDataset(parquet_path, config.tag_to_id, transform)
    kwargs = dict(batch_size=config.batch_size, shuffle=shuffle, pin_memory=True, num_workers=config.num_workers)
    if config.num_workers > 0: kwargs.update(persistent_workers=True, prefetch_factor=config.prefetch_factor)
    return DataLoader(dataset, **kwargs)
