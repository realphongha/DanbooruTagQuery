import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from .utils import load_json

class DanbooruDataset(Dataset):
    def __init__(self, parquet_path, tag_to_id_path, transform=None):
        self.frame = pd.read_parquet(parquet_path); self.tag_to_id = load_json(tag_to_id_path); self.transform = transform
        if "image_path" not in self.frame: raise ValueError(f"{parquet_path} has no image_path column; run src.download_images first")
        exists = self.frame["image_path"].apply(os.path.isfile)
        n_dropped = (~exists).sum()
        if n_dropped:
            print(f"Dropped {n_dropped} rows with missing images from {parquet_path.stem}")
            self.frame = self.frame[exists].reset_index(drop=True)
        self.num_classes = len(self.tag_to_id)
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as image: image = image.convert("RGB")
        if self.transform: image = self.transform(image)
        labels = torch.zeros(self.num_classes, dtype=torch.float32)
        ids = torch.tensor([self.tag_to_id[t] for t in row["tags"] if t in self.tag_to_id], dtype=torch.long)
        labels.scatter_(0, ids, 1.0)
        return {"image": image, "labels": labels}

def make_loader(parquet_path, config, transform, shuffle):
    dataset = DanbooruDataset(parquet_path, config.tag_to_id, transform)
    kwargs = dict(batch_size=config.batch_size, shuffle=shuffle, pin_memory=True, num_workers=config.num_workers)
    if config.num_workers > 0: kwargs.update(persistent_workers=True, prefetch_factor=config.prefetch_factor)
    return DataLoader(dataset, **kwargs)


if __name__ == "__main__":
    import matplotlib
    import os
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from .config import TrainConfig
    from .transforms import train_transforms

    config = TrainConfig()
    dataset = DanbooruDataset(config.train_parquet, config.tag_to_id, train_transforms(config.image_size))
    images, n = [], min(100, len(dataset))
    for i in range(n):
        images.append(dataset[i]["image"])
    cols = 10; rows = (n + cols - 1) // cols
    figure, axes = plt.subplots(rows, cols, figsize=(20, 2 * rows))
    for i, ax in enumerate(axes.flat):
        if i < n:
            img = images[i].permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            ax.imshow(img)
        ax.axis("off")
    figure.tight_layout()
    os.makedirs("runs", exist_ok=True)
    figure.savefig("runs/augmented_samples.png", dpi=150)
    plt.close(figure)
    print(f"Saved runs/augmented_samples.png with {n} images")
