from torchvision import transforms
from torchvision.transforms import functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResizeAndPad:
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = F.resize(img, (new_h, new_w))
        pad_left = (self.size - new_w) // 2
        pad_top = (self.size - new_h) // 2
        padding = (pad_left, pad_top, self.size - new_w - pad_left, self.size - new_h - pad_top)
        return F.pad(img, padding, fill=self.fill)


def train_transforms(image_size):
    return transforms.Compose([ResizeAndPad(image_size + 16), transforms.RandomCrop(image_size), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def val_transforms(image_size):
    return transforms.Compose([ResizeAndPad(image_size), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
