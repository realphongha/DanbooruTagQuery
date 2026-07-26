from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class BaseModel(ABC):
    """Interface for an arena-contender tagger.

    Subclasses must implement: name, tag_names, load(), predict().
    All produce sigmoid probabilities in [0,1].
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable short name, e.g. 'DeepDanbooru', 'JoyTag'."""

    @property
    @abstractmethod
    def tag_names(self) -> list[str]:
        """Full list of tag names in output-index order."""

    @abstractmethod
    def load(self) -> None:
        """Load weights, tag list, move to device."""

    @abstractmethod
    def predict(self, image: Image.Image) -> np.ndarray:
        """Run inference on a single PIL RGB image.

        Returns 1-D float32 array of shape (len(tag_names),)
        with sigmoid probabilities in [0,1].
        """

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} '{self.name}' {len(self.tag_names)} tags>"
