#!/usr/bin/env python3
"""Easy run entry point for danbooru-deploy straight from the repo.

No pip install needed — just:

    python deploy/main.py --cpu          # CPU; omit --cpu → CUDA-if-available
    python deploy/main.py models/model   # local model dir (auto CUDA)
    python deploy/main.py models/model image.jpg   # tag a single image

Model resolution: optional CLI arg -> MODEL_DIR env -> HF hub default.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from danbooru_deploy.cli import main  # noqa: E402


if __name__ == "__main__":
    main()