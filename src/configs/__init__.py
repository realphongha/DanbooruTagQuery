"""Per-experiment training config files.

Each module in this package exports a single `config` attribute, a
`TrainConfig` dataclass instance.  Load with:

    uv run python -m src.train --config src/configs/config_dinov3_l16_to_b16_transfer.py

Paths inside config files are relative to the repo root (same convention as
the existing parquet paths).  CLI flags (--epochs, --batch-size, --checkpoint,
--teacher-path, --kd-weight) override fields set here.
"""
