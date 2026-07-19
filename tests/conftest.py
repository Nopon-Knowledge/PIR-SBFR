"""Test environment setup before Ultralytics/Matplotlib imports."""

import os
from pathlib import Path

import torch


_CACHE_ROOT = Path(__file__).resolve().parents[1] / "tmp" / "test-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(_CACHE_ROOT / "ultralytics"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))
torch.set_num_threads(1)
