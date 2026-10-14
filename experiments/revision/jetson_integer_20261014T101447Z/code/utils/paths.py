"""Dataset locations.

DATA_ROOT (default ~/data) holds every dataset: sen1floods11/, STURM-Flood/,
WorldFloodsv2/ and flood_unlabeled/. A per-dataset variable such as S11_ROOT
overrides one entry. See docs/DATA_LOCATION.md.
"""

from __future__ import annotations

import os
from pathlib import Path

SUBDIRS = {
    "S11_ROOT":      "sen1floods11",
    "STURM_ROOT":    "STURM-Flood/Dataset",
    "WF_ROOT":       "WorldFloodsv2",
    "UNLABELED_DIR": "flood_unlabeled",
    "S1S2W_ROOT":    "S1S2Water",
    "FP_ROOT":       "FloodPlanet",
    "SFF_ROOT":      "SenForFlood",
}


def data_root() -> Path:
    """The directory holding all datasets."""
    return Path(os.environ.get("DATA_ROOT") or Path.home() / "data").expanduser()


def dataset_root(env: str) -> Path:
    """One dataset directory: $<env> if set, else DATA_ROOT/<subdir>."""
    if os.environ.get(env):
        return Path(os.environ[env]).expanduser()
    return data_root() / SUBDIRS[env]
