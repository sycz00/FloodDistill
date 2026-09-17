import os
from pathlib import Path

# Where the per-scene caches live. runs/ by default; $PERSCENE_ROOT points it at a mounted
# volume in the container, where the repo copy under /workspace is root-owned. perscene_eval.py
# writes through it and every collector reads through it, so both use the same root.
PERSCENE_ROOT = Path(os.environ.get("PERSCENE_ROOT", "runs"))
