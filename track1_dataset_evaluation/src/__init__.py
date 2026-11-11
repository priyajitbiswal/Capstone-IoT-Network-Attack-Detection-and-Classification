"""Track 1 leak-free data partitioning, training, and evaluation pipeline."""

import sys

from . import config, evaluate, models, preprocessing

# Preserve backward-compatible unpickling for pre-split checkpoints serialized under `src_v2.*`.
sys.modules.setdefault("src_v2", sys.modules[__name__])
sys.modules.setdefault("src_v2.config", config)
sys.modules.setdefault("src_v2.models", models)
sys.modules.setdefault("src_v2.preprocessing", preprocessing)
sys.modules.setdefault("src_v2.evaluate", evaluate)
