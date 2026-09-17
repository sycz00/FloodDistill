"""Dispatch pseudo-pool selection and feature extraction.

Selectors: diversity (FPS), confmatch (matched-pool mean confidence), random,
glister and r2core. confidence is the older 4*f*(1-f)*mean_conf selector with
different gates. features and consistency produce inputs for additional selectors.

Build one index at exactly the training N. The dataset samples uniformly from
its index. Compare ranking rules on the same candidates; raw versus filtered
arms instead test the pre-filter. Use <strategy> --help for its options.
"""
from __future__ import annotations

import os
import sys

# Allow running as `python scripts/curate.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# strategy -> (module, one-line description). Dispatch only; every module keeps its own
# argparse, so a flag can never drift between the CLI and the implementation.
STRATEGIES = {
    "diversity":   ("data.diversity_pseudo",          "FPS over the scene embedding"),
    "r2core":      ("data.r2core_pseudo",             "reliability x robustness core-set"),
    "glister":     ("data.glister_pseudo",            "GLISTER Alg. 2 bilevel influence"),
    "confidence":  ("data.curate_pseudo_gdacs",       "legacy 4p(1-p)*conf, ff-filter only"),
    "confmatch":   ("data.confidence_filtered_index", "mean confidence, matched pool"),
    "random":      ("data.random_index",              "uniform draw, matched pool (control)"),
    "features":    ("data.pool_features",          "pre-pass: pool_features.csv"),
    "consistency": ("data.teacher_consistency",    "pre-pass: consistency.csv"),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    strategy = sys.argv.pop(1)  # remaining argv goes to the module's main()
    if strategy not in STRATEGIES:
        print(f"unknown strategy: {strategy!r}\n\navailable:", file=sys.stderr)
        for name, (_, desc) in STRATEGIES.items():
            print(f"  {name:12s} {desc}", file=sys.stderr)
        sys.exit(2)

    module_name, _ = STRATEGIES[strategy]
    import importlib
    importlib.import_module(module_name).main()


if __name__ == "__main__":
    main()
