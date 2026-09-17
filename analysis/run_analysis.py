"""Dispatch CKA and gradient-conflict analyses.

Use python analysis/run_analysis.py {cka,gradconf} --help.
Benchmark evaluation and result collection are under evaluation/.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    task = sys.argv.pop(1)  # remaining args go to the module's main()
    if task == "cka":
        from analysis.cka import main as run
    elif task == "gradconf":
        from analysis.grad_conflict import main as run
    else:
        print(f"unknown analysis task: {task!r} (choose 'cka' or 'gradconf')",
              file=sys.stderr)
        sys.exit(2)
    run()


if __name__ == "__main__":
    main()
