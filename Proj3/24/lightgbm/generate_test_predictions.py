#!/usr/bin/env python3
"""Generate test-set predictions for the homework hack split.

Reads ``Proj3/data/test.csv`` (same user–product pairs as ``hjr/prediction_hack.csv``,
stars are placeholders) via ``main.py``, which loads data from ``Proj3/data/``.

Examples::

    # From Proj3/
    python sirui/generate_test_predictions.py

    # From sirui/
    python generate_test_predictions.py

    # Stronger test line (refit on train+validation, longer run)
    python sirui/generate_test_predictions.py --retrain-with-val

Any extra CLI flags are forwarded to ``main.py`` (e.g. ``--no-neural``, ``--out PATH``).

Outputs by default:
  * ``sirui/val_pred.csv`` — predictions for ``data/validation.csv``
  * ``sirui/prediction.csv`` — predictions for ``data/test.csv`` (submit / hack split)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MAIN = _HERE / "main.py"


def main() -> None:
    cmd = [sys.executable, str(_MAIN), *sys.argv[1:]]
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    main()
