"""Evaluate rating predictions against the validation set (RMSE).

Usage (run from the project_3/ directory):
    python evaluate.py --pred val_pred_ncf.csv

The prediction CSV must have columns: ReviewerID, ProductID, Star.
Predictions are clipped to [1.0, 5.0]. Missing (ReviewerID, ProductID)
pairs are filled with the global mean rating from the validation set
and a warning is printed.

For the held-out test set the same script is used by the TA after the
deadline by passing --truth pointing at the hidden labels file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_TRUTH = ROOT / "data" / "validation.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute RMSE of rating predictions."
    )
    parser.add_argument(
        "--pred", default="prediction.csv",
        help="Path to prediction CSV (cols: ReviewerID, ProductID, Star).",
    )
    parser.add_argument(
        "--truth", default=str(DEFAULT_TRUTH),
        help="Ground-truth CSV (defaults to data/validation.csv).",
    )
    args = parser.parse_args()

    truth = pd.read_csv(args.truth, usecols=["ReviewerID", "ProductID", "Star"])
    pred  = pd.read_csv(args.pred,  usecols=["ReviewerID", "ProductID", "Star"])

    df = truth.merge(
        pred, on=["ReviewerID", "ProductID"], how="left",
        suffixes=("_true", "_pred"),
    )

    missing = int(df["Star_pred"].isna().sum())
    if missing:
        fill = float(df["Star_true"].mean())
        print(
            f"WARNING: {missing} of {len(df)} pairs missing in --pred; "
            f"filling with global mean = {fill:.4f}",
            file=sys.stderr,
        )
        df["Star_pred"] = df["Star_pred"].fillna(fill)

    y_true = df["Star_true"].to_numpy(dtype=np.float64)
    y_pred = np.clip(df["Star_pred"].to_numpy(dtype=np.float64), 1.0, 5.0)

    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    mae  = float(np.abs(y_true - y_pred).mean())

    print("=" * 45)
    print(f"  Truth file     : {args.truth}")
    print(f"  Prediction file: {args.pred}")
    print(f"  Pairs evaluated: {len(df)}")
    print(f"  RMSE           : {rmse:.4f}   ← primary metric")
    print("=" * 45)


if __name__ == "__main__":
    main()
