from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error

ROOT = Path(__file__).resolve().parent
CLS_ROOT = ROOT / "data" / "cls"
REG_ROOT = ROOT / "data" / "reg"
IDS = ["1", "2", "3", "4", "5"]


def load_series(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    result_path = dataset_dir / "result.csv"
    predict_path = dataset_dir / "predict.csv"

    y_true_df = pd.read_csv(result_path)

    # Now we don't have predict.csv, so we use result.csv instead for an example.
    if predict_path.exists():
        y_pred_df = pd.read_csv(predict_path)
    else:
        y_pred_df = pd.read_csv(result_path)

    y_true = y_true_df.iloc[:, -1].to_numpy()
    y_pred = y_pred_df.iloc[:, -1].to_numpy()

    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch in {dataset_dir}: {len(y_true)} != {len(y_pred)}")
    return y_true, y_pred


def eval_cls(dataset_id: str) -> dict[str, str | float]:
    d = CLS_ROOT / dataset_id
    y_true, y_pred = load_series(d)
    acc = accuracy_score(y_true, y_pred)
    return {"task": "cls", "dataset": dataset_id, "metric": "accuracy", "value": float(acc)}


def eval_reg(dataset_id: str) -> dict[str, str | float]:
    d = REG_ROOT / dataset_id
    y_true, y_pred = load_series(d)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"task": "reg", "dataset": dataset_id, "metric": "rmse", "value": rmse}


def main() -> None:
    rows: list[dict[str, str | float]] = []

    for i in IDS:
        rows.append(eval_cls(i))
    for i in IDS:
        rows.append(eval_reg(i))

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
