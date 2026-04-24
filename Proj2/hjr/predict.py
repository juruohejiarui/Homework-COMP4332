# predict_all.py
import os
os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier, TabPFNRegressor

def load_xy(train_path: Path, test_path: Path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 最后一列是 target
    target_col = train_df.columns[-1]

    scaler = StandardScaler()
    # only scale float columns except target_col
    feature_cols = train_df.select_dtypes(include=["float"]).columns.tolist()
    if len(feature_cols) > 1 :
        feature_cols = [col for col in feature_cols if col != target_col]
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    X_train = train_df.iloc[:, :-1].to_numpy()
    y_train = train_df.iloc[:, -1].to_numpy()

    # test 有两种常见格式：
    # 1) test.csv 也带 target 列（可能为空/占位） -> 用除最后一列以外作为特征
    # 2) test.csv 不带 target 列 -> 用全部列作为特征
    if list(test_df.columns) == list(train_df.columns):
        X_test = test_df.iloc[:, :-1].to_numpy()
    else:
        X_test = test_df.to_numpy()

    return X_train, y_train, X_test, target_col

import numpy as np

import numpy as np

def stratified_min_per_class_sample(
    X: np.ndarray,
    y: np.ndarray,
    max_context: int,
    min_per_class: int = 40,
    seed : int = 0,
    oversample_small_classes: bool = True,
):
    """
    分层采样（分类）：
    - 尽量保持类别比例
    - 每个类至少 min_per_class 个
    - 若某类样本不足 min_per_class：
        oversample_small_classes=True  -> 允许有放回采样复制补足
        oversample_small_classes=False -> 只使用该类所有样本，不补足
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)
    if max_context <= 0 or n <= max_context:
        return X, y

    rng = np.random.default_rng(seed)

    classes, counts = np.unique(y, return_counts=True)
    C = len(classes)

    # 如果 max_context 小到连每类最小都塞不下：按比例缩小每类下限
    # （尽量平均分配）
    if max_context < C * min_per_class:
        scaled_min = max(1, max_context // C)
    else:
        scaled_min = min_per_class

    # 先给每类分配“基础配额”（至少 scaled_min，且不超过该类样本数 unless oversample）
    base = np.full(C, scaled_min, dtype=int)
    if not oversample_small_classes:
        base = np.minimum(base, counts)

    # 如果 base 总量超过 max_context，就再缩减（保证每类至少 1）
    if base.sum() > max_context:
        base = np.maximum(1, (base * (max_context / base.sum())).astype(int))
        # 调整到恰好 max_context
        while base.sum() > max_context:
            i = rng.integers(0, C)
            if base[i] > 1:
                base[i] -= 1
        while base.sum() < max_context:
            i = rng.integers(0, C)
            base[i] += 1

    remaining = max_context - base.sum()
    # 剩余配额按类别比例分配（基于 counts）
    if remaining > 0:
        prop = counts / counts.sum()
        extra = np.floor(prop * remaining).astype(int)
        # 调整使 extra.sum() == remaining
        while extra.sum() < remaining:
            i = rng.choice(C, p=prop)
            extra[i] += 1
        while extra.sum() > remaining:
            i = rng.integers(0, C)
            if extra[i] > 0:
                extra[i] -= 1
        alloc = base + extra
    else:
        alloc = base

    # 真正采样
    idx_all = np.arange(n)
    chosen = []
    for i, c in enumerate(classes):
        idx_c = idx_all[y == c]
        k = int(alloc[i])

        if len(idx_c) == 0:
            continue

        if len(idx_c) >= k:
            chosen.extend(rng.choice(idx_c, size=k, replace=False).tolist())
        else:
            if oversample_small_classes:
                # 有放回复制补足
                chosen.extend(rng.choice(idx_c, size=k, replace=True).tolist())
            else:
                # 只用全部，不补足
                chosen.extend(idx_c.tolist())

    chosen = np.array(chosen, dtype=int)
    rng.shuffle(chosen)

    # 若 oversample_small_classes=False 可能导致数量不足 max_context，补齐（整体随机有放回）
    if len(chosen) < max_context:
        fill = rng.choice(chosen, size=max_context - len(chosen), replace=True)
        chosen = np.concatenate([chosen, fill])

    # 若超了则截断
    if len(chosen) > max_context:
        chosen = chosen[:max_context]

    return X[chosen], y[chosen]

def quantile_bucket_sample_regression(
    X: np.ndarray,
    y: np.ndarray,
    max_context: int,
    n_bins: int = 10,
    min_per_bin: int = 100,
    seed: int = 0,
    oversample_small_bins: bool = False,
):
    """
    回归分桶采样：
    - y 按分位数切成 n_bins 桶
    - 每桶尽量采样 min_per_bin 个
    - 桶内不足则可有放回复制补足
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)
    if max_context <= 0 or n <= max_context:
        return X, y

    rng = np.random.default_rng(seed)

    # 若 max_context 太小，缩放每桶目标
    target_total = n_bins * min_per_bin
    if max_context < target_total:
        min_per_bin_scaled = max(1, max_context // n_bins)
    else:
        min_per_bin_scaled = min_per_bin

    # 分位数边界
    # 注意：np.quantile 在重复值很多时可能出现相同边界，这会导致空桶；我们要处理
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(y, qs)

    # 用 digitize 分桶：bin in [0, n_bins-1]
    # 为避免边界重复导致“挤在同一桶”，这里用 right=True 并把最后一个边界设为 +inf
    edges2 = edges.copy()
    edges2[0] = -np.inf
    edges2[-1] = np.inf
    bin_id = np.digitize(y, edges2[1:-1], right=True)  # 0..n_bins-1

    idx_all = np.arange(n)
    chosen = []

    # 先每桶采 min_per_bin_scaled
    alloc = np.full(n_bins, min_per_bin_scaled, dtype=int)
    base_sum = alloc.sum()

    # 如果 base_sum 还没到 max_context，余量按桶大小比例再分配
    remaining = max_context - base_sum
    if remaining > 0:
        bin_counts = np.array([(bin_id == b).sum() for b in range(n_bins)], dtype=float)
        if bin_counts.sum() == 0:
            # 极端情况
            extra = np.zeros(n_bins, dtype=int)
        else:
            prop = bin_counts / bin_counts.sum()
            extra = np.floor(prop * remaining).astype(int)
            while extra.sum() < remaining:
                i = rng.choice(n_bins, p=prop)
                extra[i] += 1
            while extra.sum() > remaining:
                i = rng.integers(0, n_bins)
                if extra[i] > 0:
                    extra[i] -= 1
        alloc = alloc + extra
    elif remaining < 0:
        # 缩减到 max_context
        alloc = np.maximum(1, (alloc * (max_context / base_sum)).astype(int))
        while alloc.sum() > max_context:
            i = rng.integers(0, n_bins)
            if alloc[i] > 1:
                alloc[i] -= 1
        while alloc.sum() < max_context:
            i = rng.integers(0, n_bins)
            alloc[i] += 1

    # 桶内采样（不足则 oversample）
    for b in range(n_bins):
        idx_b = idx_all[bin_id == b]
        k = int(alloc[b])
        if len(idx_b) == 0:
            continue
        if len(idx_b) >= k:
            chosen.extend(rng.choice(idx_b, size=k, replace=False).tolist())
        else:
            if oversample_small_bins:
                chosen.extend(rng.choice(idx_b, size=k, replace=True).tolist())
            else:
                chosen.extend(idx_b.tolist())

    chosen = np.array(chosen, dtype=int)
    rng.shuffle(chosen)

    # 如果因为空桶导致数量不足，整体补齐
    if len(chosen) < max_context:
        # 用已选样本有放回补齐（或者从全体采样也行）
        fill = rng.choice(chosen, size=max_context - len(chosen), replace=True)
        chosen = np.concatenate([chosen, fill])

    if len(chosen) > max_context:
        chosen = chosen[:max_context]

    return X[chosen], y[chosen]

def maybe_subsample_context(X, y, max_context: int, task: str, seed: int):
    if task == "cls":
        return stratified_min_per_class_sample(
            X, y,
            max_context=max_context,
            min_per_class=30,
            seed=seed,
            oversample_small_classes=True,  # 或 False：不复制
        )
    else:
        return quantile_bucket_sample_regression(
            X, y,
            max_context=max_context,
            n_bins=20,
            min_per_bin=100,
            seed=seed,
            oversample_small_bins=True,
        )

def run_one_dataset(dataset_dir: Path, task: str, out_dir: Path, max_context: int, seed: int, device: str):
    train_path = dataset_dir / "train.csv"
    test_path = dataset_dir / "test.csv"

    X_train, y_train, X_test, target_col = load_xy(train_path, test_path)
    X_ctx, y_ctx = maybe_subsample_context(X_train, y_train, max_context, task, seed)

    if task == "cls":
        model = TabPFNClassifier(device=device)
        model.fit(X_ctx, y_ctx)
        pred = model.predict(X_test)
    else:
        model = TabPFNRegressor(device=device)
        model.fit(X_ctx, y_ctx)
        pred = model.predict(X_test)

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predict.csv"
    pd.DataFrame({target_col: pred}).to_csv(pred_path, index=False)
    return pred_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data", help="data 目录，包含 reg/ 和 cls/")
    ap.add_argument("--out_root", type=str, default="predictions", help="输出目录")
    ap.add_argument("--max_context", type=int, default=3000, help="上下文最大训练行数（太大可能超上下文限制/变慢）")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default="cpu", help="cpu 或 cuda")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    for task in ["reg", "cls"]:
        task_dir = data_root / task
        if not task_dir.exists():
            continue

        for dataset_dir in sorted([p for p in task_dir.iterdir() if p.is_dir()]):
            out_dir = out_root / task / dataset_dir.name
            pred_path = run_one_dataset(
                dataset_dir=dataset_dir,
                task=task,
                out_dir=out_dir,
                max_context=args.max_context,
                seed=args.seed,
                device=args.device,
            )
            print(f"[{task}] {dataset_dir.name}: wrote {pred_path}")

    print("Done.")


if __name__ == "__main__":
    main()