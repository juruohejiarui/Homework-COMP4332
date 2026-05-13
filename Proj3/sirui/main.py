"""COMP4332 Project 3 — strong tabular + neural ensemble for rating prediction.

* Residual **LightGBM** on a shrinkage CF prior, rich numeric/categorical features,
  and **TF-IDF → SVD** vectors from `product.json` text (available at test time).
* Optional **Wide & Deep** checkpoint blended with LightGBM; blend weight is
  grid-searched on the public validation split to minimise RMSE.
* CF prior shrinks cold **items** toward each product's catalog `average_rating`,
  cold **users** toward the train global mean; WideDeep uses the same per-product baseline.

Data layout (homework repo): ``DATA_DIR`` = ``Proj3/data/`` (``train.csv``,
``validation.csv``, ``test.csv``, ``product.json``). ``test.csv`` matches the hack
split pairs; run this script then read ``prediction.csv`` for test stars.

Run:
    pip install -r requirements.txt
    python main.py                       # writes ``val_pred.csv`` + ``prediction.csv``
    python generate_test_predictions.py  # same, from ``Proj3/`` or ``sirui/``
    python main.py --retrain-with-val    # refit on train+val before test preds
    python main.py --no-neural           # LightGBM (+SVD) only
    python main.py --svd-dim 0           # disable product SVD
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    _HAS_TORCH = True
except ImportError:
    import types

    torch = None  # type: ignore[assignment]
    nn = types.SimpleNamespace(Module=object)  # type: ignore
    Dataset = object
    DataLoader = None  # type: ignore[assignment]
    _HAS_TORCH = False

warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

TEXT_COL = "Text"
SUMMARY_COL = "Summary"
STAR_COL = "Star"


def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _product_document(p: dict) -> str:
    parts: list[str] = []
    t = p.get("title")
    if isinstance(t, str) and t.strip():
        parts.append(t.strip())
    mc = p.get("main_category")
    if isinstance(mc, str) and mc.strip():
        parts.append(mc.strip())
    for c in (p.get("categories") or [])[:8]:
        if isinstance(c, str) and c.strip():
            parts.append(c.strip())
    for feat in (p.get("features") or [])[:5]:
        if isinstance(feat, str) and feat.strip():
            parts.append(feat.strip()[:500])
    desc = p.get("description")
    if isinstance(desc, list) and desc and isinstance(desc[0], str):
        parts.append(desc[0][:3000])
    d = _clean_text(" . ".join(parts))
    return d if d else str(p.get("ProductID", ""))


def load_products(
    path: Path,
    *,
    svd_components: int = 56,
    seed: int = 42,
    use_cache: bool = True,
) -> pd.DataFrame:
    cache_dir = SCRIPT_DIR / ".cache"
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        st = path.stat()
        cache_key = hashlib.sha256(
            f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}|{svd_components}|{seed}".encode()
        ).hexdigest()[:24]
        cache_path = cache_dir / f"products_{cache_key}.pkl"
        if cache_path.is_file():
            print(f"Loading cached product features from {cache_path.name} …")
            return pd.read_pickle(cache_path)
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    docs: list[str] = []
    for p in raw:
        pid = p.get("ProductID")
        if pid is None:
            continue
        docs.append(_product_document(p))
        price = p.get("price")
        try:
            price_f = float(price) if price is not None and price != "" else np.nan
        except (TypeError, ValueError):
            price_f = np.nan
        main_cat = p.get("main_category")
        if not isinstance(main_cat, str) or not main_cat.strip():
            main_cat = "Unknown"
        else:
            main_cat = main_cat.strip()
        avg_r = p.get("average_rating")
        try:
            avg_r = float(avg_r) if avg_r is not None else np.nan
        except (TypeError, ValueError):
            avg_r = np.nan
        rn = p.get("rating_number")
        try:
            rn = float(rn) if rn is not None else np.nan
        except (TypeError, ValueError):
            rn = np.nan
        rows.append(
            {
                "ProductID": str(pid),
                "Category": main_cat,
                "price": price_f,
                "product_avg_rating": avg_r,
                "product_rating_number": rn,
            }
        )
    df = pd.DataFrame(rows)
    med_price = float(np.nanmedian(df["price"].to_numpy()))
    df["price_log"] = np.where(
        df["price"].astype(float) > 0,
        np.log1p(df["price"].astype(float)),
        np.nan,
    )
    df["price_log"] = df["price_log"].fillna(float(np.log1p(med_price)) if med_price > 0 else 0.0)
    df["product_avg_rating"] = df["product_avg_rating"].fillna(df["product_avg_rating"].median())
    df["product_rating_number"] = df["product_rating_number"].fillna(0.0)
    df["log_rating_number"] = np.log1p(df["product_rating_number"].clip(lower=0.0))

    if svd_components > 0 and len(docs) == len(df):
        vec = TfidfVectorizer(
            max_features=12000,
            min_df=2,
            max_df=0.98,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        X_sp = vec.fit_transform(docs)
        k = int(min(svd_components, max(1, X_sp.shape[1] - 1)))
        svd = TruncatedSVD(n_components=k, random_state=seed, n_iter=12)
        Z = svd.fit_transform(X_sp)
        for j in range(k):
            df[f"p_svd_{j}"] = Z[:, j].astype(np.float32)

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        st = path.stat()
        cache_key = hashlib.sha256(
            f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}|{svd_components}|{seed}".encode()
        ).hexdigest()[:24]
        cache_path = cache_dir / f"products_{cache_key}.pkl"
        df.to_pickle(cache_path)
        print(f"Cached enriched product table → {cache_path.name}")

    return df


def train_aggregates(train: pd.DataFrame) -> tuple[dict, dict, float]:
    gmean = float(train[STAR_COL].mean())
    tg = train.assign(
        ReviewerID=train["ReviewerID"].astype(str),
        ProductID=train["ProductID"].astype(str),
    )
    ug = tg.groupby("ReviewerID", observed=False)[STAR_COL].agg(["sum", "count"])
    user_mean = (ug["sum"] / ug["count"]).astype(float)
    user_mean = {str(i): float(v) for i, v in user_mean.items()}
    user_n = {str(i): float(v) for i, v in ug["count"].items()}

    ig = tg.groupby("ProductID", observed=False)[STAR_COL].agg(["sum", "count"])
    item_mean = (ig["sum"] / ig["count"]).astype(float)
    item_mean = {str(i): float(v) for i, v in item_mean.items()}
    item_n = {str(i): float(v) for i, v in ig["count"].items()}

    return (
        {"mean": user_mean, "count": user_n},
        {"mean": item_mean, "count": item_n},
        gmean,
    )


def build_train_text_aggregates(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Per-user / per-item mean review-text statistics (train split only). Usable at inference."""

    t = train.assign(
        ReviewerID=train["ReviewerID"].astype(str),
        ProductID=train["ProductID"].astype(str),
    )
    if TEXT_COL in t.columns:
        tc = t[TEXT_COL].map(_clean_text)
        t["_tx_len"] = tc.str.len().fillna(0).astype(np.float64)
        t["_tx_wc"] = tc.str.split().str.len().fillna(0).astype(np.float64)
    else:
        t["_tx_len"] = 0.0
        t["_tx_wc"] = 0.0
    if SUMMARY_COL in t.columns:
        sc = t[SUMMARY_COL].map(_clean_text)
        t["_sm_len"] = sc.str.len().fillna(0).astype(np.float64)
    else:
        t["_sm_len"] = 0.0

    text_glo = {
        "tx_len": float(np.median(t["_tx_len"].to_numpy())),
        "tx_wc": float(np.median(t["_tx_wc"].to_numpy())),
        "sm_len": float(np.median(t["_sm_len"].to_numpy())),
    }

    u = t.groupby("ReviewerID", observed=False).agg(
        u_tx_len=("_tx_len", "mean"),
        u_tx_wc=("_tx_wc", "mean"),
        u_sm_len=("_sm_len", "mean"),
    ).reset_index()
    it = t.groupby("ProductID", observed=False).agg(
        i_tx_len=("_tx_len", "mean"),
        i_tx_wc=("_tx_wc", "mean"),
        i_sm_len=("_sm_len", "mean"),
    ).reset_index()
    return u, it, text_glo


SHRINK_STRENGTH = 14.0


def featurize(
    df: pd.DataFrame,
    *,
    products: pd.DataFrame,
    user_stats: dict,
    item_stats: dict,
    global_mean: float,
    text_u: pd.DataFrame,
    text_i: pd.DataFrame,
    text_glo: dict[str, float],
) -> pd.DataFrame:
    out = df.copy()
    out["ReviewerID"] = out["ReviewerID"].astype(str)
    out["ProductID"] = out["ProductID"].astype(str)

    out = out.merge(products, on="ProductID", how="left")
    out["Category"] = out["Category"].fillna("Unknown").astype(str)
    prod_prior = out["product_avg_rating"].astype(float).fillna(global_mean)

    um = user_stats["mean"]
    uc = user_stats["count"]
    im = item_stats["mean"]
    ic = item_stats["count"]

    out["user_avg_star"] = out["ReviewerID"].map(um).astype(float)
    out["item_avg_star"] = out["ProductID"].map(im).astype(float)
    out["user_count"] = out["ReviewerID"].map(uc).fillna(0.0).astype(float)
    out["item_count"] = out["ProductID"].map(ic).fillna(0.0).astype(float)

    out["user_avg_star"] = out["user_avg_star"].fillna(global_mean)
    out["item_avg_star"] = out["item_avg_star"].fillna(prod_prior)

    s = SHRINK_STRENGTH
    out["user_avg_shrunk"] = (out["user_count"] * out["user_avg_star"] + s * global_mean) / (
        out["user_count"] + s
    )
    out["item_avg_shrunk"] = (out["item_count"] * out["item_avg_star"] + s * prod_prior) / (
        out["item_count"] + s
    )
    out["cf_prior"] = out["user_avg_shrunk"] + out["item_avg_shrunk"] - prod_prior

    out["user_count_log"] = np.log1p(out["user_count"])
    out["item_count_log"] = np.log1p(out["item_count"])
    out["user_global_delta"] = out["user_avg_star"] - global_mean
    out["item_product_delta"] = out["item_avg_star"] - prod_prior

    out = out.merge(text_u, on="ReviewerID", how="left").merge(text_i, on="ProductID", how="left")
    glo = text_glo
    out["u_tx_len"] = out["u_tx_len"].fillna(glo["tx_len"])
    out["u_tx_wc"] = out["u_tx_wc"].fillna(glo["tx_wc"])
    out["u_sm_len"] = out["u_sm_len"].fillna(glo["sm_len"])
    out["i_tx_len"] = out["i_tx_len"].fillna(glo["tx_len"])
    out["i_tx_wc"] = out["i_tx_wc"].fillna(glo["tx_wc"])
    out["i_sm_len"] = out["i_sm_len"].fillna(glo["sm_len"])

    return out


NUMERICALS_CORE = [
    "price_log",
    "user_avg_star",
    "item_avg_star",
    "user_avg_shrunk",
    "item_avg_shrunk",
    "cf_prior",
    "user_count_log",
    "item_count_log",
    "user_global_delta",
    "item_product_delta",
    "product_avg_rating",
    "log_rating_number",
    "u_tx_len",
    "u_tx_wc",
    "u_sm_len",
    "i_tx_len",
    "i_tx_wc",
    "i_sm_len",
]


CATEGORICALS = ["ReviewerID", "ProductID", "Category"]


def numerical_columns_from_products(products: pd.DataFrame) -> list[str]:
    svd = sorted(c for c in products.columns if c.startswith("p_svd_"))
    return NUMERICALS_CORE + svd


def lgb_params() -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.04,
        "num_leaves": 255,
        "min_data_in_leaf": 35,
        "feature_fraction": 0.88,
        "bagging_fraction": 0.88,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 2.5,
        "verbosity": -1,
        "force_col_wise": True,
    }


def to_lgb_frame(df: pd.DataFrame, numericals: list[str]) -> pd.DataFrame:
    X = df[CATEGORICALS + numericals].copy()
    for c in CATEGORICALS:
        X[c] = X[c].astype("category")
    for c in numericals:
        if c not in X.columns:
            X[c] = 0.0
        X[c] = X[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def train_lgb_residual(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    base_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    base_val: np.ndarray,
    *,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[lgb.Booster, np.ndarray, np.ndarray]:
    """Fit booster on (y - baseline) and return (model, val_pred, best_iteration)."""

    y_res = y_train - base_train
    params = lgb_params()
    dtr = lgb.Dataset(
        X_train, label=y_res, categorical_feature=CATEGORICALS, free_raw_data=False
    )
    dva = lgb.Dataset(
        X_val,
        label=y_val - base_val,
        categorical_feature=CATEGORICALS,
        reference=dtr,
        free_raw_data=False,
    )
    model = lgb.train(
        params,
        dtr,
        num_boost_round=num_boost_round,
        valid_sets=[dtr, dva],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
    )
    best_it = model.best_iteration
    res_val = model.predict(X_val, num_iteration=best_it)
    val_pred = np.clip(base_val + res_val, 1.0, 5.0)
    return model, val_pred, best_it


def cross_validate(train_X: pd.DataFrame, y: np.ndarray, *, n_splits: int, seed: int) -> float:
    params = lgb_params()
    bases = np.clip(train_X["cf_prior"].to_numpy(dtype=np.float64), 1.0, 5.0)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_X), start=1):
        X_tr, X_va = train_X.iloc[tr_idx], train_X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        b_tr, b_va = bases[tr_idx], bases[va_idx]
        y_tr_res, y_va_res = y_tr - b_tr, y_va - b_va
        dtr = lgb.Dataset(
            X_tr, label=y_tr_res, categorical_feature=CATEGORICALS, free_raw_data=False
        )
        dva = lgb.Dataset(
            X_va,
            label=y_va_res,
            categorical_feature=CATEGORICALS,
            reference=dtr,
            free_raw_data=False,
        )
        model = lgb.train(
            params,
            dtr,
            num_boost_round=5000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)],
        )
        res = model.predict(X_va, num_iteration=model.best_iteration)
        pred = np.clip(b_va + res, 1.0, 5.0)
        scores.append(rmse(y_va, pred))
        print(f"  Fold {fold}/{n_splits} RMSE (train-CV): {scores[-1]:.4f}")
    mean_cv = float(np.mean(scores))
    print(f"  Mean {n_splits}-fold CV RMSE on train: {mean_cv:.4f}")
    return mean_cv


def _build_id_map(values: pd.Series) -> dict[str, int]:
    return {v: i for i, v in enumerate(sorted(set(values.astype(str))))}


class _RatingDS(Dataset):
    def __init__(self, u, i, r, base):
        self.u = torch.LongTensor(u)
        self.i = torch.LongTensor(i)
        self.r = torch.FloatTensor(r)
        self.base = torch.FloatTensor(base)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.r[idx], self.base[idx]


class _WideDeep(nn.Module):
    def __init__(self, n_u, n_i, emb_dim, hidden, dropout):
        super().__init__()
        self.user_bias = nn.Embedding(n_u, 1)
        self.item_bias = nn.Embedding(n_i, 1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        self.user_emb = nn.Embedding(n_u, emb_dim)
        self.item_emb = nn.Embedding(n_i, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.02)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        layers, d = [], 2 * emb_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.deep = nn.Sequential(*layers)

    def forward(self, u, i, prod_base):
        wide = prod_base + self.user_bias(u).squeeze(-1) + self.item_bias(i).squeeze(-1)
        x = torch.cat([self.user_emb(u), self.item_emb(i)], dim=-1)
        return wide + self.deep(x).squeeze(-1)


def _product_baseline_series(df: pd.DataFrame, pmap: dict[str, float], gmean: float) -> np.ndarray:
    s = df["ProductID"].astype(str).map(pmap)
    return s.fillna(gmean).to_numpy(dtype=np.float64)


def _nn_predict(
    model,
    df,
    u_map,
    i_map,
    baseline: np.ndarray,
    device,
    batch=4096,
):
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, len(df), batch):
            chunk = df.iloc[s : s + batch]
            u_idx = chunk["ReviewerID"].astype(str).map(u_map)
            i_idx = chunk["ProductID"].astype(str).map(i_map)
            seen = u_idx.notna() & i_idx.notna()
            row = baseline[s : s + len(chunk)].copy()
            if bool(seen.any()):
                ut = torch.LongTensor(u_idx[seen].astype(int).to_numpy()).to(device)
                it = torch.LongTensor(i_idx[seen].astype(int).to_numpy()).to(device)
                bt = torch.FloatTensor(baseline[s : s + len(chunk)][seen.to_numpy()]).to(device)
                pred = model(ut, it, bt).clamp(1.0, 5.0).cpu().numpy()
                pos = np.where(seen.to_numpy())[0]
                row[pos] = pred
            chunks.append(row)
    return np.concatenate(chunks)


def _product_rating_map(products: pd.DataFrame) -> dict[str, float]:
    return dict(
        zip(
            products["ProductID"].astype(str),
            products["product_avg_rating"].astype(float),
        )
    )


def train_neural_blend(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    products: pd.DataFrame,
    global_mean: float,
    *,
    seed: int,
    epochs: int = 36,
    batch_size: int = 768,
    patience: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    if not _HAS_TORCH:
        raise RuntimeError("torch is required for --neural ensemble")
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pmap = _product_rating_map(products)
    tr = train_df.assign(
        ReviewerID=train_df["ReviewerID"].astype(str),
        ProductID=train_df["ProductID"].astype(str),
    )
    val_id = val_df.assign(
        ReviewerID=val_df["ReviewerID"].astype(str),
        ProductID=val_df["ProductID"].astype(str),
    )
    test_id = test_df.assign(
        ReviewerID=test_df["ReviewerID"].astype(str),
        ProductID=test_df["ProductID"].astype(str),
    )
    u_map = _build_id_map(tr["ReviewerID"])
    i_map = _build_id_map(tr["ProductID"])
    u_tr = tr["ReviewerID"].map(u_map).astype(int).to_numpy()
    i_tr = tr["ProductID"].map(i_map).astype(int).to_numpy()
    r_tr = tr[STAR_COL].astype(float).to_numpy()
    base_tr = _product_baseline_series(tr, pmap, global_mean)

    emb_dim, hidden, dropout = 44, [88, 44, 18], 0.3
    model = _WideDeep(len(u_map), len(i_map), emb_dim, hidden, dropout).to(device)
    bias_p, other_p = [], []
    for name, p in model.named_parameters():
        if "bias" in name and p.dim() == 2:
            bias_p.append(p)
        else:
            other_p.append(p)
    opt = torch.optim.Adam(
        [{"params": bias_p, "weight_decay": 7e-3}, {"params": other_p, "weight_decay": 1.8e-4}],
        lr=1.05e-3,
    )
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        _RatingDS(u_tr, i_tr, r_tr, base_tr),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    base_va = _product_baseline_series(val_id, pmap, global_mean)
    base_te = _product_baseline_series(test_id, pmap, global_mean)

    best_rmse, best_state = float("inf"), None
    y_va = val_df[STAR_COL].to_numpy(dtype=np.float64)
    stagnant = 0
    for ep in range(1, epochs + 1):
        model.train()
        for u_b, i_b, r_b, b_b in loader:
            u_b, i_b, r_b, b_b = u_b.to(device), i_b.to(device), r_b.to(device), b_b.to(device)
            opt.zero_grad()
            loss = loss_fn(model(u_b, i_b, b_b), r_b)
            loss.backward()
            opt.step()

        v_pred = _nn_predict(model, val_id, u_map, i_map, base_va, device)
        vr = rmse(y_va, v_pred)
        if vr < best_rmse - 1e-5:
            best_rmse = vr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= patience:
                print(f"  WideDeep early stop at epoch {ep} (no val gain {patience} epochs)")
                break
        if ep % 4 == 0 or ep == 1:
            print(f"  WideDeep epoch {ep:02d}/{epochs} | val RMSE={vr:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    val_p = _nn_predict(model, val_id, u_map, i_map, base_va, device)
    test_p = _nn_predict(model, test_id, u_map, i_map, base_te, device)
    print(f"  WideDeep best val RMSE (train-only NN): {best_rmse:.4f}")
    return val_p, test_p


def find_lgb_nn_cf_blend(
    y_true: np.ndarray,
    lgb_p: np.ndarray,
    nn_p: np.ndarray,
    baseline_p: np.ndarray,
    *,
    steps: int = 41,
) -> tuple[float, float, float]:
    """Grid-search non-negative weights (w_lgb, w_nn, w_base) summing to 1."""

    best_wl, best_wn, best_r = 1.0, 0.0, rmse(y_true, lgb_p)
    grid = np.linspace(0.0, 1.0, steps)
    for wl in grid:
        for wn in grid:
            if wl + wn > 1.0 + 1e-9:
                continue
            wb = 1.0 - wl - wn
            pred = np.clip(wl * lgb_p + wn * nn_p + wb * baseline_p, 1.0, 5.0)
            r = rmse(y_true, pred)
            if r < best_r:
                best_r, best_wl, best_wn = r, float(wl), float(wn)
        # minor pruning: none
    return best_wl, best_wn, best_r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "prediction.csv")
    ap.add_argument("--val-pred", type=Path, default=SCRIPT_DIR / "val_pred.csv")
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument(
        "--retrain-with-val",
        action="store_true",
        help="After choosing rounds on validation, refit on train+validation for test `prediction.csv`.",
    )
    ap.add_argument(
        "--no-neural",
        action="store_true",
        help="Skip WideDeep training and LGBM+NN blending (tabular only).",
    )
    ap.add_argument(
        "--svd-dim",
        type=int,
        default=56,
        help="TruncatedSVD dims for product.json text (0 disables).",
    )
    ap.add_argument(
        "--nn-epochs",
        type=int,
        default=40,
        help="Max WideDeep epochs (early stopping applies).",
    )
    ap.add_argument(
        "--no-product-cache",
        action="store_true",
        help="Recompute TF-IDF/SVD on product.json (ignore sirui/.cache).",
    )
    args = ap.parse_args()

    data_dir: Path = args.data_dir.expanduser().resolve()
    train_path = data_dir / "train.csv"
    val_path = data_dir / "validation.csv"
    test_path = data_dir / "test.csv"
    product_path = data_dir / "product.json"

    print("Loading tables …")
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    products = load_products(
        product_path,
        svd_components=args.svd_dim,
        seed=args.seed,
        use_cache=not args.no_product_cache,
    )
    num_cols = numerical_columns_from_products(products)

    user_stats, item_stats, global_mean = train_aggregates(train)
    text_u, text_i, text_glo = build_train_text_aggregates(train)
    print(f"Train global mean (cold users / unknown products): {global_mean:.4f}")

    train_f = featurize(
        train,
        products=products,
        user_stats=user_stats,
        item_stats=item_stats,
        global_mean=global_mean,
        text_u=text_u,
        text_i=text_i,
        text_glo=text_glo,
    )
    val_f = featurize(
        val,
        products=products,
        user_stats=user_stats,
        item_stats=item_stats,
        global_mean=global_mean,
        text_u=text_u,
        text_i=text_i,
        text_glo=text_glo,
    )
    test_f = featurize(
        test,
        products=products,
        user_stats=user_stats,
        item_stats=item_stats,
        global_mean=global_mean,
        text_u=text_u,
        text_i=text_i,
        text_glo=text_glo,
    )

    X_train = to_lgb_frame(train_f, num_cols)
    y_train = train_f[STAR_COL].to_numpy(dtype=np.float64)
    X_val = to_lgb_frame(val_f, num_cols)
    y_val = val_f[STAR_COL].to_numpy(dtype=np.float64)
    X_test = to_lgb_frame(test_f, num_cols)

    if not args.skip_cv:
        print(f"\n{args.cv_splits}-fold cross-validation on training split:")
        cross_validate(X_train, y_train, n_splits=args.cv_splits, seed=args.seed)

    base_tr = np.clip(train_f["cf_prior"].to_numpy(dtype=np.float64), 1.0, 5.0)
    base_va = np.clip(val_f["cf_prior"].to_numpy(dtype=np.float64), 1.0, 5.0)

    print("\nTraining final model (residual LightGBM; early stopping on validation) …")
    model, val_pred, best_it = train_lgb_residual(
        X_train,
        y_train,
        base_tr,
        X_val,
        y_val,
        base_va,
        num_boost_round=8000,
        early_stopping_rounds=120,
    )
    val_pred_lgb = val_pred
    print(f"Validation RMSE — residual LightGBM: {rmse(y_val, val_pred_lgb):.4f}")

    nn_test: np.ndarray | None = None
    blend_wl, blend_wn = 1.0, 0.0
    if not args.no_neural and _HAS_TORCH:
        print("\nTraining WideDeep for LGBM + neural blend …")
        val_nn, nn_test = train_neural_blend(
            train,
            val,
            test,
            products,
            global_mean,
            seed=args.seed,
            epochs=args.nn_epochs,
        )
        blend_wl, blend_wn, blend_rmse = find_lgb_nn_cf_blend(
            y_val, val_pred_lgb, val_nn, base_va, steps=45
        )
        blend_wb = 1.0 - blend_wl - blend_wn
        val_pred = np.clip(
            blend_wl * val_pred_lgb + blend_wn * val_nn + blend_wb * base_va,
            1.0,
            5.0,
        )
        print(
            f"Validation RMSE — 3-way blend: {rmse(y_val, val_pred):.4f} "
            f"(weights LGB={blend_wl:.3f}, NN={blend_wn:.3f}, CF={blend_wb:.3f}; "
            f"grid-best RMSE={blend_rmse:.4f})"
        )
    elif not args.no_neural:
        print("\n`torch` not installed — neural ensemble skipped. `pip install torch` to enable.")

    val_out = val[["ReviewerID", "ProductID"]].copy()
    val_out["Star"] = val_pred
    args.val_pred.parent.mkdir(parents=True, exist_ok=True)
    val_out.to_csv(args.val_pred, index=False)
    print(f"Wrote validation predictions → {args.val_pred}")
    print("  Check with: python ../evaluate.py --pred sirui/val_pred.csv")

    base_te = np.clip(test_f["cf_prior"].to_numpy(dtype=np.float64), 1.0, 5.0)

    if args.retrain_with_val:
        print("\nRefitting on train+validation for test predictions …")
        tv = pd.concat([train, val], axis=0, ignore_index=True)
        user2, item2, g2 = train_aggregates(tv)
        tu2, ti2, tg2 = build_train_text_aggregates(tv)
        tv_f = featurize(
            tv,
            products=products,
            user_stats=user2,
            item_stats=item2,
            global_mean=g2,
            text_u=tu2,
            text_i=ti2,
            text_glo=tg2,
        )
        test_f2 = featurize(
            test,
            products=products,
            user_stats=user2,
            item_stats=item2,
            global_mean=g2,
            text_u=tu2,
            text_i=ti2,
            text_glo=tg2,
        )
        X_tv = to_lgb_frame(tv_f, num_cols)
        y_tv = tv_f[STAR_COL].to_numpy(dtype=np.float64)
        X_test2 = to_lgb_frame(test_f2, num_cols)
        base_tv = np.clip(tv_f["cf_prior"].to_numpy(dtype=np.float64), 1.0, 5.0)
        y_res_tv = y_tv - base_tv
        d_all = lgb.Dataset(
            X_tv, label=y_res_tv, categorical_feature=CATEGORICALS, free_raw_data=False
        )
        model_final = lgb.train(lgb_params(), d_all, num_boost_round=int(best_it))
        base_te2 = np.clip(test_f2["cf_prior"].to_numpy(dtype=np.float64), 1.0, 5.0)
        test_lgb = np.clip(base_te2 + model_final.predict(X_test2), 1.0, 5.0)
    else:
        res_te = model.predict(X_test, num_iteration=best_it)
        test_lgb = np.clip(base_te + res_te, 1.0, 5.0)

    test_pred = test_lgb
    if nn_test is not None:
        blend_wb = 1.0 - blend_wl - blend_wn
        if args.retrain_with_val:
            cf_b = base_te2
        else:
            cf_b = base_te
        test_pred = np.clip(
            blend_wl * test_lgb + blend_wn * nn_test + blend_wb * cf_b,
            1.0,
            5.0,
        )

    out = test[["ReviewerID", "ProductID"]].copy()
    out["Star"] = test_pred
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote test predictions → {args.out}")


if __name__ == "__main__":
    main()
