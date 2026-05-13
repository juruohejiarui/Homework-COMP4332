# train_rating_model.py
import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    pred = pred.astype(np.float64)
    y = y.astype(np.float64)
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def clamp_ratings(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 1.0, 5.0)


# ----------------------------
# Text building
# ----------------------------
def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return " ".join([safe_str(t) for t in x])
    if isinstance(x, dict):
        # flatten dict keys/values
        parts = []
        for k, v in x.items():
            parts.append(f"{k}: {safe_str(v)}")
        return " ".join(parts)
    return str(x)


def build_item_meta_text(prod: dict) -> str:
    # You can customize this template freely
    fields = []
    fields.append(f"TITLE: {safe_str(prod.get('title',''))}")
    fields.append(f"MAIN_CATEGORY: {safe_str(prod.get('main_category',''))}")
    fields.append(f"CATEGORIES: {safe_str(prod.get('categories',''))}")
    fields.append(f"BRAND: {safe_str(prod.get('store',''))}")
    fields.append(f"PRICE: {safe_str(prod.get('price',''))}")
    fields.append(f"AVG_RATING: {safe_str(prod.get('average_rating',''))}")
    fields.append(f"RATING_NUM: {safe_str(prod.get('rating_number',''))}")
    fields.append(f"FEATURES: {safe_str(prod.get('features',''))}")
    fields.append(f"DESCRIPTION: {safe_str(prod.get('description',''))}")
    # details can be huge; keep short
    details = prod.get("details", {})
    if isinstance(details, dict):
        # keep top-k keys only to avoid extremely long text
        keep_keys = list(details.keys())[:20]
        short_details = {k: details[k] for k in keep_keys}
        fields.append(f"DETAILS: {safe_str(short_details)}")
    else:
        fields.append(f"DETAILS: {safe_str(details)}")

    text = " [SEP] ".join([f for f in fields if f.strip()])
    return text


def build_user_profile_text(
    user_reviews: List[Tuple[str, str, float]],
    max_reviews: int = 20,
    per_review_max_chars: int = 400,
) -> str:
    """
    user_reviews: list of (Summary, Text, Star) from training interactions of this user.
    We sample/keep first N reviews (you can change to most recent if you have timestamp).
    """
    # simplest: keep first max_reviews. You can also random sample.
    reviews = user_reviews[:max_reviews]
    parts = []
    for s, t, star in reviews:
        s = safe_str(s)[:per_review_max_chars]
        t = safe_str(t)[:per_review_max_chars]
        parts.append(f"STAR {star}. SUMMARY: {s} TEXT: {t}")
    return " [SEP] ".join(parts)


# ----------------------------
# Transformer encoder with caching
# ----------------------------
@torch.no_grad()
def encode_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 256,
) -> torch.Tensor:
    """
    Returns (N, hidden) float32 tensor on CPU.
    Uses mean pooling over last_hidden_state with attention mask.
    """
    model.eval()
    all_out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        last = out.last_hidden_state  # (B, L, H)
        mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
        summed = torch.sum(last * mask, dim=1)  # (B, H)
        denom = torch.clamp(mask.sum(dim=1), min=1e-6)  # (B, 1)
        pooled = summed / denom  # (B, H)
        all_out.append(pooled.detach().cpu().float())
    return torch.cat(all_out, dim=0)


def maybe_cache_embeddings(
    cache_path: str,
    texts: List[str],
    tokenizer_name: str,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    """
    Cache embeddings to disk as .pt:
      { "tokenizer_name": ..., "max_length": ..., "emb": Tensor(N,H) }
    """
    if os.path.exists(cache_path):
        obj = torch.load(cache_path, map_location="cpu")
        emb = obj["emb"]
        return emb

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    model = AutoModel.from_pretrained(tokenizer_name, dtype=torch.bfloat16)
    model.to(device)

    emb = encode_texts(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    torch.save(
        {
            "tokenizer_name": tokenizer_name,
            "max_length": max_length,
            "emb": emb,
        },
        cache_path,
    )
    return emb


# ----------------------------
# Data prep
# ----------------------------
@dataclass
class PreparedData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    user2idx: Dict[str, int]
    item2idx: Dict[str, int]
    user_text_emb: Optional[torch.Tensor]  # (num_users, H) or None
    item_text_emb: Optional[torch.Tensor]  # (num_items, H) or None
    global_mean: float


def load_product_json(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    mp = {}
    for p in arr:
        pid = p.get("ProductID")
        if pid is not None:
            mp[pid] = p
    return mp


def prepare(
    data_dir: str,
    tokenizer_name: str,
    cache_dir: str,
    device: torch.device,
    text_batch_size: int,
    text_max_length: int,
    user_max_reviews: int,
) -> PreparedData:
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "validation.csv")
    test_path = os.path.join(data_dir, "test.csv")
    prod_path = os.path.join(data_dir, "product.json")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Ensure expected columns
    for col in ["ReviewerID", "ProductID", "Star"]:
        assert col in train_df.columns
        assert col in val_df.columns
        assert col in test_df.columns

    global_mean = float(train_df["Star"].mean())

    # build vocab from all splits (so test users/items are not OOV if present)
    all_users = pd.concat([train_df["ReviewerID"], val_df["ReviewerID"], test_df["ReviewerID"]]).astype(str).unique()
    all_items = pd.concat([train_df["ProductID"], val_df["ProductID"], test_df["ProductID"]]).astype(str).unique()

    # 0 reserved for OOV (just in case)
    user2idx = {"<OOV>": 0}
    item2idx = {"<OOV>": 0}
    for u in all_users:
        if u not in user2idx:
            user2idx[u] = len(user2idx)
    for i in all_items:
        if i not in item2idx:
            item2idx[i] = len(item2idx)

    # map ids to indices
    def map_ids(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["u_idx"] = df["ReviewerID"].astype(str).map(lambda x: user2idx.get(x, 0)).astype(np.int64)
        df["i_idx"] = df["ProductID"].astype(str).map(lambda x: item2idx.get(x, 0)).astype(np.int64)
        return df

    train_df = map_ids(train_df)
    val_df = map_ids(val_df)
    test_df = map_ids(test_df)

    # Text embeddings
    prod_map = load_product_json(prod_path)

    # Build item meta texts aligned by item index
    num_users = len(user2idx)
    num_items = len(item2idx)

    item_texts = [""] * num_items
    # OOV item at idx 0
    item_texts[0] = "UNKNOWN_ITEM"
    for pid, idx in item2idx.items():
        if idx == 0:
            continue
        prod = prod_map.get(pid)
        if prod is None:
            item_texts[idx] = f"TITLE: {pid}"
        else:
            item_texts[idx] = build_item_meta_text(prod)

    # Build user profile texts aligned by user index
    user_texts = [""] * num_users
    user_texts[0] = "UNKNOWN_USER"

    # gather user reviews from train only
    grouped = train_df.groupby("ReviewerID", sort=False)
    user_reviews_dict: Dict[str, List[Tuple[str, str, float]]] = {}
    for uid, g in grouped:
        # Keep order as in file (not strictly time)
        lst = []
        for _, row in g.iterrows():
            lst.append((row.get("Summary", ""), row.get("Text", ""), float(row["Star"])))
        user_reviews_dict[str(uid)] = lst

    for uid, idx in user2idx.items():
        if idx == 0:
            continue
        reviews = user_reviews_dict.get(uid, [])
        if len(reviews) == 0:
            user_texts[idx] = "NO_HISTORY"
        else:
            user_texts[idx] = build_user_profile_text(reviews, max_reviews=user_max_reviews)

    # cache
    user_cache = os.path.join(cache_dir, f"user_text_emb_{tokenizer_name.replace('/','_')}_L{text_max_length}.pt")
    item_cache = os.path.join(cache_dir, f"item_text_emb_{tokenizer_name.replace('/','_')}_L{text_max_length}.pt")

    print(f"[Info] Encoding/caching user texts -> {user_cache}")
    user_text_emb = maybe_cache_embeddings(
        cache_path=user_cache,
        texts=user_texts,
        tokenizer_name=tokenizer_name,
        device=device,
        batch_size=text_batch_size,
        max_length=text_max_length,
    )
    print(f"[Info] Encoding/caching item texts -> {item_cache}")
    item_text_emb = maybe_cache_embeddings(
        cache_path=item_cache,
        texts=item_texts,
        tokenizer_name=tokenizer_name,
        device=device,
        batch_size=text_batch_size,
        max_length=text_max_length,
    )

    return PreparedData(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        user2idx=user2idx,
        item2idx=item2idx,
        user_text_emb=user_text_emb,
        item_text_emb=item_text_emb,
        global_mean=global_mean,
    )


# ----------------------------
# Dataset
# ----------------------------
class RatingsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, with_label: bool = True):
        self.u = df["u_idx"].values.astype(np.int64)
        self.i = df["i_idx"].values.astype(np.int64)
        self.with_label = with_label
        self.y = None
        if with_label:
            self.y = df["Star"].values.astype(np.float32)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        if self.with_label:
            return self.u[idx], self.i[idx], self.y[idx]
        return self.u[idx], self.i[idx]


# ----------------------------
# Model: Wide&Deep + Text
# ----------------------------
class WideDeepText(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        text_dim: int,
        emb_dim: int = 32,
        mlp_hidden: Tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        global_mean: float = 4.5,
        use_text: bool = True,
    ):
        super().__init__()
        self.use_text = use_text
        self.global_mean = nn.Parameter(torch.tensor([global_mean], dtype=torch.float32))

        # Wide (bias)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # ID embeddings
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        in_dim = emb_dim * 2
        if use_text:
            # project text to emb_dim
            self.user_text_proj = nn.Linear(text_dim, emb_dim)
            self.item_text_proj = nn.Linear(text_dim, emb_dim)

            # gating based on (u_text, i_text, u_id, i_id)
            self.gate = nn.Sequential(
                nn.Linear(emb_dim * 4, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, 1),
                nn.Sigmoid(),
            )
            in_dim = emb_dim * 4  # u_id, i_id, u_text, i_text

        # Deep MLP predicts residual
        layers: List[nn.Module] = []
        prev = in_dim
        for h in mlp_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.mlp = nn.Sequential(*layers)

        # init biases to 0
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(
        self,
        u_idx: torch.Tensor,
        i_idx: torch.Tensor,
        user_text_emb: Optional[torch.Tensor] = None,
        item_text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        u_idx, i_idx: (B,)
        user_text_emb/item_text_emb: full tables on device, shape (num_users, H)
        """
        u_b = self.user_bias(u_idx).squeeze(-1)
        i_b = self.item_bias(i_idx).squeeze(-1)

        u_e = self.user_emb(u_idx)
        i_e = self.item_emb(i_idx)

        wide = self.global_mean.squeeze(0) + u_b + i_b  # (B,)

        if self.use_text:
            assert user_text_emb is not None and item_text_emb is not None
            u_t = self.user_text_proj(user_text_emb[u_idx])  # (B, emb_dim)
            i_t = self.item_text_proj(item_text_emb[i_idx])  # (B, emb_dim)

            # gate decides how much to trust text vs ID embeddings (simple but effective)
            g_in = torch.cat([u_e, i_e, u_t, i_t], dim=-1)
            g = self.gate(g_in)  # (B,1)
            u_mix = (1 - g) * u_e + g * u_t
            i_mix = (1 - g) * i_e + g * i_t

            deep_in = torch.cat([u_e, i_e, u_mix, i_mix], dim=-1)
        else:
            deep_in = torch.cat([u_e, i_e], dim=-1)

        residual = self.mlp(deep_in).squeeze(-1)  # (B,)
        pred = wide + residual
        return clamp_ratings(pred)


# ----------------------------
# Train / Eval / Predict
# ----------------------------
def run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    user_text_emb_dev: Optional[torch.Tensor],
    item_text_emb_dev: Optional[torch.Tensor],
) -> float:
    model.eval()
    preds = []
    ys = []
    with torch.no_grad():
        for batch in loader:
            u, i, y = batch
            u = u.to(device)
            i = i.to(device)
            y = y.to(device)
            p = model(u, i, user_text_emb_dev, item_text_emb_dev)
            preds.append(p.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
    return rmse(preds, ys)


def run_predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    user_text_emb_dev: Optional[torch.Tensor],
    item_text_emb_dev: Optional[torch.Tensor],
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            u, i = batch
            u = u.to(device)
            i = i.to(device)
            p = model(u, i, user_text_emb_dev, item_text_emb_dev)
            preds.append(p.detach().cpu().numpy())
    return np.concatenate(preds)


def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    user_text_emb_dev: Optional[torch.Tensor],
    item_text_emb_dev: Optional[torch.Tensor],
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 5,
    grad_clip: float = 1.0,
    amp: bool = True,
    save_path: str = "checkpoints/best.pt",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(amp and device.type == "cuda"))
    loss_fn = nn.MSELoss()

    best = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")
        for batch in pbar:
            u, i, y = batch
            u = u.to(device)
            i = i.to(device)
            y = y.to(device)

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(amp and device.type == "cuda")):
                pred = model(u, i, user_text_emb_dev, item_text_emb_dev)
                loss = loss_fn(pred, y)

            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()
            pbar.set_postfix(loss=float(loss.item()))

        val_rmse = run_eval(model, val_loader, device, user_text_emb_dev, item_text_emb_dev)
        print(f"[Val] RMSE = {val_rmse:.6f}")

        if val_rmse < best:
            best = val_rmse
            torch.save({"model": model.state_dict(), "best_rmse": best}, save_path)
            print(f"[Save] best -> {save_path}")

    print(f"[Done] Best Val RMSE = {best:.6f}")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="../data")
    ap.add_argument("--cache_dir", type=str, default="cache")
    ap.add_argument("--out_csv", type=str, default="prediction.csv")
    ap.add_argument("--tokenizer", type=str, default="microsoft/deberta-v3-large")
    ap.add_argument("--text_max_length", type=int, default=256)
    ap.add_argument("--text_batch_size", type=int, default=64)
    ap.add_argument("--user_max_reviews", type=int, default=20)

    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--no_text", action="store_true", help="Disable text features (ID-only Wide&Deep baseline-ish)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"[Device] {device}")

    prepared = prepare(
        data_dir=args.data_dir,
        tokenizer_name=args.tokenizer,
        cache_dir=args.cache_dir,
        device=device,
        text_batch_size=args.text_batch_size,
        text_max_length=args.text_max_length,
        user_max_reviews=args.user_max_reviews,
    )

    # Move embedding tables to device once
    user_text_emb = prepared.user_text_emb
    item_text_emb = prepared.item_text_emb
    if args.no_text:
        user_text_emb_dev = None
        item_text_emb_dev = None
    else:
        user_text_emb_dev = user_text_emb.to(device)
        item_text_emb_dev = item_text_emb.to(device)

    text_dim = int(user_text_emb.shape[1]) if user_text_emb is not None else 0
    model = WideDeepText(
        num_users=len(prepared.user2idx),
        num_items=len(prepared.item2idx),
        text_dim=text_dim,
        emb_dim=args.emb_dim,
        mlp_hidden=(128, 64),
        dropout=args.dropout,
        global_mean=prepared.global_mean,
        use_text=(not args.no_text),
    ).to(device)

    train_ds = RatingsDataset(prepared.train_df, with_label=True)
    val_ds = RatingsDataset(prepared.val_df, with_label=True)
    test_ds = RatingsDataset(prepared.test_df, with_label=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # Train
    train_one(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        user_text_emb_dev=user_text_emb_dev,
        item_text_emb_dev=item_text_emb_dev,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        grad_clip=1.0,
        amp=True,
        save_path=args.ckpt,
    )

    # Load best
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"[Load] best rmse={ckpt.get('best_rmse', None)}")

    # Predict test
    preds = run_predict(model, test_loader, device, user_text_emb_dev, item_text_emb_dev)
    out_df = prepared.test_df[["ReviewerID", "ProductID"]].copy()
    out_df["Star"] = preds.astype(np.float32)

    out_df.to_csv(args.out_csv, index=False)
    print(f"[Output] wrote {args.out_csv} with {len(out_df)} rows")


if __name__ == "__main__":
    main()