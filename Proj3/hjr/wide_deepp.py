"""Wide & Deep baseline for rating prediction.

  Wide  : global bias + user bias + item bias        (memorisation)
  Deep  : MLP over [user_emb, item_emb] concatenation (generalisation)
  Output: wide + deep, clamped to [1.0, 5.0]

Writes:
    val_pred_wide_deep.csv     (for evaluate.py sanity check on validation)
    prediction_wide_deep.csv   (the submission — rename to prediction.csv)

Run from the project_3/ directory:
    python baselines/wide_deep.py
"""

from __future__ import annotations

import json
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ── Hyper-parameters ──────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
EMB_DIM     = 16
HIDDEN      = [64, 32]
DROPOUT     = 0.5
EPOCHS      = 100
BATCH_SIZE  = 512
LR          = 2e-3
WD_BIAS     = 1e-2     # heavy reg on bias terms — forces them small but consistent
WD_OTHER    = 1e-5     # light reg on embeddings + MLP
SEED        = 42
UNKNOWN     = "__UNK__"
# ──────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)


def build_id_map(values):
    return {v: i for i, v in enumerate(sorted(set([*values, UNKNOWN])))}

def encode(df, u_map, i_map):
    u = df["ReviewerID"].map(u_map).fillna(u_map[UNKNOWN]).astype(int).to_numpy()
    i = df["ProductID"].map(i_map).fillna(i_map[UNKNOWN]).astype(int).to_numpy()
    r = (df["Star"].astype(float).to_numpy()
         if "Star" in df.columns else [0.0] * len(df))
    return u, i, r

class RatingDataset(Dataset):
    def __init__(self, users, items, ratings, item_data : pd.DataFrame, s_map):
        self.u = torch.LongTensor(users)
        self.i = torch.LongTensor(items)
        self.r = torch.FloatTensor(ratings)
        self.store = torch.LongTensor(item_data['store'].astype(int).to_numpy())
        self.meta = torch.FloatTensor(item_data.drop(columns=['store']).astype(float).to_numpy())

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.r[idx], self.store[self.i[idx]], self.meta[self.i[idx]]

class WideAndDeep(nn.Module):
    """Bias model (wide) + MLP over embeddings (deep)."""

    def __init__(self, n_users, n_items, n_stores : int, emb_dim, hidden, dropout, global_mean):
        super().__init__()
        # ── Wide: per-user / per-item / global bias ───────────────────────────
        self.user_bias   = nn.Embedding(n_users, 1)
        self.item_bias   = nn.Embedding(n_items, 1)
        self.store_bias  = nn.Embedding(n_stores, 1)
        self.global_bias = nn.Parameter(torch.tensor(float(global_mean)))
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.zeros_(self.store_bias.weight)

        # ── Deep: embeddings + MLP ────────────────────────────────────────────
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.store_emb = nn.Embedding(n_stores, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.store_emb.weight, std=0.01)

        self.meta_l = nn.Linear(3, emb_dim)

        layers, in_dim = [], 4 * emb_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.deep = nn.Sequential(*layers)

    def forward(self, u, i, s, meta):
        wide = (self.global_bias
                + self.user_bias(u).squeeze(-1)
                + self.item_bias(i).squeeze(-1)
                + self.store_bias(s).squeeze(-1))
        x = torch.cat([self.user_emb(u), self.item_emb(i), self.store_emb(s), self.meta_l(meta)], dim=-1)
        deep = self.deep(x).squeeze(-1)
        return wide + deep


def predict_with_fallback(model, valid_loader : DataLoader, global_mean):
    model.eval()
    preds = []
    with torch.no_grad():
        for u, i, r, s, meta in valid_loader :
            u, i, r, s, meta = u.to(DEVICE), i.to(DEVICE), r.to(DEVICE), s.to(DEVICE), meta.to(DEVICE)
            pred : torch.Tensor = model(u, i, s, meta)
            pred = pred.clip(1.0, 5.0)
            preds.extend(pred.tolist())
    return preds

def safe_float(x):
    try:
        if x is None: return -1
        return float(x)
    except:
        return -1

def main():
    train = pd.read_csv("../data/train.csv",      usecols=["ReviewerID", "ProductID", "Star"])
    valid = pd.read_csv("../data/validation.csv", usecols=["ReviewerID", "ProductID", "Star"])
    test  = pd.read_csv("../data/test.csv",       usecols=["ReviewerID", "ProductID", "Star"])
    with open("../data/product.json", 'r') as f :
        items = json.load(f)

    u_map = build_id_map(train["ReviewerID"])
    i_map = build_id_map(pd.concat([train["ProductID"]]).tolist()
                     + [item['ProductID'] for item in items])
    print(f"Users: {len(u_map):,}  Items: {len(i_map):,}  Train rows: {len(train):,}")

    item_data = [None] * len(i_map)
    store_map = build_id_map([(item['store'] if item['store'] is not None else UNKNOWN) for item in items])
    for item in items :
        item_data[i_map[item['ProductID']]] = {
            # "id": i_map[item['ProductID']],
            "price": safe_float(item['price']),
            "average_rating": safe_float(item['average_rating']),
            'rating_number': safe_float(item['rating_number']),
            "store": store_map[item['store'] if item['store'] is not None else UNKNOWN]
        }
    for idx, i in enumerate(item_data) :
        if i is None :
            item_data[idx] = {
                # "id": i_map[UNKNOWN],
                "price": -1,
                "average_rating": -1,
                "rating_number": -1,
                "store": store_map[UNKNOWN]
            }
    item_data = pd.DataFrame(item_data)
    print(item_data.head(5))

    global_mean = float(train["Star"].mean())
    print(f"Train mean rating: {global_mean:.4f}")

    u_tr, i_tr, r_tr = encode(train, u_map, i_map)
    u_vl, i_vl, r_vl = encode(valid, u_map, i_map)
    u_te, i_te, r_te = encode(test, u_map, i_map)

    train_loader = DataLoader(
        RatingDataset(u_tr, i_tr, r_tr, item_data, store_map),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    valid_loader = DataLoader(
        RatingDataset(u_vl, i_vl, r_vl, item_data, store_map),
        batch_size=4096, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        RatingDataset(u_te, i_te, r_te, item_data, store_map),
        batch_size=4096, shuffle=False, num_workers=0
    )

    model = WideAndDeep(
        len(u_map), len(i_map), len(store_map),
        EMB_DIM, HIDDEN, DROPOUT, global_mean).to(DEVICE)

    # Group parameters: heavy weight decay on bias embeddings (memorisation
    # part), light decay on the deep branch + global bias scalar.
    bias_params, other_params = [], []
    for name, p in model.named_parameters():
        if "bias" in name and p.dim() == 2:   # user_bias.weight, item_bias.weight
            bias_params.append(p)
        else:
            other_params.append(p)
    opt = torch.optim.Adam([
        {"params": bias_params,  "weight_decay": WD_BIAS},
        {"params": other_params, "weight_decay": WD_OTHER},
    ], lr=LR)
    loss_fn = nn.MSELoss()
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    best_rmse, best_state = math.inf, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for u, i, r, s, meta in train_loader:
            u, i, r, s, meta = u.to(DEVICE), i.to(DEVICE), r.to(DEVICE), s.to(DEVICE), meta.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(u, i, s, meta), r)
            loss.backward()
            opt.step()
            total += loss.item() * len(r)
        
        sch.step()

        v_pred = predict_with_fallback(model, valid_loader, global_mean)
        rmse = math.sqrt(((valid["Star"].to_numpy() - v_pred) ** 2).mean())
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train MSE={total/len(u_tr):.4f} | val RMSE={rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\nBest validation RMSE: {best_rmse:.4f}")
    model.load_state_dict(best_state)

    valid_out = valid.copy()
    valid_out["Star"] = predict_with_fallback(model, valid_loader, global_mean)
    valid_out.to_csv("../val_pred_wide_deep.csv", index=False)
    print("Saved val_pred_wide_deep.csv")

    test_out = test.copy()
    test_out["Star"] = predict_with_fallback(model, test_loader, global_mean)
    test_out.to_csv("../prediction_wide_deep.csv", index=False)
    print("Saved prediction_wide_deep.csv")


if __name__ == "__main__":
    main()
