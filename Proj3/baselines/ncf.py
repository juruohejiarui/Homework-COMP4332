"""Neural Collaborative Filtering baseline (rating regression).

Trains user / item embeddings, concatenates them and feeds the result
through a 3-layer MLP to predict the star rating. Reports validation
RMSE per epoch and writes filled CSVs:

    val_pred_ncf.csv     (for evaluate.py sanity check on validation)
    prediction_ncf.csv   (the submission — rename to prediction.csv)

Run from the project_3/ directory:
    python baselines/ncf.py
"""

from __future__ import annotations

import math
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ── Hyper-parameters ──────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EMB_DIM    = 16
HIDDEN     = [32, 16]
DROPOUT    = 0.3
EPOCHS     = 20
BATCH_SIZE = 512
LR         = 5e-4
WD         = 1e-4
SEED       = 42
# ──────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)


def build_id_map(values):
    return {v: i for i, v in enumerate(sorted(set(values)))}


class RatingDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.u = torch.LongTensor(users)
        self.i = torch.LongTensor(items)
        self.r = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.r[idx]


class NCF(nn.Module):
    """User and item embeddings → concat → MLP → scalar rating.

    The MLP output is added to a learnable global mean so the network
    only has to model the (small) residual around the mean, not the
    entire 1-5 scale. This gives the optimiser a much easier starting
    point and significantly lowers validation RMSE.
    """

    def __init__(self, n_users, n_items, emb_dim, hidden, dropout, global_mean):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        layers, in_dim = [], 2 * emb_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self.global_mean = nn.Parameter(torch.tensor(float(global_mean)))

    def forward(self, u, i):
        x = torch.cat([self.user_emb(u), self.item_emb(i)], dim=-1)
        return self.mlp(x).squeeze(-1) + self.global_mean


def encode(df, u_map, i_map):
    """Map IDs to integers; unseen IDs become -1 (handled at predict time)."""
    u = df["ReviewerID"].map(u_map).fillna(-1).astype(int).to_numpy()
    i = df["ProductID"].map(i_map).fillna(-1).astype(int).to_numpy()
    r = (df["Star"].astype(float).to_numpy()
         if "Star" in df.columns else [0.0] * len(df))
    return u, i, r


def predict_with_fallback(model, df, u_map, i_map, global_mean):
    """Predict star for every row; cold-start (unseen ID) → global mean."""
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(df), 4096):
            chunk = df.iloc[start:start + 4096]
            u_idx = chunk["ReviewerID"].map(u_map)
            i_idx = chunk["ProductID"].map(i_map)
            seen = u_idx.notna() & i_idx.notna()
            out = [global_mean] * len(chunk)
            if seen.any():
                u_t = torch.LongTensor(u_idx[seen].astype(int).to_numpy()).to(DEVICE)
                i_t = torch.LongTensor(i_idx[seen].astype(int).to_numpy()).to(DEVICE)
                p = model(u_t, i_t).clamp(1.0, 5.0).cpu().numpy()
                for pos, val in zip(seen.to_numpy().nonzero()[0], p):
                    out[pos] = float(val)
            preds.extend(out)
    return preds


def main():
    train = pd.read_csv("data/train.csv",      usecols=["ReviewerID", "ProductID", "Star"])
    valid = pd.read_csv("data/validation.csv", usecols=["ReviewerID", "ProductID", "Star"])
    test  = pd.read_csv("data/test.csv",       usecols=["ReviewerID", "ProductID", "Star"])

    u_map = build_id_map(train["ReviewerID"])
    i_map = build_id_map(train["ProductID"])
    print(f"Users: {len(u_map):,}  Items: {len(i_map):,}  Train rows: {len(train):,}")

    global_mean = float(train["Star"].mean())
    print(f"Train mean rating: {global_mean:.4f}")

    u_tr, i_tr, r_tr = encode(train, u_map, i_map)
    train_loader = DataLoader(
        RatingDataset(u_tr, i_tr, r_tr),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )

    model = NCF(len(u_map), len(i_map), EMB_DIM, HIDDEN, DROPOUT, global_mean).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()

    best_rmse, best_state = math.inf, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for u, i, r in train_loader:
            u, i, r = u.to(DEVICE), i.to(DEVICE), r.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(u, i), r)
            loss.backward()
            opt.step()
            total += loss.item() * len(r)

        v_pred = predict_with_fallback(model, valid, u_map, i_map, global_mean)
        rmse = math.sqrt(((valid["Star"].to_numpy() - v_pred) ** 2).mean())
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train MSE={total/len(u_tr):.4f} | val RMSE={rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\nBest validation RMSE: {best_rmse:.4f}")
    model.load_state_dict(best_state)

    # Validation predictions — sanity check via `python evaluate.py --pred val_pred_ncf.csv`
    valid_out = valid.copy()
    valid_out["Star"] = predict_with_fallback(model, valid, u_map, i_map, global_mean)
    valid_out.to_csv("val_pred_ncf.csv", index=False)
    print("Saved val_pred_ncf.csv")

    # Test predictions — rename to prediction.csv before submitting
    test_out = test.copy()
    test_out["Star"] = predict_with_fallback(model, test, u_map, i_map, global_mean)
    test_out.to_csv("prediction_ncf.csv", index=False)
    print("Saved prediction_ncf.csv")


if __name__ == "__main__":
    main()
