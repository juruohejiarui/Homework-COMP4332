#!/usr/bin/env python3
import argparse, json, random
from collections import defaultdict
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class SVDppMeta(nn.Module):
    def __init__(self, n_u, n_i, k, mu, n_brand, n_mc, cat_hash, cat_k, num_k, mlp_h, drop):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(float(mu)))
        self.bu = nn.Embedding(n_u, 1); self.bi = nn.Embedding(n_i, 1)
        self.pu = nn.Embedding(n_u, k); self.qi = nn.Embedding(n_i, k)
        self.yi = nn.EmbeddingBag(n_i, k, mode="mean", include_last_offset=True)
        self.bemb = nn.Embedding(max(1, n_brand), cat_k); self.mcemb = nn.Embedding(max(1, n_mc), cat_k)
        self.catemb = nn.EmbeddingBag(cat_hash, cat_k, mode="mean", include_last_offset=True)
        self.numproj = nn.Linear(num_k, mlp_h)
        self.mlp = nn.Sequential(nn.ReLU(), nn.Dropout(drop), nn.Linear(mlp_h + 3 * cat_k + 2 * k, mlp_h), nn.ReLU(), nn.Dropout(drop), nn.Linear(mlp_h, 1))
        for m in [self.bu, self.bi, self.pu, self.qi, self.bemb, self.mcemb]: nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.normal_(self.yi.weight, 0.0, 0.02); nn.init.normal_(self.catemb.weight, 0.0, 0.02)
        nn.init.normal_(self.numproj.weight, 0.0, 0.02); nn.init.zeros_(self.numproj.bias)
        for x in self.mlp:
            if isinstance(x, nn.Linear): nn.init.normal_(x.weight, 0.0, 0.02); nn.init.zeros_(x.bias)

    def forward(self, u, i, bag_idx, bag_off, brand, mc, cidx, coff, num):
        ybar = self.yi(bag_idx, bag_off); pu = self.pu(u) + ybar; qi = self.qi(i)
        svdpp = self.mu + self.bu(u).squeeze(-1) + self.bi(i).squeeze(-1) + (pu * qi).sum(-1)
        b = self.bemb(brand); m = self.mcemb(mc); c = self.catemb(cidx, coff); n = self.numproj(num)
        x = torch.cat([pu, qi, b, m, c, n], dim=-1)
        res = self.mlp(x).squeeze(-1)
        return svdpp + res

def rmse(pred, y): return float(torch.sqrt(F.mse_loss(pred, y)).item())

def make_maps(df_tr, df_va, df_te):
    us = pd.concat([df_tr["ReviewerID"], df_va["ReviewerID"], df_te["ReviewerID"]]).unique().tolist()
    it = pd.concat([df_tr["ProductID"], df_va["ProductID"], df_te["ProductID"]]).unique().tolist()
    u2i = {u: j for j, u in enumerate(us)}; i2i = {p: j for j, p in enumerate(it)}
    return u2i, i2i

def build_user_hist(df_tr, u2i, i2i):
    hist = defaultdict(list)
    for u, p in zip(df_tr["ReviewerID"].values, df_tr["ProductID"].values): hist[u2i[u]].append(i2i[p])
    for u in list(hist.keys()):
        if len(hist[u]) == 0: hist[u] = [0]
        else:
            seen = set(); uniq = []
            for x in hist[u]:
                if x not in seen: seen.add(x); uniq.append(x)
            hist[u] = uniq
    return hist

def pack_bag(u_batch, hist):
    idx = []; off = [0]
    for u in u_batch:
        xs = hist.get(int(u), None); xs = xs if xs is not None and len(xs) > 0 else [0]
        idx.extend(xs); off.append(off[-1] + len(xs))
    return torch.tensor(idx, dtype=torch.long), torch.tensor(off, dtype=torch.long)

def stable_hash(s, mod):
    h = 2166136261
    for ch in s.encode("utf-8", errors="ignore"): h = (h ^ ch) * 16777619 & 0xffffffff
    return int(h % mod)

def load_product_meta(path, cat_hash, min_rating_number, use_store, use_mc, use_cats):
    with open(path, "r", encoding="utf-8") as f: data = json.load(f)
    store2i = {"__UNK__": 0}; mc2i = {"__UNK__": 0}
    meta = {}
    for x in data:
        pid = x.get("ProductID")
        store = (x.get("store") or "__UNK__") if use_store else "__UNK__"
        mc = (x.get("main_category") or "__UNK__") if use_mc else "__UNK__"
        if store not in store2i: store2i[store] = len(store2i)
        if mc not in mc2i: mc2i[mc] = len(mc2i)
        price = x.get("price", 0.0); ar = x.get("average_rating", 0.0); rn = x.get("rating_number", 0.0)
        try: price = float(price) if price is not None else 0.0
        except: price = 0.0
        try: ar = float(ar) if ar is not None else 0.0
        except: ar = 0.0
        try: rn = float(rn) if rn is not None else 0.0
        except: rn = 0.0
        if rn < min_rating_number: ar = 0.0; rn = 0.0  # 太小的全局统计容易噪声大，直接当缺失处理
        cats = []
        if use_cats:
            for c in (x.get("categories") or []):
                if c is None: continue
                c = str(c).strip()
                if len(c) > 0: cats.append(stable_hash("cat:" + c, cat_hash))
        meta[pid] = (store2i[store], mc2i[mc], cats, (math_log1p(price), ar, math_log1p(rn)))
    return meta, store2i, mc2i

def math_log1p(v):
    try: return float(np.log1p(max(0.0, float(v))))
    except: return 0.0

def build_item_tensors(all_items, meta, i2i, n_store, n_mc, cat_hash):
    n_i = len(i2i)
    brand = np.zeros(n_i, dtype=np.int64); mc = np.zeros(n_i, dtype=np.int64)
    num = np.zeros((n_i, 3), dtype=np.float32)
    cat_idx = []; cat_off = [0]
    for pid, idx in i2i.items():
        if pid in meta: b, m, cats, nums = meta[pid]
        else: b, m, cats, nums = 0, 0, [], (0.0, 0.0, 0.0)
        brand[idx] = min(int(b), max(0, n_store - 1)); mc[idx] = min(int(m), max(0, n_mc - 1))
        num[idx, :] = np.array(nums, dtype=np.float32)
        if len(cats) == 0: cats = [0]
        cat_idx.extend([int(x) % cat_hash for x in cats]); cat_off.append(cat_off[-1] + len(cats))
    return brand, mc, num, np.array(cat_idx, dtype=np.int64), np.array(cat_off, dtype=np.int64)

def pack_item_bag(i_batch, item_cat_idx, item_cat_off):
    idx = []; off = [0]
    for it in i_batch:
        a = int(item_cat_off[it]); b = int(item_cat_off[it + 1])
        xs = item_cat_idx[a:b]; xs = xs if len(xs) > 0 else [0]
        idx.extend(xs); off.append(off[-1] + len(xs))
    return torch.tensor(idx, dtype=torch.long), torch.tensor(off, dtype=torch.long)

@torch.no_grad()
def eval_loop(model, df, u2i, i2i, hist, item_brand, item_mc, item_num, item_cat_idx, item_cat_off, device, bs):
    model.eval(); ys = []; ps = []
    for s in range(0, len(df), bs):
        b = df.iloc[s:s+bs]
        u = torch.tensor([u2i[x] for x in b["ReviewerID"].values], dtype=torch.long, device=device)
        it = torch.tensor([i2i[x] for x in b["ProductID"].values], dtype=torch.long, device=device)
        bag_idx, bag_off = pack_bag(u.detach().cpu().tolist(), hist); bag_idx = bag_idx.to(device); bag_off = bag_off.to(device)
        br = torch.tensor(item_brand[it.detach().cpu().numpy()], dtype=torch.long, device=device)
        mc = torch.tensor(item_mc[it.detach().cpu().numpy()], dtype=torch.long, device=device)
        num = torch.tensor(item_num[it.detach().cpu().numpy()], dtype=torch.float32, device=device)
        cidx, coff = pack_item_bag(it.detach().cpu().tolist(), item_cat_idx, item_cat_off); cidx = cidx.to(device); coff = coff.to(device)
        p = model(u, it, bag_idx, bag_off, br, mc, cidx, coff, num).clamp(1.0, 5.0)
        y = torch.tensor(b["Star"].values, dtype=torch.float32, device=device)
        ps.append(p); ys.append(y)
    ps = torch.cat(ps); ys = torch.cat(ys); return rmse(ps, ys)

@torch.no_grad()
def predict_loop(model, df, u2i, i2i, hist, item_brand, item_mc, item_num, item_cat_idx, item_cat_off, device, bs):
    model.eval(); out = []
    for s in range(0, len(df), bs):
        b = df.iloc[s:s+bs]
        u = torch.tensor([u2i[x] for x in b["ReviewerID"].values], dtype=torch.long, device=device)
        it = torch.tensor([i2i[x] for x in b["ProductID"].values], dtype=torch.long, device=device)
        bag_idx, bag_off = pack_bag(u.detach().cpu().tolist(), hist); bag_idx = bag_idx.to(device); bag_off = bag_off.to(device)
        br = torch.tensor(item_brand[it.detach().cpu().numpy()], dtype=torch.long, device=device)
        mc = torch.tensor(item_mc[it.detach().cpu().numpy()], dtype=torch.long, device=device)
        num = torch.tensor(item_num[it.detach().cpu().numpy()], dtype=torch.float32, device=device)
        cidx, coff = pack_item_bag(it.detach().cpu().tolist(), item_cat_idx, item_cat_off); cidx = cidx.to(device); coff = coff.to(device)
        p = model(u, it, bag_idx, bag_off, br, mc, cidx, coff, num).clamp(1.0, 5.0).detach().cpu().numpy()
        out.append(p)
    return np.concatenate(out, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default="../data/train.csv"); ap.add_argument("--valid", type=str, default="../data/validation.csv"); ap.add_argument("--test", type=str, default="../data/test.csv")
    ap.add_argument("--product_json", type=str, default="../data/product.json"); ap.add_argument("--out", type=str, default="prediction.csv")
    ap.add_argument("--k", type=int, default=64); ap.add_argument("--cat_k", type=int, default=16); ap.add_argument("--cat_hash", type=int, default=20000)
    ap.add_argument("--mlp_h", type=int, default=64); ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=0.01); ap.add_argument("--wd", type=float, default=0.0); ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--min_rating_number", type=float, default=5.0)
    ap.add_argument("--no_store", action="store_true"); ap.add_argument("--no_main_category", action="store_true"); ap.add_argument("--no_categories", action="store_true")
    ap.add_argument("--device", type=str, default="cuda"); ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_all(args.seed); device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    df_tr = pd.read_csv(args.train); df_va = pd.read_csv(args.valid); df_te = pd.read_csv(args.test)
    mu = float(df_tr["Star"].mean())

    u2i, i2i = make_maps(df_tr, df_va, df_te); n_u, n_i = len(u2i), len(i2i)
    hist = build_user_hist(df_tr, u2i, i2i)

    meta, store2i, mc2i = load_product_meta(args.product_json, args.cat_hash, args.min_rating_number, not args.no_store, not args.no_main_category, not args.no_categories)
    item_brand, item_mc, item_num, item_cat_idx, item_cat_off = build_item_tensors(list(i2i.keys()), meta, i2i, len(store2i), len(mc2i), args.cat_hash)

    model = SVDppMeta(n_u, n_i, args.k, mu, len(store2i), len(mc2i), args.cat_hash, args.cat_k, 3, args.mlp_h, args.drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    tr_u = np.array([u2i[x] for x in df_tr["ReviewerID"].values], dtype=np.int64)
    tr_i = np.array([i2i[x] for x in df_tr["ProductID"].values], dtype=np.int64)
    tr_y = df_tr["Star"].values.astype(np.float32)

    best = 1e9
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(len(df_tr))
        for s in range(0, len(df_tr), args.bs):
            ix = perm[s:s+args.bs]
            u = torch.tensor(tr_u[ix], dtype=torch.long, device=device)
            it = torch.tensor(tr_i[ix], dtype=torch.long, device=device)
            y = torch.tensor(tr_y[ix], dtype=torch.float32, device=device)
            bag_idx, bag_off = pack_bag(u.detach().cpu().tolist(), hist); bag_idx = bag_idx.to(device); bag_off = bag_off.to(device)
            br = torch.tensor(item_brand[it.detach().cpu().numpy()], dtype=torch.long, device=device)
            mc = torch.tensor(item_mc[it.detach().cpu().numpy()], dtype=torch.long, device=device)
            num = torch.tensor(item_num[it.detach().cpu().numpy()], dtype=torch.float32, device=device)
            cidx, coff = pack_item_bag(it.detach().cpu().tolist(), item_cat_idx, item_cat_off); cidx = cidx.to(device); coff = coff.to(device)
            p = model(u, it, bag_idx, bag_off, br, mc, cidx, coff, num)
            loss = F.mse_loss(p, y)
            if args.reg > 0:
                loss = loss + args.reg * (model.pu(u).pow(2).mean() + model.qi(it).pow(2).mean() + model.bu(u).pow(2).mean() + model.bi(it).pow(2).mean())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        v = eval_loop(model, df_va, u2i, i2i, hist, item_brand, item_mc, item_num, item_cat_idx, item_cat_off, device, args.bs)
        if v < best:
            best = v
            torch.save({"state": model.state_dict(), "u2i": u2i, "i2i": i2i, "mu": mu, "store2i": store2i, "mc2i": mc2i}, args.out + ".pt")
        print(f"epoch = {ep}  valid_rmse = {v:.6f}  best = {best:.6f}")

    ck = torch.load(args.out + ".pt", map_location=device); model.load_state_dict(ck["state"])
    pred = predict_loop(model, df_te, u2i, i2i, hist, item_brand, item_mc, item_num, item_cat_idx, item_cat_off, device, args.bs)
    sub = df_te[["ReviewerID", "ProductID"]].copy(); sub["Star"] = pred.astype(np.float32)
    sub.to_csv(args.out, index=False); print("saved:", args.out)

if __name__ == "__main__": main()