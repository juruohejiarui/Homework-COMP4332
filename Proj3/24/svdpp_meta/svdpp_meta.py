#!/usr/bin/env python3
import argparse, json, random, os, re
from collections import defaultdict
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

def seed_all(s): 
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def rmse(pred, y):
    return float(torch.sqrt(F.mse_loss(pred, y)).item())

def stable_hash(s, mod):
    h = 2166136261
    for ch in s.encode("utf-8", errors="ignore"): 
        h = (h ^ ch) * 16777619 & 0xffffffff
    return int(h % mod)

def safe_float(x) -> float :
    try :
        return float(x) if x is not None else 0.0
    except :
        return 0.0

def log1p_safe(v):
    try: 
        return float(np.log1p(max(0.0, float(v))))
    except: 
        return 0.0

def clean_text(s):
    if s is None: return ""
    s = str(s)
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def make_maps(df_tr, df_va, df_te):
    us = pd.concat([df_tr["ReviewerID"], df_va["ReviewerID"], df_te["ReviewerID"]]).unique().tolist()
    it = pd.concat([df_tr["ProductID"], df_va["ProductID"], df_te["ProductID"]]).unique().tolist()
    u2i = {u: j for j, u in enumerate(us)}; i2i = {p: j for j, p in enumerate(it)}
    return u2i, i2i

def build_user_hist(df_tr, u2i, i2i):
    hist = defaultdict(list)
    for u, p in zip(df_tr["ReviewerID"].values, df_tr["ProductID"].values):
        hist[u2i[u]].append(i2i[p])
    for u in list(hist.keys()):
        xs = hist[u]
        if len(xs) == 0: hist[u] = [0]
        else:
            seen = set(); uniq = []
            for x in xs:
                if x not in seen: seen.add(x); uniq.append(x)
            hist[u] = uniq
    return hist

def pack_bag(u_batch, hist):
    idx = []; off = [0]
    for u in u_batch:
        xs = hist.get(int(u), None); xs = xs if xs is not None and len(xs) > 0 else [0]
        idx.extend(xs); off.append(off[-1] + len(xs))
    return torch.tensor(idx, dtype=torch.long), torch.tensor(off, dtype=torch.long)

def load_product_meta(path, cat_hash, min_rating_number, use_store, use_mc, use_cats):
    with open(path, "r", encoding="utf-8") as f: data = json.load(f)
    store2i = {"__UNK__": 0}; mc2i = {"__UNK__": 0}; meta = {}
    for x in data:
        pid = x.get("ProductID")
        store = (x.get("store") or "__UNK__") if use_store else "__UNK__"
        mc = (x.get("main_category") or "__UNK__") if use_mc else "__UNK__"
        if store not in store2i: store2i[store] = len(store2i)
        if mc not in mc2i: mc2i[mc] = len(mc2i)
        price = x.get("price", 0.0)
        ar = x.get("average_rating", 0.0)
        rn = x.get("rating_number", 0.0)
        price, ar, rn = safe_float(price), safe_float(ar), safe_float(rn)
        if rn < min_rating_number: 
            ar, rn = 0.0, 0.0
        cats = []
        if use_cats:
            for c in (x.get("categories") or []):
                if c is None: continue
                c = str(c).strip()
                if len(c) > 0: cats.append(stable_hash("cat:" + c, cat_hash))
        meta[pid] = (store2i[store], mc2i[mc], cats, (log1p_safe(price), ar, log1p_safe(rn)))
    return meta, store2i, mc2i

def build_item_tensors(meta, i2i, n_store, n_mc, cat_hash):
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
        xs = item_cat_idx[a:b]
        if len(xs) == 0: 
            xs = np.array([0], dtype=np.int64)
        idx.extend(xs.tolist())
        off.append(off[-1] + int(len(xs)))
    return torch.tensor(idx, dtype=torch.long), torch.tensor(off, dtype=torch.long)

def build_item_static_text(product_json, i2i, max_chars):
    with open(product_json, "r", encoding="utf-8") as f: data = json.load(f)
    pid2txt = {}
    for x in data:
        pid = x.get("ProductID")
        if pid is None: continue
        title = clean_text(x.get("title", ""))
        store = clean_text(x.get("store", ""))
        mc = clean_text(x.get("main_category", ""))
        desc = " ".join([clean_text(t) for t in (x.get("description") or []) if t is not None])
        feats = " ".join([clean_text(t) for t in (x.get("features") or []) if t is not None])
        cats = " ".join([clean_text(c) for c in (x.get("categories") or []) if c is not None])
        txt = " | ".join([t for t in [
                                      store, 
                                      mc, 
                                      title, 
                                      desc, 
                                      feats,
                                      cats
                                    ] if len(t) > 0])
        if max_chars > 0: txt = txt[:max_chars]
        pid2txt[pid] = txt
    n_i = len(i2i)
    i_static = ["" for _ in range(n_i)]
    for pid, idx in i2i.items(): i_static[idx] = pid2txt.get(pid, "")
    return i_static

def build_docs_from_train(df_tr, u2i, i2i, i_static_text, max_reviews, max_chars):
    u_docs = defaultdict(list); i_docs = defaultdict(list)
    for u, p, t, s in zip(df_tr["ReviewerID"].values, df_tr["ProductID"].values, df_tr.get("Text", "").values, df_tr.get("Summary", "").values):
        uu = u2i[u]; ii = i2i[p]
        txt = clean_text(s) + " | " + clean_text(t)
        if len(txt) == 0: continue
        if len(u_docs[uu]) < max_reviews: u_docs[uu].append(txt)
        if len(i_docs[ii]) < max_reviews: i_docs[ii].append(txt)
    def join_cap(prefix, lst):
        parts = []
        if prefix is not None and len(prefix) > 0: parts.append(prefix)
        if len(lst) > 0: parts.append(" ".join(lst))
        s = " ".join(parts)
        return s[:max_chars] if max_chars > 0 else s
    n_u = len(u2i); n_i = len(i2i)
    u_text = [join_cap("", u_docs.get(i, [])) for i in range(n_u)]
    i_text = [join_cap(i_static_text[j], i_docs.get(j, [])) for j in range(n_i)]
    return u_text, i_text

def encode_or_load(st_name, cache_dir, tag, texts, bs, device, normalize):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{tag}.{st_name.replace('/', '_')}.npy")
    if os.path.exists(path):
        emb = np.load(path)
        return emb
    model = SentenceTransformer(st_name, device=device)
    emb = model.encode(texts, batch_size=bs, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=normalize)
    np.save(path, emb.astype(np.float32))
    return emb.astype(np.float32)

class SVDppMetaText(nn.Module):
    def __init__(self, n_u : int, n_i : int, k : int, glo_mean : float, n_brand : int, n_mc : int, cat_hash : int, cat_k : int, mlp_h : int, drop : float, text_dim : int):
        super().__init__()
        self.glo_mean = nn.Parameter(torch.tensor(float(glo_mean)))
        self.bias_u = nn.Embedding(n_u, 1)
        self.bias_i = nn.Embedding(n_i, 1)
        self.emb_u = nn.Embedding(n_u, k)
        self.emb_i = nn.Embedding(n_i, k)
        self.emb_h = nn.EmbeddingBag(n_i, k, mode="sum", include_last_offset=True)
        self.emb_b = nn.Embedding(max(1, n_brand), cat_k)
        self.emb_mc = nn.Embedding(max(1, n_mc), cat_k)
        self.emb_c = nn.EmbeddingBag(cat_hash, cat_k, mode="mean", include_last_offset=True)
        self.numproj = nn.Linear(3, mlp_h)
        self.utproj = nn.Linear(text_dim, mlp_h); self.itproj = nn.Linear(text_dim, mlp_h)
        in_dim = 2 * k + 3 * cat_k + 3 * mlp_h  # pu, qi, brand, mc, cats, num, u_text, i_text
        self.mlp = nn.Sequential(nn.ReLU(), nn.Dropout(drop), nn.Linear(in_dim, mlp_h), nn.ReLU(), nn.Dropout(drop), nn.Linear(mlp_h, 1))
        for m in [self.bias_u, self.bias_i, self.emb_u, self.emb_i, self.emb_b, self.emb_mc]: 
            nn.init.normal_(m.weight, 0.0, 0.01)
        nn.init.normal_(self.emb_h.weight, 0.0, 0.01)
        nn.init.normal_(self.emb_c.weight, 0.0, 0.01)
        nn.init.normal_(self.emb_u.weight, 0.0, 0.01)
        nn.init.normal_(self.emb_i.weight, 0.0, 0.01)
        nn.init.zeros_(self.bias_u.weight)
        nn.init.zeros_(self.bias_i.weight)   # 将被外部 init_bi 覆盖
        for lin in [self.numproj, self.utproj, self.itproj]:
            nn.init.normal_(lin.weight, 0.0, 0.01); nn.init.zeros_(lin.bias)
        for x in self.mlp:
            if isinstance(x, nn.Linear): nn.init.normal_(x.weight, 0.0, 0.01); nn.init.zeros_(x.bias)
        
        self.text_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, u : torch.Tensor, it : torch.Tensor, bag_idx : torch.Tensor, bag_off : torch.Tensor, brand : torch.Tensor, mc : torch.Tensor, cidx : torch.Tensor, coff : torch.Tensor, num, utext : torch.Tensor, itext : torch.Tensor):
        lens : torch.Tensor = (bag_off[1:] - bag_off[:-1]).clamp_min(1).to(self.emb_u.weight.dtype)
        ybar : torch.Tensor = self.emb_h(bag_idx, bag_off) / torch.sqrt(lens).unsqueeze(-1)
        eu : torch.Tensor = self.emb_u(u) + ybar
        ei : torch.Tensor = self.emb_i(it)
        base = self.glo_mean + self.bias_u(u).squeeze(-1) + self.bias_i(it).squeeze(-1) + (eu * ei).sum(-1)
        eb = self.emb_b(brand)
        emc = self.emb_mc(mc)
        ec = self.emb_c(cidx, coff)
        n = self.numproj(num)
        ut = self.utproj(utext) * self.text_alpha
        itx = self.itproj(itext) * self.text_alpha
        x = torch.cat([eu, ei, eb, emc, ec, n, ut, itx], dim=-1)
        res = self.mlp(x).squeeze(-1)
        return base + res

@torch.no_grad()
def eval_loop(model, df, u2i, i2i, hist, item_brand, item_mc, item_num, item_cat_idx, item_cat_off, u_text_emb, i_text_emb, device, bs) -> tuple[torch.Tensor, float]:
    model.eval(); ys = []; ps = []
    for s in range(0, len(df), bs):
        b = df.iloc[s:s+bs]
        u = torch.tensor([u2i[x] for x in b["ReviewerID"].values], dtype=torch.long, device=device)
        it = torch.tensor([i2i[x] for x in b["ProductID"].values], dtype=torch.long, device=device)
        bag_idx, bag_off = pack_bag(u.detach().cpu().tolist(), hist)
        bag_idx = bag_idx.to(device)
        bag_off = bag_off.to(device)
        br = torch.tensor(item_brand[it.detach().cpu().numpy()], dtype=torch.long, device=device)
        mc = torch.tensor(item_mc[it.detach().cpu().numpy()], dtype=torch.long, device=device)
        num = torch.tensor(item_num[it.detach().cpu().numpy()], dtype=torch.float32, device=device)
        cidx, coff = pack_item_bag(it.detach().cpu().tolist(), item_cat_idx, item_cat_off)
        cidx = cidx.to(device)
        coff = coff.to(device)
        ut = torch.tensor(u_text_emb[u.detach().cpu().numpy()], dtype=torch.float32, device=device)
        itx = torch.tensor(i_text_emb[it.detach().cpu().numpy()], dtype=torch.float32, device=device)
        p = model(u, it, bag_idx, bag_off, br, mc, cidx, coff, num, ut, itx).clamp(1.0, 5.0)
        y = torch.tensor(b["Star"].values, dtype=torch.float32, device=device)
        ps.append(p)
        ys.append(y)
    ps = torch.cat(ps); ys = torch.cat(ys)
    return ps, rmse(ps, ys)

@torch.no_grad()
def predict_loop(model, df, u2i, i2i, hist, item_brand, item_mc, item_num, item_cat_idx, item_cat_off, u_text_emb, i_text_emb, device, bs):
    model.eval(); out = []
    for s in range(0, len(df), bs):
        b = df.iloc[s:s+bs]
        u = torch.tensor([u2i[x] for x in b["ReviewerID"].values], dtype=torch.long, device=device)
        it = torch.tensor([i2i[x] for x in b["ProductID"].values], dtype=torch.long, device=device)
        bag_idx, bag_off = pack_bag(u.detach().cpu().tolist(), hist)
        bag_idx = bag_idx.to(device)
        bag_off = bag_off.to(device)
        br = torch.tensor(item_brand[it.detach().cpu().numpy()], dtype=torch.long, device=device)
        mc = torch.tensor(item_mc[it.detach().cpu().numpy()], dtype=torch.long, device=device)
        num = torch.tensor(item_num[it.detach().cpu().numpy()], dtype=torch.float32, device=device)
        cidx, coff = pack_item_bag(it.detach().cpu().tolist(), item_cat_idx, item_cat_off)
        cidx = cidx.to(device)
        coff = coff.to(device)
        ut = torch.tensor(u_text_emb[u.detach().cpu().numpy()], dtype=torch.float32, device=device)
        itx = torch.tensor(i_text_emb[it.detach().cpu().numpy()], dtype=torch.float32, device=device)
        p = model(u, it, bag_idx, bag_off, br, mc, cidx, coff, num, ut, itx).clamp(1.0, 5.0).detach().cpu().numpy()
        out.append(p)
    return np.concatenate(out, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default="../data/train.csv")
    ap.add_argument("--valid", type=str, default="../data/validation.csv")
    ap.add_argument("--test", type=str, default="../data/test.csv")
    ap.add_argument("--product_json", type=str, default="../data/product.json")
    ap.add_argument("--valid_out", type=str, default="val_pred.csv")
    ap.add_argument("--test_out", type=str, default="prediction.csv")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--cat_k", type=int, default=256)
    ap.add_argument("--cat_hash", type=int, default=20000)
    ap.add_argument("--mlp_h", type=int, default=256)
    ap.add_argument("--drop", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=350)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--no_init_bias_i", action='store_true')
    ap.add_argument("--min_rating_number", type=float, default=5.0)
    ap.add_argument("--no_store", action="store_true")
    ap.add_argument("--no_main_category", action="store_true")
    ap.add_argument("--no_categories", action="store_true")
    ap.add_argument("--st_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--st_bs", type=int, default=128)
    ap.add_argument("--st_norm", action="store_true")
    ap.add_argument("--doc_max_reviews", type=int, default=20)
    ap.add_argument("--doc_max_chars", type=int, default=16384)
    ap.add_argument("--cache_dir", type=str, default="cache_st")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=84)
    args = ap.parse_args()

    seed_all(args.seed)
    dev = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    device = torch.device(dev)

    df_tr = pd.read_csv(args.train); df_va = pd.read_csv(args.valid); df_te = pd.read_csv(args.test)
    mu = float(df_tr["Star"].mean())

    u2i, i2i = make_maps(df_tr, df_va, df_te)
    n_u, n_i = len(u2i), len(i2i)
    hist = build_user_hist(df_tr, u2i, i2i)

    meta, store2i, mc2i = load_product_meta(args.product_json, args.cat_hash, args.min_rating_number, not args.no_store, not args.no_main_category, not args.no_categories)
    item_brand, item_mc, item_num, item_cat_idx, item_cat_off = build_item_tensors(meta, i2i, len(store2i), len(mc2i), args.cat_hash)

    # ---------- 新增：利用 product.json 中的 average_rating 初始化每个物品的偏置 ----------
    # item_num 的第1列（索引1）即为 average_rating （经过 min_rating_number 过滤后，若无效则为0）
    item_ar = item_num[:, 1]  # shape (n_i,)
    # 构造初始 bi：如果 average_rating 有效（>0），则 bi = ar - mu，否则 bi = 0
    init_bi = np.where(item_ar > 0.0, item_ar - mu, 0.0).astype(np.float32)
    # ----------------------------------------------------------------------------

    i_static = build_item_static_text(args.product_json, i2i, args.doc_max_chars)
    u_text, i_text = build_docs_from_train(df_tr, u2i, i2i, i_static, args.doc_max_reviews, args.doc_max_chars)
    u_text_emb = encode_or_load(args.st_model, args.cache_dir, "u_doc", u_text, args.st_bs, dev, args.st_norm)
    i_text_emb = encode_or_load(args.st_model, args.cache_dir, "i_doc", i_text, args.st_bs, dev, args.st_norm)
    text_dim = int(u_text_emb.shape[1])

    model = SVDppMetaText(n_u, n_i, args.k, mu, len(store2i), len(mc2i), args.cat_hash, args.cat_k, args.mlp_h, args.drop, text_dim).to(device)

    # ---------- 覆盖 bi 的初始权重 ----------
    if not args.no_init_bias_i :
        model.bias_i.weight.data = torch.tensor(init_bi, dtype=torch.float32, device=device).view(-1, 1)
    # -----------------------------------------

    opt = torch.optim.AdamW([
        {
            "params": [p for n, p in model.named_parameters() if n.startswith('bias')],
            "weight_decay": args.wd * 100
        },
        {
            "params": [p for n, p in model.named_parameters() if not n.startswith('bias')],
            "weight_decay": args.wd
        }
    ], lr=args.lr, weight_decay=args.wd)
    # opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    tr_u = np.array([u2i[x] for x in df_tr["ReviewerID"].values], dtype=np.int64)
    tr_i = np.array([i2i[x] for x in df_tr["ProductID"].values], dtype=np.int64)
    tr_y = df_tr["Star"].values.astype(np.float32)

    best, best_val_pred = 1e9, None
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(len(df_tr))
        
        train_pred = []
        train_target = []

        for s in range(0, len(df_tr), args.bs):
            ix = perm[s:s+args.bs]
            u = torch.tensor(tr_u[ix], dtype=torch.long, device=device)
            it = torch.tensor(tr_i[ix], dtype=torch.long, device=device)
            y = torch.tensor(tr_y[ix], dtype=torch.float32, device=device)

            bag_idx, bag_off = pack_bag(u.detach().cpu().tolist(), hist)
            bag_idx = bag_idx.to(device); bag_off = bag_off.to(device)
            br = torch.tensor(item_brand[it.detach().cpu().numpy()], dtype=torch.long, device=device)
            mc = torch.tensor(item_mc[it.detach().cpu().numpy()], dtype=torch.long, device=device)
            num = torch.tensor(item_num[it.detach().cpu().numpy()], dtype=torch.float32, device=device)
            cidx, coff = pack_item_bag(it.detach().cpu().tolist(), item_cat_idx, item_cat_off)
            cidx = cidx.to(device); coff = coff.to(device)
            ut = torch.tensor(u_text_emb[u.detach().cpu().numpy()], dtype=torch.float32, device=device)
            itx = torch.tensor(i_text_emb[it.detach().cpu().numpy()], dtype=torch.float32, device=device)

            p : torch.Tensor = model(u, it, bag_idx, bag_off, br, mc, cidx, coff, num, ut, itx)
            loss = F.mse_loss(p, y)

            train_pred.extend(p.detach().cpu().numpy().tolist())
            train_target.extend(y.detach().cpu().numpy().tolist())

            if args.reg > 0: 
                loss = loss + args.reg * (model.emb_u(u).pow(2).mean() + model.emb_i(it).pow(2).mean() + model.bias_u(u).pow(2).mean() + model.bias_i(it).pow(2).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()

        train_rmse = rmse(torch.Tensor(train_pred), torch.Tensor(train_target))

        val_preds, v = eval_loop(model, df_va, u2i, i2i, hist, item_brand, item_mc, item_num, item_cat_idx, item_cat_off, u_text_emb, i_text_emb, device, args.bs)
        if v < best:
            best_val_pred, best = val_preds.cpu().numpy().astype(np.float32), v
            torch.save({"state": model.state_dict(), "u2i": u2i, "i2i": i2i, "mu": mu, "store2i": store2i, "mc2i": mc2i, "st_model": args.st_model}, args.test_out + ".pt")
        print(f"epoch = {ep:>4d} train_rmse = {train_rmse:.6f} valid_rmse = {v:.6f}  best = {best:.6f}")

    ck = torch.load(args.test_out + ".pt", map_location=device)
    model.load_state_dict(ck["state"])
    # save prediction on validation set
    sub = df_va[['ReviewerID', 'ProductID']].copy()
    sub['Star'] = best_val_pred
    sub.to_csv(args.valid_out, index=False)
    print(f"validation prediction save to {args.valid_out}")

    # make prediction on test set
    pred = predict_loop(model, df_te, u2i, i2i, hist, item_brand, item_mc, item_num, item_cat_idx, item_cat_off, u_text_emb, i_text_emb, device, args.bs)
    sub = df_te[["ReviewerID", "ProductID"]].copy()
    sub["Star"] = pred.astype(np.float32)
    sub.to_csv(args.test_out, index=False)
    print(f"test prediction save to {args.test_out}")

if __name__ == "__main__": 
    main()