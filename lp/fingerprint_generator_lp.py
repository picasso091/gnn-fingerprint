# fingerprint_generator_lp.py
"""
This file creates graph fingerprints specialized for link prediction.
- One graph per fingerprint (I = {I}), with n nodes and sparse/random init A
- Optimize X (and optionally A) s.t. concatenated LP outputs from target/pos/neg models
  are separable for the Univerifier
"""

from __future__ import annotations
import argparse, importlib, glob, os, json, math, random
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def device_of():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_edge_index_from_adj(adj_bin: torch.Tensor) -> torch.Tensor:
    """adj_bin: (n, n) {0,1} with zeros on diagonal. Returns edge_index [2, E] (undirected)."""
    n = adj_bin.size(0)
    up = torch.triu(adj_bin, diagonal=1)
    rows, cols = torch.nonzero(up, as_tuple=True)
    edge_index = torch.stack([torch.cat([rows, cols]), torch.cat([cols, rows])], dim=0)
    return edge_index

def sample_non_self_pairs(n: int, m: int) -> torch.Tensor:
    """Return m node pairs (u,v), u!=v, as [m,2] long tensor."""
    pairs = []
    for _ in range(m):
        u = random.randrange(n)
        v = random.randrange(n-1)
        if v >= u: v += 1
        pairs.append((u, v))
    return torch.tensor(pairs, dtype=torch.long)

def inner_product_lp(z: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Sigmoid(inner product) scores for link prediction over selected node pairs."""
    u = z[pairs[:,0]]
    v = z[pairs[:,1]]
    score = (u * v).sum(-1)
    return torch.sigmoid(score)


def load_model(module: str, classname: str, ckpt_path: Optional[str], **kwargs) -> nn.Module:
    mod = importlib.import_module(module)
    Model = getattr(mod, classname)
    model = Model(**kwargs)
    if ckpt_path and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        # allow both raw state dicts and wrapped dicts
        state_dict = sd.get("state_dict", sd)
        model.load_state_dict(state_dict, strict=False)
    return model

# @torch.no_grad()
def forward_node_reps(model: nn.Module, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model(x, edge_index)

# Loss shaping for LP fingerprints
def concat_lp_outputs(models: List[nn.Module],
                      x: torch.Tensor,
                      edge_index: torch.Tensor,
                      pairs: torch.Tensor) -> torch.Tensor:
    """
    Concatenate LP probabilities from a list of models over the same fingerprint.
    Output: [len(models) * m], where m = #pairs sampled.
    """
    outs = []
    for m_ in models:
        z = forward_node_reps(m_, x, edge_index)
        p = inner_product_lp(z, pairs)  # [m]
        outs.append(p.view(-1))
    return torch.cat(outs, dim=0)  # [models*m]

def pos_neg_target_loss(target_out: torch.Tensor,
                        pos_outs: List[torch.Tensor],
                        neg_outs: List[torch.Tensor]) -> torch.Tensor:
    """
    Encourage target and 'positive' (pirated/obfuscated) models to be similar;
    discourage 'negative' (irrelevant) models from being similar to target.
    """
    def sim(a, b): return F.cosine_similarity(a, b, dim=0)
    loss_pos = 0.0
    for po in pos_outs:
        loss_pos = loss_pos + (1.0 - sim(target_out, po))  # pull together

    loss_neg = 0.0
    for no in neg_outs:
        loss_neg = loss_neg + torch.clamp(sim(target_out, no), min=0.0)  # push apart

    return loss_pos + loss_neg

# Fingerprint construction
def build_init_graph(n: int, d: int, edge_init_ratio: float, x_low: float, x_high: float, device) -> Tuple[torch.Tensor, torch.Tensor]:
    x = (x_high - x_low) * torch.rand(n, d, device=device) + x_low
    # Sparse random A
    adj = torch.zeros(n, n, device=device)
    mask = torch.rand(n, n, device=device) < edge_init_ratio
    mask = torch.triu(mask, diagonal=1)
    adj[mask] = 1.0
    adj = adj + adj.t()  # undirected
    return x.requires_grad_(True), adj  # only X needs grad by default

def do_ranked_adj_flips(adj: torch.Tensor, gradA: torch.Tensor, k: int) -> torch.Tensor:
    """
    Operate on upper triangle, then mirror.
    """
    n = adj.size(0)
    up_mask = torch.ones_like(adj, dtype=torch.bool).triu(diagonal=1)
    g = gradA.clone()
    g[~up_mask] = 0.0
    # pick top-k by absolute gradient
    flat = g.abs().view(-1)
    k = min(k, int(up_mask.sum().item()))
    if k <= 0: 
        return adj
    idx = torch.topk(flat, k=k, largest=True).indices
    # map flat idx -> 2D coords
    rows = idx // adj.size(1)
    cols = idx % adj.size(1)
    # flip
    adj_new = adj.clone()
    adj_new[rows, cols] = 1.0 - adj_new[rows, cols]
    # mirror
    adj_new = torch.triu(adj_new, diagonal=1)
    adj_new = adj_new + adj_new.t()
    return adj_new

def optimize_fingerprints_lp(
    target: nn.Module,
    positives: List[nn.Module],
    negatives: List[nn.Module],
    n_nodes: int = 32,
    feat_dim: int = 32,
    n_fps: int = 64,
    iters: int = 1000,
    lr: float = 1e-3,
    m_pairs: int = 64,
    edge_init_ratio: float = 0.05,
    x_low: float = -0.5,
    x_high: float = 0.5,
    update_adj: bool = False,
    adj_flips_per_step: int = 8,
    seed: int = 42,
) -> List[dict]:
    """
    Return a list of dicts per fingerprint: {'X': np.array[n,d], 'A': np.array[n,n], 'pairs': np.array[m,2]}
    """
    set_seed(seed)
    dev = device_of()
    target.to(dev)
    for m in positives + negatives: m.to(dev)

    fps: List[dict] = []
    x_opt = torch.optim.Adam 

    for fp_idx in range(n_fps):
        x, adj = build_init_graph(n_nodes, feat_dim, edge_init_ratio, x_low, x_high, dev)
        opt = x_opt([x], lr=lr)
        # fixed sample of node-pairs for this fingerprint
        pairs = sample_non_self_pairs(n_nodes, m_pairs).to(dev)

        for t in range(iters):
            opt.zero_grad()

            edge_index = build_edge_index_from_adj(adj)

            tgt_out = concat_lp_outputs([target], x, edge_index, pairs)  # [m]
            pos_outs = [concat_lp_outputs([pm], x, edge_index, pairs) for pm in positives]
            neg_outs = [concat_lp_outputs([nm], x, edge_index, pairs) for nm in negatives]

            loss = pos_neg_target_loss(tgt_out, pos_outs, neg_outs)
            loss.backward()

            opt.step()

            # Optional: ranked adjacency flips
            if update_adj:
                with torch.no_grad():
                    # sample K candidate edges and evaluate loss delta
                    k = adj_flips_per_step
                    n = adj.size(0)
                    cand = []
                    tries = 4 * k
                    cnt = 0
                    while cnt < tries:
                        i = random.randrange(n)
                        j = random.randrange(n-1)
                        if j >= i: j += 1
                        if i > j: i, j = j, i
                        cand.append((i, j))
                        cnt += 1
                    # evaluate deltas
                    deltas = []
                    base_loss = loss.item()
                    for (i, j) in cand:
                        a_flip = adj.clone()
                        a_flip[i, j] = 1.0 - a_flip[i, j]
                        a_flip[j, i] = a_flip[i, j]
                        ei = build_edge_index_from_adj(a_flip)
                        tgt = concat_lp_outputs([target], x, ei, pairs)
                        pos = [concat_lp_outputs([pm], x, ei, pairs) for pm in positives]
                        neg = [concat_lp_outputs([nm], x, ei, pairs) for nm in negatives]
                        l = pos_neg_target_loss(tgt, pos, neg).item()
                        deltas.append((l - base_loss, i, j))
                    # flip k edges that most reduce loss
                    deltas.sort(key=lambda z: z[0])
                    for d, i, j in deltas[:adj_flips_per_step]:
                        adj[i, j] = 1.0 - adj[i, j]
                        adj[j, i] = adj[i, j]

        fps.append({
            "X": x.detach().cpu().numpy(),
            "A": adj.detach().cpu().numpy(),
            "pairs": pairs.detach().cpu().numpy(),  # store which node-pairs this FP uses for LP outputs
        })

    return fps


def parse_args():
    p = argparse.ArgumentParser(description="GNNFingers LP Fingerprint Generator (separate file)")

    p.add_argument("--target-module", type=str, required=True, help="e.g., gcn")
    p.add_argument("--target-class", type=str, required=True, help="e.g., GCN")
    p.add_argument("--target-ckpt", type=str, default="fingerprints/lp/enc_citeseer.pt")
    p.add_argument("--target-kwargs", type=str, default='{"in_dim": 3703, "hid": 64, "out": 64, "num_layers": 3}', help='JSON dict of ctor kwargs')

    p.add_argument("--pos-glob", type=str, default="models/positives/lp/*.pt", help="glob for positive model checkpoints (same module/class/kwargs as target)")
    p.add_argument("--neg-glob", type=str, default="models/negatives/lp/*.pt", help="glob for negative model checkpoints (same module/class/kwargs as target)")

    p.add_argument("--n-fps", type=int, default=64)
    p.add_argument("--n-nodes", type=int, default=32)
    p.add_argument("--feat-dim", type=int, default=32)
    p.add_argument("--iters", type=int, default=1) #default=1000
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--m-pairs", type=int, default=64)
    p.add_argument("--edge-init-ratio", type=float, default=0.05)
    p.add_argument("--x-low", type=float, default=-0.5)
    p.add_argument("--x-high", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)

    # Optional adjacency updates
    p.add_argument("--update-adj", action="store_true", help="Enable discrete ranked flips on adjacency (slower).")
    p.add_argument("--adj-flips-per-step", type=int, default=8)

    p.add_argument("--out", type=str, default="fingerprints/lp/fingerprints_lp_citeseer.npz")
    p.add_argument("--meta", type=str, default="fingerprints/lpfingerprints_lp_citeseer_meta.json")
    return p.parse_args()

def main():
    args = parse_args()
    ctor_kwargs = json.loads(args.target_kwargs)

    target = load_model(args.target_module, args.target_class, args.target_ckpt, **ctor_kwargs)

    pos_paths = sorted(glob.glob(args.pos_glob)) if args.pos_glob else []
    neg_paths = sorted(glob.glob(args.neg_glob)) if args.neg_glob else []

    positives = [load_model(args.target_module, args.target_class, p, **ctor_kwargs) for p in pos_paths]
    negatives = [load_model(args.target_module, args.target_class, n, **ctor_kwargs) for n in neg_paths]

    # Build fingerprints
    fps = optimize_fingerprints_lp(
        target, positives, negatives,
        n_nodes=args.n_nodes, feat_dim=args.feat_dim, n_fps=args.n_fps,
        iters=args.iters, lr=args.lr, m_pairs=args.m_pairs, edge_init_ratio=args.edge_init_ratio,
        x_low=args.x_low, x_high=args.x_high,
        update_adj=args.update_adj, adj_flips_per_step=args.adj_flips_per_step,
        seed=args.seed
    )

    # Save
    npz = {}
    for i, fp in enumerate(fps):
        npz[f"X_{i}"] = fp["X"]
        npz[f"A_{i}"] = fp["A"]
        npz[f"pairs_{i}"] = fp["pairs"]
    np.savez_compressed(args.out, **npz)

    meta = {
        "task": "link_prediction",
        "dataset_hint": "Citeseer",
        "n_fps": args.n_fps,
        "n_nodes": args.n_nodes,
        "feat_dim": args.feat_dim,
        "iters": args.iters,
        "lr": args.lr,
        "m_pairs": args.m_pairs,
        "edge_init_ratio": args.edge_init_ratio,
        "update_adj": args.update_adj,
        "adj_flips_per_step": args.adj_flips_per_step,
        "target": {"module": args.target_module, "class": args.target_class, "ckpt": args.target_ckpt, "kwargs": ctor_kwargs},
        "positives": pos_paths,
        "negatives": neg_paths,
    }
    with open(args.meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved LP fingerprints to {args.out}")
    print(f"Saved meta to {args.meta}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("time taken (min): ", (end_time-start_time)/60)
