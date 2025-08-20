import os, glob, argparse, random
from typing import List
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from graphsage_gc import build_model_from_args

def set_seed(s: int):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def ensure_dir(p: str):
    if p: os.makedirs(p, exist_ok=True)

def list_ckpts(dir_path: str) -> List[str]:
    return sorted(glob.glob(os.path.join(dir_path, "*.pt"))) if os.path.isdir(dir_path) else []

def load_ckpt_model(ckpt_path: str, device: str) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model_from_args(
        in_channels=ckpt["in_channels"], out_channels=ckpt["out_channels"],
        hidden_channels=ckpt["hidden"], num_layers=ckpt["layers"],
        sage_agg=ckpt["agg"], readout=ckpt["readout"], dropout=ckpt["dropout"], use_bn=True
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model

@torch.no_grad()
def model_fp_vector(model: nn.Module, fps: List[Data], device: str) -> torch.Tensor:
    vecs = []
    for g in fps:
        x = g.x.to(device)
        ei = g.edge_index.to(device)
        b = g.batch.to(device) if getattr(g, "batch", None) is not None else torch.zeros(g.num_nodes, dtype=torch.long, device=device)
        logits = model(x, ei, b)          # [1, C]
        vecs.append(logits.softmax(dim=-1).flatten())
    return torch.cat(vecs, dim=0)          # [N*C]

class Univerifier(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,128), nn.LeakyReLU(0.01,inplace=True),
            nn.Linear(128,64), nn.LeakyReLU(0.01,inplace=True),
            nn.Linear(64,32), nn.LeakyReLU(0.01,inplace=True),
            nn.Linear(32,2),
        )
    def forward(self,x): return self.net(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprints_path", type=str, default="fingerprints/gc/fingerprints_gc_enzyme.pt")
    ap.add_argument("--verifier_ckpt",    type=str, default="fingerprints/gc/univerifier_gc_enzyme.pt")
    ap.add_argument("--pos_dir",          type=str, default="models/positives/gc")
    ap.add_argument("--neg_dir",          type=str, default="models/negatives/gc")
    ap.add_argument("--device",           type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed",             type=int, default=42)
    ap.add_argument("--out_plot",         type=str, default="plots/aruc_gc_enzyme.png")
    ap.add_argument("--out_csv",          type=str, default="plots/aruc_gc_enzyme.csv")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(os.path.dirname(args.out_plot))
    ensure_dir(os.path.dirname(args.out_csv))

    # Load fingerprints
    pack = torch.load(args.fingerprints_path, map_location="cpu")
    fps: List[Data] = pack["fingerprints"]
    # ensure batch vectors exist
    fixed = []
    for g in fps:
        if getattr(g, "batch", None) is None:
            fixed.append(Data(
                x=g.x, edge_index=g.edge_index,
                y=getattr(g, "y", torch.zeros((), dtype=torch.long)),
                batch=torch.zeros(g.num_nodes, dtype=torch.long),
            ))
        else:
            fixed.append(g)
    fps = fixed

    pos_ckpts = list_ckpts(args.pos_dir)
    neg_ckpts = list_ckpts(args.neg_dir)
    if len(pos_ckpts)==0 or len(neg_ckpts)==0:
        raise RuntimeError("No positives or negatives found under models/positives/gc or models/negatives/gc")

    # Determine input dim
    probe = load_ckpt_model(pos_ckpts[0], args.device)
    with torch.no_grad():
        C = probe(fps[0].x.to(args.device), fps[0].edge_index.to(args.device),
                  fps[0].batch.to(args.device)).size(-1)
    N = len(fps); in_dim = N * C

    # Load verifier
    vpack = torch.load(args.verifier_ckpt, map_location="cpu")
    verifier = Univerifier(in_dim=in_dim).to(args.device)
    verifier.load_state_dict(vpack["state_dict"], strict=True)
    verifier.eval()

    positives = [load_ckpt_model(p, args.device) for p in pos_ckpts]
    negatives = [load_ckpt_model(n, args.device) for n in neg_ckpts]
    print(f"Loaded suspects : positives={len(positives)}, negatives={len(negatives)} | N={N}, C={C}, in_dim={in_dim}")

    X_pos = torch.stack([model_fp_vector(m, fps, args.device).cpu() for m in positives], dim=0)
    X_neg = torch.stack([model_fp_vector(m, fps, args.device).cpu() for m in negatives], dim=0)
    y_pos = torch.ones(X_pos.size(0), dtype=torch.long)
    y_neg = torch.zeros(X_neg.size(0), dtype=torch.long)
    X = torch.cat([X_pos, X_neg], dim=0)
    y = torch.cat([y_pos, y_neg], dim=0).numpy()

    with torch.no_grad():
        probs = verifier(X.to(args.device)).softmax(dim=-1)[:,1].cpu().numpy()
    pos_scores = probs[y == 1]
    neg_scores = probs[y == 0]

    ts = np.linspace(0.0, 1.0, 1001)
    robustness = np.array([(pos_scores >= t).mean() if pos_scores.size else 0.0 for t in ts])
    uniqueness = np.array([(neg_scores <  t).mean() if neg_scores.size else 0.0 for t in ts])
    overlap = np.minimum(robustness, uniqueness)
    aruc = np.trapz(overlap, ts)
    t_best = ts[np.argmax(overlap)]
    best_overlap = float(overlap.max())

    header = "threshold,robustness,uniqueness,overlap"
    data = np.column_stack([ts, robustness, uniqueness, overlap])
    np.savetxt(args.out_csv, data, delimiter=",", header=header, comments="")
    print(f"Saved ARUC CSV: {args.out_csv}")

    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f2f2f2') 

    # grid style
    ax.grid(True, which='both', axis='both',
            color='#c0c0c0', linestyle='--', linewidth=0.7, alpha=0.7)

    ax.plot(ts, robustness, color='red', linewidth=2.5, label='Robustness (TPR)')
    ax.plot(ts, uniqueness, color='blue', linewidth=2.5, linestyle='--', label='Uniqueness (TNR)')

    # overlap region
    ax.fill_between(ts, np.minimum(robustness, uniqueness),
                    color='#d9d9d9', alpha=1.0, label='Overlap (ARUC region)')

    # best threshold vertical line
    ax.axvline(t_best, color='grey', linewidth=2.0, alpha=0.85)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Threshold (τ)', fontsize=16)
    ax.set_ylabel('Score', fontsize=16)

    ax.set_title(f'ENZYMES graph-classification • ARUC={aruc:.3f}', fontsize=22, pad=12)

    leg = ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=14)

    plt.tight_layout()
    plt.savefig(args.out_plot, bbox_inches="tight")
    print(f"Saved ARUC plot: {args.out_plot}")

if __name__ == "__main__":
    main()
