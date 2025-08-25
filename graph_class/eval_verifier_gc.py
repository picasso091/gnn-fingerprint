"""
Evaluate a trained Univerifier on GRAPH CLASSIFICATION (ENZYMES) positives ({target ∪ F+})
and negatives (F−) using saved GC fingerprints. Produces Robustness/Uniqueness curves and ARUC.
"""

import argparse, glob, json, os, torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import dense_to_sparse, to_undirected

from gsage_gc import get_model  # GraphSAGE GC with pooling

import torch.nn as nn


class FPVerifier(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def list_paths_from_globs(globs_str):
    globs = [g.strip() for g in globs_str.split(",") if g.strip()]
    paths = []
    for g in globs:
        paths.extend(glob.glob(g))
    return sorted(paths)


def load_model_from_pt(pt_path, in_dim, num_classes):
    meta_path = pt_path.replace(".pt", ".json")
    j = json.load(open(meta_path, "r"))
    m = get_model(
        j.get("arch", "gsage"),
        in_dim,
        j.get("hidden", 64),
        num_classes,
        num_layers=j.get("layers", 3),
        dropout=j.get("dropout", 0.5),
        pool="mean",
    )
    m.load_state_dict(torch.load(pt_path, map_location="cpu"))
    m.eval()
    return m


# GC fingerprint forward: model -> graph logits
@torch.no_grad()
def forward_on_fp(model, fp):
    X = fp["X"]
    A = fp["A"]
    n = X.size(0)

    A_bin = (A > 0.5).float()
    A_sym = torch.maximum(A_bin, A_bin.t())
    edge_index = dense_to_sparse(A_sym)[0]
    if edge_index.numel() == 0:
        idx = torch.arange(n, dtype=torch.long)
        edge_index = torch.stack([idx, (idx + 1) % n], dim=0)
    edge_index = to_undirected(edge_index)

    batch = X.new_zeros(n, dtype=torch.long)
    logits = model(X, edge_index, batch=batch)
    return logits.squeeze(0)


@torch.no_grad()
def concat_for_model(model, fps):
    parts = [forward_on_fp(model, fp) for fp in fps]
    return torch.cat(parts, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fingerprints_path', type=str, default='fingerprints/fingerprints_gc.pt')
    ap.add_argument('--verifier_path', type=str, default='fingerprints/univerifier_gc.pt')
    ap.add_argument('--target_path', type=str, default='models/target_model_gc.pt')
    ap.add_argument('--target_meta', type=str, default='models/target_meta_gc.json')
    ap.add_argument('--positives_glob', type=str,
                    default='models/positives/gc_ftpr_*.pt,models/positives/distill_gc_*.pt')
    ap.add_argument('--negatives_glob', type=str, default='models/negatives/negative_gc_*.pt')
    ap.add_argument('--out_plot', type=str, default='plots/enzymes_gc_aruc.png')
    ap.add_argument('--save_csv', type=str, default='',
                    help='Optional: path to save thresholds/robustness/uniqueness CSV')
    args = ap.parse_args()

    # Dataset dims
    ds = TUDataset(root="data/ENZYMES", name="ENZYMES",
                   use_node_attr=True, transform=NormalizeFeatures())
    in_dim = ds.num_features
    num_classes = ds.num_classes

    # Load fingerprints (list of tiny graph specs)
    pack = torch.load(args.fingerprints_path, map_location="cpu")
    fps = pack["fingerprints"]
    ver_in_dim_saved = int(pack.get("ver_in_dim", 0))

    # Load models (target + positives + negatives)
    tmeta = json.load(open(args.target_meta, "r"))
    target = get_model(
        tmeta.get("arch", "gsage"), in_dim, tmeta.get("hidden", 64), num_classes,
        num_layers=tmeta.get("layers", 3), dropout=tmeta.get("dropout", 0.5), pool="mean"
    )
    target.load_state_dict(torch.load(args.target_path, map_location="cpu"))
    target.eval()

    pos_paths = list_paths_from_globs(args.positives_glob)
    neg_paths = sorted(glob.glob(args.negatives_glob))

    models_pos = [target] + [load_model_from_pt(p, in_dim, num_classes) for p in pos_paths]
    models_neg = [load_model_from_pt(n, in_dim, num_classes) for n in neg_paths]

    # Infer verifier input dim from a probe concat
    z0 = concat_for_model(models_pos[0], fps)
    D = z0.numel()
    if ver_in_dim_saved and ver_in_dim_saved != D:
        raise RuntimeError(f"Verifier input mismatch: D={D} vs ver_in_dim_saved={ver_in_dim_saved}")

    # Load verifier
    V = FPVerifier(D)
    ver_path = Path(args.verifier_path)
    if ver_path.exists():
        V.load_state_dict(torch.load(str(ver_path), map_location='cpu'))
        print(f"Loaded verifier from {ver_path}")
    elif "verifier" in pack:
        V.load_state_dict(pack["verifier"])
        print("Loaded verifier from fingerprints pack.")
    else:
        raise FileNotFoundError(
            f"No verifier found at {args.verifier_path} and no 'verifier' key in {args.fingerprints_path}"
        )
    V.eval()

    # Collect scores
    with torch.no_grad():
        pos_scores = []
        for m in models_pos:
            z = concat_for_model(m, fps).unsqueeze(0)  # [1, D]
            pos_scores.append(float(V(z)))
        neg_scores = []
        for m in models_neg:
            z = concat_for_model(m, fps).unsqueeze(0)
            neg_scores.append(float(V(z)))

    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)

    ts = np.linspace(0.0, 1.0, 201)
    robustness = np.array([(pos_scores >= t).mean() for t in ts])  # TPR on positives
    uniqueness = np.array([(neg_scores <  t).mean() for t in ts])  # TNR on negatives
    overlap = np.minimum(robustness, uniqueness)

    aruc = np.trapz(overlap, ts)

    # Best threshold (maximize min(robustness, uniqueness))
    idx_best = int(np.argmax(overlap))
    t_best = float(ts[idx_best])
    rob_best = float(robustness[idx_best])
    uniq_best = float(uniqueness[idx_best])
    acc_best = 0.5 * (rob_best + uniq_best)

    print(f"Models: +{len(models_pos)} | -{len(models_neg)} | D={D}")
    print(f"ARUC = {aruc:.4f}")
    print(f"Best threshold = {t_best:.3f} | Robustness={rob_best:.3f} | Uniqueness={uniq_best:.3f} | Acc={acc_best:.3f}")

    if args.save_csv:
        import csv
        Path(os.path.dirname(args.save_csv)).mkdir(parents=True, exist_ok=True)
        with open(args.save_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['threshold', 'robustness', 'uniqueness', 'min_curve'])
            for t, r, u, s in zip(ts, robustness, uniqueness, overlap):
                w.writerow([f"{t:.5f}", f"{r:.6f}", f"{u:.6f}", f"{s:.6f}"])
        print(f"Saved CSV to {args.save_csv}")

    # Plot
    Path(os.path.dirname(args.out_plot)).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.0, 3.0), dpi=160)
    title = f"ENZYMES graph classification (ARUC={aruc:.3f})"
    plt.title(title)
    plt.plot(ts, robustness, color="#ff0000", linewidth=2.2, label="Robustness (TPR)")
    plt.plot(ts, uniqueness, color="#0000ff", linestyle="--", linewidth=2.0, label="Uniqueness (TNR)")
    plt.fill_between(ts, overlap, color="#888888", alpha=0.25, label="Overlap (ARUC region)")
    plt.axvline(t_best, alpha=0.25, linewidth=1.0)
    plt.xlabel('Threshold (τ)')
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    plt.xlim(0, 1.0)
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.savefig(args.out_plot, bbox_inches='tight')
    print(f"Saved plot to {args.out_plot}")


if __name__ == "__main__":
    main()
