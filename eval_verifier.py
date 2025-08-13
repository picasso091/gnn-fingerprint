"""
Evaluate a trained Univerifier on positives ({target ∪ F+}) and negatives (F−)
using saved fingerprints. Produces Robustness/Uniqueness curves and ARUC.

- Uses the same node-level protocol as fingerprint_generator.py:
  for each fingerprint I_p, select fp["node_idx"] node logits and concatenate.

Outputs:
  - A PNG plot (--out_plot)
  - Prints ARUC and best-threshold stats
"""

import argparse, glob, json, math, torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dense_to_sparse
from gcn import get_model

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


# ----------------------------
# Fingerprint forward (fixed nodes, no mean-pool)
# ----------------------------
@torch.no_grad()
def forward_on_fp(model, fp):
    """
    fp: dict with 'X' [n,in_dim], 'A' [n,n] in [0,1], 'node_idx' [m]
    returns: [m*d] flattened selected-node logits
    """
    X = fp["X"]
    A = fp["A"]
    idx = fp["node_idx"]

    A_bin = (A > 0.5).float()
    A_sym = torch.triu(A_bin, diagonal=1)
    A_sym = A_sym + A_sym.t()
    edge_index = dense_to_sparse(A_sym)[0]

    # Fallback if graph became empty
    if edge_index.numel() == 0:
        n = X.size(0)
        edge_index = torch.arange(n, dtype=torch.long).repeat(2, 1)

    logits = model(X, edge_index)   # [n, d]
    sel = logits[idx, :]            # [m, d]
    return sel.reshape(-1)           # [m*d]


@torch.no_grad()
def concat_for_model(model, fps):
    parts = [forward_on_fp(model, fp) for fp in fps]
    return torch.cat(parts, dim=0)   # [P*m*d]


def list_paths_from_globs(globs_str):
    globs = [g.strip() for g in globs_str.split(",") if g.strip()]
    paths = []
    for g in globs:
        paths.extend(glob.glob(g))
    return sorted(paths)


def load_model_from_pt(pt_path, in_dim):
    meta_path = pt_path.replace(".pt", ".json")
    j = json.load(open(meta_path, "r"))
    m = get_model(j["arch"], in_dim, j["hidden"], j["num_classes"])
    m.load_state_dict(torch.load(pt_path, map_location="cpu"))
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fingerprints_path', type=str, default='fingerprints/fingerprints.pt')
    ap.add_argument('--verifier_path', type=str, default='fingerprints/univerifier.pt',
                    help='Trained standalone Univerifier weights (from train_univerifier.py)')
    ap.add_argument('--target_path', type=str, default='models/target_model.pt')
    ap.add_argument('--target_meta', type=str, default='models/target_meta.json')
    ap.add_argument('--positives_glob', type=str,
                    default='models/positives/ftpr_*.pt,models/positives/distill_*.pt')
    ap.add_argument('--negatives_glob', type=str, default='models/negatives/negative_*.pt')
    ap.add_argument('--out_plot', type=str, default='plots/cora_gcn_aruc.png')
    ap.add_argument('--save_csv', type=str, default='',
                    help='Optional: path to save thresholds/robustness/uniqueness CSV')
    args = ap.parse_args()

    # Ensure dataset dims for model reconstruction
    ds = Planetoid(root="data/cora", name="Cora")
    in_dim = ds.num_features
    num_classes = ds.num_classes

    # Load fingerprints (with node_idx)
    pack = torch.load(args.fingerprints_path, map_location="cpu")
    fps = pack["fingerprints"]
    ver_in_dim_saved = int(pack.get("ver_in_dim", 0))

    # Load models (target + positives + negatives)
    tmeta = json.load(open(args.target_meta, "r"))
    target = get_model(tmeta["arch"], in_dim, tmeta["hidden"], tmeta["num_classes"])
    target.load_state_dict(torch.load(args.target_path, map_location="cpu"))
    target.eval()

    pos_paths = list_paths_from_globs(args.positives_glob)
    neg_paths = sorted(glob.glob(args.negatives_glob))

    models_pos = [target] + [load_model_from_pt(p, in_dim) for p in pos_paths]
    models_neg = [load_model_from_pt(n, in_dim) for n in neg_paths]

    # Infer verifier input dim from a probe concat
    z0 = concat_for_model(models_pos[0], fps)
    D = z0.numel()
    if ver_in_dim_saved and ver_in_dim_saved != D:
        raise RuntimeError(f"Verifier input mismatch: D={D} vs ver_in_dim_saved={ver_in_dim_saved}")

    # Load verifier
    V = FPVerifier(D)
    V.load_state_dict(torch.load(args.verifier_path, map_location='cpu'))
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

    # Sweep thresholds
    ts = np.linspace(0.0, 1.0, 201)
    robustness = np.array([(pos_scores >= t).mean() for t in ts])  # TPR on positives
    uniqueness = np.array([(neg_scores <  t).mean() for t in ts])  # TNR on negatives
    shade = np.minimum(robustness, uniqueness)

    # ARUC (area under shaded min curve)
    aruc = np.trapz(shade, ts)

    # Best threshold (maximize min(robustness, uniqueness))
    idx_best = int(np.argmax(shade))
    t_best = float(ts[idx_best])
    rob_best = float(robustness[idx_best])
    uniq_best = float(uniqueness[idx_best])
    # Accuracy at best threshold
    acc_best = 0.5 * (rob_best + uniq_best)

    print(f"Models: +{len(models_pos)} | -{len(models_neg)} | D={D}")
    print(f"ARUC = {aruc:.4f}")
    print(f"Best threshold τ* = {t_best:.3f} | Robustness={rob_best:.3f} | Uniqueness={uniq_best:.3f} | Acc={acc_best:.3f}")

    # Optional CSV dump
    if args.save_csv:
        import csv
        with open(args.save_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['threshold', 'robustness', 'uniqueness', 'min_curve'])
            for t, r, u, s in zip(ts, robustness, uniqueness, shade):
                w.writerow([f"{t:.5f}", f"{r:.6f}", f"{u:.6f}", f"{s:.6f}"])
        print(f"Saved CSV to {args.save_csv}")

    # Plot
    import os
    os.makedirs(os.path.dirname(args.out_plot), exist_ok=True)
    plt.figure(figsize=(4.0, 3.0), dpi=160)
    title = f"Cora node-classification (ARUC={aruc:.3f})"
    plt.title(title)
    plt.plot(ts, robustness, label='Robustness (TPR)', linewidth=1.8)
    plt.plot(ts, uniqueness, '--', label='Uniqueness (TNR)', linewidth=1.8)
    plt.fill_between(ts, 0, shade, alpha=0.15)
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
