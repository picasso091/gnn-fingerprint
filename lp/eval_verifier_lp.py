import argparse, os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt


class VerifierMLP(nn.Module):
    """3-layer MLP used for the Univerifier
    """
    def __init__(self, in_dim: int, hidden=(128, 64, 32)):
        super().__init__()
        h1, h2, h3 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h1, h2), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h2, h3), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h3, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def tpr_tnr_vs_threshold(y_true: np.ndarray, probs: np.ndarray, num=1001):
    """Compute TPR and TNR across thresholds in [0,1]."""
    thresholds = np.linspace(0.0, 1.0, num)
    P = np.clip(np.sum(y_true == 1), 1, None)
    N = np.clip(np.sum(y_true == 0), 1, None)
    TPR = np.empty_like(thresholds)
    TNR = np.empty_like(thresholds)
    for i, t in enumerate(thresholds):
        pred = probs >= t
        TP = np.sum((pred == 1) & (y_true == 1))
        TN = np.sum((pred == 0) & (y_true == 0))
        TPR[i] = TP / P
        TNR[i] = TN / N
    return thresholds, TPR, TNR


def aruc_from_probs(y_true: np.ndarray, probs: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, probs)
    U = 1.0 - fpr
    order = np.argsort(U)
    return float(np.trapz(tpr[order], U[order]))


def save_threshold_plot(thr: np.ndarray, TPR: np.ndarray, TNR: np.ndarray, aruc: float, out_png: str):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    overlap = np.minimum(TPR, TNR)

    J = TPR + TNR - 1.0
    j_idx = int(np.argmax(J))
    tau = float(thr[j_idx])

    plt.figure(figsize=(6, 4))
    plt.plot(thr, TPR, linewidth=2.2, color="#ff0000", label="Robustness (TPR)")
    plt.plot(thr, TNR, linewidth=2.0, linestyle="--", color="#0000ff", label="Uniqueness (TNR)")

    plt.fill_between(thr, overlap, color="#888888", alpha=0.18, label="Overlap (ARUC region)")

    # Vertical marker at best threshold
    plt.axvline(tau, color="#444444", alpha=0.3)

    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel("Threshold (τ)")
    plt.ylabel("Score")
    plt.title(f"CiteSeer link-prediction • ARUC={aruc:.3f}")
    plt.grid(True, linestyle=":", linewidth=0.7)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_threshold_csv(thr: np.ndarray, TPR: np.ndarray, TNR: np.ndarray, out_csv: str):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("threshold,robustness_tpr,uniqueness_tnr")
        for t, r, u in zip(thr, TPR, TNR):
            f.write(f"{t:.6f},{r:.6f},{u:.6f}")


def main():
    ap = argparse.ArgumentParser("Evaluate Univerifier (LP)")
    ap.add_argument("--data", required=True, help="univerifier dataset .pt (X,y)")
    ap.add_argument("--verifier", required=True, help="verifier checkpoint .pt")
    ap.add_argument("--split-json", default=None, help='Optional JSON with {"train_idx":[], "test_idx":[]}.')
    ap.add_argument("--test-size", type=float, default=0.5, help="If no split JSON, fraction to use as test.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-png", default="plots/citeseer_gcn_aruc.png")
    ap.add_argument("--out-csv", default="plots/citeseer_gcn_aruc.csv")
    ap.add_argument("--points", type=int, default=1001, help="#threshold points in [0,1]")
    args = ap.parse_args()

    # Load data
    d = torch.load(args.data, map_location="cpu")
    X = d["X"].float()
    y = d["y"].long().numpy()

    # Load verifier
    ck = torch.load(args.verifier, map_location="cpu")
    in_dim = int(ck.get("in_dim", X.shape[1]))
    hidden = tuple(ck.get("hidden", [128, 64, 32]))
    ver = VerifierMLP(in_dim=in_dim, hidden=hidden)
    ver.load_state_dict(ck["state_dict"], strict=False)
    ver.eval()

    # Build split
    if args.split_json and os.path.exists(args.split_json):
        import json
        with open(args.split_json, "r") as f:
            sp = json.load(f)
        idx_tr = np.asarray(sp.get("train_idx"))
        idx_te = np.asarray(sp.get("test_idx"))
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        idx_tr, idx_te = next(sss.split(X, y))

    # Probabilities on test set only
    with torch.no_grad():
        probs_te = torch.sigmoid(ver(X[idx_te])).cpu().numpy()

    # Curves vs threshold
    thr, TPR, TNR = tpr_tnr_vs_threshold(y_true=y[idx_te], probs=probs_te, num=args.points)

    # ARUC from ROC
    aruc = aruc_from_probs(y_true=y[idx_te], probs=probs_te)

    save_threshold_plot(thr, TPR, TNR, aruc, args.out_png)
    save_threshold_csv(thr, TPR, TNR, args.out_csv)

    print(f"Saved PNG: {args.out_png}")
    print(f"Saved CSV: {args.out_csv}")
    print(f"ARUC:     {aruc:.4f}")
    print(f"Points:   {len(thr)} (τ∈[0,1])")


if __name__ == "__main__":
    main()
