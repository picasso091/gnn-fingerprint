# eval_verifier.py
import argparse, glob, json, torch, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dense_to_sparse
from gcn import get_model

def load_model_from_pair(pt_path, in_dim):
    j = json.load(open(pt_path.replace('.pt','.json'),'r'))
    m = get_model(j["arch"], in_dim, j["hidden"], j["num_classes"])
    m.load_state_dict(torch.load(pt_path, map_location='cpu')); m.eval()
    return m, j

def forward_on_fp(model, fp):
    edge_index = dense_to_sparse(fp["A"])[0]
    logits = model(fp["X"], edge_index)  # [n, C]
    return logits.mean(dim=0)

def concat_for_model(model, fps):
    with torch.no_grad():
        return torch.cat([forward_on_fp(model, fp) for fp in fps], dim=-1)

def split_models(positive_pts, negative_pts, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.array(sorted(positive_pts))
    neg = np.array(sorted(negative_pts))
    rng.shuffle(pos); rng.shuffle(neg)
    pos_mid, neg_mid = len(pos)//2, len(neg)//2
    return (pos[:pos_mid], pos[pos_mid:]), (neg[:neg_mid], neg[neg_mid:])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fingerprints', default='fingerprints/fingerprints.pt')
    ap.add_argument('--univerifier', default='fingerprints/univerifier.pt')
    ap.add_argument('--positives_glob', default='models/positives/ftpr_*.pt,models/positives/distill_*.pt')
    ap.add_argument('--negatives_glob', default='models/negatives/negative_*.pt')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--outdir', default='eval')
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # load fingerprints
    pack = torch.load(args.fingerprints, map_location='cpu')
    fps = [{"X": fp["X"].detach(), "A": fp["A"].detach()} for fp in pack["fingerprints"]]

    # dataset info (for in_dim)
    ds = Planetoid(root='data/cora', name='Cora')
    in_dim = ds.num_features

    # collect model paths
    pos_globs = [g.strip() for g in args.positives_glob.split(',') if g.strip()]
    pos_paths = sorted(sum([glob.glob(g) for g in pos_globs], []))
    neg_paths = sorted(glob.glob(args.negatives_glob))

    # split 1:1 (verifier train vs test); we evaluate on the held-out half (test)
    (pos_train, pos_test), (neg_train, neg_test) = split_models(pos_paths, neg_paths, seed=args.seed)
    print(f"[models] pos total={len(pos_paths)} neg total={len(neg_paths)} | eval on pos={len(pos_test)} neg={len(neg_test)}")

    # load univerifier arch
    class Verifier(torch.nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_dim, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(32, 1),
                torch.nn.Sigmoid()
            )
        def forward(self, x): return self.net(x)

    # infer in-dim from fingerprints and #classes
    num_classes = ds.num_classes
    ver_in_dim = len(fps) * num_classes
    V = Verifier(ver_in_dim)
    V.load_state_dict(torch.load(args.univerifier, map_location='cpu'))
    V.eval()

    # compute logits for each model on fingerprints
    with torch.no_grad():
        # positives (target itself could be included too)
        X_pos = []
        for p in pos_test:
            m,_ = load_model_from_pair(p, in_dim)
            X_pos.append(concat_for_model(m, fps))
        X_pos = torch.stack(X_pos) if X_pos else torch.empty(0, ver_in_dim)

        X_neg = []
        for n in neg_test:
            m,_ = load_model_from_pair(n, in_dim)
            X_neg.append(concat_for_model(m, fps))
        X_neg = torch.stack(X_neg) if X_neg else torch.empty(0, ver_in_dim)

        s_pos = V(X_pos).squeeze(-1).cpu().numpy()
        s_neg = V(X_neg).squeeze(-1).cpu().numpy()

    # sweep thresholds and compute metrics
    ths = np.linspace(0.0, 1.0, 101)
    robust, unique, accs, inter = [], [], [], []
    for t in ths:
        tp = (s_pos >= t).mean() if len(s_pos)>0 else 0.0   # robustness (TPR)
        tn = (s_neg <  t).mean() if len(s_neg)>0 else 0.0   # uniqueness (TNR)
        robust.append(tp); unique.append(tn)
        accs.append( (tp*len(s_pos) + tn*len(s_neg)) / (len(s_pos)+len(s_neg)) if (len(s_pos)+len(s_neg))>0 else 0.0 )
        inter.append(min(tp, tn))  # overlap region at this threshold

    # ARUC = area under overlap region (R vs U curves intersect region)
    # approximate with trapezoid on threshold axis
    aruc = np.trapz(inter, ths)

    print(f"ARUC: {aruc:.3f}")
    print(f"Mean Test Accuracy (avg over thresholds): {np.mean(accs):.3f}")
    print(f"Max Test Accuracy (best threshold): {np.max(accs):.3f}")

    # plot curves
    plt.figure()
    plt.plot(ths, robust, label='Robustness (TPR)')
    plt.plot(ths, unique, label='Uniqueness (TNR)')
    # shade intersection region
    plt.fill_between(ths, np.minimum(robust, unique), 0, alpha=0.2)
    plt.xlabel('threshold')
    plt.ylabel('score')
    plt.ylim(0,1.01)
    plt.title(f'Robustness vs Uniqueness (ARUC={aruc:.3f})')
    plt.legend()
    out_png = f"{args.outdir}/curves_cora_nodecls.png"
    plt.savefig(out_png, dpi=180, bbox_inches='tight')
    print(f"Saved {out_png}")
    # also save per-threshold CSV
    np.savetxt(f"{args.outdir}/metrics_cora_nodecls.csv",
               np.column_stack([ths, robust, unique, accs, inter]),
               delimiter=",", header="threshold,robustness,uniqueness,accuracy,intersection", comments="")
if __name__ == '__main__':
    main()
