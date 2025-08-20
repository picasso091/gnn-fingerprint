import argparse, os
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

class VerifierMLP(nn.Module):
    def __init__(self, in_dim, hidden=(128,64,32)):
        super().__init__()
        h1,h2,h3 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h1, h2), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h2, h3), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h3, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

def train_eval(X, y, epochs=100, lr=1e-3, val_split=0.5, seed=0, out="fingerprints/lp/univerifier_lp.pt"):
    torch.manual_seed(seed)

    X = X.detach().float().contiguous()
    y = y.detach().float().contiguous()

    # stratified split (exact 50/50 pos/neg in val)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    idx_tr, idx_val = next(sss.split(X, y))

    Xtr, ytr = X[idx_tr], y[idx_tr]
    Xva, yva = X[idx_val], y[idx_val]

    model = VerifierMLP(in_dim=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bcelogits = nn.BCEWithLogitsLoss()

    best_state = None
    best_auc = -1.0

    for ep in range(1, epochs+1):
        # train
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(Xtr)                 # [N_tr]
        loss = bcelogits(logits, ytr)       # binary labels in {0,1}
        loss.backward()
        opt.step()

        # validate (no graph)
        model.eval()
        with torch.no_grad():
            va_logits = model(Xva)
            va_prob = torch.sigmoid(va_logits).cpu().numpy()
            va_y = yva.cpu().numpy()
            auc = roc_auc_score(va_y, va_prob)
            acc = accuracy_score(va_y, (va_prob >= 0.5).astype("float32"))

        if auc > best_auc:
            best_auc = auc
            # store a pure Tensor copy (no graph)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0 or ep == epochs:
            print(f"[{ep}] train_loss={loss.item():.4f} valAUC={auc:.3f} valACC={acc:.3f}")

        del logits, loss, va_logits

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_dim": X.shape[1], "hidden": [128,64,32]}, out)
    print(f"Saved verifier to {out}  bestAUC={best_auc:.3f}")
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="fingerprints/lp/univerifier_lp.pt")
    args = ap.parse_args()

    d = torch.load(args.data, map_location="cpu")
    X, y = d["X"], d["y"].float()
    train_eval(X, y, epochs=args.epochs, lr=args.lr, val_split=args.val_split, seed=args.seed, out=args.out)

if __name__ == "__main__":
    main()
