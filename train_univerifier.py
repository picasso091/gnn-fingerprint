
"""
Trains the FPVerifier V_ω over the dataset created by generate_univerifier_dataset.py
This is the "post-process" verifier, but for full fidelity use fingerprint_generator's alternating loop.
"""
import argparse, torch, torch.nn as nn, torch.nn.functional as F

class Verifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, max(32, in_dim//2)),
            nn.ReLU(),
            nn.Linear(max(32, in_dim//2), 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='fingerprints/univerifier_dataset.pt')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val_split', type=float, default=0.2)
    args = ap.parse_args()

    pack = torch.load(args.dataset, map_location='cpu')
    X, y = pack['X'], pack['y'].unsqueeze(-1)
    X = X.detach()
    y = y.detach()
    
    n = X.size(0)
    n_val = int(args.val_split * n)
    perm = torch.randperm(n)
    X_train, y_train = X[perm[:-n_val]], y[perm[:-n_val]]
    X_val, y_val = X[perm[-n_val:]], y[perm[-n_val:]]

    V = Verifier(X.size(1))
    opt = torch.optim.Adam(V.parameters(), lr=args.lr)

    best_val, best_state = 0.0, None
    for ep in range(1, args.epochs+1):
        V.train(); opt.zero_grad()
        pred = V(X_train)
        loss = F.binary_cross_entropy(pred, y_train)
        loss.backward(); opt.step()
        with torch.no_grad():
            V.eval()
            pv = V(X_val)
            val_loss = F.binary_cross_entropy(pv, y_val)
            val_acc = ((pv>=0.5).float()==y_val).float().mean().item()
        if val_acc > best_val:
            best_val, best_state = val_acc, {k:v.cpu().clone() for k,v in V.state_dict().items()}
        if ep % 20 == 0 or ep == args.epochs:
            print(f"Epoch {ep:03d} | train_bce {loss.item():.4f} | val_bce {val_loss.item():.4f} | val_acc {val_acc:.4f}")

    V.load_state_dict(best_state)
    torch.save(V.state_dict(), 'fingerprints/univerifier.pt')
    print(f"Saved fingerprints/univerifier.pt | Best Val Acc {best_val:.4f}")

if __name__ == '__main__':
    main()
