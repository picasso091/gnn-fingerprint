import os, glob, argparse, random
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data

from graphsage_gc import build_model_from_args

def set_seed(s: int):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_ckpts(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path): return []
    return sorted(glob.glob(os.path.join(dir_path, "*.pt")))

def load_ckpt_model(ckpt_path: str, device: str) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model_from_args(
        in_channels=ckpt["in_channels"],
        out_channels=ckpt["out_channels"],
        hidden_channels=ckpt["hidden"],
        num_layers=ckpt["layers"],
        sage_agg=ckpt["agg"],
        readout=ckpt["readout"],
        dropout=ckpt["dropout"],
        use_bn=True,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model

@torch.no_grad()
def model_fp_vector(model: nn.Module, fps: List[Data]) -> torch.Tensor:
    # concat probs over all fingerprint graphs -> shape [N*C]
    vecs = []
    for g in fps:
        logits = model(g.x, g.edge_index, g.batch)
        vecs.append(logits.softmax(dim=-1).flatten())
    return torch.cat(vecs, dim=0)

class Univerifier(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprints_path", type=str, default="fingerprints/gc/fingerprints_gc_enzyme.pt")
    ap.add_argument("--pos_dir", type=str, default="models/positives/gc")
    ap.add_argument("--neg_dir", type=str, default="models/negatives/gc")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50) #default=50
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_dir", type=str, default="fingerprints/gc")
    ap.add_argument("--save_name", type=str, default="univerifier_gc_enzyme.pt")
    ap.add_argument("--save_dataset", action="store_true", help="Save X_train/test and y for reuse")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.save_dir)
    device = args.device

    # Load fingerprints
    pack = torch.load(args.fingerprints_path, map_location="cpu")
    fps = pack["fingerprints"]  # list of Data (x, edge_index, batch)
    for i, g in enumerate(fps):
        if getattr(g, "batch", None) is None:
            fps[i] = Data(
                x=g.x,
                edge_index=g.edge_index,
                y=getattr(g, "y", torch.zeros((), dtype=torch.long)),
                batch=torch.zeros(g.num_nodes, dtype=torch.long),
            )

    # Load positives/negatives
    pos_ckpts = list_ckpts(args.pos_dir)
    neg_ckpts = list_ckpts(args.neg_dir)
    if len(pos_ckpts) == 0 or len(neg_ckpts) == 0:
        raise RuntimeError("No positives or negatives found. Check models/positives/gc and models/negatives/gc")

    positives = [load_ckpt_model(p, device) for p in pos_ckpts]
    negatives = [load_ckpt_model(n, device) for n in neg_ckpts]
    print(f"Loaded suspects: positives={len(positives)}, negatives={len(negatives)}")

    # Compute features per model
    with torch.no_grad():
        # determine in_dim: N*C
        test_logits = positives[0](fps[0].x.to(device), fps[0].edge_index.to(device), fps[0].batch.to(device))
        C = test_logits.size(-1)
        N = len(fps)
        in_dim = N * C

    def to_device_graphs(graphs, device):
        out = []
        for g in graphs:
            out.append(type(g)(x=g.x.to(device), edge_index=g.edge_index.to(device), y=g.y, batch=g.batch.to(device)))
        return out

    fps_dev = to_device_graphs(fps, device)

    X_pos = []
    for m in positives:
        X_pos.append(model_fp_vector(m, fps_dev).cpu())
    X_neg = []
    for m in negatives:
        X_neg.append(model_fp_vector(m, fps_dev).cpu())

    X_pos = torch.stack(X_pos, dim=0)  # [P, in_dim]
    X_neg = torch.stack(X_neg, dim=0)  # [N, in_dim]
    y_pos = torch.ones(X_pos.size(0), dtype=torch.long)
    y_neg = torch.zeros(X_neg.size(0), dtype=torch.long)

    X = torch.cat([X_pos, X_neg], dim=0)
    y = torch.cat([y_pos, y_neg], dim=0)

    def stratified_half_split(y_tensor: torch.Tensor) -> Tuple[List[int], List[int]]:
        idx_pos = (y_tensor == 1).nonzero(as_tuple=True)[0].tolist()
        idx_neg = (y_tensor == 0).nonzero(as_tuple=True)[0].tolist()
        random.shuffle(idx_pos); random.shuffle(idx_neg)
        half_pos = len(idx_pos) // 2
        half_neg = len(idx_neg) // 2
        train_idx = idx_pos[:half_pos] + idx_neg[:half_neg]
        test_idx  = idx_pos[half_pos:] + idx_neg[half_neg:]
        random.shuffle(train_idx); random.shuffle(test_idx)
        return train_idx, test_idx

    train_idx, test_idx = stratified_half_split(y)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # Datasets & loaders
    train_ds = TensorDataset(X_train, y_train)
    test_ds  = TensorDataset(X_test,  y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # Model / opt
    verifier = Univerifier(in_dim=in_dim).to(device)
    opt = torch.optim.Adam(verifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    # Train loop
    best_acc = -1.0
    best_state = None
    for ep in range(1, args.epochs + 1):
        verifier.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = verifier(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * yb.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        # Eval
        verifier.eval()
        with torch.no_grad():
            corr = tot = 0
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = verifier(xb).argmax(dim=-1)
                corr += int((pred == yb).sum()); tot += yb.size(0)
            test_acc = corr / max(tot, 1)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu() for k, v in verifier.state_dict().items()}

        print(f"Epoch {ep:03d} | train_loss {train_loss:.4f} | test_acc {test_acc:.4f} | best {best_acc:.4f}")

    if best_state is None:
        best_state = {k: v.cpu() for k, v in verifier.state_dict().items()}

    # Save verifier
    out_path = os.path.join(args.save_dir, args.save_name)
    torch.save(
        {
            "state_dict": best_state,
            "in_dim": in_dim,
            "seed": args.seed,
            "N": N,
            "C": C,
            "fingerprints_path": args.fingerprints_path,
            "pos_used": len(pos_ckpts),
            "neg_used": len(neg_ckpts),
        },
        out_path,
    )
    print(f"Saved Univerifier → {out_path}")

    # save dataset
    if args.save_dataset:
        torch.save(
            {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test},
            os.path.join(args.save_dir, "univerifier_gc_dataset.pt"),
        )
        print(f"Saved dataset tensors → {os.path.join(args.save_dir, 'univerifier_gc_dataset.pt')}")

if __name__ == "__main__":
    main()
