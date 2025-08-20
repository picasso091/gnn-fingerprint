# Train unrelated (negative) GraphSAGE models on ENZYMES
# Each model trained from scratch with a different random seed

import os, argparse, math, torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Subset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from graphsage_gc import build_model_from_args

def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def maybe_add_rand_x(ds, dim=32, seed=42):
    if ds.num_features > 0:
        return ds.num_features
    g = torch.Generator().manual_seed(seed)
    for d in ds:
        d.x = torch.randn((d.num_nodes, dim), generator=g)
    return dim

def stratified_indices(labels, tr=0.7, va=0.1, te=0.2):
    classes = labels.unique().tolist()
    tr_i, va_i, te_i = [], [], []
    for c in classes:
        idx = (labels == c).nonzero(as_tuple=True)[0].tolist()
        n = len(idx)
        nt, nv = math.floor(n*tr), math.floor(n*va)
        tr_i += idx[:nt]; va_i += idx[nt:nt+nv]; te_i += idx[nt+nv:]
    return tr_i, va_i, te_i

def eval_model(model, loader, device):
    model.eval(); corr = tot = 0
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            logits = model(b.x, b.edge_index, b.batch)
            pred = logits.argmax(-1)
            corr += int((pred == b.y).sum()); tot += b.y.size(0)
    return corr / max(tot, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_models", type=int, default=20, help="Number of unrelated negatives to train")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--save_dir", type=str, default="models/negatives/gc")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1, help="starting seed")
    args = ap.parse_args()

    ensure_dir(args.save_dir)

    # dataset
    ds = TUDataset(root=args.data_root, name="ENZYMES")
    in_ch = maybe_add_rand_x(ds, dim=32, seed=args.seed)
    num_classes = ds.num_classes
    labels = torch.tensor([d.y.item() for d in ds])
    tr, va, te = stratified_indices(labels)
    trl = DataLoader(Subset(ds, tr), batch_size=args.batch_size, shuffle=True)
    val = DataLoader(Subset(ds, va), batch_size=args.batch_size)
    tes = DataLoader(Subset(ds, te), batch_size=args.batch_size)

    crit = nn.CrossEntropyLoss()

    generated = 0
    seed = args.seed
    while generated < args.num_models:
        fname = f"unrelated_seed{seed}.pt"
        save_path = os.path.join(args.save_dir, fname)
        if os.path.exists(save_path):
            print(f"skip existing {save_path}")
            generated += 1; seed += 1
            continue

        set_seed(seed)
        model = build_model_from_args(
            in_channels=in_ch,
            out_channels=num_classes,
            hidden_channels=args.hidden,
            num_layers=args.layers,
            sage_agg="mean",
            readout="mean",
            dropout=0.5,
            use_bn=True,
        ).to(args.device)

        opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val, best_state, noimp = -1.0, None, 0
        for ep in range(1, args.epochs+1):
            model.train()
            for b in trl:
                b = b.to(args.device)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(b.x, b.edge_index, b.batch), b.y)
                loss.backward(); opt.step()

            va_acc = eval_model(model, val, args.device)
            te_acc = eval_model(model, tes, args.device)
            if va_acc > best_val:
                best_val = va_acc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                noimp = 0
            else:
                noimp += 1
            print(f"[unrelated seed={seed}] epoch {ep:03d} val={va_acc:.3f} test={te_acc:.3f}")
            if noimp >= args.patience: break

        if best_state is None:
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        torch.save({
            "state_dict": best_state,
            "in_channels": in_ch,
            "out_channels": num_classes,
            "hidden": args.hidden,
            "layers": args.layers,
            "agg": "mean",
            "readout": "mean",
            "dropout": 0.5,
            "seed": seed,
            "dataset": "ENZYMES",
        }, save_path)
        print(f"saved: {save_path}")
        generated += 1; seed += 1

    print(f"Generated {generated} unrelated models in {args.save_dir}")

if __name__ == "__main__":
    main()
