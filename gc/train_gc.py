# Train GraphSAGE-Mean on ENZYMES and save target checkpoint.

import os
import math
import argparse
from typing import Tuple, List

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Subset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from graphsage_gc import build_model_from_args


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def stratified_indices_by_ratio(labels: torch.Tensor, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2) -> Tuple[List[int], List[int], List[int]]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    num_graphs = labels.size(0)
    classes = labels.unique().tolist()

    train_idx, val_idx, test_idx = [], [], []
    for c in classes:
        idx_c = (labels == c).nonzero(as_tuple=True)[0].tolist()
        n_c = len(idx_c)
        n_train = math.floor(n_c * train_ratio)
        n_val = math.floor(n_c * val_ratio)
        n_test = n_c - n_train - n_val
        train_idx.extend(idx_c[:n_train])
        val_idx.extend(idx_c[n_train:n_train + n_val])
        test_idx.extend(idx_c[n_train + n_val:])
    return train_idx, val_idx, test_idx


def maybe_add_random_features_if_missing(dataset: TUDataset, dim: int = 32, seed: int = 42) -> int:
    """
    if node attributes are missing, assign random values.
    Returns the updated in_channels.
    """
    if dataset.num_features and dataset.num_features > 0:
        return dataset.num_features
    # assign random node features
    g = torch.Generator().manual_seed(seed)
    for data in dataset:
        n = data.num_nodes
        data.x = torch.randn((n, dim), generator=g)
    return dim


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            pred = logits.argmax(dim=-1)
            correct += int((pred == batch.y).sum().item())
            total += batch.y.size(0)
    return correct / max(total, 1)


def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.y.size(0)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Train GraphSAGE-Mean on ENZYMES (graph classification)")
    parser.add_argument("--root", type=str, default="./data", help="Root to store datasets")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience on val acc")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)  # depth=3 per paper
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--readout", type=str, default="mean", choices=["mean", "sum", "max"])
    parser.add_argument("--agg", type=str, default="mean", choices=["mean", "max", "add"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_dir", type=str, default="fingerprints/gc")
    parser.add_argument("--ckpt_name", type=str, default="target_graphsage.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.ckpt_dir)

    dataset = TUDataset(root=args.root, name="ENZYMES")
    in_channels = maybe_add_random_features_if_missing(dataset, dim=32, seed=args.seed)
    num_classes = dataset.num_classes  # should be 6

    labels = torch.tensor([data.y.item() for data in dataset], dtype=torch.long)
    train_idx, val_idx, test_idx = stratified_indices_by_ratio(labels, 0.7, 0.1, 0.2)

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Build model (GraphSAGE-Mean)
    model = build_model_from_args(
        in_channels=in_channels,
        out_channels=num_classes,
        hidden_channels=args.hidden,
        num_layers=args.layers,
        sage_agg=args.agg,      # 'mean'
        readout=args.readout,   # 'mean'
        dropout=args.dropout,
        use_bn=True,
    ).to(args.device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, args.device, criterion)
        val_acc = evaluate(model, val_loader, args.device)
        test_acc = evaluate(model, test_loader, args.device)

        improved = val_acc > best_val
        if improved:
            best_val = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(f"Epoch {epoch:03d} | loss {train_loss:.4f} | val_acc {val_acc:.4f} | test_acc {test_acc:.4f} | best_val {best_val:.4f}")

        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
            break

    # Save best checkpoint (target model for fingerprinting)
    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    torch.save(
        {
            "state_dict": best_state,
            "in_channels": in_channels,
            "out_channels": num_classes,
            "hidden": args.hidden,
            "layers": args.layers,
            "agg": args.agg,
            "readout": args.readout,
            "dropout": args.dropout,
            "seed": args.seed,
            "dataset": "ENZYMES",
        },
        ckpt_path,
    )
    print(f"Saved target GraphSAGE checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
