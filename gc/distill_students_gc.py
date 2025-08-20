import os, argparse, random, torch
from torch import nn
from torch.optim import Adam
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
from torch_geometric.data import Data

from graphsage_gc import build_model_from_args

def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def maybe_add_rand_x(ds, dim=32, seed=42):
    if ds.num_features > 0: return ds.num_features
    g = torch.Generator().manual_seed(seed)
    for d in ds:
        d.x = torch.randn((d.num_nodes, dim), generator=g)
    return dim


def sample_subgraph_batch(batch, keep_ratio: float):
    new_graphs = []
    for gid in batch.batch.unique(sorted=True):
        idx = (batch.batch == gid).nonzero(as_tuple=True)[0]
        n = len(idx)
        k = max(2, int(n * keep_ratio))
        keep_nodes = idx[torch.randperm(n)[:k]]

        mask_nodes = torch.zeros(batch.num_nodes, dtype=torch.bool, device=batch.x.device)
        mask_nodes[keep_nodes] = True
        _, edge_index = subgraph(mask_nodes, batch.edge_index, relabel_nodes=True)

        # if no edges, skip this graph
        if edge_index is None or edge_index.numel() == 0:
            continue

        edge_index = edge_index.to(torch.long).contiguous()

        data = Data(
            x=batch.x[keep_nodes],
            edge_index=edge_index,
            y=batch.y[gid].unsqueeze(0),
            batch=torch.zeros(len(keep_nodes), dtype=torch.long, device=batch.x.device),
        )
        new_graphs.append(data)

    return new_graphs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_ckpt", type=str, required=True)
    ap.add_argument("--num_models", type=int, default=20, help="Number of student pirates to generate")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--student_hidden", type=int, default=128)
    ap.add_argument("--student_layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1000, help="starting seed")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--save_dir", type=str, default="models/positives/gc")
    args = ap.parse_args()

    ensure_dir(args.save_dir)

    # dataset
    ds = TUDataset(root=args.data_root, name="ENZYMES")
    in_ch = maybe_add_rand_x(ds, dim=32, seed=args.seed)
    num_classes = ds.num_classes
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # teacher (target)
    tckpt = torch.load(args.target_ckpt, map_location="cpu")
    teacher = build_model_from_args(
        in_channels=tckpt["in_channels"],
        out_channels=tckpt["out_channels"],
        hidden_channels=tckpt["hidden"],
        num_layers=tckpt["layers"],
        sage_agg=tckpt["agg"],
        readout=tckpt["readout"],
        dropout=tckpt["dropout"],
        use_bn=True,
    ).to(args.device)
    teacher.load_state_dict(tckpt["state_dict"], strict=True)
    teacher.eval()

    soft = nn.Softmax(dim=-1)
    logsoft = nn.LogSoftmax(dim=-1)
    kl = nn.KLDivLoss(reduction="batchmean")

    generated = 0
    seed = args.seed
    while generated < args.num_models:
        fname = f"pirate_distill_seed{seed}.pt"
        save_path = os.path.join(args.save_dir, fname)
        if os.path.exists(save_path):
            print(f"skip existing {save_path}")
            generated += 1
            seed += 1
            continue

        set_seed(seed)

        # student 
        student_hidden = args.student_hidden if args.student_hidden > 0 else 128
        student_layers = args.student_layers if args.student_layers > 0 else 3

        student = build_model_from_args(
            in_channels=in_ch,
            out_channels=num_classes,
            hidden_channels=student_hidden,
            num_layers=student_layers,
            sage_agg="mean",
            readout="mean",
            dropout=0.5,
            use_bn=True,
        ).to(args.device)

        opt = Adam(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for ep in range(1, args.epochs + 1):
            student.train()
            for batch in loader:
                batch = batch.to(args.device)
                keep_ratio = random.uniform(0.5, 0.8)
                subs = sample_subgraph_batch(batch, keep_ratio)

                for g in subs:
                    g = g.to(args.device)
                    with torch.no_grad():
                        t_logits = teacher(g.x, g.edge_index, g.batch)
                        t_prob = soft(t_logits)

                    s_logits = student(g.x, g.edge_index, g.batch)
                    loss = kl(logsoft(s_logits), t_prob)

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

        # save ckpt
        torch.save({
            "state_dict": {k: v.cpu() for k,v in student.state_dict().items()},
            "in_channels": in_ch,
            "out_channels": num_classes,
            "hidden": student_hidden,
            "layers": student_layers,
            "agg": "mean",
            "readout": "mean",
            "dropout": 0.5,
            "seed": seed,
            "dataset": "ENZYMES",
        }, save_path)
        print(f"saved: {save_path}")
        generated += 1
        seed += 1

    print(f"Done. Generated {generated} distilled pirates in {args.save_dir}")

if __name__ == "__main__":
    main()
