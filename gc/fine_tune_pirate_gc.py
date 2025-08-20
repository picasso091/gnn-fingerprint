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

def stratified_indices(labels, tr=0.7, va=0.1, te=0.2):
    classes = labels.unique().tolist()
    tr_i, va_i, te_i = [], [], []
    for c in classes:
        idx = (labels == c).nonzero(as_tuple=True)[0].tolist()
        n = len(idx)
        nt, nv = math.floor(n*tr), math.floor(n*va)
        tr_i += idx[:nt]; va_i += idx[nt:nt+nv]; te_i += idx[nt+nv:]
    return tr_i, va_i, te_i

def freeze_all(m):
    for p in m.parameters(): p.requires_grad = False

def unfreeze_last_layer(m):
    freeze_all(m)
    for p in m.head.parameters(): p.requires_grad = True

def unfreeze_all(m):
    for p in m.parameters(): p.requires_grad = True

def reinit_layer(layer):
    if hasattr(layer, "reset_parameters"): layer.reset_parameters()
    for mod in layer.modules():
        if isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None: nn.init.zeros_(mod.bias)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_ckpt", type=str, required=True)
    ap.add_argument("--num_models", type=int, default=20, help="Total pirated models to generate")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--save_dir", type=str, default="models/positives/gc")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ensure_dir(args.save_dir)

    ds = TUDataset(root=args.data_root, name="ENZYMES")
    in_ch = ds.num_features if ds.num_features > 0 else 32
    if ds.num_features == 0:
        g = torch.Generator().manual_seed(42)
        for d in ds:
            d.x = torch.randn((d.num_nodes, in_ch), generator=g)
    labels = torch.tensor([d.y.item() for d in ds])
    tr, va, te = stratified_indices(labels)
    trl = DataLoader(Subset(ds,tr), batch_size=args.batch_size, shuffle=True)
    val = DataLoader(Subset(ds,va), batch_size=args.batch_size)
    tes = DataLoader(Subset(ds,te), batch_size=args.batch_size)

    ckpt = torch.load(args.target_ckpt, map_location="cpu")

    modes = ["ft_last","ft_all","reinit_last","reinit_all"]
    crit = nn.CrossEntropyLoss()

    generated = 0
    seed = 1
    while generated < args.num_models:
        for mode in modes:
            if generated >= args.num_models: break
            fname = f"pirate_{mode}_seed{seed}.pt"
            save_path = os.path.join(args.save_dir, fname)
            if os.path.exists(save_path):
                print(f"skip existing {save_path}")
                generated += 1; continue

            set_seed(seed)
            model = build_model_from_args(
                in_channels=ckpt["in_channels"],
                out_channels=ckpt["out_channels"],
                hidden_channels=ckpt["hidden"],
                num_layers=ckpt["layers"],
                sage_agg=ckpt["agg"],
                readout=ckpt["readout"],
                dropout=ckpt["dropout"],
                use_bn=True,
            ).to(args.device)
            model.load_state_dict(ckpt["state_dict"], strict=True)

            # apply obfuscation
            if mode=="ft_last": unfreeze_last_layer(model)
            elif mode=="ft_all": unfreeze_all(model)
            elif mode=="reinit_last": reinit_layer(model.head); unfreeze_all(model)
            elif mode=="reinit_all":
                for c in model.convs: reinit_layer(c)
                reinit_layer(model.head); unfreeze_all(model)

            opt = Adam(filter(lambda p:p.requires_grad, model.parameters()),
                       lr=args.lr, weight_decay=args.weight_decay)

            # fine-tune loop
            for ep in range(1, args.epochs+1):
                model.train()
                for b in trl:
                    b = b.to(args.device)
                    opt.zero_grad(set_to_none=True)
                    loss = crit(model(b.x,b.edge_index,b.batch), b.y)
                    loss.backward(); opt.step()

            torch.save({
                **ckpt,
                "state_dict": {k:v.cpu() for k,v in model.state_dict().items()}
            }, save_path)
            print(f"saved: {save_path}")
            generated += 1
        seed += 1

    print(f"Generated {generated} pirated models in {args.save_dir}")

if __name__ == "__main__":
    main()
