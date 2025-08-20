import argparse, glob, os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hid=64, out=64, num_layers=3, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_dim, hid)] +
                                   [GCNConv(hid, hid) for _ in range(num_layers-2)] +
                                   [GCNConv(hid, out)])
        self.dropout = dropout
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs)-1:
                x = F.relu(x); x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class SAGEEncoder(nn.Module):
    def __init__(self, in_dim, hid=64, out=64, num_layers=3, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_dim, hid)] +
                                   [SAGEConv(hid, hid) for _ in range(num_layers-2)] +
                                   [SAGEConv(hid, out)])
        self.dropout = dropout
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs)-1:
                x = F.relu(x); x = F.dropout(x, p=self.dropout, training=self.training)
        return x

def build_edge_index_from_adj(adj_bin: torch.Tensor) -> torch.Tensor:
    up = torch.triu(adj_bin, diagonal=1)
    r, c = torch.nonzero(up, as_tuple=True)
    return torch.stack([torch.cat([r, c]), torch.cat([c, r])], dim=0)

def inner_product_scores(z: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    u, v = pairs[:,0], pairs[:,1]
    return torch.sigmoid((z[u] * z[v]).sum(dim=-1))  # [m]

def load_fps(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    # infer number of fingerprints
    keys = sorted([k for k in z.files if k.startswith("X_")], key=lambda s: int(s.split("_")[1]))
    fps = []
    for k in keys:
        i = int(k.split("_")[1])
        X = torch.tensor(z[f"X_{i}"]).float()
        A = torch.tensor(z[f"A_{i}"]).float()
        pairs = torch.tensor(z[f"pairs_{i}"]).long()
        edge_index = build_edge_index_from_adj((A > 0.5).float())
        fps.append((X, edge_index, pairs))
    return fps

def instantiate_from_meta(sd):
    arch = sd.get("arch","gcn").lower()
    in_dim = int(sd.get("in_dim"))
    hid = int(sd.get("hid", 64))
    out = int(sd.get("out", 64))
    layers = int(sd.get("layers", 3))
    enc = GCNEncoder(in_dim, hid, out, layers) if arch=="gcn" else SAGEEncoder(in_dim, hid, out, layers)
    enc.load_state_dict(sd["state_dict"], strict=False)
    enc.eval()
    return enc

def adapt_features_for_model(x: torch.Tensor, expected_in: int) -> torch.Tensor:
    n, d = x.shape
    if d == expected_in: return x
    if d > expected_in:  # slice
        return x[:, :expected_in]
    # pad
    pad = torch.zeros(n, expected_in - d, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)

def model_in_dim(model: nn.Module) -> int:
    for n,p in model.named_parameters():
        if n.endswith("convs.0.lin.weight") and p.ndim==2:
            return p.shape[1]
        if n.endswith("convs.0.weight") and p.ndim==2:  # fallback
            return p.shape[1]
    return None

def row_for_model(model, fps):
    vecs = []
    for X, edge_index, pairs in fps:
        ein = model_in_dim(model)
        X_in = adapt_features_for_model(X, ein) if ein else X
        z = model(X_in, edge_index)
        s = inner_product_scores(z, pairs)  # [m]
        vecs.append(s.view(-1))
    return torch.cat(vecs, dim=0)  # [N * m]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprints", type=str, required=True)
    ap.add_argument("--pos-glob", type=str, required=True)
    ap.add_argument("--neg-glob", type=str, required=True)
    ap.add_argument("--target-ckpt", type=str, required=False)
    ap.add_argument("--out", type=str, default="fingerprints/lp/univerifier_lp_citeseer.pt")
    args = ap.parse_args()

    fps = load_fps(args.fingerprints)
    pos_paths = sorted(glob.glob(args.pos_glob))
    neg_paths = sorted(glob.glob(args.neg_glob))

    models = []
    labels = []

    if args.target_ckpt:
        sd = torch.load(args.target_ckpt, map_location="cpu")
        models.append(instantiate_from_meta(sd)); labels.append(1)

    for p in pos_paths:
        sd = torch.load(p, map_location="cpu")
        models.append(instantiate_from_meta(sd)); labels.append(1)

    for n in neg_paths:
        sd = torch.load(n, map_location="cpu")
        models.append(instantiate_from_meta(sd)); labels.append(0)

    rows = [row_for_model(m, fps) for m in models]
    X = torch.stack(rows, dim=0)
    y = torch.tensor(labels, dtype=torch.long)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"X": X, "y": y, "num_fps": len(fps), "m_pairs": fps[0][2].shape[0]}, args.out)
    print(f"saved {args.out}  X:{tuple(X.shape)}  y:{tuple(y.shape)}")

if __name__ == "__main__":
    main()
