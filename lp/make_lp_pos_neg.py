import os, json, math, random, argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import negative_sampling

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
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # node embeddings

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
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

def build_encoder(arch:str, in_dim:int, hid:int=64, out:int=64, layers:int=3, dropout:float=0.5):
    arch = arch.lower()
    if arch in ("gcn",):
        return GCNEncoder(in_dim, hid, out, layers, dropout)
    if arch in ("sage","graphsage"):
        return SAGEEncoder(in_dim, hid, out, layers, dropout)
    raise ValueError(f"Unknown arch {arch}")


def score_pairs(z: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
    u, v = ei
    return (z[u] * z[v]).sum(dim=-1)

def bce_on_edges(enc: nn.Module, x, edge_index, pos_ei, num_nodes, num_neg=None):
    if num_neg is None: num_neg = pos_ei.size(1)
    z = enc(x, edge_index)
    neg_ei = negative_sampling(edge_index=edge_index, num_nodes=num_nodes, num_neg_samples=num_neg)
    pos_logit = score_pairs(z, pos_ei)
    neg_logit = score_pairs(z, neg_ei)
    y = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)])
    logits = torch.cat([pos_logit, neg_logit])
    return F.binary_cross_entropy_with_logits(logits, y)

@torch.no_grad()
def eval_auc(enc: nn.Module, x, edge_index, pos_ei, num_nodes):
    z = enc(x, edge_index)
    neg_ei = negative_sampling(edge_index=edge_index, num_nodes=num_nodes, num_neg_samples=pos_ei.size(1))
    pos = torch.sigmoid(score_pairs(z, pos_ei))
    neg = torch.sigmoid(score_pairs(z, neg_ei))
    from sklearn.metrics import roc_auc_score
    scores = torch.cat([pos, neg]).cpu().numpy()
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]).cpu().numpy()
    return float(__import__("sklearn.metrics").metrics.roc_auc_score(labels, scores))

def get_splits(edge_index, val_ratio=0.125, test_ratio=0.25):
    E = edge_index.size(1)
    perm = torch.randperm(E)
    n_test = int(test_ratio * E); n_val = int(val_ratio * E)
    test_e = edge_index[:, perm[:n_test]]
    val_e  = edge_index[:, perm[n_test:n_test+n_val]]
    train_e= edge_index[:, perm[n_test+n_val:]]
    return train_e, val_e, test_e

def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# Load teacher from target LPModel checkpoint
def load_teacher_encoder(target_ckpt:str, dropout=0.5):
    sd = torch.load(target_ckpt, map_location="cpu")
    state = sd.get("state_dict", sd)
    model_type = sd.get("type","gcn")
    in_dim = int(sd.get("in_dim"))
    keys = [k for k in state.keys() if k.startswith("enc.convs")]
    last_lin = [k for k in keys if k.endswith("lin.weight")]
    if not last_lin:
        raise RuntimeError("Could not locate encoder conv weights in state_dict.")
    last = sorted(last_lin)[-1]
    first = sorted([k for k in keys if k.endswith("lin.weight")])[0]
    hid = state[first].shape[0]
    out = state[last].shape[0]
    arch = model_type.lower()
    enc = build_encoder(arch, in_dim, hid, out, layers=3, dropout=dropout)
    # strip 'enc.' prefix
    enc_state = {}
    for k, v in state.items():
        if k.startswith("enc."):
            enc_state[k.replace("enc.", "", 1)] = v
    missing, unexpected = enc.load_state_dict(enc_state, strict=False)
    if missing:
        print(f"Warn: missing keys in teacher load: {missing[:4]} ...")
    if unexpected:
        print(f"Warn: unexpected keys in teacher load: {unexpected[:4]} ...")
    return enc, dict(arch=arch, in_dim=in_dim, hid=hid, out=out, layers=3)


# Training loops
def train_finetune(enc, data, tr_e, va_e, epochs=10, lr=1e-3, seed=0, wd=0.0):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = enc.to(device)
    x = data.x.to(device); num_nodes = data.num_nodes
    edge_index = tr_e.to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=lr, weight_decay=wd)
    best = None; best_auc = -1.0
    for ep in range(1, epochs+1):
        enc.train(); opt.zero_grad()
        loss = bce_on_edges(enc, x, edge_index, tr_e.to(device), num_nodes)
        loss.backward(); opt.step()
        if ep % 2 == 0 or ep == epochs:
            enc.eval()
            auc = eval_auc(enc, x, edge_index, va_e.to(device), num_nodes)
            if auc > best_auc:
                best_auc, best = auc, {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
    if best is None:
        best = enc.state_dict()
    enc.load_state_dict(best, strict=False)
    return enc

def reinit_last_layer(enc):
    # reinit final conv layer (GCNConv) weights & bias
    last = enc.convs[-1]
    for p in last.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.zeros_(p)

def train_distill(student_enc, teacher_enc, data, tr_e, va_e, epochs=10, lr=1e-3, seed=0, alpha_supervised=0.2):
    """
    Distill teacher link probabilities into the student.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_enc = student_enc.to(device)
    teacher_enc = teacher_enc.to(device); teacher_enc.eval()

    x = data.x.to(device); num_nodes = data.num_nodes
    edge_index = tr_e.to(device)

    opt = torch.optim.Adam(student_enc.parameters(), lr=lr)
    best = None; best_auc = -1.0

    for ep in range(1, epochs+1):
        student_enc.train(); opt.zero_grad()
        # sample negatives same count as positives
        neg_e = negative_sampling(edge_index=edge_index, num_nodes=num_nodes, num_neg_samples=tr_e.size(1))
        with torch.no_grad():
            tz = teacher_enc(x, edge_index)
            t_pos = torch.sigmoid(score_pairs(tz, tr_e.to(device)))
            t_neg = torch.sigmoid(score_pairs(tz, neg_e))

        sz = student_enc(x, edge_index)
        s_pos_logit = score_pairs(sz, tr_e.to(device))
        s_neg_logit = score_pairs(sz, neg_e)

        # distill to teacher probs
        distill_loss = F.binary_cross_entropy_with_logits(s_pos_logit, t_pos) + \
                       F.binary_cross_entropy_with_logits(s_neg_logit, t_neg)
        # small supervised component
        y_pos = torch.ones_like(s_pos_logit)
        y_neg = torch.zeros_like(s_neg_logit)
        sup_loss = F.binary_cross_entropy_with_logits(s_pos_logit, y_pos) + \
                   F.binary_cross_entropy_with_logits(s_neg_logit, y_neg)

        loss = (1 - alpha_supervised) * distill_loss + alpha_supervised * sup_loss
        loss.backward(); opt.step()

        if ep % 2 == 0 or ep == epochs:
            student_enc.eval()
            auc = eval_auc(student_enc, x, edge_index, va_e.to(device), num_nodes)
            if auc > best_auc:
                best_auc, best = auc, {k: v.detach().cpu().clone() for k, v in student_enc.state_dict().items()}

    if best is None:
        best = student_enc.state_dict()
    student_enc.load_state_dict(best, strict=False)
    return student_enc

def train_scratch(enc, data, tr_e, va_e, epochs=50, lr=1e-3, seed=0, wd=0.0):
    return train_finetune(enc, data, tr_e, va_e, epochs=epochs, lr=lr, seed=seed, wd=wd)


def save_encoder(enc, out_path, meta):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"state_dict": enc.state_dict(), **meta}, out_path)
    print(f"saved: {out_path}")

def main():
    ap = argparse.ArgumentParser("Generate LP suspect models (positives & negatives) for CiteSeer")
    ap.add_argument("--mode", choices=["finetune","partial","distill","scratch_neg"], required=True)
    ap.add_argument("--target-ckpt", type=str, help="Target LPModel checkpoint (for finetune/partial/distill).")
    ap.add_argument("--arch", type=str, default="gcn", help="Encoder arch for finetune/partial/scratch_neg (gcn|sage).")
    ap.add_argument("--student-arch", type=str, default="sage", help="Student arch for distill (gcn|sage).")
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--out", type=str, required=True, help="Output directory to place generated checkpoints.")
    ap.add_argument("--num-models", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha-supervised", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = Planetoid(root="data", name="CiteSeer", transform=NormalizeFeatures())[0]
    tr_e, va_e, te_e = get_splits(data.edge_index, val_ratio=0.125, test_ratio=0.25)

    if args.mode in ("finetune","partial","distill"):
        if not args.target_ckpt:
            raise SystemExit("--target-ckpt is required for finetune/partial/distill.")
        teacher, tmeta = load_teacher_encoder(args.target_ckpt)
        in_dim, out_dim = tmeta["in_dim"], tmeta["out"]
    else:
        in_dim, out_dim = data.num_features, 64

    # generate pos/neg
    os.makedirs(args.out, exist_ok=True)
    for i in range(args.num_models):
        seed_i = args.seed + i
        if args.mode == "finetune":
            enc = build_encoder(args.arch, in_dim, args.hid, out_dim, layers=3)
            # load teacher weights if same arch; otherwise initialize from teacher enc where shapes match
            if args.arch == tmeta["arch"]:
                enc.load_state_dict(teacher.state_dict(), strict=False)
            else:
                print("[info] arch mismatch; starting from random init.")
            enc = train_finetune(enc, data, tr_e, va_e, epochs=args.epochs, lr=args.lr, seed=seed_i)
            meta = {"arch": args.arch, "in_dim": in_dim, "hid": args.hid, "out": out_dim, "layers": 3, "mode": "finetune"}
            save_encoder(enc, os.path.join(args.out, f"pos_finetune_{args.arch}_seed{seed_i}.pt"), meta)

        elif args.mode == "partial":
            enc = build_encoder(args.arch, in_dim, args.hid, out_dim, layers=3)
            if args.arch == tmeta["arch"]:
                enc.load_state_dict(teacher.state_dict(), strict=False)
            else:
                print("[info] arch mismatch; starting from random init.")
            reinit_last_layer(enc)  # pirate by reinitializing last layer
            enc = train_finetune(enc, data, tr_e, va_e, epochs=args.epochs, lr=args.lr, seed=seed_i)
            meta = {"arch": args.arch, "in_dim": in_dim, "hid": args.hid, "out": out_dim, "layers": 3, "mode": "partial"}
            save_encoder(enc, os.path.join(args.out, f"pos_partial_{args.arch}_seed{seed_i}.pt"), meta)

        elif args.mode == "distill":
            student = build_encoder(args.student_arch, in_dim, args.hid, out_dim, layers=3)
            student = train_distill(student, teacher, data, tr_e, va_e,
                                    epochs=args.epochs, lr=args.lr, seed=seed_i,
                                    alpha_supervised=args.alpha_supervised)
            meta = {"arch": args.student_arch, "in_dim": in_dim, "hid": args.hid, "out": out_dim, "layers": 3, "mode": "distill"}
            save_encoder(student, os.path.join(args.out, f"pos_distill_{args.student_arch}_seed{seed_i}.pt"), meta)

        elif args.mode == "scratch_neg":
            enc = build_encoder(args.arch, in_dim, args.hid, out_dim, layers=3)
            enc = train_scratch(enc, data, tr_e, va_e, epochs=max(args.epochs, 30), lr=args.lr, seed=seed_i)
            meta = {"arch": args.arch, "in_dim": in_dim, "hid": args.hid, "out": out_dim, "layers": 3, "mode": "scratch_neg"}
            save_encoder(enc, os.path.join(args.out, f"neg_{args.arch}_seed{seed_i}.pt"), meta)

    print("Pos/Neg generation complete.")

if __name__ == "__main__":
    main()
