import torch, torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import DataLoader
import os

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hid=64, out=64, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_dim, hid)] +
                                   [GCNConv(hid, hid) for _ in range(num_layers-2)] +
                                   [GCNConv(hid, out)])
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs)-1: x = F.relu(x)
        return x

class SAGEEncoder(nn.Module):
    def __init__(self, in_dim, hid=64, out=64, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_dim, hid)] +
                                   [SAGEConv(hid, hid) for _ in range(num_layers-2)] +
                                   [SAGEConv(hid, out)])
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs)-1: x = F.relu(x)
        return x

class LPModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.enc = encoder
    def score(self, z, ei):  # dot-product decoder
        src, dst = ei
        return (z[src] * z[dst]).sum(dim=-1)
    def forward(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.enc(x, edge_index)
        pos_logit = self.score(z, pos_edge_index)
        neg_logit = self.score(z, neg_edge_index)
        return pos_logit, neg_logit

def get_splits(edge_index, num_nodes, val_ratio=0.1, test_ratio=0.2):
    E = edge_index.size(1)
    perm = torch.randperm(E)
    n_test = int(test_ratio * E); n_val = int(val_ratio * E)
    test_e = edge_index[:, perm[:n_test]]
    val_e  = edge_index[:, perm[n_test:n_test+n_val]]
    train_e= edge_index[:, perm[n_test+n_val:]]
    return train_e, val_e, test_e

def bce_loss(pos_logit, neg_logit):
    y = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)])
    logits = torch.cat([pos_logit, neg_logit])
    return F.binary_cross_entropy_with_logits(logits, y)

@torch.no_grad()
def eval_auc(model, x, edge_index, pos_e, num_nodes):
    z = model.enc(x, edge_index)
    neg_e = negative_sampling(edge_index=edge_index, num_nodes=num_nodes,
                              num_neg_samples=pos_e.size(1))
    pos = torch.sigmoid(model.score(z, pos_e))
    neg = torch.sigmoid(model.score(z, neg_e))
    scores = torch.cat([pos, neg]).cpu()
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]).cpu()
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels.numpy(), scores.numpy())

def main(model_type='gcn', epochs=200, lr=1e-3, seed=0, out='lp_citeseer.pt'):
    torch.manual_seed(seed)
    data = Planetoid(root='data', name='CiteSeer', transform=NormalizeFeatures())[0]
    tr_e, va_e, te_e = get_splits(data.edge_index, data.num_nodes, val_ratio=0.125, test_ratio=0.25)  # 7/1/2

    enc = GCNEncoder(data.num_features) if model_type=='gcn' else SAGEEncoder(data.num_features)
    model = LPModel(enc); opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        model.train(); opt.zero_grad()
        neg_e = negative_sampling(edge_index=tr_e, num_nodes=data.num_nodes, num_neg_samples=tr_e.size(1))
        pos_logit, neg_logit = model(data.x, tr_e, tr_e, neg_e)
        loss = bce_loss(pos_logit, neg_logit)
        loss.backward(); opt.step()

        if ep % 20 == 0:
            model.eval()
            val_auc = eval_auc(model, data.x, tr_e, va_e, data.num_nodes)
            print(f"[{ep}] loss={loss.item():.4f} valAUC={val_auc:.3f}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({'state_dict': model.state_dict(),
            'type': model_type,
            'in_dim': data.num_features}, out)

    test_auc = eval_auc(model, data.x, tr_e, te_e, data.num_nodes)
    print(f"Test AUC: {test_auc:.3f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['gcn','sage'], default='gcn')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=str, default='fingerprints/lp/lp_citeseer.pt')
    args = ap.parse_args()
    main(args.model, args.epochs, args.lr, args.seed, args.out)
