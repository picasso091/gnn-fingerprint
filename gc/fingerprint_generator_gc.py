# Build graph-level fingerprints for ENZYMES (GraphSAGE target)

import os, glob, argparse, random
from typing import List, Tuple
import torch, time
from torch import nn
from torch_geometric.data import Data

from graphsage_gc import build_model_from_args

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_ckpts(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
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

def init_random_graph(num_nodes: int, feat_dim: int, edge_prob: float, device: str) -> Data:
    x = (2.0 * torch.rand((num_nodes, feat_dim), device=device)) - 1.0
    mask = torch.bernoulli(torch.full((num_nodes, num_nodes), edge_prob, device=device)).bool()
    mask = torch.triu(mask, diagonal=1)
    u, v = mask.nonzero(as_tuple=True)
    edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], dim=0).to(torch.long).contiguous()
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    return Data(x=x, edge_index=edge_index, y=torch.zeros((), dtype=torch.long), batch=batch)

@torch.no_grad()
def graph_forward_probs(model: nn.Module, g: Data) -> torch.Tensor:
    logits = model(g.x, g.edge_index, g.batch)
    return logits.softmax(dim=-1).flatten()

def concat_outputs(models: List[nn.Module], graphs: List[Data]) -> torch.Tensor:
    outs = []
    for m in models:
        vecs = [graph_forward_probs(m, g) for g in graphs]
        outs.append(torch.cat(vecs, dim=0))
    return torch.stack(outs, dim=0)

def flip_edges(edge_index: torch.Tensor, to_add: torch.Tensor, to_remove: torch.Tensor) -> torch.Tensor:
    device, dtype = edge_index.device, edge_index.dtype
    if to_remove.numel() > 0 and edge_index.numel() > 0:
        s = set(map(tuple, edge_index.t().tolist()))
        for u, v in zip(to_remove[0].tolist(), to_remove[1].tolist()):
            s.discard((u, v)); s.discard((v, u))
        edge_index = torch.tensor(list(s), device=device, dtype=dtype).t().contiguous() if len(s) else torch.empty((2,0), device=device, dtype=dtype)
    if to_add.numel() > 0:
        add_uv = torch.cat([to_add, to_add.flip(0)], dim=1).to(device=device, dtype=dtype)
        edge_index = add_uv if edge_index.numel()==0 else torch.cat([edge_index, add_uv], dim=1)
        edge_index = edge_index.t().unique(dim=0).t().contiguous()
    return edge_index

def rank_topk_edges_by_proxy_grad(num_nodes:int, edge_index:torch.Tensor, node_signal:torch.Tensor, k:int)->Tuple[torch.Tensor,torch.Tensor]:
    g = node_signal @ node_signal.t()
    iu = torch.triu(torch.ones_like(g, dtype=torch.bool), diagonal=1)
    g = g * iu
    adj = torch.zeros_like(g, dtype=torch.bool)
    if edge_index.numel() > 0:
        adj[edge_index[0], edge_index[1]] = True; adj[edge_index[1], edge_index[0]] = True
    add_scores = torch.where(~adj, g, torch.zeros_like(g))
    rem_scores = torch.where(adj, g, torch.zeros_like(g))
    u,v = iu.nonzero(as_tuple=True)

    add_vals = add_scores[iu]; add_idx = torch.argsort(add_vals, descending=True)[:k]
    au,av = u[add_idx], v[add_idx]; keep = add_scores[au,av]>0
    to_add = torch.stack([au[keep], av[keep]], dim=0)

    rem_vals = rem_scores[iu]; rem_idx = torch.argsort(rem_vals, descending=False)[:k]
    ru,rv = u[rem_idx], v[rem_idx]; keep2 = rem_scores[ru,rv]<0
    to_remove = torch.stack([ru[keep2], rv[keep2]], dim=0)
    return to_add,to_remove


class Univerifier(nn.Module):
    def __init__(self,in_dim:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,128), nn.LeakyReLU(0.01,inplace=True),
            nn.Linear(128,64), nn.LeakyReLU(0.01,inplace=True),
            nn.Linear(64,32), nn.LeakyReLU(0.01,inplace=True),
            nn.Linear(32,2),
        )
    def forward(self,x): return self.net(x)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--target_ckpt",type=str,required=True)
    ap.add_argument("--pos_dir",type=str,default="models/positives/gc")
    ap.add_argument("--neg_dir",type=str,default="models/negatives/gc")
    ap.add_argument("--N",type=int,default=64)
    ap.add_argument("--n",type=int,default=32)
    ap.add_argument("--feat_dim",type=int,default=32)
    ap.add_argument("--edge_prob",type=float,default=0.05)
    ap.add_argument("--topk_edges",type=int,default=16)
    ap.add_argument("--alpha_x",type=float,default=1e-2)
    ap.add_argument("--steps",type=int,default=1) #default=200
    ap.add_argument("--e1",type=int,default=1)
    ap.add_argument("--e2",type=int,default=5)
    ap.add_argument("--lr_univerifier",type=float,default=1e-3)
    ap.add_argument("--device",type=str,default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_dir",type=str,default="fingerprints/gc")
    ap.add_argument("--save_name",type=str,default="fingerprints_gc_enzyme.pt")
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.save_dir)

    device = args.device

    _target_ckpt = torch.load(args.target_ckpt, map_location="cpu")
    target_in_channels = int(_target_ckpt["in_channels"])

    feat_dim = target_in_channels

    target = load_ckpt_model(args.target_ckpt, device)

    pos_ckpts=list_ckpts(args.pos_dir); neg_ckpts=list_ckpts(args.neg_dir)
    positives=[load_ckpt_model(p,device) for p in pos_ckpts]
    negatives=[load_ckpt_model(n,device) for n in neg_ckpts]
    models=[target]+positives+negatives
    y_models=torch.tensor([1]*(1+len(positives))+[0]*len(negatives),dtype=torch.long,device=device)
    print(f"Loaded: target=1, positives={len(positives)}, negatives={len(negatives)}")

    # init fingerprints
    fps = [
        init_random_graph(args.n, feat_dim, args.edge_prob, device=device)
        for _ in range(args.N)
    ]

    with torch.no_grad(): C=graph_forward_probs(target,fps[0]).numel()
    univerifier=Univerifier(C*args.N).to(device)
    opt_u=torch.optim.Adam(univerifier.parameters(),lr=args.lr_univerifier)
    crit=nn.CrossEntropyLoss()

    for t in range(args.steps):
        # update fingerprints
        for _ in range(args.e1):
            with torch.no_grad(): X_all=concat_outputs(models,fps)
            X_all.requires_grad_(True)
            logits_u=univerifier(X_all)
            loss=crit(logits_u,y_models)
            univerifier.zero_grad(set_to_none=True); loss.backward()
            per_graph=[]
            for i in range(args.N):
                sl=slice(i*C,(i+1)*C)
                g_i=X_all.grad[:,sl].mean(dim=0)
                per_graph.append(g_i)
            for i,gsig in enumerate(per_graph):
                d=fps[i]
                node_weight=torch.randn((d.num_nodes,1),device=device)
                x_dir=(node_weight @ gsig.unsqueeze(0)).sum(dim=1,keepdim=True)
                d.x=torch.clamp(d.x+args.alpha_x*torch.sign(x_dir),-1.0,1.0)
                node_signal=d.x.norm(p=2,dim=1,keepdim=True)
                node_signal=node_signal/(node_signal.norm(p=2)+1e-9)
                to_add,to_remove=rank_topk_edges_by_proxy_grad(d.num_nodes,d.edge_index,node_signal,args.topk_edges)
                d.edge_index=flip_edges(d.edge_index,to_add,to_remove)
                
        # update univerifier
        for _ in range(args.e2):
            with torch.no_grad(): X_all=concat_outputs(models,fps)
            logits_u=univerifier(X_all.detach())
            loss_u=crit(logits_u,y_models)
            opt_u.zero_grad(set_to_none=True); loss_u.backward(); opt_u.step()
        if (t+1)%10==0:
            with torch.no_grad():
                X_all=concat_outputs(models,fps)
                acc=(univerifier(X_all).argmax(-1)==y_models).float().mean().item()
            print(f"[{t+1:03d}/{args.steps}] univerifier_acc={acc:.3f}")

    save_path=os.path.join(args.save_dir,args.save_name)
    torch.save({"fingerprints":[Data(x=d.x.cpu(),edge_index=d.edge_index.cpu()) for d in fps],
                "N":args.N,"n":args.n,"feat_dim":args.feat_dim,"edge_prob":args.edge_prob,
                "topk_edges":args.topk_edges,"seed":args.seed,
                "pos_ckpts_used":pos_ckpts,"neg_ckpts_used":neg_ckpts},
               save_path)
    print(f"Saved fingerprints: {save_path}")

if __name__=="__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("time taken (min): ", (end_time-start_time)/60)

