
"""
Builds a dataset for the verifier: for each model, compute Concat({m(I_p)}_p) and label 1 for positives
({f} ∪ F+) and 0 for negatives F−. Saves to a .pt file with tensors X (N x D) and y (N).
"""
import argparse, torch, glob, json
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dense_to_sparse
from gcn import get_model

def forward_on_fp(model, fp):
    with torch.no_grad():  # <— ensure no graph
        edge_index = dense_to_sparse(fp["A"].detach())[0]
        logits = model(fp["X"].detach(), edge_index)
        return logits.mean(dim=0).detach()

def concat_for_model(model, fps):
    vecs = [forward_on_fp(model, fp) for fp in fps]
    return torch.cat(vecs, dim=-1).detach()

def load_model_from_meta(path_pt, in_dim):
    j = json.load(open(path_pt.replace('.pt','.json'),'r'))
    m = get_model(j["arch"], in_dim, j["hidden"], j["num_classes"])
    m.load_state_dict(torch.load(path_pt, map_location='cpu')); m.eval()
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_meta', type=str, default='models/target_meta.json')
    ap.add_argument('--target_path', type=str, default='models/target_model.pt')
    ap.add_argument('--positives_glob', type=str,
    default='models/positives/ftpr_*.pt,models/positives/distill_*.pt')
    ap.add_argument('--negatives_glob', type=str,
    default='models/negatives/negative_*.pt')
    ap.add_argument('--fingerprints_path', type=str, default='fingerprints/fingerprints.pt')
    args = ap.parse_args()

    ds = Planetoid(root='data/cora', name='Cora')
    in_dim = ds.num_features

    fps_pack = torch.load(args.fingerprints_path, map_location='cpu')
    fps = fps_pack["fingerprints"]

    # target
    tmeta = json.load(open(args.target_meta,'r'))
    target = get_model(tmeta["arch"], in_dim, tmeta["hidden"], tmeta["num_classes"])
    target.load_state_dict(torch.load(args.target_path, map_location='cpu')); target.eval()

    pos_globs = [g.strip() for g in args.positives_glob.split(',') if g.strip()]
    pos_paths = sorted(sum([glob.glob(g) for g in pos_globs], []))
    neg_paths = sorted(glob.glob(args.negatives_glob))

    models, labels = [], []
    models.append(target); labels.append(1)
    for p in pos_paths:
        models.append(load_model_from_meta(p, in_dim)); labels.append(1)
    for n in neg_paths:
        models.append(load_model_from_meta(n, in_dim)); labels.append(0)

    X = torch.stack([concat_for_model(m, fps) for m in models]).detach()
    y = torch.tensor(labels, dtype=torch.float32).detach()


    out = {'X': X, 'y': y}
    torch.save(out, 'fingerprints/univerifier_dataset.pt')
    print("Saved fingerprints/univerifier_dataset.pt with", X.shape[0], "rows and dim", X.shape[1])

if __name__ == '__main__':
    main()
