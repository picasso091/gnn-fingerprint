
"""
Quick diagnostic to evaluate verifier on the constructed dataset.
"""
import torch

pack = torch.load('fingerprints/univerifier_dataset.pt', map_location='cpu')
X, y = pack['X'], pack['y'].unsqueeze(-1)

V = torch.load('fingerprints/univerifier.pt', map_location='cpu')
# In case V was saved as sd only
try:
    # Try as state dict into same architecture
    import torch.nn as nn, torch.nn.functional as F
    class Verifier(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, max(32, in_dim//2)),
                nn.ReLU(),
                nn.Linear(max(32, in_dim//2), 1),
                nn.Sigmoid()
            )
        def forward(self, x): return self.net(x)
    V_state = V
    V = Verifier(X.size(1)); V.load_state_dict(V_state); V.eval()
except Exception:
    pass

with torch.no_grad():
    pred = (V(X)>=0.5).float()
    acc = (pred==y).float().mean().item()
print(f"Verifier full-dataset accuracy: {acc:.4f}")
