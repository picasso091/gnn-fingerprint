# GraphSAGE for Graph Classification (ENZYMES)
# Implements SAGEConv encoder with global pooling head (mean/sum/max).
# Use depth=3 by default

from typing import Literal, Optional, List
import torch
from torch import nn
from torch_geometric.nn import SAGEConv, global_mean_pool, global_add_pool, global_max_pool

ReadoutType = Literal["mean", "sum", "max"]
AggType = Literal["mean", "max", "add"]

class GraphSAGEGC(nn.Module):
    """
    GraphSAGE for graph-level classification.

    Args:
        in_channels (int):   Node feature dimension.
        hidden_channels (int): Hidden size for all GraphSAGE layers.
        out_channels (int):  Number of classes (ENZYMES has 6).
        num_layers (int):    Number of GraphSAGE layers (>= 2). Default: 3.
        sage_agg (str):      Aggregator for SAGEConv: 'mean' | 'max' | 'add'. Default: 'mean'.
        readout (str):       Graph readout: 'mean' | 'sum' | 'max'. Default: 'mean'.
        dropout (float):     Dropout probability. Default: 0.5.
        use_bn (bool):       Use BatchNorm1d after each hidden layer. Default: True.
        act (nn.Module):     Activation (default: LeakyReLU).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        sage_agg: AggType = "mean",
        readout: ReadoutType = "mean",
        dropout: float = 0.5,
        use_bn: bool = True,
        act: Optional[nn.Module] = None,
    ):
        super().__init__()

        assert num_layers >= 2, "num_layers must be >= 2"
        assert sage_agg in ("mean", "max", "add")
        assert readout in ("mean", "sum", "max")

        self.readout = readout
        self.dropout = nn.Dropout(p=dropout)
        self.use_bn = use_bn
        self.act = act if act is not None else nn.LeakyReLU(negative_slope=0.01, inplace=True)

        # Encoder: SAGEConv stack
        convs: List[SAGEConv] = []
        bns: List[nn.BatchNorm1d] = []

        # First layer
        convs.append(SAGEConv(in_channels, hidden_channels, aggr=sage_agg))
        if use_bn:
            bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=sage_agg))
            if use_bn:
                bns.append(nn.BatchNorm1d(hidden_channels))

        # Last layer
        convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=sage_agg))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns) if use_bn else None

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels, out_channels),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.use_bn:
            for bn in self.bns:
                bn.reset_parameters()
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _readout(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.readout == "mean":
            return global_mean_pool(x, batch)
        if self.readout == "sum":
            return global_add_pool(x, batch)
        if self.readout == "max":
            return global_max_pool(x, batch)
        raise ValueError(f"Unsupported readout: {self.readout}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # SAGE layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                if self.use_bn:
                    x = self.bns[i](x)
                x = self.act(x)
                x = self.dropout(x)

        # Graph-level pooling
        graph_x = self._readout(x, batch)

        # Classification head
        logits = self.head(graph_x)
        return logits


def build_model_from_args(
    in_channels: int,
    out_channels: int,
    hidden_channels: int = 128,
    num_layers: int = 3,
    sage_agg: AggType = "mean",
    readout: ReadoutType = "mean",
    dropout: float = 0.5,
    use_bn: bool = True,
) -> GraphSAGEGC:

    return GraphSAGEGC(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        sage_agg=sage_agg,
        readout=readout,
        dropout=dropout,
        use_bn=use_bn,
    )
