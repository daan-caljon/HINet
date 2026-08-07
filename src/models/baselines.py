import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import Predictor
from src.models.layers import GCNLayer, GINLayer


class GINModel(nn.Module):
    """Simple GIN-based model for outcome prediction.

    Uses a single GIN layer followed by an MLP predictor.
    Does not predict treatment (returns zeros for t_pred).
    """
    def __init__(self, Xshape, hidden, dropout=0):
        super(GINModel, self).__init__()
        self.conv1 = GINLayer(Xshape + 1, hidden, dropout)
        self.fc = Predictor(hidden + Xshape + 1, hidden, hidden, 1, dropout)

    def forward(self, x, t, z, edge_index):
        x = torch.cat([x, t.unsqueeze(1)], dim=-1)
        x_gin = self.conv1(x, edge_index)
        x_gin = F.relu(x_gin)
        x_comb = torch.cat([x_gin, x], dim=-1)
        x = self.fc(x_comb)
        t_pred = torch.zeros(x.shape[0], 1, device=x.device)
        return t_pred, x


class TARNet(nn.Module):
    """Treatment-Agnostic Representation Network.

    MLP-based model (no graph convolution) with separate outcome heads
    for treated and control groups.
    """
    def __init__(self, Xshape, hidden, dropout=0, n_in=2, n_out=2):
        super(TARNet, self).__init__()
        self.gc = nn.ModuleList([nn.Linear(Xshape, hidden)])
        for i in range(n_in - 1):
            self.gc.append(nn.Linear(hidden, hidden))
        self.n_in = n_in
        self.n_out = n_out

        self.out_t00 = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(n_out)])
        self.out_t10 = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(n_out)])
        self.out_t01 = nn.Linear(hidden, 1)
        self.out_t11 = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)
        self.pp = nn.Linear(hidden, 1)

    def forward(self, x, t, z, edge_index):
        del z, edge_index

        rep = x
        for layer in self.gc:
            rep = self.dropout(F.relu(layer(rep)))

        y0_hidden = rep
        y1_hidden = rep
        for control_layer, treated_layer in zip(self.out_t00, self.out_t10):
            y0_hidden = self.dropout(F.relu(control_layer(y0_hidden)))
            y1_hidden = self.dropout(F.relu(treated_layer(y1_hidden)))

        # Outcomes are standardized continuous targets and must remain unbounded.
        y0 = self.out_t01(y0_hidden)
        y1 = self.out_t11(y1_hidden)
        treated = t.reshape(-1, 1) > 0.5
        y = torch.where(treated, y1, y0)

        # Kept for compatibility with the shared trainer interface.
        p1 = torch.sigmoid(self.pp(rep))
        return p1, y


class GCN_DECONF(nn.Module):
    """Graph Convolutional Network Deconfounder.

    Multi-layer GCN with separate outcome heads for treated/control groups.
    The learned representation is returned for the NetDeconf balancing loss.
    """
    def __init__(self, Xshape, hidden, dropout=0, n_in=1, n_out=2, cuda=True):
        super(GCN_DECONF, self).__init__()
        del cuda
        self.gc = nn.ModuleList([GCNLayer(Xshape, hidden, dropout)])
        for i in range(n_in - 1):
            self.gc.append(GCNLayer(hidden, hidden, dropout))
        self.n_in = n_in
        self.n_out = n_out

        self.out_t00 = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(n_out)])
        self.out_t10 = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(n_out)])
        self.out_t01 = nn.Linear(hidden, 1)
        self.out_t11 = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)
        self.pp = nn.Linear(hidden, 1)

    def forward(self, x, t, z, edge_index):
        del z

        rep = x
        for layer in self.gc:
            # GCNLayer applies its configured dropout internally.
            rep = F.relu(layer(rep, edge_index))

        y0_hidden = rep
        y1_hidden = rep
        for control_layer, treated_layer in zip(self.out_t00, self.out_t10):
            y0_hidden = self.dropout(F.relu(control_layer(y0_hidden)))
            y1_hidden = self.dropout(F.relu(treated_layer(y1_hidden)))

        y0 = self.out_t01(y0_hidden)
        y1 = self.out_t11(y1_hidden)
        treated = t.reshape(-1, 1) > 0.5
        y = torch.where(treated, y1, y0)

        # Kept for compatibility; NetDeconf is trained with outcome and balance losses.
        p1 = torch.sigmoid(self.pp(rep))
        return p1, y, rep
