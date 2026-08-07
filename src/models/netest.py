import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import Encoder, Predictor, Discriminator
from src.models.layers import GCNLayer, GINLayer


class NetEst(nn.Module):
    """Network Estimator with GCN encoder and separate treatment/exposure discriminators.

    Uses three-optimizer adversarial training: one for the treatment discriminator,
    one for the exposure (z) discriminator, and one for the predictor + encoder.
    """
    def __init__(self, Xshape, hidden, dropout=0):
        super(NetEst, self).__init__()
        self.encodergc = GCNLayer(Xshape, hidden, dropout)
        self.encoder = Encoder(hidden + Xshape, hidden, dropout)
        self.predictor = Predictor(hidden + 2, hidden, hidden, 1, dropout)
        self.discriminator = Discriminator(hidden, hidden, hidden, 1, dropout)
        self.discrimnator_z = Discriminator(hidden + 1, hidden, hidden, 1, dropout)

    def forward(self, x, t, z, edge_index):
        xgc = self.encodergc(x, edge_index)
        xgc = F.relu(xgc)
        xgc = self.encoder(torch.cat([xgc, x], dim=-1))
        pred_t = self.discriminator(xgc)
        pred_z = self.discrimnator_z(torch.cat([xgc, t.unsqueeze(1)], dim=-1))
        x = torch.cat([xgc, z.unsqueeze(1), t.unsqueeze(1)], dim=-1)
        y = self.predictor(x)
        return pred_t, y, pred_z


class GINNetEst(nn.Module):
    """GIN-based variant of NetEst.

    Same architecture as NetEst but uses a GINLayer instead of GCNLayer
    for the graph encoder, providing more expressive power.
    """
    def __init__(self, Xshape, hidden, dropout=0):
        super(GINNetEst, self).__init__()
        self.encodergc = GINLayer(Xshape, hidden, dropout)
        self.encoder = Encoder(hidden + Xshape, hidden, dropout)
        self.predictor = Predictor(hidden + 2, hidden, hidden, 1, dropout)
        self.discriminator = Discriminator(hidden, hidden, hidden, 1, dropout)
        self.discrimnator_z = Discriminator(hidden + 1, hidden, hidden, 1, dropout)

    def forward(self, x, t, z, edge_index):
        xgc = self.encodergc(x, edge_index)
        xgc = F.relu(xgc)
        xgc = self.encoder(torch.cat([xgc, x], dim=-1))
        pred_t = self.discriminator(xgc)
        pred_z = self.discrimnator_z(torch.cat([xgc, t.unsqueeze(1)], dim=-1))
        x = torch.cat([xgc, z.unsqueeze(1), t.unsqueeze(1)], dim=-1)
        y = self.predictor(x)
        return pred_t, y, pred_z
