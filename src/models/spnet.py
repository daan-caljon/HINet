import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from src.models.components import Predictor, Discriminator
from src.models.layers import MaskedAttentionLayer


class SPNet(nn.Module):
    """Separating Personal and Networked Effects model.

    Based on: Huang, Q., Ma, J., Li, J., Guo, R., Sun, H., & Chang, Y. (2023).
    Modeling interference for individual treatment effect estimation from networked
    observational data. ACM TKDD, 18(3), 1-21.

    Uses two GCN pathways (outcome & treatment), masked attention, and Wasserstein
    distance to balance representations.
    """
    def __init__(self, Xshape, hidden, dropout=0):
        super(SPNet, self).__init__()
        self.part_outcome = GCNConv(Xshape, hidden, add_self_loops=True)
        self.part_treat = GCNConv(Xshape, hidden, add_self_loops=True)
        self.outcome_linear = nn.Linear(hidden, hidden)
        self.treat_linear = nn.Linear(hidden, hidden)
        self.attention = MaskedAttentionLayer(2 * hidden, 1)
        self.discriminator_t = Discriminator(hidden, hidden, hidden, 1, dropout)
        self.encoder_final = nn.Linear(hidden + hidden, hidden)
        self.predict_1 = Predictor(hidden, hidden, hidden, 1, dropout)
        self.predict_0 = Predictor(hidden, hidden, hidden, 1, dropout)

    def forward(self, x, t, z, edge_index):
        r_o = self.part_outcome(x, edge_index)
        r_t = self.part_treat(x, edge_index)
        r_t = F.relu(r_t)
        r_o = F.relu(r_o)
        cat = torch.cat([r_o, r_t], dim=-1)
        h = self.attention(cat, r_t, t.unsqueeze(1), edge_index)
        z = self.encoder_final(torch.cat([r_o, h], dim=-1))
        representations = z
        pred_1 = self.predict_1(z)
        pred_0 = self.predict_0(z)
        pred = torch.where(t > 0, pred_1.squeeze(), pred_0.squeeze())
        pred_t = self.discriminator_t(r_t)
        return pred_t, pred.unsqueeze(1), representations
