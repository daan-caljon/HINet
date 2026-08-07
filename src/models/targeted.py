import torch
import torch.nn as nn
import numpy as np

from src.models.components import Predictor
from src.models.layers import GCNLayer

"""Code from Doubly Robust Causal Effect Estimation under Networked Interference
via Targeted Learning (Chen et al. 2024):
https://github.com/WeilinChen507/targeted_interference"""

INI_NORMAL_VARIANCE = 0.1


class Truncated_power():
    def __init__(self, degree, knots):
        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)

        if self.degree == 0:
            raise ValueError('Degree should not be set to 0!')
        if not isinstance(self.degree, int):
            raise ValueError('Degree should be int')

    def forward(self, x):
        x = x.squeeze()
        out = torch.zeros(x.shape[0], self.num_of_basis, device=x.device)
        for _ in range(self.num_of_basis):
            if _ <= self.degree:
                if _ == 0:
                    out[:, _] = 1.
                else:
                    out[:, _] = x**_
            else:
                if self.degree == 1:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree]))
                else:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree - 1])) ** self.degree
        return out


class TR(nn.Module):
    def __init__(self, degree, knots):
        super(TR, self).__init__()
        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis
        self.weight = nn.Parameter(torch.rand(self.d, device='cuda'), requires_grad=True)

    def forward(self, t):
        out = self.spb.forward(t)
        out = torch.matmul(out, self.weight)
        return out

    def _initialize_weights(self):
        self.weight.data.zero_()


def comp_grid(y, num_grid):
    U = torch.ceil(y * num_grid)
    inter = 1 - (U - y * num_grid)
    L = U - 1
    L = torch.clamp(L, min=0)
    return L.long(), U.long(), inter


class Density_Block(nn.Module):
    def __init__(self, num_grid, ind, isbias=1):
        super(Density_Block, self).__init__()
        self.ind = ind
        self.num_grid = num_grid
        self.outd = num_grid + 1
        self.isbias = isbias
        self.weight = nn.Parameter(torch.rand(self.ind, self.outd, device='cuda'), requires_grad=True)
        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd, device='cuda'), requires_grad=True)
        else:
            self.bias = None
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z, x):
        out = torch.matmul(x, self.weight)
        if self.isbias:
            out += self.bias
        out = self.softmax(out)
        x1 = torch.arange(x.shape[0], device=x.device)
        L, U, inter = comp_grid(z, self.num_grid)
        L_out = out[x1, L]
        U_out = out[x1, U]
        out = L_out + (U_out - L_out) * inter
        return out

    def _initialize_weights(self):
        self.weight.data.normal_(0, INI_NORMAL_VARIANCE)
        if self.isbias:
            self.bias.data.normal_(0, INI_NORMAL_VARIANCE)


class Density_Estimator(nn.Module):
    def __init__(self, input_size, num_grid):
        super().__init__()
        self.num_grid = num_grid
        self.density_estimator_head = Density_Block(self.num_grid, input_size, isbias=1)

    def forward(self, x, z):
        g_Z = self.density_estimator_head(z, x)
        return g_Z

    def _initialize_weights(self):
        self.density_estimator_head._initialize_weights()


class Discriminator_simplified(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(Discriminator_simplified, self).__init__()
        self.disc1 = nn.Linear(input_size, hidden_size1, device='cuda')
        self.disc3 = nn.Linear(hidden_size1, output_size, device='cuda')
        self.act = nn.LeakyReLU(0.2, inplace=True).cuda()

    def forward(self, x):
        x = self.disc1(x)
        x = self.act(x)
        x = self.disc3(x)
        x = torch.sigmoid(x)
        return x

    def _initialize_weights(self):
        self.disc1.weight.data.normal_(0, INI_NORMAL_VARIANCE)
        self.disc3.weight.data.normal_(0, INI_NORMAL_VARIANCE)


class TargetedModel_DoubleBSpline(nn.Module):
    """Doubly robust targeted learning model for causal effect estimation
    under networked interference.

    Uses GCN encoding, separate density estimators for treatment and exposure,
    and truncated power basis splines for the fluctuation model.
    Two-step training: base model training + fluctuation parameter optimization.
    """
    def __init__(self, Xshape, hidden, dropout=False, num_grid=None, init_weight=True, tr_knots=0.25, cfg_density=None):
        super(TargetedModel_DoubleBSpline, self).__init__()
        if num_grid is None:
            num_grid = 20

        self.encoder = GCNLayer(in_channels=Xshape, hidden_channels=hidden)
        self.X_XN = Predictor(input_size=hidden + Xshape, hidden_size1=hidden, hidden_size2=hidden, output_size=int(hidden/2))
        self.Q1 = Predictor(input_size=int(hidden/2) + 1, hidden_size1=int(hidden*2), hidden_size2=hidden, output_size=1)
        self.Q0 = Predictor(input_size=int(hidden/2) + 1, hidden_size1=int(hidden*2), hidden_size2=hidden, output_size=1)
        self.g_T = Discriminator_simplified(input_size=int(hidden/2), hidden_size1=hidden, output_size=1)
        self.g_Z = Density_Estimator(input_size=int(hidden/2), num_grid=num_grid)
        tr_knots = list(np.arange(tr_knots, 1, tr_knots))
        tr_degree = 2
        self.tr_reg_t1 = TR(tr_degree, tr_knots)
        self.tr_reg_t0 = TR(tr_degree, tr_knots)

        if init_weight:
            self.g_Z._initialize_weights()
            self.g_T._initialize_weights()
            self.tr_reg_t1._initialize_weights()
            self.tr_reg_t0._initialize_weights()

    def parameter_base(self):
        return (list(self.encoder.parameters()) +
                list(self.X_XN.parameters()) +
                list(self.Q1.parameters()) + list(self.Q0.parameters()) +
                list(self.g_T.parameters()) +
                list(self.g_Z.parameters()))

    def parameter_trageted(self):
        return list(self.tr_reg_t0.parameters()) + list(self.tr_reg_t1.parameters())

    def tr_reg(self, T, neighborAverageT):
        tr_reg_t1 = self.tr_reg_t1(neighborAverageT)
        tr_reg_t0 = self.tr_reg_t0(neighborAverageT)
        regur = torch.where(T == 1, tr_reg_t1, tr_reg_t0)
        return regur

    def forward(self, x, t, z, edge_index):
        embeddings = self.encoder(x, edge_index)
        embeddings = self.X_XN(torch.cat((embeddings, x), dim=1))
        g_T_hat = self.g_T(embeddings)
        neighborAverageT = z
        g_Z_hat = self.g_Z(embeddings, neighborAverageT)
        g_Z_hat = g_Z_hat.unsqueeze(1)
        embed_avgT = torch.cat((embeddings, neighborAverageT.reshape(-1, 1)), 1)
        Q_hat = t.reshape(-1, 1) * self.Q1(embed_avgT) + (1-t.reshape(-1, 1)) * self.Q0(embed_avgT)
        epsilon = self.tr_reg(t, neighborAverageT)
        return g_T_hat, g_Z_hat, Q_hat, epsilon, embeddings, neighborAverageT

    def infer_potential_outcome(self, x, t, z, edge_index):
        embeddings = self.encoder(x, edge_index)
        embeddings = self.X_XN(torch.cat((embeddings, x), dim=1))
        g_T_hat = self.g_T(embeddings)
        g_T_hat = g_T_hat.squeeze(1)
        neighborAverageT = z
        g_Z_hat = self.g_Z(embeddings, neighborAverageT)
        embed_avgT = torch.cat((embeddings, neighborAverageT.reshape(-1, 1)), 1)
        Q_hat = t.reshape(-1, 1) * self.Q1(embed_avgT) + (1-t.reshape(-1, 1)) * self.Q0(embed_avgT)
        epsilon = self.tr_reg(t, neighborAverageT)
        return Q_hat.reshape(-1) + (epsilon.reshape(-1) * 1/(g_Z_hat.reshape(-1)*g_T_hat.reshape(-1) + 1e-6))
