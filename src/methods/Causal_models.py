
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv
import numpy as np

import src.methods.Attention_layer as Attention_layer
from src.methods.utils import GradientReversalLayer


# GAT Layer
class GATLayer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.0, heads=1):
        super(GATLayer, self).__init__()
        self.gat = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True  # if True, outputs heads*hidden_channels; else average
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # apply GAT convolution + dropout
        x = self.gat(x, edge_index)
        x = F.elu(x)  # typical nonlinearity in GAT
        return self.dropout(x)


# GraphSAGE Layer
class GraphSAGELayer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.0, aggr='mean'):
        super(GraphSAGELayer, self).__init__()
        self.sage = SAGEConv(
            in_channels,
            hidden_channels,
            aggr=aggr
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.sage(x, edge_index)
        x = F.relu(x)  # standard nonlinearity in GraphSAGE
        return self.dropout(x)
    
class GCNLayer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        return self.dropout(self.conv(x, edge_index))


class GINLayer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,dropout=0):
        super(GINLayer, self).__init__()
        # Define the MLP for GIN
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.Tanh(),
            torch.nn.Dropout(dropout),
            # torch.nn.Linear(hidden_channels, hidden_channels)
        )
        # GIN layer
        self.gin = GINConv(self.mlp)

    def forward(self, x, edge_index):
        return self.gin(x, edge_index)


class GINModel(torch.nn.Module):
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
        t_pred = torch.zeros(x.shape[0], 1).cuda()

        return t_pred, x


class NetEst(nn.Module):
    def __init__(self, Xshape, hidden, dropout=0):
        super(NetEst, self).__init__()
        # self.encodergc = GINLayer(Xshape, hidden)
        self.encodergc = GCNLayer(Xshape, hidden, dropout)
        # self.encoder = nn.Linear(hidden+Xshape,hidden)
        self.encoder = Encoder(hidden + Xshape, hidden, dropout)
        self.predictor = Predictor(hidden + 2, hidden, hidden, 1, dropout)
        self.discriminator = Discriminator(hidden, hidden, hidden, 1, dropout)
        self.discrimnator_z = Discriminator(hidden + 1, hidden, hidden, 1, dropout)

    def forward(self, x, t, z, edge_index):
        xgc = self.encodergc(x, edge_index)
        xgc = F.relu(xgc)
        xgc = self.encoder(torch.cat([xgc, x], dim=-1))
        # average of neighbor t
        pred_t = self.discriminator(xgc)
        pred_z = self.discrimnator_z(torch.cat([xgc, t.unsqueeze(1)], dim=-1))
        x = torch.cat([xgc, z.unsqueeze(1), t.unsqueeze(1)], dim=-1)
        y = self.predictor(x)

        return pred_t, y, pred_z


class GINNetEst(nn.Module):
    def __init__(self, Xshape, hidden, dropout=0):
        super(GINNetEst, self).__init__()
        self.encodergc = GINLayer(Xshape, hidden, dropout)
        # self.encodergc = GCNLayer(Xshape, hidden)
        # self.encoder = nn.Linear(hidden+Xshape,hidden)
        self.encoder = Encoder(hidden + Xshape, hidden, dropout)
        self.predictor = Predictor(hidden + 2, hidden, hidden, 1, dropout)
        self.discriminator = Discriminator(hidden, hidden, hidden, 1, dropout)
        self.discrimnator_z = Discriminator(hidden + 1, hidden, hidden, 1, dropout)

    def forward(self, x, t, z, edge_index):
        xgc = self.encodergc(x, edge_index)
        xgc = F.relu(xgc)
        xgc = self.encoder(torch.cat([xgc, x], dim=-1))
        # average of neighbor t
        pred_t = self.discriminator(xgc)
        pred_z = self.discrimnator_z(torch.cat([xgc, t.unsqueeze(1)], dim=-1))
        x = torch.cat([xgc, z.unsqueeze(1), t.unsqueeze(1)], dim=-1)
        y = self.predictor(x)
        return pred_t, y, pred_z


class HINet(nn.Module):
    def __init__(self, Xshape, hidden, dropout=0):
        super(HINet, self).__init__()
        self.encoder = Encoder(Xshape, hidden, dropout)
        self.gin_predict = GINLayer(hidden, hidden, dropout)
        # self.gin_predict = GATLayer(hidden, hidden, dropout)
        self.grl = GradientReversalLayer(lambda_=1)

        self.discriminator = Discriminator(hidden + hidden, hidden, hidden, 1, dropout)
        self.gin_y = GINLayer(hidden + 1, hidden, dropout)
        # self.gin_y = GATLayer(hidden + 1, hidden, dropout)

        self.pred_y = Predictor(hidden + hidden + 1, hidden, hidden, 1, dropout)

    def forward(self, x, t, z, edge_index):
        embed = self.encoder(x)
        # embed = x
        reversed_embed = self.grl(embed)
        t_pred_embed = self.gin_predict(reversed_embed, edge_index)
        t_pred_embed = torch.cat([t_pred_embed, reversed_embed], dim=1)
        t_pred = self.discriminator(t_pred_embed)

        embed_y = self.gin_y(torch.cat([embed, t.unsqueeze(1)], dim=1), edge_index)

        embed_y = torch.cat([embed_y, embed, t.unsqueeze(1)], dim=1)
        y = self.pred_y(embed_y)

        return t_pred, y

class HINet_only_network(nn.Module):
    def __init__(self, Xshape, hidden, dropout=0):
        super(HINet, self).__init__()
        self.encoder = Encoder(Xshape, hidden, dropout)
        self.gin_predict = GINLayer(hidden, hidden, dropout)
        # self.gin_predict = GATLayer(hidden, hidden, dropout)
        self.grl = GradientReversalLayer(lambda_=1)

        self.discriminator = Discriminator(hidden, hidden, hidden, 1, dropout)
        self.gin_y = GINLayer(hidden + 1, hidden, dropout)
        # self.gin_y = GATLayer(hidden + 1, hidden, dropout)

        self.pred_y = Predictor(hidden + hidden + 1, hidden, hidden, 1, dropout)

    def forward(self, x, t, z, edge_index):
        embed = self.encoder(x)
        # embed = x
        reversed_embed = self.grl(embed)
        t_pred_embed = self.gin_predict(reversed_embed, edge_index)
        # t_pred_embed = torch.cat([t_pred_embed, reversed_embed], dim=1)
        t_pred = self.discriminator(t_pred_embed)

        embed_y = self.gin_y(torch.cat([embed, t.unsqueeze(1)], dim=1), edge_index)

        embed_y = torch.cat([embed_y, embed, t.unsqueeze(1)], dim=1)
        y = self.pred_y(embed_y)

        return t_pred, y



class HINet_no_net_conf(nn.Module):
    def __init__(self, Xshape, hidden, dropout=0):
        super(HINet_no_net_conf, self).__init__()
        self.encoder = Encoder(Xshape, hidden, dropout)
        # self.gin_predict = GINLayer(hidden, hidden)
        self.grl = GradientReversalLayer(lambda_=1)
        # self.gin_predict = GINLayer(Xshape, hidden)
        self.discriminator = Discriminator(hidden, hidden, hidden, 1, dropout)
        self.gin_y = GINLayer(hidden + 1, hidden, dropout)
        # self.gin_y = GINLayer(Xshape+1, hidden)
        self.pred_y = Predictor(hidden + hidden + 1, hidden, hidden, 1, dropout)

    def forward(self, x, t, z, edge_index):
        embed = self.encoder(x)
        # embed = x
        reversed_embed = self.grl(embed)
        # t_pred_embed = self.gin_predict(reversed_embed, edge_index)
        t_pred_embed = reversed_embed
        t_pred = self.discriminator(t_pred_embed)

        # Reshape the result to [t.shape[0], t.shape[0]]
        # Continue with the rest of your forward pass
        embed_y = self.gin_y(torch.cat([embed, t.unsqueeze(1)], dim=1), edge_index)

        # embed_y = F.relu(embed_y)
        embed_y = torch.cat([embed_y, embed, t.unsqueeze(1)], dim=1)
        y = self.pred_y(embed_y)

        return t_pred, y


class Encoder(nn.Module):
    def __init__(self, input_size, output_size,dropout=0):
        super(Encoder, self).__init__()

        self.predict1 = nn.Linear(input_size, output_size)  # .cuda()
        self.predict2 = nn.Linear(output_size, output_size)  # .cuda()
        self.dropout = nn.Dropout(dropout)  # .cuda()   
        self.act = nn.LeakyReLU(0.2, inplace=False)  # .cuda()

    def forward(self, x):
        x = self.predict1(x)        
        x = self.act(x)
        x = self.dropout(x)
        x = self.predict2(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class Predictor(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size,dropout=0):
        super(Predictor, self).__init__()

        self.predict1 = nn.Linear(input_size, hidden_size1)  # .cuda()
        self.predict2 = nn.Linear(hidden_size1, hidden_size2)  # .cuda()
        self.predict3 = nn.Linear(hidden_size2, output_size)  # .cuda()
        self.act = nn.LeakyReLU(0.2, inplace=False)  # .cuda()
        self.dropout = nn.Dropout(dropout)  # .cuda()
        self.sigmoid = nn.Sigmoid()  # .cuda()

    def forward(self, x):
        x = self.predict1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.predict2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.predict3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size,dropout=0):
        super(Discriminator, self).__init__()

        self.disc1 = nn.Linear(input_size, hidden_size1)  # .cuda()
        self.disc2 = nn.Linear(hidden_size1, hidden_size2)  # .cuda()
        self.disc3 = nn.Linear(hidden_size2, output_size)  # .cuda()
        self.act = nn.LeakyReLU(0.2, inplace=False)  # .cuda()
        self.dropout = nn.Dropout(dropout)  # .cuda().cuda()
        self.sigmoid = nn.Sigmoid()  # .cuda()

    def forward(self, x):
        x = self.disc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.disc2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.disc3(x)
        x = torch.sigmoid(x)
        return x



"""This code is based on the description from the paper:
Huang, Q., Ma, J., Li, J., Guo, R., Sun, H., & Chang, Y. (2023). 
Modeling interference for individual treatment effect estimation from networked observational data. 
ACM Transactions on Knowledge Discovery from Data, 18(3), 1-21.
"""
class SPNet(nn.Module):
    def __init__(self, Xshape, hidden, dropout=0):
        super(SPNet, self).__init__()
        self.part_outcome = GCNConv(Xshape, hidden, add_self_loops=True,)
        self.part_treat = GCNConv(Xshape, hidden, add_self_loops=True)
        self.outcome_linear = nn.Linear(hidden, hidden)
        self.treat_linear = nn.Linear(hidden, hidden)
        self.attention = Attention_layer.MaskedAttentionLayer(2 * hidden, 1)
        self.discriminator_t = Discriminator(hidden, hidden, hidden, 1, dropout)
        self.encoder_final = nn.Linear(hidden + hidden, hidden)
        self.predict_1 = Predictor(hidden, hidden, hidden, 1, dropout)
        self.predict_0 = Predictor(hidden, hidden, hidden, 1, dropout)

    def forward(self, x, t, z, edge_index):
        # in principle self loops here?
        r_o = self.part_outcome(x, edge_index)
        r_t = self.part_treat(x, edge_index)
        r_t = F.relu(r_t)
        r_o = F.relu(r_o)
        cat = torch.cat([r_o, r_t], dim=-1)
        # calculate attention weights per edge
        h = self.attention(cat, r_t, t.unsqueeze(1), edge_index)
        z = self.encoder_final(torch.cat([r_o, h], dim=-1))
        representations = z
        pred_1 = self.predict_1(z)
        pred_0 = self.predict_0(z)
        pred = torch.where(t > 0, pred_1.squeeze(), pred_0.squeeze())
        pred_t = self.discriminator_t(r_t)
        return pred_t, pred.unsqueeze(1), representations


class GCN_DECONF(nn.Module):
    def __init__(self, Xshape, hidden, dropout=0, n_in=1, n_out=2, cuda=True):
        super(GCN_DECONF, self).__init__()

        # if cuda:
        #     self.gc = nn.ModuleList([GCNLayer(nfeat, nhid)]).cuda()
        #     for i in range(n_in - 1):
        #         self.gc.append(GCNLayer(nhid, nhid).cuda())
        # else:
        self.gc = nn.ModuleList([GCNLayer(Xshape, hidden, dropout)])
        for i in range(n_in - 1):
            self.gc.append(GCNLayer(hidden, hidden, dropout))

        self.n_in = n_in
        self.n_out = n_out

        # if cuda:

        #     self.out_t00 = nn.ModuleList([nn.Linear(nhid,nhid).cuda() for i in range(n_out)])
        #     self.out_t10 = nn.ModuleList([nn.Linear(nhid,nhid).cuda() for i in range(n_out)])
        #     self.out_t01 = nn.Linear(nhid,1).cuda()
        #     self.out_t11 = nn.Linear(nhid,1).cuda()

        # else:
        self.out_t00 = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(n_out)])
        self.out_t10 = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(n_out)])
        self.out_t01 = nn.Linear(hidden, 1)
        self.out_t11 = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)
        # self.dropout = dropout

        # a linear layer for propensity prediction
        self.pp = nn.Linear(hidden, 1)

        # if cuda:
        #     self.pp = self.pp.cuda()
        self.pp_act = nn.Sigmoid()

    def forward(self, x, t, z, edge_index):
        # if Z is None:
        #     neighbors = torch.sum(adj, 1)
        #     neighborAverageT = torch.div(torch.matmul(adj, t.reshape(-1)), neighbors)
        # else:
        #     neighborAverageT = Z

        rep = F.relu(self.gc[0](x, edge_index))
        # rep = F.dropout(rep, self.dropout, training=self.training)
        for i in range(1, self.n_in):
            rep = F.relu(self.gc[i](rep, edge_index))
            rep = self.dropout(rep)
            # rep = F.dropout(rep, self.dropout, training=self.training)

        for i in range(self.n_out):
            y00 = F.relu(self.out_t00[i](rep))
            y00 = self.dropout(y00)
            # y00 = F.dropout(y00, self.dropout, training=self.training)
            y10 = F.relu(self.out_t10[i](rep))
            y10 = self.dropout(y10)
            # y10 = F.dropout(y10, self.dropout, training=self.training)

        y0 = self.out_t01(y00).view(-1)
        y1 = self.out_t11(y10).view(-1)

        y = torch.where(t > 0, y1, y0)

        p1 = self.pp_act(self.pp(rep)).view(-1)

        return p1.unsqueeze(1), y.unsqueeze(1)


class TARNet(nn.Module):
    def __init__(self, Xshape, hidden, dropout=0, n_in=2, n_out=2):
        super(TARNet, self).__init__()

        # if cuda:
        #     self.gc = nn.ModuleList([nn.Linear(nfeat, nhid)]).cuda()
        #     for i in range(n_in - 1):
        #         self.gc.append(nn.Linear(nhid,nhid).cuda())

        self.gc = nn.ModuleList([nn.Linear(Xshape, hidden)])
        for i in range(n_in - 1):
            self.gc.append(nn.Linear(hidden, hidden))

        self.n_in = n_in
        self.n_out = n_out

        # if cuda:

        #     self.out_t00 = nn.ModuleList([nn.Linear(nhid,nhid).cuda() for i in range(n_out)])
        #     self.out_t10 = nn.ModuleList([nn.Linear(nhid,nhid).cuda() for i in range(n_out)])
        #     self.out_t01 = nn.Linear(nhid,1).cuda()
        #     self.out_t11 = nn.Linear(nhid,1).cuda()

        # else:
        self.out_t00 = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(n_out)])
        self.out_t10 = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(n_out)])
        self.out_t01 = nn.Linear(hidden, 1)
        self.out_t11 = nn.Linear(hidden, 1)

        # self.dropout = dropout
        self.dropout = nn.Dropout(dropout)
        # a linear layer for propensity prediction
        self.pp = nn.Linear(hidden, 1)

        # if cuda:
        #     self.pp = self.pp.cuda()
        self.pp_act = nn.Sigmoid()

    def forward(self, x, t, z, edge_index):
        # if Z is None:
        #     neighbors = torch.sum(adj, 1)
        #     neighborAverageT = torch.div(torch.matmul(adj, t.reshape(-1)), neighbors)
        # else:
        #     neighborAverageT = Z

        rep = F.relu(self.gc[0](x))
        rep = self.dropout(rep)
        for i in range(1, self.n_in):
            rep = F.relu(self.gc[i](rep))
            rep = self.dropout(rep)

        for i in range(self.n_out):
            y00 = F.relu(self.out_t00[i](rep))

            y00 = self.dropout(y00)
            y10 = F.relu(self.out_t10[i](rep))
            y10 = self.dropout(y10)
        # print("y00",y00.shape)
        # print("y10",y10.shape)
        y0 = self.out_t01(y00).view(-1)
        y1 = self.out_t11(y10).view(-1)
        # print("t",t.shape)
        # print("y0",y0.shape)
        # print("y1",y1.shape)
        y = torch.where(t > 0, y1, y0)
        # print("y",y.shape)
        y = self.pp_act(y).view(-1, 1)
        p1 = self.pp_act(self.pp(rep)).view(-1, 1)
        # print("p1",p1.shape)
        # print("y",y.shape)
        # stop
        return p1, y

"""Code from Doubly Robust Causal Effect Estimation under Networked Interference via Targeted Learning (Chen et al. 2024):
https://github.com/WeilinChen507/targeted_interference"""

ini_normal_variance=0.1
class Truncated_power():
    def __init__(self, degree, knots):
        """
        This class construct the truncated power basis; the data is assumed in [0,1]
        :param degree: int, the degree of truncated basis
        :param knots: list, the knots of the spline basis; two end points (0,1) should not be included
        """
        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)

        if self.degree == 0:
            print('Degree should not set to be 0!')
            raise ValueError

        if not isinstance(self.degree, int):
            print('Degree should be int')
            raise ValueError

    def forward(self, x):
        """
        :param x: torch.tensor, batch_size * 1
        :return: the value of each basis given x; batch_size * self.num_of_basis
        """
        x = x.squeeze()
        out = torch.zeros(x.shape[0], self.num_of_basis).cuda()
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

        return out # bs, num_of_basis


class TR(nn.Module):
    def __init__(self, degree, knots):
        super(TR, self).__init__()
        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis # num of basis
        self.weight = torch.nn.Parameter(torch.rand(self.d, device='cuda'), requires_grad=True)

    def forward(self, t):
        out = self.spb.forward(t)
        out = torch.matmul(out, self.weight)
        return out

    def _initialize_weights(self):
        # self.weight.data.normal_(0, 0.1)
        self.weight.data.zero_()


class TargetedModel_DoubleBSpline(nn.Module):

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
            # self.encoder._initialize_weights()
            # self.X_XN._initialize_weights()
            # self.Q1._initialize_weights()
            # self.Q0._initialize_weights()
            self.g_Z._initialize_weights()
            self.g_T._initialize_weights()
            self.tr_reg_t1._initialize_weights()
            self.tr_reg_t0._initialize_weights()

    def parameter_base(self):
        return list(self.encoder.parameters()) +\
            list(self.X_XN.parameters()) +\
            list(self.Q1.parameters())+list(self.Q0.parameters())+\
            list(self.g_T.parameters())+\
            list(self.g_Z.parameters())

    def parameter_trageted(self):
        return list(self.tr_reg_t0.parameters()) + list(self.tr_reg_t1.parameters())

    def tr_reg(self, T, neighborAverageT):
        tr_reg_t1 = self.tr_reg_t1(neighborAverageT)
        tr_reg_t0 = self.tr_reg_t0(neighborAverageT)
        regur = torch.where(T==1, tr_reg_t1, tr_reg_t0)
        return regur



    def forward(self, x, t, z, edge_index):
        embeddings = self.encoder(x, edge_index)  # X_i,X_N
        embeddings = self.X_XN(torch.cat((embeddings,x), dim=1))

        g_T_hat = self.g_T(embeddings)  # X_i,X_N -> T_i

        neighborAverageT = z


        g_Z_hat = self.g_Z(embeddings, neighborAverageT)  # X_i,X_N -> Z
        g_Z_hat = g_Z_hat.unsqueeze(1)


        embed_avgT = torch.cat((embeddings, neighborAverageT.reshape(-1, 1)), 1)

        Q_hat = t.reshape(-1, 1) * self.Q1(embed_avgT) + (1-t.reshape(-1, 1)) * self.Q0(embed_avgT)

        epsilon = self.tr_reg(t, neighborAverageT)  # epsilon(T,Z)


        return g_T_hat, g_Z_hat, Q_hat, epsilon, embeddings, neighborAverageT

    def infer_potential_outcome(self, x, t, z, edge_index):
        embeddings = self.encoder(x, edge_index)  # X_i,X_N
        embeddings = self.X_XN(torch.cat((embeddings,x), dim=1))
        g_T_hat = self.g_T(embeddings)  # X_i,X_N -> T_i
        g_T_hat = g_T_hat.squeeze(1)

        # if Z is None:
        #     neighbors = torch.sum(A, 1)
        #     neighborAverageT = torch.div(torch.matmul(A, T.reshape(-1)), neighbors)  # treated_neighbors / all_neighbors
        # else:
        #     neighborAverageT = Z
        neighborAverageT = z

        g_Z_hat = self.g_Z(embeddings, neighborAverageT)  # X_i,X_N -> Z



        embed_avgT = torch.cat((embeddings, neighborAverageT.reshape(-1, 1)), 1)

        Q_hat = t.reshape(-1, 1) * self.Q1(embed_avgT) + (1-t.reshape(-1, 1)) * self.Q0(embed_avgT)

        epsilon = self.tr_reg(t, neighborAverageT)  # epsilon(T,Z)
        # epsilon = epsilon.squeeze(1)


        return Q_hat.reshape(-1) + (epsilon.reshape(-1) * 1/(g_Z_hat.reshape(-1)*g_T_hat.reshape(-1) + 1e-6))

class Discriminator_simplified(nn.Module):
    def __init__(self,input_size,hidden_size1,output_size):
        super(Discriminator_simplified,self).__init__()

        self.disc1 = nn.Linear(input_size,hidden_size1, device='cuda')
        self.disc3 = nn.Linear(hidden_size1,output_size, device='cuda')
        self.act = nn.LeakyReLU(0.2, inplace=True).cuda()

    def forward(self,x):
        x = self.disc1(x)
        x = self.act(x)
        x = self.disc3(x)
        x = torch.sigmoid(x)
        return x

    def _initialize_weights(self):
        self.disc1.weight.data.normal_(0, ini_normal_variance)
        self.disc3.weight.data.normal_(0, ini_normal_variance)

def comp_grid(y, num_grid):

    # L gives the lower index
    # U gives the upper index
    # inter gives the distance to the lower int

    U = torch.ceil(y * num_grid)
    inter = 1 - (U - y * num_grid)
    L = U - 1
    L += (L < 0).int()

    return L.int().tolist(), U.int().tolist(), inter


class Density_Block(nn.Module):
    def __init__(self, num_grid, ind, isbias=1):
        super(Density_Block, self).__init__()
        """
        Assume the variable is bounded by [0,1]
        the output grid: 0, 1/B, 2/B, ..., B/B; output dim = B + 1; num_grid = B
        """
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

        x1 = list(torch.arange(0, x.shape[0]))
        L, U, inter = comp_grid(z, self.num_grid)

        L_out = out[x1, L]
        U_out = out[x1, U]

        out = L_out + (U_out - L_out) * inter

        return out

    def _initialize_weights(self):
        self.weight.data.normal_(0, ini_normal_variance)
        if self.isbias:
            self.bias.data.normal_(0, ini_normal_variance)
        # self.disc3.weight.data.zero_()
        return


class Density_Estimator(nn.Module):

    def __init__(self, input_size, num_grid):
        super().__init__()
        # input_size is the size of embeddings of X_i,X_N
        self.num_grid=num_grid
        self.density_estimator_head = Density_Block(self.num_grid, input_size, isbias=1)

    def forward(self, x, z):
        g_Z = self.density_estimator_head(z, x)
        return g_Z

    def _initialize_weights(self):
        self.density_estimator_head._initialize_weights()


