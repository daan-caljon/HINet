from math import e
from numpy import r_
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool,GCNConv, GATConv
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data import Data

from src.methods.utils import GradientReversalLayer
import src.methods.Attention_layer as Attention_layer
class GCNLayer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
    
    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class GINLayer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GINLayer, self).__init__()
        # Define the MLP for GIN
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.Tanh(),
            # torch.nn.Linear(hidden_channels, hidden_channels)

        )
        # GIN layer
        self.gin = GINConv(self.mlp)
    
    def forward(self, x, edge_index):
        return self.gin(x, edge_index)

class GINModel(torch.nn.Module):
    def __init__(self, Xshape, hidden):
        super(GINModel, self).__init__()
        self.conv1 = GINLayer(Xshape+1, hidden)
        self.fc = Predictor(hidden+Xshape+1, hidden, hidden, 1)
    def forward(self, x,t, z,edge_index):
        x = torch.cat([x, t.unsqueeze(1)], dim=-1)
        x_gin = self.conv1(x, edge_index)
        x_gin = F.relu(x_gin)

        x_comb = torch.cat([x_gin, x], dim=-1)
        x = self.fc(x_comb)
        t_pred = torch.zeros(x.shape[0],1).cuda()
        
        return t_pred, x
    
class NetEst(nn.Module):
    def __init__(self, Xshape,hidden):
        super(NetEst, self).__init__()
        #self.encodergc = GINLayer(Xshape, hidden)
        self.encodergc = GCNLayer(Xshape, hidden)
        # self.encoder = nn.Linear(hidden+Xshape,hidden)
        self.encoder = Encoder(hidden+Xshape,hidden)
        self.predictor = Predictor(hidden+2,hidden,hidden,1)
        self.discriminator = Discriminator(hidden,hidden,hidden,1)
        self.discrimnator_z = Discriminator(hidden+1,hidden,hidden,1)
    
    def forward(self, x,t,z, edge_index):
        xgc = self.encodergc(x, edge_index)
        xgc = F.relu(xgc)
        xgc = self.encoder(torch.cat([xgc,x],dim=-1))
        #average of neighbor t
        pred_t = self.discriminator(xgc)
        pred_z = self.discrimnator_z(torch.cat([xgc, t.unsqueeze(1)], dim=-1))
        x = torch.cat([xgc, z.unsqueeze(1), t.unsqueeze(1)], dim=-1)
        y = self.predictor(x)
        
        

        return pred_t, y, pred_z
    
class GINNetEst(nn.Module):
    def __init__(self, Xshape,hidden):
        super(GINNetEst, self).__init__()
        self.encodergc = GINLayer(Xshape, hidden)
        # self.encodergc = GCNLayer(Xshape, hidden)
        # self.encoder = nn.Linear(hidden+Xshape,hidden)
        self.encoder = Encoder(hidden+Xshape,hidden)
        self.predictor = Predictor(hidden+2,hidden,hidden,1)
        self.discriminator = Discriminator(hidden,hidden,hidden,1)
        self.discrimnator_z = Discriminator(hidden+1,hidden,hidden,1)
    
    def forward(self, x,t,z, edge_index):
        xgc = self.encodergc(x, edge_index)
        xgc = F.relu(xgc)
        xgc = self.encoder(torch.cat([xgc,x],dim=-1))
        #average of neighbor t
        pred_t = self.discriminator(xgc)
        pred_z = self.discrimnator_z(torch.cat([xgc, t.unsqueeze(1)], dim=-1))
        x = torch.cat([xgc, z.unsqueeze(1), t.unsqueeze(1)], dim=-1)
        y = self.predictor(x)
        return pred_t, y, pred_z

class HINet(nn.Module):
    def __init__(self, Xshape,hidden):
        super(HINet, self).__init__()
        self.encoder = Encoder(Xshape,hidden)
        self.gin_predict = GINLayer(hidden, hidden)
        self.grl = GradientReversalLayer(lambda_=1)

        self.discriminator = Discriminator(hidden+hidden,hidden,hidden,1)
        self.gin_y = GINLayer(hidden+1, hidden)

        self.pred_y = Predictor(hidden+hidden+1,hidden,hidden,1)


    
    def forward(self, x, t, z, edge_index):
        embed = self.encoder(x)
        # embed = x
        reversed_embed = self.grl(embed)
        t_pred_embed = self.gin_predict(reversed_embed, edge_index)
        t_pred_embed = torch.cat([t_pred_embed, reversed_embed], dim=1)
        t_pred = self.discriminator(t_pred_embed)


        embed_y = self.gin_y(torch.cat([embed, t.unsqueeze(1)], dim=1), edge_index)


        embed_y = torch.cat([embed_y,embed,t.unsqueeze(1)], dim=1)
        y = self.pred_y(embed_y)

        
        return t_pred, y


class HINet_no_net_conf(nn.Module):
    def __init__(self, Xshape,hidden):
        super(HINet_no_net_conf, self).__init__()
        self.encoder = Encoder(Xshape,hidden)
        # self.gin_predict = GINLayer(hidden, hidden)
        self.grl = GradientReversalLayer(lambda_=1)
        # self.gin_predict = GINLayer(Xshape, hidden)
        self.discriminator = Discriminator(hidden,hidden,hidden,1)
        self.gin_y = GINLayer(hidden+1, hidden)
        # self.gin_y = GINLayer(Xshape+1, hidden)
        self.pred_y = Predictor(hidden+hidden+1,hidden,hidden,1)


    
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
        embed_y = torch.cat([embed_y,embed,t.unsqueeze(1)], dim=1)
        y = self.pred_y(embed_y)

        
        return t_pred, y


class Encoder(nn.Module):
    def __init__(self,input_size,output_size):
        super(Encoder, self).__init__()

        self.predict1 = nn.Linear(input_size,output_size)#.cuda()
        self.predict2 = nn.Linear(output_size,output_size)#.cuda()

        self.act = nn.LeakyReLU(0.2, inplace=False)#.cuda()
    def forward(self,x):
        x = self.predict1(x)
        x = self.act(x)
        x = self.predict2(x)
        x = self.act(x)
        return  x

class Predictor(nn.Module):

    def __init__(self,input_size,hidden_size1,hidden_size2,output_size):
        super(Predictor, self).__init__()

        self.predict1 = nn.Linear(input_size,hidden_size1)#.cuda()
        self.predict2 = nn.Linear(hidden_size1,hidden_size2)#.cuda()
        self.predict3 = nn.Linear(hidden_size2,output_size)#.cuda()
        self.act = nn.LeakyReLU(0.2, inplace=False)#.cuda()
        self.dropout = nn.Dropout(0.1)#.cuda()
        self.sigmoid = nn.Sigmoid()#.cuda()

    def forward(self,x):
        x = self.predict1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.predict2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.predict3(x)
        return  x
    
class Discriminator(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,output_size):
        super(Discriminator,self).__init__()

        self.disc1 = nn.Linear(input_size,hidden_size1)#.cuda()
        self.disc2 = nn.Linear(hidden_size1,hidden_size2)#.cuda()
        self.disc3 = nn.Linear(hidden_size2,output_size)#.cuda()
        self.act = nn.LeakyReLU(0.2, inplace=False)#.cuda()
        self.dropout = nn.Dropout(0.1)#.cuda().cuda()
        self.sigmoid = nn.Sigmoid()#.cuda()

    def forward(self,x):
        x = self.disc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.disc2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.disc3(x)
        x = torch.sigmoid(x)
        return x
    
import src.methods.Attention_layer as Attention_layer
class SPNet(nn.Module):
    def __init__(self, Xshape,hidden):
        super(SPNet, self).__init__()
        self.part_outcome = GCNConv(Xshape, hidden,add_self_loops=True)
        self.part_treat =GCNConv(Xshape, hidden,add_self_loops=True)
        self.outcome_linear = nn.Linear(hidden,hidden)
        self.treat_linear = nn.Linear(hidden,hidden)
        self.attention = Attention_layer.MaskedAttentionLayer(2*hidden,1)
        self.discriminator_t = Discriminator(hidden,hidden,hidden,1)
        self.encoder_final = nn.Linear(hidden+hidden,hidden)
        self.predict_1 = Predictor(hidden,hidden,hidden,1)
        self.predict_0 = Predictor(hidden,hidden,hidden,1)

    
    def forward(self, x,t,z, edge_index):
        #in principle self loops here?
        r_o= self.part_outcome(x, edge_index)
        r_t = self.part_treat(x, edge_index)
        r_t = F.relu(r_t)
        r_o = F.relu(r_o)
        cat = torch.cat([r_o, r_t], dim=-1)
        #calculate attention weights per edge
        h = self.attention(cat,r_t,t.unsqueeze(1), edge_index)
        z = self.encoder_final(torch.cat([r_o, h], dim=-1))
        representations=z
        pred_1 = self.predict_1(z)
        pred_0 = self.predict_0(z)
        pred = torch.where(t > 0, pred_1.squeeze(), pred_0.squeeze())
        pred_t = self.discriminator_t(r_t)
        return  pred_t, pred.unsqueeze(1),representations

class GCN_DECONF(nn.Module):
    def __init__(self, Xshape,hidden, n_in=1, n_out=2, cuda=True):
        super(GCN_DECONF, self).__init__()

        # if cuda:
        #     self.gc = nn.ModuleList([GCNLayer(nfeat, nhid)]).cuda()
        #     for i in range(n_in - 1):
        #         self.gc.append(GCNLayer(nhid, nhid).cuda())
        # else:
        self.gc = nn.ModuleList([GCNLayer(Xshape, hidden)])
        for i in range(n_in - 1):
            self.gc.append(GCNLayer(hidden, hidden))
        
        self.n_in = n_in
        self.n_out = n_out

        # if cuda:

        #     self.out_t00 = nn.ModuleList([nn.Linear(nhid,nhid).cuda() for i in range(n_out)])
        #     self.out_t10 = nn.ModuleList([nn.Linear(nhid,nhid).cuda() for i in range(n_out)])
        #     self.out_t01 = nn.Linear(nhid,1).cuda()
        #     self.out_t11 = nn.Linear(nhid,1).cuda()

        # else:
        self.out_t00 = nn.ModuleList([nn.Linear(hidden,hidden) for i in range(n_out)])
        self.out_t10 = nn.ModuleList([nn.Linear(hidden,hidden) for i in range(n_out)])
        self.out_t01 = nn.Linear(hidden,1)
        self.out_t11 = nn.Linear(hidden,1)

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
            # rep = F.dropout(rep, self.dropout, training=self.training)

        for i in range(self.n_out):

            y00 = F.relu(self.out_t00[i](rep))
            # y00 = F.dropout(y00, self.dropout, training=self.training)
            y10 = F.relu(self.out_t10[i](rep))
            # y10 = F.dropout(y10, self.dropout, training=self.training)
        
        y0 = self.out_t01(y00).view(-1)
        y1 = self.out_t11(y10).view(-1)

        y = torch.where(t > 0,y1,y0)

        p1 = self.pp_act(self.pp(rep)).view(-1)

        return p1.unsqueeze(1), y.unsqueeze(1)


class TARNet(nn.Module):
    def __init__(self, Xshape,hidden,n_in=2,n_out=2):
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
        self.out_t00 = nn.ModuleList([nn.Linear(hidden,hidden) for i in range(n_out)])
        self.out_t10 = nn.ModuleList([nn.Linear(hidden,hidden) for i in range(n_out)])
        self.out_t01 = nn.Linear(hidden,1)
        self.out_t11 = nn.Linear(hidden,1)

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

        rep = F.relu(self.gc[0](x))
        # rep = F.dropout(rep, self.dropout, training=self.training)
        for i in range(1, self.n_in):
            rep = F.relu(self.gc[i](rep))
            # rep = F.dropout(rep, self.dropout, training=self.training)

        for i in range(self.n_out):

            y00 = F.relu(self.out_t00[i](rep))
            # y00 = F.dropout(y00, self.dropout, training=self.training)
            y10 = F.relu(self.out_t10[i](rep))
            # y10 = F.dropout(y10, self.dropout, training=self.training)
        # print("y00",y00.shape)
        # print("y10",y10.shape)
        y0 = self.out_t01(y00).view(-1)
        y1 = self.out_t11(y10).view(-1)
        # print("t",t.shape)
        # print("y0",y0.shape)
        # print("y1",y1.shape)
        y = torch.where(t > 0,y1,y0)
        # print("y",y.shape)
        y = self.pp_act(y).view(-1,1)
        p1 = self.pp_act(self.pp(rep)).view(-1,1)
        # print("p1",p1.shape)
        # print("y",y.shape)
        # stop
        return p1, y
    