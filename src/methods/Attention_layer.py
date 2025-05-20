from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import torch
import torch.nn.functional as F

class MaskedAttentionLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MaskedAttentionLayer, self).__init__(aggr='add')  # 'add' aggregation (or sum)
        self.fc = torch.nn.Linear(2 * in_channels, 1)  # Linear transformation
        # self.att = torch.nn.Parameter(torch.Tensor(out_channels, 1))  # Attention weight vector
        # torch.nn.init.xavier_uniform_(self.att)  # Initialize attention weights

    def forward(self, x_cat, x,t,edge_index):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix (edge list) of shape [2, num_edges]

        # Step 1: Start propagating messages
        return self.propagate(edge_index, x_cat = x_cat,x=x,t=t)

    def message(self, x_cat_i, x_cat_j,x_i,x_j, t_i,t_j,edge_index, size_i):
        # x_i, x_j: Node features for source and target nodes in each edge, [num_edges, in_channels]

        # Concatenate features of source and target nodes
        r_ij = torch.cat([x_cat_i, x_cat_j], dim=-1)  # [num_edges, 2 * in_channels]
        # Step 2: Apply linear transformation and ReLU activation
        alpha = F.relu(self.fc(r_ij))  # [num_edges, out_channels]

        # Step 3: Compute attention weights
        alpha = softmax(alpha, edge_index[0], num_nodes=size_i)  # Normalize attention coefficients

        #take the first half of x_j ()
        # print("alpha shape:{}".format(alpha.shape))
        # print("x_j shape:{}".format(x_j.shape))
        
        output = x_j * alpha #*t_j
        return output  # Apply attention weights

    def update(self, aggr_out):
        # Step 4: Return the aggregated message (interference representation)

        return aggr_out
