"""IDE-Net model adapted to PyG with original INE-style functionality.

Adapted from the IDE-Net implementation of Adhikari & Zheleva (2025), which is
distributed under the GNU General Public License v3.0 (see LICENSE).

Implements exposure modes from the original code:
    - exposure_type=1: z_IDENet = average neighbor treatment
    - exposure_type=2: VanillaHeterogeneousExposure mapping
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree, to_scipy_sparse_matrix


class MLPLayer(nn.Module):
    """Lightweight residual MLP block used in exposure mapping."""

    def __init__(self, indim, outdim, num_layers=1, dropout=0.0):
        super().__init__()
        self.num_layers = max(1, num_layers)
        self.dropout = dropout
        self.emb = nn.Linear(indim, outdim)
        self.mlp = nn.ModuleList()
        for _ in range(self.num_layers - 1):
            self.mlp.append(nn.Linear(outdim, outdim))

    def forward(self, x):
        emb = self.emb(x)
        out = emb
        for i, layer in enumerate(self.mlp):
            out = F.dropout(out, self.dropout, self.training)
            out = torch.tanh(out)
            out = layer(out)
            if i == len(self.mlp) - 1:
                out = emb + out
        out = F.dropout(out, self.dropout, self.training)
        return out


class CANEFeatureEmbedding(nn.Module):
    """CANE-style feature embedding used by original INE models."""

    def __init__(
        self,
        node_dim,
        node_hidden,
        edge_dim,
        edge_hidden=4,
        num_layers=2,
        dropout=0.0,
    ):
        super().__init__()
        ego_hidden = node_hidden // 2
        peer_hidden = node_hidden - ego_hidden

        self.edge_dim = edge_dim
        self.edge_hidden = edge_hidden
        self.dropout = dropout

        self.peer_embs = MLPLayer(
            node_dim + edge_dim,
            peer_hidden + edge_hidden,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.ego_emb = MLPLayer(node_dim, ego_hidden, num_layers=num_layers, dropout=dropout)
        self.edge_emb = MLPLayer(edge_dim, edge_hidden, num_layers=num_layers, dropout=dropout)

        # Matches original CANE output dimensionality.
        self.hdim = node_hidden + 3 * edge_hidden

    def _prepare_edge_attr(self, edge_index, edge_attr_IDE, dtype, device):
        num_edges = edge_index.shape[1]
        if edge_attr_IDE is None:
            edge_attr_IDE = torch.ones((num_edges, 1), dtype=dtype, device=device)

        edge_attr_IDE = edge_attr_IDE.to(device=device, dtype=dtype)
        if edge_attr_IDE.dim() == 1:
            edge_attr_IDE = edge_attr_IDE.unsqueeze(1)

        cur_dim = edge_attr_IDE.shape[1]
        if cur_dim == self.edge_dim:
            return edge_attr_IDE
        if cur_dim > self.edge_dim:
            return edge_attr_IDE[:, : self.edge_dim]

        pad = torch.zeros((num_edges, self.edge_dim - cur_dim), dtype=dtype, device=device)
        return torch.cat([edge_attr_IDE, pad], dim=1)

    def forward(self, x, edge_index, edge_attr_IDE=None):
        row, col = edge_index
        num_nodes = x.shape[0]
        dtype = x.dtype
        device = x.device

        edge_attr = self._prepare_edge_attr(edge_index, edge_attr_IDE, dtype=dtype, device=device)

        # Row-normalized adjacency weights (same spirit as original column=False normalization).
        deg = torch.bincount(row, minlength=num_nodes).to(device=device, dtype=dtype)
        norm = torch.pow(deg + 1e-8, -0.5)

        h_ego = F.relu(self.ego_emb(x))

        peer_features = torch.cat([x[col], edge_attr], dim=1)
        peer_msg = self.peer_embs(peer_features) * norm[row].unsqueeze(1)
        h_peer = torch.zeros((num_nodes, peer_msg.shape[1]), dtype=dtype, device=device)
        h_peer.index_add_(0, row, peer_msg)
        h_peer = F.relu(h_peer)

        edge_msg = F.relu(self.edge_emb(edge_attr))
        h_edge = torch.zeros((num_nodes, edge_msg.shape[1]), dtype=dtype, device=device)
        h_edge.index_add_(0, row, edge_msg)

        h_edge_prop = torch.zeros_like(h_edge)
        h_edge_prop.index_add_(0, row, h_edge[col] * norm[row].unsqueeze(1))
        h_edge = torch.cat([h_edge, h_edge_prop], dim=1)

        out = torch.cat([h_ego, h_edge, h_peer], dim=1)
        return out, None


class VanillaGCNFeatureEmbedding(nn.Module):
    """Vanilla GCN feature embedding branch."""

    def __init__(self, num_features, num_hidden, num_layers=2, dropout=0.0):
        super().__init__()
        self.hdim = num_hidden
        self.dropout = dropout

        self.gc = nn.ModuleList([GCNConv(num_features, num_hidden)])
        for _ in range(max(1, num_layers) - 1):
            self.gc.append(GCNConv(num_hidden, num_hidden))

    def forward(self, x, edge_index, edge_attr_IDE=None):
        del edge_attr_IDE
        h = F.relu(self.gc[0](x, edge_index))
        h = F.dropout(h, self.dropout, self.training)
        for layer in self.gc[1:]:
            h = F.relu(layer(h, edge_index))
            h = F.dropout(h, self.dropout, self.training)
        return h, None


class VanillaHeterogeneousExposure(nn.Module):
    """Exposure mapping reproduced from the original IDE-Net implementation.

    This maps node embeddings, treatment, and edge attributes to a
    heterogeneous exposure vector per node.
    """

    def __init__(self, node_hidden, edge_dim=2, dropout=0.0, num_layers=1):
        super().__init__()
        self.edge_dim = edge_dim
        self.hdim = 2 * node_hidden + (edge_dim + 1)
        self.edge_emb = MLPLayer(edge_dim + 1, edge_dim + 1, num_layers=num_layers, dropout=dropout)
        self._cached_aa_edge = None
        self._cached_aa_key = None

    def _prepare_edge_attr(self, edge_index, edge_attr_IDE, dtype, device):
        num_edges = edge_index.shape[1]
        if edge_attr_IDE is None:
            edge_attr_IDE = torch.ones((num_edges, 1), dtype=dtype, device=device)

        edge_attr_IDE = edge_attr_IDE.to(device=device, dtype=dtype)
        if edge_attr_IDE.dim() == 1:
            edge_attr_IDE = edge_attr_IDE.unsqueeze(1)

        cur_dim = edge_attr_IDE.shape[1]
        if cur_dim == self.edge_dim:
            return edge_attr_IDE
        if cur_dim > self.edge_dim:
            return edge_attr_IDE[:, : self.edge_dim]

        pad = torch.zeros((num_edges, self.edge_dim - cur_dim), dtype=dtype, device=device)
        return torch.cat([edge_attr_IDE, pad], dim=1)

    def _get_aa_edge(self, edge_index, num_nodes, dtype, device):
        """Compute AA feature on observed edges only and cache it.

        AA mirrors the original:
            AA = (A @ A^T) * A
        then we keep AA[row, col] for each observed edge (row, col).
        """
        key = (num_nodes, int(edge_index.shape[1]))
        if self._cached_aa_edge is None or self._cached_aa_key != key:
            edge_index_cpu = edge_index.detach().cpu()
            # Use explicit writable NumPy buffers for SciPy advanced indexing.
            row_cpu = np.asarray(edge_index_cpu[0].numpy(), dtype=np.int64).copy()
            col_cpu = np.asarray(edge_index_cpu[1].numpy(), dtype=np.int64).copy()
            A_sp = to_scipy_sparse_matrix(edge_index_cpu, num_nodes=num_nodes).tocsr()
            AA_sp = A_sp @ A_sp.T
            aa_np = np.asarray(AA_sp[row_cpu, col_cpu]).reshape(-1).astype(np.float32)
            self._cached_aa_edge = torch.from_numpy(aa_np)
            self._cached_aa_key = key
        return self._cached_aa_edge.to(device=device, dtype=dtype)

    def forward(self, node_embeddings, treatment, edge_index, edge_attr_IDE=None):
        row, col = edge_index
        num_nodes = node_embeddings.shape[0]
        dtype = node_embeddings.dtype
        device = node_embeddings.device

        treatment = treatment.to(device=device, dtype=dtype).view(-1, 1)
        edge_attr = self._prepare_edge_attr(edge_index, edge_attr_IDE, dtype=dtype, device=device)
        aa_edge = self._get_aa_edge(edge_index, num_nodes, dtype=dtype, device=device).view(-1, 1)

        edge_plus_aa = torch.cat([edge_attr, aa_edge], dim=1)
        edge_plus_aa = edge_plus_aa + F.relu(self.edge_emb(edge_plus_aa))

        x_i, x_j = node_embeddings[row], node_embeddings[col]
        sim = torch.exp(-((x_j - x_i) ** 2))
        edge_features = torch.cat([edge_plus_aa, x_j, sim], dim=1)

        weighted_num = edge_features * treatment[col]
        num = torch.zeros((num_nodes, edge_features.shape[1]), dtype=dtype, device=device)
        den = torch.zeros_like(num)
        num.index_add_(0, row, weighted_num)
        den.index_add_(0, row, edge_features)
        z_IDENet = num / (den + 1e-8)
        return z_IDENet


class IDENet(nn.Module):
    """IDE-Net with GCN feature embedding and T-learner.

    Architecture:
        1. GCN encoder: x -> h (node embeddings via graph convolution)
          2. Representation:
              - exposure_type=1: cat(h, z_IDENet_scalar)
              - exposure_type=2: cat(h, z_IDENet_vector)
        3. T-learner: separate MLP heads for y0 (control) and y1 (treated)
        4. Propensity head: predicts treatment from embeddings (for logging only)

    Forward returns (t_pred, y_pred) matching the existing interface,
    where y_pred = y1 if t>0 else y0.
    """

    def __init__(
        self,
        Xshape,
        hidden,
        dropout=0,
        exposure_type=2,
        edge_dim=1,
        exposure_layers=2,
        vanilla=False,
        edge_hidden=4,
        feature_layers=2,
    ):
        super(IDENet, self).__init__()
        self.exposure_type = exposure_type
        self.edge_dim = edge_dim
        self.vanilla = vanilla
        self.dropout = nn.Dropout(dropout)

        if self.vanilla:
            self.featuremap = VanillaGCNFeatureEmbedding(
                Xshape,
                hidden,
                num_layers=feature_layers,
                dropout=dropout,
            )
        else:
            self.featuremap = CANEFeatureEmbedding(
                node_dim=Xshape,
                node_hidden=hidden,
                edge_dim=edge_dim,
                edge_hidden=edge_hidden,
                num_layers=feature_layers,
                dropout=dropout,
            )

        feature_dim = self.featuremap.hdim

        if self.exposure_type == 2:
            self.exposuremap = VanillaHeterogeneousExposure(
                node_hidden=feature_dim,
                edge_dim=edge_dim,
                dropout=dropout,
                num_layers=exposure_layers,
            )
            rep_dim = feature_dim + self.exposuremap.hdim
        elif self.exposure_type == 1:
            rep_dim = feature_dim + 1
        else:
            rep_dim = feature_dim

        # Shared embedding for T-learner
        self.tl_embed = nn.Linear(rep_dim, hidden)
        self.tl_bn = nn.BatchNorm1d(hidden)

        # Separate outcome heads
        self.out_t0 = nn.Linear(hidden, 1)
        self.out_t1 = nn.Linear(hidden, 1)

        # Propensity score head (for compatibility / logging)
        self.prop_head = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def encoder_parameters(self):
        params = list(self.featuremap.parameters()) + list(self.prop_head.parameters())
        if hasattr(self, "exposuremap"):
            params += list(self.exposuremap.parameters())
        return params

    def estimator_parameters(self):
        return (
            list(self.tl_embed.parameters())
            + list(self.tl_bn.parameters())
            + list(self.out_t0.parameters())
            + list(self.out_t1.parameters())
        )

    @staticmethod
    def compute_z_IDENet(t, edge_index, num_nodes=None, eps=1e-8):
        """Compute IDENet exposure as average neighbor treatment.

        Matches original IDE-Net implementation:
            z_i = (sum_{j in N(i)} t_j) / (deg(i) + eps)
        """
        if num_nodes is None:
            num_nodes = t.shape[0]
        row, col = edge_index
        z_IDENet = torch.zeros(num_nodes, device=t.device, dtype=t.dtype)
        z_IDENet.scatter_add_(0, row, t[col])
        deg = degree(row, num_nodes=num_nodes, dtype=t.dtype)
        z_IDENet = z_IDENet / (deg + eps)
        return z_IDENet

    def _build_representation(self, h, t, edge_index, z_IDENet=None, edge_attr_IDE=None):
        if self.exposure_type == 0:
            return h

        if self.exposure_type == 2:
            if z_IDENet is None:
                z_IDENet = self.exposuremap(h, t, edge_index, edge_attr_IDE=edge_attr_IDE)
            if z_IDENet.dim() == 1:
                z_IDENet = z_IDENet.unsqueeze(1)
            return torch.cat([h, z_IDENet], dim=1)

        # exposure_type == 1 (average neighbor treatment scalar)
        if z_IDENet is None:
            z_IDENet = self.compute_z_IDENet(t, edge_index, num_nodes=h.shape[0])
        if z_IDENet.dim() == 1:
            z_IDENet = z_IDENet.unsqueeze(1)
        return torch.cat([h, z_IDENet], dim=1)

    def _encode_features(self, x, edge_index, edge_attr_IDE=None):
        h, _ = self.featuremap(x, edge_index, edge_attr_IDE=edge_attr_IDE)
        h = F.relu(h)
        h = self.dropout(h)
        return h

    def forward(self, x, t, z_IDENet=None, edge_index=None, edge_attr_IDE=None):
        if edge_index is None:
            raise ValueError("edge_index must be provided for IDENet forward pass.")
        # Feature encoding
        h = self._encode_features(x, edge_index, edge_attr_IDE=edge_attr_IDE)

        # Propensity prediction from embeddings
        t_pred = self.prop_head(h)

        # Build representation: [embedding, exposure]
        rep = self._build_representation(
            h,
            t,
            edge_index,
            z_IDENet=z_IDENet,
            edge_attr_IDE=edge_attr_IDE,
        )

        # T-learner
        emb = F.relu(self.tl_embed(rep))
        emb = torch.tanh(self.tl_bn(emb))

        y0 = self.out_t0(emb).view(-1)
        y1 = self.out_t1(emb).view(-1)

        # Select factual outcome based on treatment
        y_pred = torch.where(t > 0, y1, y0)

        return t_pred, y_pred.unsqueeze(1)

    def forward_y0_y1(self, x, t, z_IDENet=None, edge_index=None, edge_attr_IDE=None):
        """Return both potential outcome predictions (for ITE estimation)."""
        if edge_index is None:
            raise ValueError("edge_index must be provided for IDENet forward pass.")
        h = self._encode_features(x, edge_index, edge_attr_IDE=edge_attr_IDE)

        t_pred = self.prop_head(h)

        rep = self._build_representation(
            h,
            t,
            edge_index,
            z_IDENet=z_IDENet,
            edge_attr_IDE=edge_attr_IDE,
        )
        emb = F.relu(self.tl_embed(rep))
        emb = torch.tanh(self.tl_bn(emb))

        y0 = self.out_t0(emb).view(-1)
        y1 = self.out_t1(emb).view(-1)

        # Return TLearner embedding to match original CFR regularization target.
        return t_pred, y0, y1, emb
