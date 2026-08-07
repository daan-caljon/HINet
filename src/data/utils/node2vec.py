"""Use node2vec to generate node embeddings given an adjacency matrix A"""
import torch
import numpy as np
import torch_geometric
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.nn import Node2Vec
from scipy.sparse import csr_matrix


def generate_node_embeddings(adjacency_matrix, embedding_dim=10, epochs=100,
                             walk_length=20, context_size=10, walks_per_node=5,
                             num_negative_samples=1, p=2, q=0.5, lr=0.01,
                             seed=None):
    sparse_A = csr_matrix(adjacency_matrix)
    edge_index, edge_attr = from_scipy_sparse_matrix(sparse_A)
    data = torch_geometric.data.Data(edge_index=edge_index, edge_attr=edge_attr)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    node2vec = Node2Vec(data.edge_index, embedding_dim=embedding_dim,
                        walk_length=walk_length, context_size=context_size,
                        walks_per_node=walks_per_node, num_negative_samples=num_negative_samples,
                        p=p, q=q, sparse=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    node2vec = node2vec.to(device)
    loader = node2vec.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(node2vec.parameters(), lr=lr)

    for epoch in range(epochs):
        node2vec.train()
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    node_embeddings = node2vec.embedding.weight.detach().cpu().numpy()
    return node_embeddings
