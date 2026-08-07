import pickle as pkl

import numpy as np
import torch
from torch_geometric.data import Data

"""
    This is code from Song Jiang: https://github.com/songjiang0909/Causal-Inference-on-Networked-Data
    MIT License
    Copyright (c) 2022 Song Jiang
"""


def dataTransform(data, cuda):
    A, X, T, cfT, PO, cfPO, ITTE, X_random, PO_random = (
        data["network"],
        data["features"],
        data["T"],
        data["cfT"],
        data["PO"],
        data["cfPO"],
        data["ITTE"],
        data["X_random"],
        data["PO_random"],
    )
    X = torch.tensor(X, dtype=torch.float)
    # Handle both sparse (new) and dense (old) adjacency matrices
    if hasattr(A, 'toarray'):
        A = A.toarray()
    A, T, cfT, PO, cfPO = (
        torch.tensor(A, dtype=torch.float),
        torch.tensor(T, dtype=torch.float),
        torch.tensor(cfT, dtype=torch.float),
        torch.tensor(PO, dtype=torch.float),
        torch.tensor(cfPO, dtype=torch.float),
    )
    ITTE = torch.tensor(ITTE, dtype=torch.float)
    X_random, PO_random = (
        torch.tensor(data["X_random"], dtype=torch.float),
        torch.tensor(data["PO_random"], dtype=torch.float),
    )
    return A, X, T, cfT, PO, cfPO, ITTE, X_random, PO_random


def _build_split(split_data):
    """Process one data split while avoiding dense adjacency expansion when possible."""
    A = split_data["network"]
    X = torch.tensor(split_data["features"], dtype=torch.float)
    T = torch.tensor(split_data["T"], dtype=torch.float)
    cfT = torch.tensor(split_data["cfT"], dtype=torch.float)
    PO = torch.tensor(split_data["PO"], dtype=torch.float)
    cfPO = torch.tensor(split_data["cfPO"], dtype=torch.float)
    ITTE = torch.tensor(split_data["ITTE"], dtype=torch.float)
    X_random = torch.tensor(split_data["X_random"], dtype=torch.float)
    PO_random = torch.tensor(split_data["PO_random"], dtype=torch.float)

    if hasattr(A, "tocsr"):
        A = A.tocsr()
        T_np = T.numpy()
        neighbors = np.asarray(A.sum(axis=1)).flatten()
        z_np = np.divide(
            A @ T_np,
            neighbors,
            out=np.zeros_like(T_np, dtype=np.float32),
            where=neighbors != 0,
        )
        rows, cols = A.nonzero()
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        z = torch.tensor(z_np, dtype=torch.float)
        # Dataset has no edge features: use binary edge indicator for IDENet.
        edge_attr_IDE = torch.ones((edge_index.shape[1], 1), dtype=torch.float)
    else:
        # Backward compatibility for legacy dense pickles.
        A = torch.tensor(A, dtype=torch.float)
        neighbors = torch.sum(A, 1)
        z = torch.div(torch.matmul(A, T.reshape(-1)), neighbors)
        edge_index = A.nonzero().t().contiguous()
        # Dataset has no edge features: use binary edge indicator for IDENet.
        edge_attr_IDE = torch.ones((edge_index.shape[1], 1), dtype=torch.float)

    return Data(
        x=X, edge_index=edge_index, y=PO, t=T,
        cf_t=cfT, cf_y=cfPO, z=z, ITTE=ITTE,
        X_random=X_random, PO_random=PO_random,
        edge_attr_IDE=edge_attr_IDE,
    )


def loadData(setting):
    file = "data/simulated/" + setting + ".pkl"
    with open(file, "rb") as f:
        data = pkl.load(f)

    # Process each split sequentially so only one dense A is in memory at a time
    train_data = _build_split(data["train"])
    val_data = _build_split(data["val"])
    test_data = _build_split(data["test"])

    return train_data, val_data, test_data
