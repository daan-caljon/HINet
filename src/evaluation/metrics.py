import numpy as np
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
from src.data.data_generator import calculate_ITTE, potentialOutcomeSimulation


def PEHNE(config, data, predict_fn, num_networks, train_mean_y, train_std_y, uses_dynamic_z=False):
    """Precision in Estimation of Heterogeneous Network Effects.

    Evaluates ability to predict heterogeneous treatment effects across
    multiple randomly sampled counterfactual treatment networks.

    Args:
        config: Experiment configuration dict.
        data: PyG Data object.
        predict_fn: Callable(x, t, z, edge_index, edge_attr_IDE=None) -> y_pred (denormalized).
        num_networks: Number of counterfactual networks to sample.
        train_mean_y: Mean of training outcomes (for denormalization).
        train_std_y: Std of training outcomes (for denormalization).
        uses_dynamic_z: If True, recompute z from treatment vector (for NetEst, GINNetEst, TargetedModel).
    """
    MSE_loss = torch.nn.MSELoss()
    edge_index = data.edge_index

    # Convert edge_index to scipy sparse once (not on GPU, not per-iteration)
    A_sp = to_scipy_sparse_matrix(edge_index).tocsr()
    X_np = data.x.cpu().numpy()
    neighbors_np = np.asarray(A_sp.sum(axis=1)).flatten()

    metric = 0
    length = data.t.shape[0]
    T_list = np.linspace(0, length, num_networks)

    edge_attr_IDE = getattr(data, "edge_attr_IDE", None)

    for T in T_list:
        T = int(T)
        # Keep torch.randperm to preserve RNG sequence
        T_tensor = torch.zeros(length).cuda()
        indices = torch.randperm(length)[:T]
        T_tensor[indices] = 1
        zero_tensor = torch.zeros(length).cuda()

        if uses_dynamic_z:
            T_np = T_tensor.cpu().numpy()
            z_np = (A_sp @ T_np) / neighbors_np
            z = torch.from_numpy(z_np.astype(np.float32)).cuda()
        else:
            z = data.z

        y_pred_T = predict_fn(
            data.x,
            T_tensor,
            z,
            data.edge_index,
            edge_attr_IDE=edge_attr_IDE,
        )
        y_pred_0 = predict_fn(
            data.x,
            zero_tensor,
            zero_tensor,
            data.edge_index,
            edge_attr_IDE=edge_attr_IDE,
        )
        ITTE_pred = y_pred_T - y_pred_0

        # Call numpy version directly — no repeated .cpu().numpy() on A and X
        T_np_for_itte = T_tensor.cpu().numpy()
        ITTE_data = calculate_ITTE(config, X_np, A_sp, T_np_for_itte)
        ITTE_data = torch.tensor(ITTE_data, dtype=torch.float32)
        metric += MSE_loss(ITTE_pred.cpu(), ITTE_data)

    return metric / num_networks


def CNEE(config, data, predict_fn, num_networks, train_mean_y, train_std_y, uses_dynamic_z=False):
    """Counterfactual Network Estimation Error.

    Evaluates ability to predict outcomes under different treatment assignments
    across multiple randomly sampled counterfactual networks.

    Args:
        config: Experiment configuration dict.
        data: PyG Data object.
        predict_fn: Callable(x, t, z, edge_index, edge_attr_IDE=None) -> y_pred (denormalized).
        num_networks: Number of counterfactual networks to sample.
        train_mean_y: Mean of training outcomes (for denormalization).
        train_std_y: Std of training outcomes (for denormalization).
        uses_dynamic_z: If True, recompute z from treatment vector.
    """
    MSE_loss = torch.nn.MSELoss()
    edge_index = data.edge_index

    # Convert edge_index to scipy sparse once (not on GPU, not per-iteration)
    A_sp = to_scipy_sparse_matrix(edge_index).tocsr()
    X_np = data.x.cpu().numpy()
    neighbors_np = np.asarray(A_sp.sum(axis=1)).flatten()

    metric = 0
    length = data.t.shape[0]
    T_list = np.linspace(0, length, num_networks)

    edge_attr_IDE = getattr(data, "edge_attr_IDE", None)

    for T in T_list:
        T = int(T)
        T_tensor = torch.zeros(length).cuda()
        indices = torch.randperm(length)[:T]
        T_tensor[indices] = 1

        if uses_dynamic_z:
            T_np = T_tensor.cpu().numpy()
            z_np = (A_sp @ T_np) / neighbors_np
            z = torch.from_numpy(z_np.astype(np.float32)).cuda()
        else:
            z = data.z

        y_pred_T = predict_fn(
            data.x,
            T_tensor,
            z,
            data.edge_index,
            edge_attr_IDE=edge_attr_IDE,
        )

        # Call numpy version directly — no repeated .cpu().numpy() on A and X
        T_np_for_po = T_tensor.cpu().numpy()
        y_T = potentialOutcomeSimulation(config, X_np, A_sp, T_np_for_po)
        y_T = torch.tensor(y_T, dtype=torch.float32)
        loss = MSE_loss(y_pred_T.cpu(), y_T)
        metric += loss

    return metric / num_networks
