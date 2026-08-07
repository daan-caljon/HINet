import os
import pickle as pkl

import networkx as nx
import numpy as np
import torch
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics.pairwise import cosine_similarity

from src.data.datatools import readData, dataSplit, covariateTransform, adjMatrixSplit
from src.data.utils import node2vec


def _neighbor_max(A, values):
    """Per-node maximum of ``values`` over graph neighbours.

    Companion to the neighbour-averaging used for ``mean`` aggregation: for each
    node ``i`` this returns ``max_{k in N_i} values[k]`` (``0.0`` for isolated
    nodes, matching the ``out=zeros`` convention of the mean path). Only actual
    neighbours are considered, so negative ``values`` are handled correctly --
    a sparse row-max would instead treat the structural zeros as candidates.
    """
    A = csr_matrix(A)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.zeros(A.shape[0], dtype=np.float64)
    indptr, indices = A.indptr, A.indices
    for i in range(A.shape[0]):
        start, end = indptr[i], indptr[i + 1]
        if end > start:
            out[i] = values[indices[start:end]].max()
    return out


def _neighbor_softmax(A, values, temperature=1.0):
    """Per-node softmax-weighted aggregation of ``values`` over graph neighbours.

    Each neighbour ``k`` contributes a score ``values[k]``; node ``i`` aggregates
    them as ``sum_{k in N_i} w_ik * values[k]`` with weights ``w_ik`` proportional
    to ``exp(values[k] / temperature)``. This is a *soft* maximum: it leans on the
    highest-scoring neighbours (like ``max``) but smoothly, so the assignment
    signal stays bounded between the neighbourhood mean and max instead of
    tracking a single extreme neighbour. The ``temperature`` tau sets how soft:
    ``tau -> 0`` approaches ``max`` (peaked weights), ``tau -> inf`` approaches the
    ``mean`` (uniform weights); larger tau spreads the weights and so preserves the
    overlap a hard maximum can erode on degree-heterogeneous graphs. Numerically
    mirrors ``masked_softmax`` (row max subtracted for stability); ``0.0`` for
    isolated nodes.
    """
    A = csr_matrix(A)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.zeros(A.shape[0], dtype=np.float64)
    indptr, indices = A.indptr, A.indices
    for i in range(A.shape[0]):
        start, end = indptr[i], indptr[i + 1]
        if end > start:
            s = values[indices[start:end]]
            w = np.exp((s - s.max()) / temperature)
            w /= w.sum()
            out[i] = float(w @ s)
    return out


def treatmentSimulation(config, X, A, rng=None, return_propensity=False):
    if rng is None:
        rng = np.random
    if config["betaConfounding"] == 0 and config["betaNeighborConfounding"] == 0:
        propensityT = sigmoid(np.zeros(len(X)))
        random_values = np.ones(len(propensityT))
        indices = rng.choice(
            len(propensityT),
            int(len(propensityT) * config["percent_treated"]),
            replace=False,
        )
        random_values[indices] = 0
        T = (random_values < np.array(propensityT)).astype(int)
        if return_propensity:
            return T, np.mean(T), propensityT
        return T, np.mean(T)

    X_extended = np.concatenate((X[:, :5], X[:, -5:]), axis=1)
    covariate2TreatmentMechanism = np.matmul(config["w_c"], X_extended.T)
    covariate2NeighborTreatmentMechanism = np.matmul(config["w_c_n"], X_extended.T)
    neighbors = np.asarray(A.sum(axis=1)).flatten()
    neighborValues = covariate2NeighborTreatmentMechanism.reshape(-1)
    # How a node aggregates its neighbours' features into the assignment signal.
    # "mean" is the mechanism used throughout the paper and is the default.
    aggregation = config.get("neighborTreatmentAggregation", "mean")
    if aggregation == "mean":
        neighbor_sum = A @ neighborValues
        neighborAgg = np.divide(
            neighbor_sum,
            neighbors,
            out=np.zeros_like(neighbor_sum, dtype=np.float64),
            where=neighbors != 0,
        )
    elif aggregation == "max":
        neighborAgg = _neighbor_max(A, neighborValues)
    elif aggregation == "softmax":
        neighborAgg = _neighbor_softmax(
            A, neighborValues,
            temperature=config.get("neighborSoftmaxTemperature", 1.0),
        )
    else:
        raise ValueError(
            "neighborTreatmentAggregation must be 'mean', 'max', or 'softmax', "
            "got %r" % (aggregation,)
        )
    propensityT = (
        config["betaConfounding"] * covariate2TreatmentMechanism
        + config["betaNeighborConfounding"] * neighborAgg
    )
    percentile = np.percentile(propensityT, 100 * (1 - config["percent_treated"]))
    propensityT = propensityT - percentile
    propensityT = sigmoid(propensityT)

    random_values = rng.rand(len(propensityT))
    T = (random_values < propensityT).astype(int)
    print("T", np.mean(T), np.std(T))

    mean_T = np.mean(T)
    if return_propensity:
        return T, mean_T, propensityT
    return T, mean_T


def masked_softmax(x):
    mask = x != 0
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    exp_x = np.where(mask, exp_x, 0)
    softmax = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return softmax


def exposure_mapping(X, A, T, exposure_type="average", w_exposure=None, biasNT2Y=3):
    def _mask_adjacency(A_in, T_in):
        """Support masking for both scipy sparse and dense adjacency."""
        if hasattr(A_in, "multiply"):
            return A_in.multiply(T_in.reshape(1, -1))
        return np.asarray(A_in) * T_in.reshape(1, -1)

    neighbors = np.asarray(A.sum(axis=1)).flatten()
    if exposure_type == "sum":
        exposure = A @ T.reshape(-1)

    elif exposure_type == "average":
        sum_exposure = A @ T.reshape(-1)
        exposure = np.divide(
            sum_exposure,
            neighbors,
            out=np.zeros_like(sum_exposure, dtype=np.float64),
            where=neighbors != 0,
        )

    elif exposure_type == "weight":
        exposure_mech = np.matmul(w_exposure, X.T) + biasNT2Y
        A_masked = _mask_adjacency(A, T)
        exposure_num = A_masked @ exposure_mech
        exposure = np.divide(
            exposure_num,
            neighbors,
            out=np.zeros_like(exposure_num, dtype=np.float64),
            where=neighbors != 0,
        )

    elif exposure_type == "weight_squared":
        exposure_mech = np.matmul(w_exposure, X.T) + biasNT2Y
        exposure_mech = np.square(exposure_mech)
        A_masked = _mask_adjacency(A, T)
        exposure_num = A_masked @ exposure_mech
        exposure = np.divide(
            exposure_num,
            neighbors,
            out=np.zeros_like(exposure_num, dtype=np.float64),
            where=neighbors != 0,
        )

    elif exposure_type == "weight_sigmoid":
        exposure_mech = np.matmul(w_exposure, X.T) + biasNT2Y
        A_masked = _mask_adjacency(A, T)
        exposure = A_masked @ exposure_mech
        exposure = np.exp(exposure)
        exposure = exposure / np.sum(exposure, axis=0, keepdims=True)

    elif exposure_type == "full_correlated":
        exposure_mech_1 = np.matmul(w_exposure, X.T) + biasNT2Y
        w_exposure_0 = -np.array(w_exposure) * 0.5
        w_exposure_0 = w_exposure_0.tolist()
        exposure_mech_0 = np.matmul(w_exposure_0, X.T)
        A_masked = _mask_adjacency(A, T)
        exposure_num = np.where(
            T.reshape(-1) == 1,
            A_masked @ exposure_mech_1,
            A_masked @ exposure_mech_0,
        )
        exposure = np.divide(
            exposure_num,
            neighbors,
            out=np.zeros_like(exposure_num, dtype=np.float64),
            where=neighbors != 0,
        )

    elif exposure_type == "entropy":
        treated_neighbors = A @ T.reshape(-1)
        prob_treated = np.divide(
            treated_neighbors,
            neighbors,
            out=np.zeros_like(treated_neighbors, dtype=np.float64),
            where=neighbors != 0,
        )
        entropy = -(prob_treated * np.log2(prob_treated + 1e-10) +
                   (1 - prob_treated) * np.log2(1 - prob_treated + 1e-10))
        exposure = entropy
        exposure = exposure - 0.5

    return exposure


def flipTreatment(T, rate, rng=None):
    if rng is None:
        rng = np.random
    numToFlip = int(len(T) * rate)
    flip_indices = rng.choice(len(T), numToFlip, replace=False)
    cfT = np.array(T, dtype=np.float32, copy=True)
    cfT[flip_indices] = 1 - cfT[flip_indices]
    nodesToFlip = set(flip_indices.tolist())
    return cfT, nodesToFlip


def calculate_ITTE(config, X, A, T):
    T_treat_0 = np.zeros(len(T))
    PO_0_treat = potentialOutcomeSimulation(config, X, A, T_treat_0)
    PO_T = potentialOutcomeSimulation(config, X, A, T)
    ITTE = PO_T - PO_0_treat
    return ITTE


def calculate_ITTE_torch(config, X, A, T):
    X = X.cpu().numpy()
    A = A.cpu().numpy()
    T = T.cpu().numpy()
    ITTE = calculate_ITTE(config, X, A, T)
    return torch.tensor(ITTE, dtype=torch.float32)


def potentialOutcomeSimulation_torch(config, X, A, T, epsilon=0):
    X = X.cpu().numpy()
    A = A.cpu().numpy()
    T = T.cpu().numpy()
    po = potentialOutcomeSimulation(config, X, A, T, epsilon)
    return torch.tensor(po, dtype=torch.float32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def potentialOutcomeSimulation(config, X, A, T, epsilon=0):
    w = np.array(config["w"])
    w_beta_T2Y = np.array(config["w_beta_T2Y"])
    T = np.array(T, dtype=np.float32)

    X_sigmoid = sigmoid(X)

    X_extended = np.concatenate((X[:, :5], X_sigmoid[:, -5:]), axis=1)

    covariate2OutcomeMechanism = np.matmul(w, X_extended.T)
    covariate2NeighborOutcomeMechanism = np.matmul(config["w_n"], X_extended.T)

    neighbors = np.asarray(A.sum(axis=1)).flatten()
    neighbor_sum = A @ covariate2NeighborOutcomeMechanism.reshape(-1)
    neighborAverage = np.divide(
        neighbor_sum,
        neighbors,
        out=np.zeros_like(neighbor_sum, dtype=np.float64),
        where=neighbors != 0,
    )

    beta_T2Y = np.matmul(w_beta_T2Y, X_extended.T) + config["bias_T2Y"]
    total_Treat2Outcome = config["betaTreat2Outcome"] * beta_T2Y

    exposure = exposure_mapping(
        X_extended,
        A,
        T,
        exposure_type=config["exposure_type"],
        w_exposure=config["w_exposure"],
        biasNT2Y=config["bias_NT2Y"],
    )

    T = np.array(T)
    potentialOutcome = (
        config["beta0"]
        + total_Treat2Outcome * T
        + config["betaCovariate2Outcome"] * covariate2OutcomeMechanism
        + config["betaNeighborCovariate2Outcome"] * neighborAverage
        + config["betaNeighborTreatment2Outcome"] * exposure
        + config["betaNoise"] * epsilon
    )

    return potentialOutcome


def generate_data(config, gen_type, nx_seed, watts_strogatz=False):
    # Per-split RNG: every stochastic draw inside this function (X, T, eps, cf_T,
    # X_random) must come from `rng`, not the global numpy state, so train/val/test
    # are reproducible independently of what ran before in the process.
    rng = np.random.RandomState(nx_seed)

    do_node2vec = config["node2vec"]
    do_homophily = config["homophily"]

    if config["dataset"] == "full_sim" and not do_homophily:
        if watts_strogatz:
            G = nx.connected_watts_strogatz_graph(
                config["num_nodes"], 4, 0.1, seed=nx_seed
            )
        else:
            G = nx.barabasi_albert_graph(
                config["num_nodes"], config["edges_new_node"], seed=nx_seed
            )
        adj_matrix = nx.adjacency_matrix(G)
    elif config["dataset"] in ("Flickr", "BC", "CS"):
        data, parts = readData(config["dataset"])
        trainIndex, valIndex, testIndex = dataSplit(parts)
        trainX, valX, testX = covariateTransform(
            data, config["covariate_dim"], trainIndex, valIndex, testIndex,
            random_state=config["seed"],
        )
        mean_trainX = np.mean(trainX, axis=(0))
        std_trainX = np.std(trainX, axis=(0))
        trainX = (trainX - mean_trainX) / std_trainX
        valX = (valX - np.mean(valX, axis=0)) / np.std(valX, axis=0)
        testX = (testX - np.mean(testX, axis=0)) / np.std(testX, axis=0)
        trainA, valA, testA = adjMatrixSplit(
            data, trainIndex, valIndex, testIndex, config["dataset"]
        )
        do_homophily = False

    if do_homophily:
        path = (
            "data/simulated/"
            + "num_nodes_"
            + str(config["num_nodes"])
            + "_homophilous_"
            + gen_type
            + ".pkl"
        )
        if os.path.exists(path):
            with open(path, "rb") as f:
                A, X = pkl.load(f)
            # Backward compat: old pickles stored dense A
            if not issparse(A):
                A = csr_matrix(A)
        else:
            A, X = create_homophilous_network(
                config["num_nodes"], config["covariate_dim"], rng=rng
            )
            with open(path, "wb") as f:
                pkl.dump((csr_matrix(A), X), f)
            A = csr_matrix(A)
    elif config["dataset"] == "full_sim":
        A = csr_matrix(adj_matrix)
    else:
        if gen_type == "train":
            A = csr_matrix(trainA)
        elif gen_type == "val":
            A = csr_matrix(valA)
        elif gen_type == "test":
            A = csr_matrix(testA)

    if do_node2vec:
        if config["dataset"] == "full_sim":
            file = (
                "data/simulated/"
                + config["dataset"]
                + "_num_nodes_"
                + str(config["num_nodes"])
                + "_"
                + gen_type
                + "X.pkl"
            )
        else:
            file = (
                "data/semi_synthetic/"
                + config["dataset"]
                + "/"
                + config["dataset"]
                + "_"
                + gen_type
                + "X.pkl"
            )
        if os.path.exists(file):
            with open(file, "rb") as f:
                X = pkl.load(f)
        else:
            X = node2vec.generate_node_embeddings(
                A, embedding_dim=config["covariate_dim"], seed=nx_seed
            )
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            X = np.clip(X, -5, 5)
            os.makedirs(os.path.dirname(file), exist_ok=True)
            with open(file, "wb") as f:
                pkl.dump(X, f)
        trainX = X
        valX = X
        testX = X
    elif do_homophily:
        pass
    elif config["dataset"] == "full_sim" and not do_homophily:
        X = rng.randn(A.shape[1], config["covariate_dim"])
    else:
        if gen_type == "train":
            X = trainX
        elif gen_type == "val":
            X = valX
        elif gen_type == "test":
            X = testX

    epsilon = rng.normal(0, 1, X.shape[0])

    T, meanT = treatmentSimulation(config, X=np.asarray(X), A=A, rng=rng)
    T = torch.tensor(T, dtype=torch.float32)

    PO = potentialOutcomeSimulation(config, X, A, T, epsilon)

    cfT, nodesToFlip = flipTreatment(T, config["flipRate"], rng=rng)
    cfT = torch.from_numpy(cfT)
    cfPOTrain = potentialOutcomeSimulation(config, X, A, cfT)

    ITTE = calculate_ITTE(config, X, A, cfT)

    num = X.shape[0]
    X_random = rng.randn(num, config["covariate_dim"])
    PO_random = potentialOutcomeSimulation(config, X_random, A, T)

    # Optional diagnostics: expensive for large graphs.
    if config.get("compute_assortativity", False):
        T_array = np.asarray(T)
        PO_array = np.asarray(PO)
        PO_array = (PO_array - np.mean(PO_array)) / np.std(PO_array)
        G = nx.from_scipy_sparse_array(A)
        for i in range(G.number_of_nodes()):
            G.nodes[i]["T"] = T_array[i]
            G.nodes[i]["Y"] = PO_array[i]
        treatment_assortativity = nx.attribute_assortativity_coefficient(G, "T")
        print("treatment_assortativity", treatment_assortativity)
        outcome_assortativity = nx.numeric_assortativity_coefficient(G, "Y")
        print("outcome_assortativity", outcome_assortativity)

    my_data = {
        "T": T.numpy(),
        "cfT": cfT.numpy(),
        "features": np.asarray(X),
        "PO": np.asarray(PO),
        "cfPO": np.asarray(cfPOTrain),
        "nodesToFlip": nodesToFlip,
        "network": csr_matrix(A),
        "meanT": meanT,
        "ITTE": np.asarray(ITTE),
        "X_random": X_random,
        "PO_random": PO_random,
    }
    return my_data


def simulate_data(config, setting, watts_strogatz=False):
    base = config["seed"]
    train = generate_data(
        config, gen_type="train", nx_seed=base, watts_strogatz=watts_strogatz
    )
    val = generate_data(
        config, gen_type="val", nx_seed=base + 1, watts_strogatz=watts_strogatz
    )
    test = generate_data(
        config, gen_type="test", nx_seed=base + 2, watts_strogatz=watts_strogatz
    )
    data = {"train": train, "val": val, "test": test}
    file = "data/simulated/" + setting + ".pkl"
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "wb") as f:
        pkl.dump(data, f)


def create_homophilous_network(num_nodes, num_features, rng=None):
    if rng is None:
        rng = np.random
    X = rng.randn(num_nodes, num_features)
    similarity_matrix = cosine_similarity(X)
    np.fill_diagonal(similarity_matrix, -1)

    loc = 0.80
    scale = 0.025
    tolerance = 0.1
    goal = 4

    np.fill_diagonal(similarity_matrix, -1)
    for i in range(100):
        threshold_num_matrix = rng.normal(
            loc=loc, scale=scale, size=(num_nodes, num_nodes)
        )
        A = (similarity_matrix > threshold_num_matrix).astype(int)
        upper = np.triu(A)
        A = upper + upper.T
        np.fill_diagonal(A, 0)

        avg_deg = np.mean(np.sum(A, 1))
        print("avg_deg", avg_deg)
        if abs(avg_deg - goal) < tolerance:
            max_sim = np.argmax(similarity_matrix, 1)
            for i in range(num_nodes):
                A[i, max_sim[i]] = 1
                A[max_sim[i], i] = 1
            G = nx.from_numpy_array(A)
            if nx.is_connected(G):
                print("connected")
            else:
                print("not connected")
                largest_cc = max(nx.connected_components(G), key=len)
                print("largest_cc", len(largest_cc))
            break

        if avg_deg < goal - tolerance:
            loc -= 0.002
        elif avg_deg > goal + tolerance:
            loc += 0.002

    return csr_matrix(A), X
