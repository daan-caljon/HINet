import yaml
import numpy as np
import math


def load_config(yaml_path, cli_overrides=None):
    """Load configuration from a YAML file with optional CLI overrides.

    Args:
        yaml_path: Path to YAML config file.
        cli_overrides: List of "dotted.key=value" strings, e.g. ["model.type=HINet", "data.num_nodes=1000"].

    Returns:
        Nested dict of configuration.
    """
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    if cli_overrides:
        for override in cli_overrides:
            key, value = override.split("=", 1)
            value = _parse_value(value)
            _set_nested(config, key, value)

    return config


def _parse_value(value_str):
    """Parse a CLI value string into the appropriate Python type."""
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    if value_str.lower() == "null" or value_str.lower() == "none":
        return None
    # Try list (e.g. "[1,2,3]")
    if value_str.startswith("[") and value_str.endswith("]"):
        inner = value_str[1:-1]
        return [_parse_value(v.strip()) for v in inner.split(",")]
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str


def _set_nested(d, dotted_key, value):
    """Set a value in a nested dict using a dotted key like 'data.num_nodes'."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def to_flat_dict(config):
    """Convert nested YAML config to the flat dict format expected by existing code.

    This bridges the clean YAML structure to the legacy config keys used by
    data_generator, Trainer, and metrics code.
    """
    exp = config.get("experiment", {})
    data = config.get("data", {})
    model = config.get("model", {})
    training = config.get("training", {})
    tuning = config.get("tuning", {})
    evaluation = config.get("evaluation", {})
    targeted = config.get("targeted_model", {})
    idenet = config.get("idenet", {})
    netdeconf = config.get("netdeconf", {})

    seed = exp.get("seed", 2000)
    num_nodes = data.get("num_nodes", 5000)
    covariate_dim = data.get("covariate_dim", 10)

    # DGP weight vectors, drawn Unif(-1, 1) deterministically from the seed.
    rng = np.random.RandomState(seed)
    w_c = 2 * rng.random_sample(covariate_dim) - 1
    w_c_n = 2 * rng.random_sample(covariate_dim) - 1
    w = 2 * rng.random_sample(covariate_dim) - 1
    w_n = 2 * rng.random_sample(covariate_dim) - 1
    w_beta_T2Y = 2 * rng.random_sample(covariate_dim) - 1
    w_exposure = 2 * rng.random_sample(covariate_dim) - 1

    # Auto-compute targeted model beta if not set
    beta = targeted.get("beta")
    if beta is None:
        beta = 20 * (num_nodes ** -0.5)

    flat = {
        # Experiment
        "seed": seed,
        "num_seeds": exp.get("num_seeds", 5),
        "run": 1,
        # Data
        "dataset": data.get("dataset", "full_sim"),
        "num_nodes": num_nodes,
        "covariate_dim": covariate_dim,
        "homophily": data.get("homophily", False),
        "node2vec": data.get("node2vec", False),
        "compute_assortativity": data.get("compute_assortativity", False),
        "edges_new_node": data.get("edges_new_node", 2),
        "exposure_type": data.get("exposure_type", "weight"),
        "percent_treated": data.get("percent_treated", 0.25),
        "flipRate": data.get("flip_rate", 0.5),
        # DGP parameters
        "betaConfounding": data.get("beta_confounding", 3),
        "betaNeighborConfounding": data.get("beta_neighbor_confounding", 0),
        "betaTreat2Outcome": data.get("beta_treat_to_outcome", 2),
        "betaNeighborTreatment2Outcome": data.get("beta_neighbor_treatment_to_outcome", 2),
        "betaCovariate2Outcome": data.get("beta_covariate_to_outcome", 1.5),
        "betaNeighborCovariate2Outcome": data.get("beta_neighbor_covariate_to_outcome", 1.5),
        "betaNoise": data.get("beta_noise", 0.2),
        "beta0": data.get("beta0", 0),
        "bias_T2Y": data.get("bias_T2Y", 0),
        "bias_NT2Y": data.get("bias_NT2Y", 0),
        # Random weight vectors
        "w_c": w_c,
        "w": w,
        "w_beta_T2Y": w_beta_T2Y,
        "w_c_n": w_c_n,
        "w_n": w_n,
        "w_exposure": w_exposure,
        # Model
        "model_type": model.get("type", "HINet"),
        "hidden": model.get("hidden", 16),
        "dropout": model.get("dropout", 0.0),
        "gnn_layer": model.get("gnn_layer", "GIN"),  # HINet only: "GIN", "GAT", "GCN", "GraphSAGE"
        # Training
        "num_epochs": training.get("num_epochs", 1500),
        "batch_size": training.get("batch_size", -1),
        "learning_rate": training.get("learning_rate", 0.005),
        "weight_decay": training.get("weight_decay", 0.001),
        "alpha": training.get("alpha", 1),
        "gamma": training.get("gamma", 1),
        # NetDeconf
        "netdeconf_balance_weight": netdeconf.get("balance_weight", 0.5),
        # Tuning
        "epochs_range": tuning.get("epochs_range", [500, 1000, 2000]),
        "hidden_range": tuning.get("hidden_range", [16, 32]),
        "alpha_range": tuning.get("alpha_range", [0, 0.025, 0.05, 0.1, 0.2, 0.3]),
        "n_alpha_tune": tuning.get("n_alpha_tune", 2),
        "lr_range": tuning.get("lr_range", [0.001, 0.0005, 0.0001]),
        "dropout_range": tuning.get("dropout_range", [0, 0.1, 0.2]),
        "p_alpha": tuning.get("p_alpha", 0.1),
        # Evaluation
        "num_networks": evaluation.get("num_networks", 50),
        "track_loss": evaluation.get("track_loss", False),
        # Targeted model
        "beta": beta,
        "num_grid": targeted.get("num_grid", 20),
        "tr_knots": targeted.get("tr_knots", 0.1),
        "lr_1_step": targeted.get("lr_2_step", 1e-3),
        "lr_2_step": targeted.get("lr_2_step", 1e-3),
        "pre_train_epochs": targeted.get("pre_train_epochs", 0),
        "fluctuation_train_epochs": targeted.get("fluctuation_train_epochs", 50),
        "loss_2step_with_ly": targeted.get("loss_2step_with_ly", 0),
        "loss_2step_with_ltz": targeted.get("loss_2step_with_ltz", 0),
        # IDENet
        "idenet_lr_est_multiplier": idenet.get("lr_est_multiplier", 10.0),
        "idenet_lr_step_size": idenet.get("lr_step_size", 50),
        "idenet_lr_step_gamma": idenet.get("lr_step_gamma", 0.5),
        "idenet_grad_clip": idenet.get("grad_clip", 3.0),
        "idenet_exposure_type": idenet.get("exposure_type", 2),
        "idenet_edge_dim": idenet.get("edge_dim", 1),
        "idenet_exposure_layers": idenet.get("exposure_layers", 2),
        "idenet_vanilla": idenet.get("vanilla", False),
        "idenet_edge_hidden": idenet.get("edge_hidden", 4),
        "idenet_feature_layers": idenet.get("feature_layers", 2),
        "idenet_precompute_train_z": idenet.get("precompute_train_z", True),
        "idenet_smooth_lambda": idenet.get("smooth_lambda", 0.1),
        "idenet_smooth_start_epoch": idenet.get("smooth_start_epoch", 150),
        "idenet_early_stop_patience": idenet.get("early_stop_patience", 300),
        "idenet_early_stop_warmup": idenet.get("early_stop_warmup", 50),
        "idenet_early_stop_min_epoch": idenet.get("early_stop_min_epoch", 150),
    }
    return flat


def build_setting_string(flat_config):
    """Build the data file setting string from a flat config dict."""
    return (
        flat_config["dataset"]
        + "_num_nodes" + str(flat_config["num_nodes"])
        + "_T2O_" + str(flat_config["betaTreat2Outcome"])
        + "_NT2O_" + str(flat_config["betaNeighborTreatment2Outcome"])
        + "_seed_" + str(flat_config["seed"])
    )
