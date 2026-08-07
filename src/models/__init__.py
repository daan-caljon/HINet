from src.models.hinet import HINet, HINet_no_net_conf
from src.models.netest import NetEst, GINNetEst
from src.models.baselines import GINModel, TARNet, GCN_DECONF
from src.models.spnet import SPNet
from src.models.targeted import TargetedModel_DoubleBSpline
from src.models.idenet import IDENet

MODEL_REGISTRY = {
    "HINet": HINet,
    "HINet_no_net_conf": HINet_no_net_conf,
    "NetEst": NetEst,
    "GINNetEst": GINNetEst,
    "GINModel": GINModel,
    "TARNet": TARNet,
    "GCN_DECONF": GCN_DECONF,
    "SPNet": SPNet,
    "TargetedModel_DoubleBSpline": TargetedModel_DoubleBSpline,
    "IDENet": IDENet,
}


def create_model(model_type, covariate_dim, hidden, dropout=0, **kwargs):
    """Instantiate a model by name.

    Args:
        model_type: Key in MODEL_REGISTRY.
        covariate_dim: Input feature dimension.
        hidden: Hidden layer dimension.
        dropout: Dropout rate.
        **kwargs: Additional model-specific args (e.g. tr_knots for TargetedModel).
    """
    cls = MODEL_REGISTRY[model_type]
    if model_type == "TargetedModel_DoubleBSpline":
        return cls(
            Xshape=covariate_dim, hidden=hidden, dropout=dropout,
            tr_knots=kwargs.get("tr_knots", 0.1)
        )
    if model_type == "IDENet":
        return cls(
            Xshape=covariate_dim,
            hidden=hidden,
            dropout=dropout,
            exposure_type=kwargs.get("idenet_exposure_type", 2),
            edge_dim=kwargs.get("idenet_edge_dim", 1),
            exposure_layers=kwargs.get("idenet_exposure_layers", 2),
            vanilla=kwargs.get("idenet_vanilla", False),
            edge_hidden=kwargs.get("idenet_edge_hidden", 4),
            feature_layers=kwargs.get("idenet_feature_layers", 2),
        )
    if model_type in ("HINet", "HINet_no_net_conf"):
        return cls(
            Xshape=covariate_dim,
            hidden=hidden,
            dropout=dropout,
            gnn_layer=kwargs.get("gnn_layer", "GIN"),
        )
    return cls(Xshape=covariate_dim, hidden=hidden, dropout=dropout)
