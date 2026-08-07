from src.trainers.simple_trainer import SimpleTrainer
from src.trainers.grl_trainer import GRLTrainer
from src.trainers.adversarial_trainer import AdversarialTrainer
from src.trainers.spnet_trainer import SPNetTrainer
from src.trainers.targeted_trainer import TargetedTrainer
from src.trainers.idenet_trainer import IDENetTrainer
from src.trainers.netdeconf_trainer import NetDeconfTrainer

TRAINER_REGISTRY = {
    "HINet": GRLTrainer,
    "HINet_no_net_conf": GRLTrainer,
    "NetEst": AdversarialTrainer,
    "GINNetEst": AdversarialTrainer,
    "GINModel": SimpleTrainer,
    "TARNet": SimpleTrainer,
    "GCN_DECONF": NetDeconfTrainer,
    "SPNet": SPNetTrainer,
    "TargetedModel_DoubleBSpline": TargetedTrainer,
    "IDENet": IDENetTrainer,
}


def create_trainer(config, train_data, val_data, test_data, device=True):
    """Create the appropriate trainer for the given model type."""
    model_type = config["model_type"]
    trainer_cls = TRAINER_REGISTRY[model_type]
    return trainer_cls(
        config=config,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        device=device,
    )
