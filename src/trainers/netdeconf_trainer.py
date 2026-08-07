import torch
import wandb

from src.trainers.base import BaseTrainer
from src.utils.utils import wasserstein


class NetDeconfTrainer(BaseTrainer):
    """Train NetDeconf with outcome loss and representation balancing.

    The objective is factual-outcome MSE plus a Wasserstein discrepancy
    between treated and control node representations.
    """

    alpha_tuning_enabled = False
    default_gamma = 0

    def __init__(self, config, train_data, val_data, test_data, device=True):
        super().__init__(config, train_data, val_data, test_data, device=device)
        self.balance_weight = self.config.get("netdeconf_balance_weight", 0.5)

    def _create_optimizers(self, lr, weight_decay):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def _train_step(self, batch, batch_size):
        self.optimizer.zero_grad()
        _, y_pred, representations = self.model(
            batch.x, batch.t, batch.z, batch.edge_index
        )

        # NeighborLoader places target nodes first. Restrict both losses to
        # those nodes so sampled neighbors are not repeatedly overweighted.
        y_pred = y_pred[:batch_size]
        y_true = batch.y[:batch_size]
        rep = representations[:batch_size]
        treatment = batch.t[:batch_size]

        y_loss = self.criterion(y_pred.squeeze(1), y_true)
        rep_treated = rep[treatment > 0.5]
        rep_control = rep[treatment <= 0.5]

        # The shared Wasserstein helper expects at least two rows per group.
        if rep_treated.shape[0] < 2 or rep_control.shape[0] < 2:
            balance_loss = rep.sum() * 0.0
        else:
            balance_loss, _ = wasserstein(
                rep_treated, rep_control, cuda=self.device
            )

        total_loss = y_loss + self.balance_weight * balance_loss
        total_loss.backward()
        self.optimizer.step()

        if self._log_this_epoch:
            wandb.log({
                "epoch": self._current_epoch,
                "train_y_loss": y_loss.detach().cpu().item(),
                "train_balance_loss": balance_loss.detach().cpu().item(),
                "train_total_loss": total_loss.detach().cpu().item(),
            })

    def _predict(self, data):
        t_pred, y_pred, _ = self.model(
            data.x, data.t, data.z, data.edge_index
        )
        return t_pred, y_pred

    def _raw_predict(self, x, t, z, edge_index, edge_attr_IDE=None):
        del edge_attr_IDE
        t_pred, y_pred, _ = self.model(x, t, z, edge_index)
        return t_pred, y_pred
