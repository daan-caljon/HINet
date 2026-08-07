import torch
import wandb

from src.trainers.base import BaseTrainer
from src.utils.utils import wasserstein


class SPNetTrainer(BaseTrainer):
    """Trainer for SPNet: Wasserstein distance representation balancing.

    Single optimizer. Loss = y_loss + alpha * t_loss + gamma * wasserstein_distance.
    Balances representations between treated and control groups using
    the sliced Wasserstein distance.
    """

    alpha_tuning_enabled = False
    default_gamma = None  # Uses config gamma (default 1)

    def _create_optimizers(self, lr, weight_decay):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def _train_step(self, batch, batch_size):
        self.optimizer.zero_grad()
        t_pred, y_pred, representations = self.model(batch.x, batch.t, batch.z, batch.edge_index)
        y_loss = self.criterion(y_pred[:batch_size].squeeze(1), batch.y[:batch_size])
        t_loss = self.bce_loss(t_pred[:batch_size].squeeze(1), batch.t[:batch_size])

        rep_t1 = representations[(batch.t > 0).nonzero()]
        rep_t0 = representations[(batch.t < 1).nonzero()]
        d_loss, _ = wasserstein(rep_t1, rep_t0)

        total_loss = y_loss + self.alpha * t_loss + self.gamma * d_loss
        total_loss.backward()
        self.optimizer.step()
        if self._log_this_epoch:
            wandb.log({
                "epoch": self._current_epoch,
                "train_y_loss": y_loss.cpu().item(),
                "train_t_loss": t_loss.cpu().item(),
                "train_wasserstein_loss": d_loss.cpu().item(),
                "train_total_loss": total_loss.cpu().item(),
            })

    def _predict(self, data):
        t_pred, y_pred, _ = self.model(data.x, data.t, data.z, data.edge_index)
        return t_pred, y_pred

    def _raw_predict(self, x, t, z, edge_index, edge_attr_IDE=None):
        del edge_attr_IDE
        t_pred, y_pred, _ = self.model(x, t, z, edge_index)
        return t_pred, y_pred
