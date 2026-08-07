import torch
import wandb

from src.trainers.base import BaseTrainer


class SimpleTrainer(BaseTrainer):
    """Trainer for models with simple outcome-only loss: GINModel and TARNet.

    Single optimizer. Loss = MSE(y_pred, y_true).
    No treatment-prediction or representation-balancing objective.
    """

    alpha_tuning_enabled = False
    default_gamma = 0

    def _create_optimizers(self, lr, weight_decay):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def _train_step(self, batch, batch_size):
        self.optimizer.zero_grad()
        t_pred, y_pred = self.model(batch.x, batch.t, batch.z, batch.edge_index)
        y_loss = self.criterion(y_pred[:batch_size].squeeze(1), batch.y[:batch_size])
        y_loss.backward()
        self.optimizer.step()
        if self._log_this_epoch:
            wandb.log({
                "epoch": self._current_epoch,
                "train_y_loss": y_loss.cpu().item(),
                "train_t_loss": self.bce_loss(t_pred[:batch_size].squeeze(1), batch.t[:batch_size]).cpu().item(),
                "train_total_loss": y_loss.cpu().item(),
            })

    def _predict(self, data):
        t_pred, y_pred = self.model(data.x, data.t, data.z, data.edge_index)
        return t_pred, y_pred
