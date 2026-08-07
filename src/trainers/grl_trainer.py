import torch
import wandb

from src.trainers.base import BaseTrainer


class GRLTrainer(BaseTrainer):
    """Trainer for gradient-reversal-based models: HINet, HINet_no_net_conf.

    Single optimizer. Loss = y_loss + alpha * t_loss.
    The gradient reversal layer in the model automatically reverses gradients
    for the treatment prediction branch, creating adversarial training.
    Gamma is forced to 0 (no separate z loss).
    """

    alpha_tuning_enabled = True
    default_gamma = 0

    def _create_optimizers(self, lr, weight_decay):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def _train_step(self, batch, batch_size):
        self.optimizer.zero_grad()
        t_pred, y_pred = self.model(batch.x, batch.t, batch.z, batch.edge_index)
        y_loss = self.criterion(y_pred[:batch_size].squeeze(1), batch.y[:batch_size])
        t_loss = self.bce_loss(t_pred[:batch_size].squeeze(1), batch.t[:batch_size])
        total_loss = y_loss + self.alpha * t_loss
        total_loss.backward()
        self.optimizer.step()
        if self._log_this_epoch:
            wandb.log({
                "epoch": self._current_epoch,
                "train_y_loss": y_loss.cpu().item(),
                "train_t_loss": t_loss.cpu().item(),
                "train_total_loss": total_loss.cpu().item(),
            })

    def _predict(self, data):
        t_pred, y_pred = self.model(data.x, data.t, data.z, data.edge_index)
        return t_pred, y_pred
