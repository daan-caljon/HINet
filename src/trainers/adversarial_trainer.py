import torch
import wandb

from src.trainers.base import BaseTrainer


class AdversarialTrainer(BaseTrainer):
    """Trainer for multi-optimizer adversarial models: NetEst, GINNetEst.

    Three separate optimizers:
    1. optimizer_t: trains treatment discriminator (maximize t prediction)
    2. optimizer_z: trains exposure discriminator (maximize z prediction)
    3. optimizer_p: trains encoder + predictor (minimize y_loss, maximize confusion of t and z)

    Each training step involves three sequential forward-backward passes.
    """

    alpha_tuning_enabled = True
    default_gamma = 0

    @property
    def uses_dynamic_z(self):
        return True

    def _create_optimizers(self, lr, weight_decay):
        self.optimizer_t = torch.optim.Adam(
            self.model.discriminator.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.optimizer_z = torch.optim.Adam(
            self.model.discrimnator_z.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.optimizer_p = torch.optim.Adam(
            [
                {"params": self.model.encoder.parameters()},
                {"params": self.model.predictor.parameters()},
            ],
            lr=lr, weight_decay=weight_decay
        )

    def _train_step(self, batch, batch_size):
        # Step 1: Train treatment discriminator
        self.optimizer_t.zero_grad()
        self.optimizer_z.zero_grad()
        self.optimizer_p.zero_grad()

        t_pred, y_pred, z_pred = self.model(batch.x, batch.t, batch.z, batch.edge_index)
        t_loss = self.bce_loss(t_pred[:batch_size].squeeze(1), batch.t[:batch_size])

        self.optimizer_t.zero_grad()
        t_loss.backward()
        self.optimizer_t.step()

        # Step 2: Train exposure discriminator
        t_pred, y_pred, z_pred = self.model(batch.x, batch.t, batch.z, batch.edge_index)
        z_loss = self.bce_loss(z_pred[:batch_size].squeeze(1), batch.z)
        self.optimizer_z.zero_grad()
        z_loss.backward()
        self.optimizer_z.step()

        # Step 3: Train predictor (adversarial: minimize y_loss, maximize t_loss and z_loss)
        t_pred, y_pred, z_pred = self.model(batch.x, batch.t, batch.z, batch.edge_index)
        y_loss = self.criterion(y_pred[:batch_size].squeeze(1), batch.y[:batch_size])
        t_loss = self.bce_loss(t_pred[:batch_size].squeeze(1), batch.t[:batch_size])
        z_loss = self.bce_loss(z_pred[:batch_size].squeeze(1), batch.z)
        total_loss = y_loss - self.alpha * t_loss - self.gamma * z_loss
        self.optimizer_p.zero_grad()
        total_loss.backward()
        self.optimizer_p.step()

        if self._log_this_epoch:
            wandb.log({
                "epoch": self._current_epoch,
                "train_y_loss": y_loss.cpu().item(),
                "train_t_loss": t_loss.cpu().item(),
                "train_z_loss": z_loss.cpu().item(),
                "train_total_loss": total_loss.cpu().item(),
            })

    def _predict(self, data):
        t_pred, y_pred, z_pred = self.model(data.x, data.t, data.z, data.edge_index)
        return t_pred, y_pred

    def _raw_predict(self, x, t, z, edge_index, edge_attr_IDE=None):
        del edge_attr_IDE
        t_pred, y_pred, z_pred = self.model(x, t, z, edge_index)
        return t_pred, y_pred
