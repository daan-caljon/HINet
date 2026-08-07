import torch
import wandb

from src.trainers.base import BaseTrainer
from src.utils.utils import Normalize_outcome_recover


class TargetedTrainer(BaseTrainer):
    """Trainer for TargetedModel_DoubleBSpline: doubly robust targeted learning.

    Two optimizers:
    1. optimizer_base: trains the base model (encoder, outcome heads, density estimators)
    2. optimizer_fluc: trains the fluctuation parameters (TR splines)

    Loss = Q_Loss + alpha * g_T_Loss + gamma * g_Z_Loss + beta * Loss_TR

    After pre_train_epochs, a second fluctuation optimization step is applied per batch.
    """

    alpha_tuning_enabled = False
    default_gamma = None  # Uses config gamma (default 1)

    @property
    def uses_dynamic_z(self):
        return True

    def _create_optimizers(self, lr, weight_decay):
        self.optimizer_base = torch.optim.Adam(
            self.model.parameter_base(), lr=lr, weight_decay=weight_decay
        )
        self.optimizer_fluc = torch.optim.Adam(
            self.model.parameters(), lr=self.config["lr_2_step"], weight_decay=weight_decay
        )
        self.epoch_count = 0

    def _train_step(self, batch, batch_size):
        # Base training step
        self.optimizer_base.zero_grad()
        self.optimizer_fluc.zero_grad()

        g_T_hat, g_Z_hat, Q_hat, epsilon, embeddings, neighborAverageT = self.model(
            batch.x, batch.t, batch.z, batch.edge_index
        )
        Q_Loss = self.criterion(Q_hat.reshape(-1), batch.y)
        g_T_Loss = self.bce_loss(g_T_hat.reshape(-1), batch.t)
        g_Z_Loss = -torch.log(g_Z_hat + 1e-6).mean()
        Loss_base = Q_Loss + self.alpha * g_T_Loss + self.gamma * g_Z_Loss

        target = Q_hat.reshape(-1) + epsilon.reshape(-1) * (
            1 / (g_T_hat.detach().reshape(-1) * g_Z_hat.detach().reshape(-1) + 1e-6)
        )
        Loss_TR = self.criterion(target, batch.y)
        loss_train = Loss_base + self.config["beta"] * Loss_TR

        loss_train.backward()
        self.optimizer_base.step()

        # Fluctuation step (after pre-training epochs)
        if self.epoch_count > self.config["pre_train_epochs"]:
            self._train_fluctuation_step(batch)
        self.epoch_count += 1

        if self._log_this_epoch:
            wandb.log({
                "epoch": self._current_epoch,
                "train_y_loss": Q_Loss.cpu().item(),
                "train_t_loss": g_T_Loss.cpu().item(),
                "train_z_loss": g_Z_Loss.cpu().item(),
                "train_total_loss": loss_train.cpu().item(),
            })

    def _train_fluctuation_step(self, batch):
        """Train fluctuation parameters for the targeted learning correction."""
        self.model.train()
        self.optimizer_fluc.zero_grad()

        g_T_hat, g_Z_hat, Q_hat, epsilon, _, _ = self.model(
            batch.x, batch.t, batch.z, batch.edge_index
        )
        Loss_TR = self.criterion(
            Q_hat.reshape(-1) + epsilon.reshape(-1) * (
                1 / (g_T_hat.reshape(-1).detach() * g_Z_hat.reshape(-1).detach() + 1e-6)
            ),
            batch.y,
        )
        loss_train = self.config["beta"] * Loss_TR

        if self.config["loss_2step_with_ly"] == 1:
            Q_Loss = self.criterion(Q_hat.reshape(-1), batch.y)
            loss_train = loss_train + Q_Loss

        if self.config["loss_2step_with_ltz"] == 1:
            g_T_Loss = self.bce_loss(g_T_hat.reshape(-1), batch.t)
            g_Z_Loss = -torch.log(g_Z_hat + 1e-6).mean()
            loss_train = loss_train + self.alpha * g_T_Loss + self.gamma * g_Z_Loss

        loss_train.backward()
        self.optimizer_fluc.step()

    def _forward_to_potential_outcome(self, x, t, z, edge_index):
        """Single forward pass that returns both t_pred and the potential outcome."""
        g_T_hat, g_Z_hat, Q_hat, epsilon, _, _ = self.model(x, t, z, edge_index)
        y_pred = Q_hat.reshape(-1) + epsilon.reshape(-1) / (
            g_Z_hat.reshape(-1) * g_T_hat.reshape(-1) + 1e-6
        )
        return g_T_hat, y_pred.unsqueeze(1)

    def _predict(self, data):
        return self._forward_to_potential_outcome(data.x, data.t, data.z, data.edge_index)

    def _predict_with_t(self, data, t, z):
        return self._forward_to_potential_outcome(data.x, t, z, data.edge_index)

    def _raw_predict(self, x, t, z, edge_index, edge_attr_IDE=None):
        del edge_attr_IDE
        return self._forward_to_potential_outcome(x, t, z, edge_index)

    def _make_predict_fn(self, mean_y_train, std_y_train):
        """Override: TargetedModel uses infer_potential_outcome for predictions."""
        def predict_fn(x, t, z, edge_index, edge_attr_IDE=None):
            del edge_attr_IDE
            y_pred = self.model.infer_potential_outcome(x, t, z, edge_index).squeeze()
            y_pred = Normalize_outcome_recover(y_pred, mean_y_train, std_y_train)
            return y_pred
        return predict_fn
