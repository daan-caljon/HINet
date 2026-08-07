"""Trainer for IDE-Net.

Adapted from the IDE-Net implementation of Adhikari & Zheleva (2025), which is
distributed under the GNU General Public License v3.0 (see LICENSE).

Two optimizers with separate learning rates:
  1. Encoder optimizer: GCN parameters (slower LR)
  2. Estimator optimizer: T-learner head parameters (faster LR)

LR step schedulers, gradient clipping, and optional variance smoothing
regularization.

Hyperparameters are read from config["idenet_*"] keys.

Important: IDENet uses a T-learner (separate y0/y1 heads) and takes z
(exposure) as direct input.  This means:
    - z_IDENet is computed from (A, t) inside the model to match the original
        IDE-Net implementation.
    - for exposure_type=1, z_IDENet can be cached in training when using full
        batch mode.
    - _predict_with_t passes treatment interventions to the model so that
    counterfactual and ITTE evaluations use the correct exposure.
"""
import torch
import wandb
from torch_geometric.loader import NeighborLoader

from src.trainers.base import BaseTrainer
from src.utils.utils import Normalize_outcome


class IDENetTrainer(BaseTrainer):
    """Trainer for IDENet: GCN encoder + T-learner with two optimizers.

    Loss = MSE(y_pred, y_true) + optional smoothing regularization.
    """

    alpha_tuning_enabled = False
    default_gamma = 0

    @property
    def uses_dynamic_z(self):
        """IDENet recomputes z_IDENet inside the model, not in metrics."""
        return False

    def _create_optimizers(self, lr, weight_decay):
        # lr and weight_decay come from hyperparameter tuning / config.
        # The estimator (T-learner) uses a higher LR than the encoder (GCN).
        # lr_est_multiplier controls the ratio (default 10x, matching the
        # IDE-Net paper's 0.2/0.02 defaults).
        lr_est_multiplier = self.config.get("idenet_lr_est_multiplier", 10.0)
        lr_enc = lr
        lr_est = lr * lr_est_multiplier

        step_size = self.config.get("idenet_lr_step_size", 50)
        step_gamma = self.config.get("idenet_lr_step_gamma", 0.5)
        grad_clip = self.config.get("idenet_grad_clip", 3.0)

        # Store for use in _train_step / train loop
        self._grad_clip = grad_clip
        self._cache_train_z_IDENet = self.config.get("idenet_precompute_train_z", True)
        self._cached_train_z_IDENet = None
        self._use_cached_train_z_IDENet = False
        self._smooth_lambda = self.config.get("idenet_smooth_lambda", 0.1)
        self._smooth_start_epoch = self.config.get("idenet_smooth_start_epoch", 150)
        self._early_stop_patience = self.config.get("idenet_early_stop_patience", 300)
        self._early_stop_min_epoch = self.config.get("idenet_early_stop_min_epoch", 150)
        self._early_stop_warmup = self.config.get("idenet_early_stop_warmup", 50)

        # Encoder: feature mapping (+ exposure mapping) + propensity head.
        encoder_params = self.model.encoder_parameters()
        self.optimizer_enc = torch.optim.Adam(
            encoder_params, lr=lr_enc, weight_decay=weight_decay
        )
        # Estimator: T-learner heads.
        estimator_params = self.model.estimator_parameters()
        self.optimizer_est = torch.optim.Adam(
            estimator_params, lr=lr_est, weight_decay=weight_decay
        )
        # LR schedulers
        self.scheduler_enc = torch.optim.lr_scheduler.StepLR(
            self.optimizer_enc, step_size=step_size, gamma=step_gamma
        )
        self.scheduler_est = torch.optim.lr_scheduler.StepLR(
            self.optimizer_est, step_size=step_size, gamma=step_gamma
        )

    def _train_step(self, batch, batch_size):
        self.optimizer_enc.zero_grad()
        self.optimizer_est.zero_grad()

        z_IDENet_cached = None
        if (
            self._use_cached_train_z_IDENet
            and self._cached_train_z_IDENet is not None
            and self._cached_train_z_IDENet.shape[0] == batch.x.shape[0]
        ):
            z_IDENet_cached = self._cached_train_z_IDENet

        t_pred, y0, y1, _ = self.model.forward_y0_y1(
            batch.x,
            batch.t,
            z_IDENet=z_IDENet_cached,
            edge_index=batch.edge_index,
            edge_attr_IDE=getattr(batch, "edge_attr_IDE", None),
        )
        y_pred = torch.where(batch.t[:batch_size] > 0,
                             y1[:batch_size], y0[:batch_size])
        y_loss = self.criterion(y_pred, batch.y[:batch_size])

        # Original code includes variance smoothing after warm-up epochs.
        ite_var = torch.var(y1[:batch_size] - y0[:batch_size])
        if self._current_epoch >= self._smooth_start_epoch:
            smooth_coeff = torch.exp(-3 * ite_var).detach()
        else:
            smooth_coeff = torch.tensor(0.0, device=y_loss.device)
        smooth_loss = self._smooth_lambda * smooth_coeff * ite_var

        total_loss = y_loss + smooth_loss
        total_loss.backward()

        # Gradient clipping
        if self._grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clip)

        self.optimizer_enc.step()
        self.optimizer_est.step()
        self.scheduler_enc.step()
        self.scheduler_est.step()

        if self._log_this_epoch:
            wandb.log({
                "epoch": self._current_epoch,
                "train_y_loss": y_loss.cpu().item(),
                "train_smooth_loss": smooth_loss.cpu().item(),
                "train_total_loss": total_loss.cpu().item(),
            })

    def train_model(self, train_data, val_data, test=False):
        """IDENet training loop with original-style early stopping behavior."""
        self.alpha = self.config["alpha"]
        self.gamma = self.config["gamma"]

        learning_rate = self.config["learning_rate"]
        epochs = self.config["num_epochs"]
        batch_size = self.config["batch_size"]

        if batch_size == -1:
            batch_size = train_data.x.shape[0]
        if test:
            wandb.log({
                "T_train": train_data.t.sum().cpu().item(),
                "T_val": val_data.t.sum().cpu().item(),
            })

        mean_y_train = train_data.y.mean()
        std_y_train = train_data.y.std()
        train_data.y = Normalize_outcome(train_data.y, mean_y_train, std_y_train)
        val_data.y = Normalize_outcome(val_data.y, mean_y_train, std_y_train)
        train_data.cf_y = Normalize_outcome(train_data.cf_y, mean_y_train, std_y_train)
        val_data.cf_y = Normalize_outcome(val_data.cf_y, mean_y_train, std_y_train)

        try:
            loader = NeighborLoader(
                train_data,
                num_neighbors=[-1],
                batch_size=batch_size,
                shuffle=True,
                subgraph_type="induced",
            )
            _test_iter = iter(loader)
            next(_test_iter)
            del _test_iter
            loader = NeighborLoader(
                train_data,
                num_neighbors=[-1],
                batch_size=batch_size,
                shuffle=True,
                subgraph_type="induced",
            )
        except (ImportError, Exception):
            loader = [train_data]

        if self.device:
            self.model.to("cuda")
            val_data.to("cuda")
            train_data.to("cuda")
        else:
            self.model.to("cpu")
            val_data.to("cpu")
            train_data.to("cpu")

        self.criterion = torch.nn.MSELoss()
        self.bce_loss = torch.nn.BCELoss()

        self._create_optimizers(learning_rate, self.config["weight_decay"])

        # Cache train exposure once when it is static across all training steps.
        # This is only valid for exposure_type=1 in full-batch training.
        self._use_cached_train_z_IDENet = (
            self.model.exposure_type == 1
            and self._cache_train_z_IDENet
            and batch_size == train_data.x.shape[0]
        )
        if self._use_cached_train_z_IDENet:
            with torch.no_grad():
                self._cached_train_z_IDENet = self.model.compute_z_IDENet(
                    train_data.t,
                    train_data.edge_index,
                    num_nodes=train_data.x.shape[0],
                )
        else:
            self._cached_train_z_IDENet = None

        log_every = max(1, epochs // 20)
        best_loss = float("inf")
        patience = 0

        self.model.train()
        for epoch in range(epochs):
            self._current_epoch = epoch
            self._log_this_epoch = (epoch % log_every == 0) or (epoch == epochs - 1)
            for batch in loader:
                self.model.train()
                self.model.zero_grad()
                self._train_step(batch, batch_size)

            with torch.no_grad():
                self.model.eval()
                _, y_val_out = self._predict(val_data)
                val_loss = self.criterion(y_val_out.squeeze(1), val_data.y).cpu().item()

            if epoch > self._early_stop_warmup and val_loss < best_loss:
                best_loss = val_loss
                patience = 0
            else:
                patience += 1

            if self.config["track_loss"]:
                self._log_validation_during_training(
                    val_data, train_data, mean_y_train, std_y_train, epoch
                )

            if patience >= self._early_stop_patience and epoch >= self._early_stop_min_epoch:
                break

        return self._evaluate(train_data, val_data, mean_y_train, std_y_train, test)

    # ------------------------------------------------------------------
    # Prediction helpers
    #
    # IDENet uses a T-learner: the y0/y1 head is selected by t via
    # torch.where(t > 0, y1, y0).  z_IDENet is computed internally from
    # adjacency and treatment so it stays consistent under interventions.
    # ------------------------------------------------------------------

    def _predict_idenet(
        self,
        x,
        t,
        edge_index,
        edge_attr_IDE=None,
        z_IDENet=None,
    ):
        """Single helper used by all IDENet prediction entry points."""
        return self.model(
            x,
            t,
            z_IDENet=z_IDENet,
            edge_index=edge_index,
            edge_attr_IDE=edge_attr_IDE,
        )

    def _predict(self, data):
        t_pred, y_pred = self._predict_idenet(
            data.x,
            data.t,
            data.edge_index,
            edge_attr_IDE=getattr(data, "edge_attr_IDE", None),
            z_IDENet=None,
        )
        return t_pred, y_pred

    def _predict_with_t(self, data, t, z):
        """Predict under a hypothetical treatment vector t.

        z_IDENet is recomputed inside the model from (edge_index, t), matching
        the original IDE-Net behavior.
        """
        del z  # Not used for IDENet.
        t_pred, y_pred = self._predict_idenet(
            data.x,
            t,
            data.edge_index,
            edge_attr_IDE=getattr(data, "edge_attr_IDE", None),
            z_IDENet=None,
        )
        return t_pred, y_pred

    def _predict_cf(self, data):
        """Predict with counterfactual treatment, recomputing z from cf_t."""
        return self._predict_with_t(data, data.cf_t, data.z)

    def _raw_predict(self, x, t, z, edge_index, edge_attr_IDE=None):
        """Low-level prediction. Called by _make_predict_fn for PEHNE/CNEE."""
        del z  # IDENet computes z_IDENet from treatment and graph.
        t_pred, y_pred = self._predict_idenet(
            x,
            t,
            edge_index,
            edge_attr_IDE=edge_attr_IDE,
            z_IDENet=None,
        )
        return t_pred, y_pred
