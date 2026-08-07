import numpy as np
import torch
from torch_geometric.loader import NeighborLoader, DataLoader

import wandb
from src.models import create_model
from src.utils.utils import Normalize_outcome, Normalize_outcome_recover
from src.evaluation.metrics import CNEE, PEHNE


class BaseTrainer:
    """Base class for all model trainers.

    Implements the shared training pipeline: hyperparameter tuning, alpha selection,
    multi-seed evaluation, data normalization, and metric computation.

    Subclasses must implement:
        - alpha_tuning_enabled (property): whether this model type does alpha tuning
        - default_gamma (property): default gamma value for this model type
        - uses_dynamic_z (property): whether metrics should recompute z from treatment vector
        - _create_optimizers(lr, weight_decay): set up optimizer(s)
        - _train_step(batch, batch_size): one batch forward + backward + step
        - _predict(data): model inference returning (t_pred, y_pred)
    """

    def __init__(self, config, train_data, val_data, test_data, device=True):
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = config.as_dict()
        if device:
            self.train_data = train_data.cuda()
            self.val_data = val_data.cuda()
            self.test_data = test_data.cuda()
        self.device = device
        self.model = None
        self.model_type = self.config["model_type"]

    # ========================================================================
    # Abstract interface — subclasses must implement these
    # ========================================================================

    @property
    def alpha_tuning_enabled(self):
        raise NotImplementedError

    @property
    def default_gamma(self):
        raise NotImplementedError

    @property
    def uses_dynamic_z(self):
        """Whether metrics should recompute z from the treatment vector."""
        return False

    def _create_optimizers(self, lr, weight_decay):
        raise NotImplementedError

    def _train_step(self, batch, batch_size):
        raise NotImplementedError

    def _predict(self, data):
        """Run model inference. Returns (t_pred, y_pred)."""
        raise NotImplementedError

    # ========================================================================
    # Model creation
    # ========================================================================

    def _create_model(self, hidden, dropout):
        return create_model(
            self.model_type,
            covariate_dim=self.config["covariate_dim"],
            hidden=hidden,
            dropout=dropout,
            tr_knots=self.config.get("tr_knots", 0.1),
            gnn_layer=self.config.get("gnn_layer", "GIN"),
            idenet_exposure_type=self.config.get("idenet_exposure_type", 2),
            idenet_edge_dim=self.config.get("idenet_edge_dim", 1),
            idenet_exposure_layers=self.config.get("idenet_exposure_layers", 2),
            idenet_vanilla=self.config.get("idenet_vanilla", False),
            idenet_edge_hidden=self.config.get("idenet_edge_hidden", 4),
            idenet_feature_layers=self.config.get("idenet_feature_layers", 2),
        )

    # ========================================================================
    # Outer pipeline
    # ========================================================================

    def train_test_best_model(self, epochs_range, lr_range, alpha_range, hidden_range, num_seeds):
        if self.alpha_tuning_enabled:
            best_val_loss, best_epoch, best_lr, best_alpha, best_hidden, best_dropout = (
                self.hyperparameter_tuning(epochs_range, lr_range, [0], hidden_range,
                                           dropout_range=self.config["dropout_range"])
            )
            best_alpha = self.select_alpha(best_epoch, best_lr, best_hidden, best_dropout, alpha_range)
        else:
            best_alpha = 0
            if self.model_type == "TargetedModel_DoubleBSpline":
                best_alpha = self.config["alpha"]
                print("Using fixed alpha for TargetedModel_DoubleBSpline:", best_alpha)
            best_val_loss, best_epoch, best_lr, best_alpha, best_hidden, best_dropout = (
                self.hyperparameter_tuning(epochs_range, lr_range, [best_alpha], hidden_range,
                                           dropout_range=self.config["dropout_range"])
            )

        wandb.log({
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "best_lr": best_lr,
            "best_alpha": best_alpha,
            "best_hidden": best_hidden,
            "best_dropout": best_dropout,
        })
        self.config["num_epochs"] = best_epoch
        self.config["learning_rate"] = best_lr
        self.config["alpha"] = best_alpha
        self.config["hidden"] = best_hidden
        self.config["dropout"] = best_dropout

        test_y_loss_list = []
        test_pehne_list = []
        test_cf_y_loss_list = []
        ITTE_loss_list = []
        cnee_list = []

        for seed in range(num_seeds):
            train_seed = seed + self.config["seed"] + 1
            torch.manual_seed(train_seed)
            np.random.seed(train_seed)
            self.model = self._create_model(best_hidden, best_dropout)

            test_y_loss, test_pehne, test_cf_y_loss, ITTE_loss, cnee = self.train_model(
                self.train_data, self.test_data, test=True
            )
            test_y_loss_list.append(test_y_loss)
            test_pehne_list.append(test_pehne)
            test_cf_y_loss_list.append(test_cf_y_loss)
            ITTE_loss_list.append(ITTE_loss)
            cnee_list.append(cnee)

        avg_test_y_loss = np.mean(test_y_loss_list)
        avg_pehne = np.mean(test_pehne_list)
        avg_test_cf_y_loss = np.mean(test_cf_y_loss_list)
        avg_ITTE_loss = np.mean(ITTE_loss_list)
        avg_cnee = np.mean(cnee_list)

        print("avg_test_y_loss", avg_test_y_loss)
        print("avg_PEHNE", avg_pehne)
        print("avg_test_cf_y_loss", avg_test_cf_y_loss)
        print("avg_ITTE_loss", avg_ITTE_loss)
        print("avg_cnee", avg_cnee)
        wandb.log({
            "avg_test_y_loss": avg_test_y_loss,
            "avg_pehne": avg_pehne,
            "avg_test_cf_y_loss": avg_test_cf_y_loss,
            "avg_ITTE_loss": avg_ITTE_loss,
            "avg_cnee": avg_cnee,
            "st_dev_test_y_loss": np.std(test_y_loss_list),
            "st_dev_test_pehne": np.std(test_pehne_list),
            "st_dev_test_cf_y_loss": np.std(test_cf_y_loss_list),
            "st_dev_ITTE_loss": np.std(ITTE_loss_list),
            "st_dev_cnee": np.std(cnee_list),
        })
        return avg_test_y_loss, avg_pehne, avg_test_cf_y_loss, avg_ITTE_loss, avg_cnee

    # ========================================================================
    # Alpha selection
    # ========================================================================

    def select_alpha(self, epochs, lr, hidden, dropout, alpha_range):
        p = self.config["p_alpha"]
        n_alpha_tune = max(1, int(self.config.get("n_alpha_tune", 2)))
        loss_dict = {}
        self.config["num_epochs"] = epochs
        self.config["learning_rate"] = lr
        self.config["hidden"] = hidden
        self.config["dropout"] = dropout

        for alpha in alpha_range:
            self.config["alpha"] = alpha
            val_y_loss_runs = []

            for run_idx in range(n_alpha_tune):
                tune_seed = self.config["seed"] + run_idx
                torch.manual_seed(tune_seed)
                np.random.seed(tune_seed)

                self.model = self._create_model(hidden, dropout)
                val_y_loss = self.train_model(self.train_data, self.val_data, test=False)[0]
                val_y_loss_runs.append(val_y_loss)
                print(
                    "alpha", alpha,
                    "run", run_idx + 1,
                    "of", n_alpha_tune,
                    "val_y_loss", val_y_loss,
                )

            avg_val_y_loss = float(np.mean(val_y_loss_runs))
            print("alpha", alpha, "avg_val_y_loss", avg_val_y_loss)
            loss_dict[alpha] = avg_val_y_loss

        alpha_list = list(loss_dict.keys())
        loss_list = list(loss_dict.values())
        if len(alpha_list) == 1:
            return alpha_list[0]
        max_loss = (1 + p) * loss_list[0]
        for i in range(len(loss_list)):
            if loss_list[i] > max_loss:
                best_alpha = alpha_list[i - 1]
                print("best_alpha", best_alpha)
                return best_alpha
        return alpha_list[-1]

    # ========================================================================
    # Hyperparameter tuning
    # ========================================================================

    def hyperparameter_tuning(self, epoch_range, lr_range, alpha_range, hidden_range, dropout_range):
        best_val_loss = np.inf
        best_epoch = 0
        best_lr = 0
        best_alpha = 0
        best_hidden = 0
        best_dropout = 0

        for epoch in epoch_range:
            for lr in lr_range:
                for alpha in alpha_range:
                    for hidden in hidden_range:
                        for dropout in dropout_range:
                            self.config["hidden"] = hidden
                            self.config["num_epochs"] = epoch
                            self.config["learning_rate"] = lr
                            self.config["alpha"] = alpha
                            self.config["dropout"] = dropout

                            torch.manual_seed(self.config["seed"])
                            np.random.seed(self.config["seed"])
                            self.model = self._create_model(hidden, dropout)

                            val_y_loss = self.train_model(self.train_data, self.val_data, test=False)[0]
                            print("configuration", "epoch", epoch, "lr", lr,
                                  "alpha", alpha, "hidden", hidden, "dropout", dropout)
                            print("val_y_loss", val_y_loss, "best_val_loss", best_val_loss)

                            if val_y_loss < best_val_loss:
                                best_val_loss = val_y_loss
                                best_epoch = epoch
                                best_lr = lr
                                best_alpha = alpha
                                best_hidden = hidden
                                best_dropout = dropout

        print("best_val_loss", best_val_loss, "best_epoch", best_epoch,
              "best_lr", best_lr, "best_alpha", best_alpha,
              "best_hidden", best_hidden, "best_dropout", best_dropout)
        return best_val_loss, best_epoch, best_lr, best_alpha, best_hidden, best_dropout

    # ========================================================================
    # Core training loop
    # ========================================================================

    def train_model(self, train_data, val_data, test=False):
        self.alpha = self.config["alpha"]
        self.gamma = self.config["gamma"]
        if self.default_gamma is not None:
            self.gamma = self.default_gamma

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

        # Normalize outcomes
        mean_y_train = train_data.y.mean()
        std_y_train = train_data.y.std()
        train_data.y = Normalize_outcome(train_data.y, mean_y_train, std_y_train)
        val_data.y = Normalize_outcome(val_data.y, mean_y_train, std_y_train)
        train_data.cf_y = Normalize_outcome(train_data.cf_y, mean_y_train, std_y_train)
        val_data.cf_y = Normalize_outcome(val_data.cf_y, mean_y_train, std_y_train)

        # Use NeighborLoader if pyg-lib/torch-sparse is available, otherwise fall back
        use_neighbor_loader = True
        try:
            loader = NeighborLoader(
                train_data,
                num_neighbors=[-1],
                batch_size=batch_size,
                shuffle=True,
                subgraph_type="induced",
            )
            # Test if the loader actually works (pyg-lib check)
            _test_iter = iter(loader)
            next(_test_iter)
            del _test_iter
            # Recreate loader since we consumed one batch
            loader = NeighborLoader(
                train_data,
                num_neighbors=[-1],
                batch_size=batch_size,
                shuffle=True,
                subgraph_type="induced",
            )
        except (ImportError, Exception):
            use_neighbor_loader = False
            # Fallback: use the entire graph as a single batch
            loader = [train_data]

        if self.device:
            self.model.to("cuda")

        # Set up loss functions
        self.criterion = torch.nn.MSELoss()
        self.bce_loss = torch.nn.BCELoss()
        val_data.to("cuda")
        train_data.to("cuda")

        # Create optimizers (subclass-specific)
        self._create_optimizers(learning_rate, self.config["weight_decay"])

        # Training loop
        log_every = max(1, epochs // 20)  # log at most 20 times per training run
        self.model.train()
        for epoch in range(epochs):
            self._current_epoch = epoch
            self._log_this_epoch = (epoch % log_every == 0) or (epoch == epochs - 1)
            for batch in loader:
                self.model.train()
                self.model.zero_grad()
                self._train_step(batch, batch_size)

            # Validation tracking during training
            if self.config["track_loss"]:
                self._log_validation_during_training(val_data, train_data, mean_y_train, std_y_train, epoch)

        # Post-training evaluation
        return self._evaluate(train_data, val_data, mean_y_train, std_y_train, test)

    # ========================================================================
    # Post-training evaluation
    # ========================================================================

    def _evaluate(self, train_data, val_data, mean_y_train, std_y_train, test):
        criterion = self.criterion
        bce_loss = self.bce_loss

        self.model.eval()
        with torch.no_grad():
            # Validation predictions
            t_val_out, y_val_out = self._predict(val_data)
            y_val_out = Normalize_outcome_recover(y_val_out, mean_y_train, std_y_train)
            val_data.y = Normalize_outcome_recover(val_data.y, mean_y_train, std_y_train)

            val_y_loss = criterion(y_val_out.squeeze(), val_data.y).cpu().item()
            val_t_loss = bce_loss(t_val_out, val_data.t.unsqueeze(1)).cpu().item()

            # Counterfactual validation predictions
            cf_t_val_out, cf_y_val_out = self._predict_cf(val_data)
            cf_y_val_out = Normalize_outcome_recover(cf_y_val_out, mean_y_train, std_y_train)
            val_data.cf_y = Normalize_outcome_recover(val_data.cf_y, mean_y_train, std_y_train)
            val_cf_y_loss = criterion(cf_y_val_out, val_data.cf_y.unsqueeze(1)).cpu().item()
            val_cf_t_loss = bce_loss(cf_t_val_out, val_data.cf_t.unsqueeze(1)).cpu().item()

            # Train predictions
            t_train_out, y_train_out = self._predict(train_data)
            cf_t_train_out, cf_y_train_out = self._predict_cf(train_data)
            train_y_out = Normalize_outcome_recover(y_train_out, mean_y_train, std_y_train)
            train_data.y = Normalize_outcome_recover(train_data.y, mean_y_train, std_y_train)
            train_cf_y_out = Normalize_outcome_recover(cf_y_train_out, mean_y_train, std_y_train)
            train_data.cf_y = Normalize_outcome_recover(train_data.cf_y, mean_y_train, std_y_train)

            train_y_loss = criterion(train_y_out.squeeze(), train_data.y).cpu().item()
            train_t_loss = bce_loss(t_train_out, train_data.t.unsqueeze(1)).cpu().item()
            cf_y_train_loss = criterion(train_cf_y_out, train_data.cf_y.unsqueeze(1)).cpu().item()
            cf_t_train_loss = bce_loss(cf_t_train_out, train_data.cf_t.unsqueeze(1)).cpu().item()

            # ITTE computation
            val_ITTE = val_data.ITTE
            t_0 = torch.zeros_like(val_data.t)
            _, y_val_0 = self._predict_with_t(val_data, t_0, val_data.z)
            val_ITTE_pred = cf_y_val_out.reshape(-1) - Normalize_outcome_recover(
                y_val_0.squeeze(1), mean_y_train, std_y_train
            )

            train_ITTE = train_data.ITTE
            t_0 = torch.zeros_like(train_data.t)
            _, y_train_0 = self._predict_with_t(train_data, t_0, train_data.z)
            train_ITTE_pred = train_cf_y_out.reshape(-1) - Normalize_outcome_recover(
                y_train_0.squeeze(1), mean_y_train, std_y_train
            )

            ITTE_loss_val = criterion(val_ITTE_pred, val_ITTE).cpu().item()
            ITTE_loss_train = criterion(train_ITTE_pred, train_ITTE).cpu().item()

            # X_random effect analysis
            X_randomTrain = train_data.X_random
            X_randomVal = val_data.X_random

            self.model.eval()
            train_y_pred = train_y_out.squeeze()
            _, train_random_y_pred = self._predict_with_x(train_data, X_randomTrain)
            train_random_y_pred = Normalize_outcome_recover(train_random_y_pred.squeeze(), mean_y_train, std_y_train)

            y_diff = criterion(train_y_pred, train_random_y_pred).cpu().item()
            train_y_out_treated = train_y_pred[train_data.t == 1]
            train_random_y_out_treated = train_random_y_pred[train_data.t == 1]
            train_y_out_untreated = train_y_pred[train_data.t == 0]
            train_random_y_out_untreated = train_random_y_pred[train_data.t == 0]

            y_diff_actual = train_data.PO_random - train_data.y
            y_diff_actual_treated = train_data.PO_random[train_data.t == 1] - train_data.y[train_data.t == 1]
            y_diff_actual_untreated = train_data.PO_random[train_data.t == 0] - train_data.y[train_data.t == 0]
            y_diff_out = train_random_y_pred - train_y_pred
            y_diff_out_treated = train_random_y_out_treated - train_y_out_treated
            y_diff_out_untreated = train_random_y_out_untreated - train_y_out_untreated

            y_diff_normalvs_conf = criterion(y_diff_actual, y_diff_out).cpu().item()
            y_diff_normalvs_conf_treated = criterion(y_diff_actual_treated, y_diff_out_treated).cpu().item()
            y_diff_normalvs_conf_untreated = criterion(y_diff_actual_untreated, y_diff_out_untreated).cpu().item()

            if test:
                y_diff_treated = criterion(train_y_out_treated, train_random_y_out_treated).cpu().item()
                y_diff_untreated = criterion(train_y_out_untreated, train_random_y_out_untreated).cpu().item()

                val_y_pred = y_val_out.squeeze()
                _, val_random_y_pred = self._predict_with_x(val_data, X_randomVal)
                val_random_y_pred = Normalize_outcome_recover(val_random_y_pred.squeeze(), mean_y_train, std_y_train)

                val_y_out_treated = val_y_pred[val_data.t == 1]
                val_random_y_out_treated = val_random_y_pred[val_data.t == 1]
                val_y_out_untreated = val_y_pred[val_data.t == 0]
                val_random_y_out_untreated = val_random_y_pred[val_data.t == 0]

                val_y_diff = criterion(val_y_pred, val_random_y_pred).cpu().item()
                val_y_diff_treated = criterion(val_y_out_treated, val_random_y_out_treated).cpu().item()
                val_y_diff_untreated = criterion(val_y_out_untreated, val_random_y_out_untreated).cpu().item()

                predict_fn = self._make_predict_fn(mean_y_train, std_y_train)
                test_pehne = PEHNE(self.config, val_data, predict_fn, self.config["num_networks"],
                                   mean_y_train, std_y_train, self.uses_dynamic_z)
                train_pehne = PEHNE(self.config, train_data, predict_fn, self.config["num_networks"],
                                    mean_y_train, std_y_train, self.uses_dynamic_z)
                test_cnee = CNEE(self.config, val_data, predict_fn, self.config["num_networks"],
                                 mean_y_train, std_y_train, self.uses_dynamic_z)
                train_cnee = CNEE(self.config, train_data, predict_fn, self.config["num_networks"],
                                  mean_y_train, std_y_train, self.uses_dynamic_z)

                print("test_pehne", test_pehne, "train_pehne", train_pehne)
                print("test_cnee", test_cnee, "train_cnee", train_cnee)
                wandb.log({
                    "test_pehne": test_pehne,
                    "train_pehne": train_pehne,
                    "test_cnee": test_cnee,
                    "train_cnee": train_cnee,
                })

                wandb.log({
                    "val_y_loss_final": val_y_loss,
                    "val_t_loss_final": val_t_loss,
                    "val_cf_y_loss_final": val_cf_y_loss,
                    "val_cf_t_loss_final": val_cf_t_loss,
                    "train_y_loss_final": train_y_loss,
                    "train_t_loss_final": train_t_loss,
                    "train_cf_y_loss_final": cf_y_train_loss,
                    "train_cf_t_loss_final": cf_t_train_loss,
                    "ITTE_loss_val": ITTE_loss_val,
                    "ITTE_loss_train": ITTE_loss_train,
                    "train_y_diff": y_diff,
                    "train_y_diff_treated": y_diff_treated,
                    "train_y_diff_untreated": y_diff_untreated,
                    "val_y_diff": val_y_diff,
                    "val_y_diff_treated": val_y_diff_treated,
                    "val_y_diff_untreated": val_y_diff_untreated,
                    "train_y_diff_normalvs_conf": y_diff_normalvs_conf,
                    "train_y_diff_normalvs_conf_treated": y_diff_normalvs_conf_treated,
                    "train_y_diff_normalvs_conf_untreated": y_diff_normalvs_conf_untreated,
                })
                return val_y_loss, test_pehne, val_cf_y_loss, ITTE_loss_val, test_cnee

            return val_y_loss, val_t_loss, val_cf_y_loss, val_cf_t_loss

    # ========================================================================
    # Prediction helpers
    # ========================================================================

    def _predict_cf(self, data):
        """Predict with counterfactual treatment."""
        return self._predict_with_t(data, data.cf_t, data.z)

    def _predict_with_t(self, data, t, z):
        """Predict with custom treatment vector. Subclasses may override."""
        # Default: call model with custom t and z
        t_pred, y_pred = self._raw_predict(
            data.x,
            t,
            z,
            data.edge_index,
            edge_attr_IDE=getattr(data, "edge_attr_IDE", None),
        )
        return t_pred, y_pred

    def _predict_with_x(self, data, x):
        """Predict with custom features X."""
        return self._raw_predict(
            x,
            data.t,
            data.z,
            data.edge_index,
            edge_attr_IDE=getattr(data, "edge_attr_IDE", None),
        )

    def _raw_predict(self, x, t, z, edge_index, edge_attr_IDE=None):
        """Low-level prediction. Subclasses override for different model outputs."""
        out = self.model(x, t, z, edge_index)
        return out[0], out[1]

    def _make_predict_fn(self, mean_y_train, std_y_train):
        """Create a predict function for metrics. Returns y_pred (denormalized)."""
        def predict_fn(x, t, z, edge_index, edge_attr_IDE=None):
            _, y_pred = self._raw_predict(x, t, z, edge_index, edge_attr_IDE=edge_attr_IDE)
            y_pred = Normalize_outcome_recover(y_pred.squeeze(), mean_y_train, std_y_train)
            return y_pred
        return predict_fn

    # ========================================================================
    # Validation logging during training
    # ========================================================================

    def _log_validation_during_training(self, val_data, train_data, mean_y_train, std_y_train, epoch):
        self.model.eval()
        with torch.no_grad():
            t_val_out, y_val_out = self._predict(val_data)
            y_val_out = Normalize_outcome_recover(y_val_out, mean_y_train, std_y_train)
            val_data.y = Normalize_outcome_recover(val_data.y, mean_y_train, std_y_train)
            val_y_loss = self.criterion(y_val_out.squeeze(1), val_data.y).cpu().item()
            val_t_loss = self.bce_loss(t_val_out, val_data.t.unsqueeze(1)).cpu().item()

            cf_t_val_out, cf_y_val_out = self._predict_cf(val_data)
            cf_y_val_out = Normalize_outcome_recover(cf_y_val_out, mean_y_train, std_y_train)
            val_data.cf_y = Normalize_outcome_recover(val_data.cf_y, mean_y_train, std_y_train)
            val_cf_y_loss = self.criterion(cf_y_val_out, val_data.cf_y.unsqueeze(1)).cpu().item()
            val_cf_t_loss = self.bce_loss(cf_t_val_out, val_data.cf_t.unsqueeze(1)).cpu().item()

            wandb.log({
                "epoch": epoch + 1,
                "val_y_loss": val_y_loss,
                "val_t_loss": val_t_loss,
                "val_cf_y_loss": val_cf_y_loss,
                "val_cf_t_loss": val_cf_t_loss,
            })
