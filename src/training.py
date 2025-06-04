import numpy as np

# from re import S
import torch
from torch_geometric.loader import NeighborLoader

import src.utils.utils as utils
import wandb
from src.methods.Causal_models import (
    GCN_DECONF,
    GINModel,
    GINNetEst,
    HINet,
    HINet_no_net_conf,
    NetEst,
    SPNet,
    TARNet,
)
from src.utils.metrics import CNEE, PEHNE
from src.utils.utils import Normalize_outcome, Normalize_outcome_recover


class Trainer:
    def __init__(self, config, train_data, val_data, test_data, model, device=True):
        self.config = config.as_dict()
        if device:
            self.train_data = train_data.cuda()
            self.val_data = val_data.cuda()
            self.test_data = test_data.cuda()
        self.device = device
        self.model = model
        self.model_type = config["model_type"]

    def train_test_best_model(
        self, epochs_range, lr_range, alpha_range, hidden_range, num_seeds
    ):
        if (
            self.config["model_type"] == "HINet"
            or self.config["model_type"] == "HINet_no_net_conf"
            or self.config["model_type"] == "NetEst"
            or self.config["model_type"] == "GINNetEst"
        ):
            alpha_tuning = True
        else:
            alpha_tuning = False
            best_alpha = 0

        if not alpha_tuning:
            best_val_loss, best_epoch, best_lr, best_alpha, best_hidden = (
                self.hyperparameter_tuning(
                    epochs_range, lr_range, alpha_range, hidden_range
                )
            )
        else:
            best_val_loss, best_epoch, best_lr, best_alpha, best_hidden = (
                self.hyperparameter_tuning(epochs_range, lr_range, [0], hidden_range)
            )
            # find elbow
            best_alpha = self.select_alpha(
                best_epoch, best_lr, best_hidden, alpha_range
            )

        wandb.log(
            {
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "best_lr": best_lr,
                "best_alpha": best_alpha,
                "best_hidden": best_hidden,
            }
        )
        self.config["num_epochs"] = best_epoch
        self.config["learning_rate"] = best_lr
        self.config["alpha"] = best_alpha
        self.config["hidden"] = best_hidden
        sum_test_y_loss = 0
        sum_pehne = 0
        sum_test_cf_y_loss = 0
        sum_ITTE_loss = 0
        sum_cnee = 0

        test_y_loss_list = []
        test_pehne_list = []
        test_cf_y_loss_list = []
        ITTE_loss_list = []
        cnee_list = []

        for seed in range(num_seeds):
            train_seed = seed + self.config["seed"] + 1
            torch.manual_seed(train_seed)
            np.random.seed(train_seed)
            if self.model_type == "HINet":
                self.model = HINet(
                    Xshape=self.config["covariate_dim"], hidden=best_hidden
                )
            elif self.model_type == "NetEst":
                self.model = NetEst(
                    Xshape=self.config["covariate_dim"], hidden=best_hidden
                )
            elif self.model_type == "HINet_no_net_conf":
                self.model = HINet_no_net_conf(
                    Xshape=self.config["covariate_dim"], hidden=best_hidden
                )
            elif self.model_type == "GINModel":
                self.model = GINModel(
                    Xshape=self.config["covariate_dim"], hidden=best_hidden
                )
            elif self.model_type == "GINNetEst":
                self.model = GINNetEst(
                    Xshape=self.config["covariate_dim"], hidden=best_hidden
                )
            elif self.model_type == "TARNet":
                self.model = TARNet(
                    Xshape=self.config["covariate_dim"], hidden=best_hidden
                )
            elif self.model_type == "GCN_DECONF":
                self.model = GCN_DECONF(
                    Xshape=self.config["covariate_dim"], hidden=best_hidden
                )
            elif self.model_type == "SPNet":
                self.model = SPNet(
                    Xshape=self.config["covariate_dim"], hidden=best_hidden
                )
            test_y_loss, test_pehne, test_cf_y_loss, ITTE_loss, cnee = self.train_model(
                self.train_data, self.test_data, test=True
            )
            sum_test_y_loss += test_y_loss
            sum_pehne += test_pehne
            sum_test_cf_y_loss += test_cf_y_loss
            sum_ITTE_loss += ITTE_loss
            sum_cnee += cnee
            test_y_loss_list.append(test_y_loss)
            test_pehne_list.append(test_pehne)
            test_cf_y_loss_list.append(test_cf_y_loss)
            ITTE_loss_list.append(ITTE_loss)
            cnee_list.append(cnee)
        avg_test_y_loss = sum_test_y_loss / num_seeds
        avg_pehne = sum_pehne / num_seeds
        avg_test_cf_y_loss = sum_test_cf_y_loss / num_seeds
        avg_ITTE_loss = sum_ITTE_loss / num_seeds
        avg_cnee = sum_cnee / num_seeds
        st_dev_test_y_loss = np.std(test_y_loss_list)
        st_dev_pehne = np.std(test_pehne_list)
        st_dev_test_cf_y_loss = np.std(test_cf_y_loss_list)
        st_dev_ITTE_loss = np.std(ITTE_loss_list)
        st_dev_cnee = np.std(cnee_list)

        print("avg_test_y_loss", avg_test_y_loss)
        print("avg_PEHNE", avg_pehne)
        print("avg_test_cf_y_loss", avg_test_cf_y_loss)
        print("avg_ITTE_loss", avg_ITTE_loss)
        print("avg_cnee", avg_cnee)
        wandb.log(
            {
                "avg_test_y_loss": avg_test_y_loss,
                "avg_pehne": avg_pehne,
                "avg_test_cf_y_loss": avg_test_cf_y_loss,
                "avg_ITTE_loss": avg_ITTE_loss,
                "avg_cnee": avg_cnee,
                "st_dev_test_y_loss": st_dev_test_y_loss,
                "st_dev_test_pehne": st_dev_pehne,
                "st_dev_test_cf_y_loss": st_dev_test_cf_y_loss,
                "st_dev_ITTE_loss": st_dev_ITTE_loss,
                "st_dev_cnee": st_dev_cnee,
            }
        )
        return avg_test_y_loss, avg_pehne, avg_test_cf_y_loss, avg_ITTE_loss, avg_cnee

    def select_alpha(self, epochs, lr, hidden, alpha_range):
        p = self.config["p_alpha"]
        loss_dict = {}
        self.config["num_epochs"] = epochs
        self.config["learning_rate"] = lr
        self.config["hidden"] = hidden
        for alpha in alpha_range:
            torch.manual_seed(self.config["seed"])
            np.random.seed(self.config["seed"])
            self.config["alpha"] = alpha
            if self.model_type == "HINet":
                self.model = HINet(Xshape=self.config["covariate_dim"], hidden=hidden)
            elif self.model_type == "NetEst":
                self.model = NetEst(Xshape=self.config["covariate_dim"], hidden=hidden)
            elif self.model_type == "no_net_conf":
                self.model = HINet_no_net_conf(
                    Xshape=self.config["covariate_dim"], hidden=hidden
                )
            elif self.model_type == "GINModel":
                self.model = GINModel(
                    Xshape=self.config["covariate_dim"], hidden=hidden
                )
            elif self.model_type == "GINNetEst":
                self.model = GINNetEst(
                    Xshape=self.config["covariate_dim"], hidden=hidden
                )
            elif self.model_type == "TARNet":
                self.model = TARNet(Xshape=self.config["covariate_dim"], hidden=hidden)
            elif self.model_type == "GCN_DECONF":
                self.model = GCN_DECONF(
                    Xshape=self.config["covariate_dim"], hidden=hidden
                )
            elif self.model_type == "SPNet":
                self.model = SPNet(Xshape=self.config["covariate_dim"], hidden=hidden)

            val_y_loss = self.train_model(self.train_data, self.val_data, test=False)[0]
            print("alpha", alpha)
            print("val_y_loss", val_y_loss)
            loss_dict[alpha] = val_y_loss
            print("loss_dict", loss_dict)
        print("loss_dict", loss_dict)
        # visualize loss_dict with matplotlib
        alpha_list = list(loss_dict.keys())
        loss_list = list(loss_dict.values())
        if len(alpha_list) == 1:
            return alpha_list[0]
        max_loss = (1 + p) * loss_list[0]
        # find the highest alpha still smaller than max loss
        for i in range(len(loss_list)):
            if loss_list[i] > max_loss:
                best_alpha = alpha_list[i - 1]
                print("best_alpha", best_alpha)
                return best_alpha
        return alpha_list[-1]

    def train_model(self, train_data, val_data, test=False):
        # https://medium.com/stanford-cs224w/a-tour-of-pygs-data-loaders-9f2384e48f8f

        # normalize y values
        alpha = self.config["alpha"]
        gamma = self.config["gamma"]
        if self.model_type == "HINet" or self.model_type == "HINet_no_net_conf":
            gamma = 0
        learning_rate = self.config["learning_rate"]
        epochs = self.config["num_epochs"]
        batch_size = self.config["batch_size"]

        if batch_size == -1:
            batch_size = train_data.x.shape[0]
        if test:
            wandb.log(
                {
                    "T_train": train_data.t.sum().cpu().item(),
                    "T_val": val_data.t.sum().cpu().item(),
                }
            )
        # first normalize the outcomes for the datasets

        mean_y_train = train_data.y.mean()
        std_y_train = train_data.y.std()
        train_data.y = Normalize_outcome(train_data.y, mean_y_train, std_y_train)
        val_data.y = Normalize_outcome(val_data.y, mean_y_train, std_y_train)
        train_data.cf_y = Normalize_outcome(train_data.cf_y, mean_y_train, std_y_train)
        val_data.cf_y = Normalize_outcome(val_data.cf_y, mean_y_train, std_y_train)
        # if this does not work because pyg-lib is not installed, try the following loader instead:
        # loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)

        loader = NeighborLoader(
            train_data,
            num_neighbors=[-1],  # Include all neighbors at 1-hop
            batch_size=batch_size,  # Number of nodes per batch
            shuffle=True,
            subgraph_type="induced",
        )
        torch.autograd.set_detect_anomaly(True)
        if self.device:
            self.model.to("cuda")

        if self.model_type == "NetEst" or self.model_type == "GINNetEst":
            optimizer_t = torch.optim.Adam(
                self.model.discriminator.parameters(),
                lr=learning_rate,
                weight_decay=self.config["weight_decay"],
            )
            optimizer_z = torch.optim.Adam(
                self.model.discrimnator_z.parameters(),
                lr=learning_rate,
                weight_decay=self.config["weight_decay"],
            )
            optimizer_p = torch.optim.Adam(
                [
                    {"params": self.model.encoder.parameters()},
                    {"params": self.model.predictor.parameters()},
                ],
                lr=learning_rate,
                weight_decay=self.config["weight_decay"],
            )
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=self.config["weight_decay"],
            )
        criterion = torch.nn.MSELoss()
        BCE_loss = torch.nn.BCELoss()
        val_data.to("cuda")
        train_data.to("cuda")
        # Training loop
        self.model.train()
        print("Training started")
        print("alpha", alpha)
        print("gamma", gamma)
        print("epochs", epochs)
        print("hidden", self.config["hidden"])
        print("learning_rate", self.config["learning_rate"])
        for epoch in range(epochs):
            total_loss = 0
            batch_num = 0
            for batch in loader:
                batch_num += 1
                self.model.train()
                self.model.zero_grad()

                if self.model_type == "NetEst" or self.model_type == "GINNetEst":
                    optimizer_t.zero_grad()
                    optimizer_z.zero_grad()
                    optimizer_p.zero_grad()
                else:
                    optimizer.zero_grad()

                if self.model_type == "NetEst" or self.model_type == "GINNetEst":
                    t_pred, y_pred, z_pred = self.model(
                        batch.x, batch.t, batch.z, batch.edge_index
                    )
                elif self.model_type == "SPNet":
                    t_pred, y_pred, representations = self.model(
                        batch.x, batch.t, batch.z, batch.edge_index
                    )
                else:
                    t_pred, y_pred = self.model(
                        batch.x, batch.t, batch.z, batch.edge_index
                    )

                batch_y_loss = criterion(
                    y_pred[:batch_size].squeeze(1), batch.y[:batch_size]
                )
                batch_t_loss = BCE_loss(
                    t_pred[:batch_size].squeeze(1), batch.t[:batch_size]
                )

                batch_y_loss_train = batch_y_loss
                batch_t_loss_train = batch_t_loss

                # Calculate the losses
                if self.model_type == "NetEst" or self.model_type == "GINNetEst":
                    # here we go over all the different optimizers and do a backward pass
                    # backward all the losses
                    optimizer_t.zero_grad()
                    batch_t_loss.backward()
                    # self.optimize_t.zero_grad()
                    optimizer_t.step()
                    t_pred, y_pred, z_pred = self.model(
                        batch.x, batch.t, batch.z, batch.edge_index
                    )
                    batch_z_loss = BCE_loss(z_pred[:batch_size].squeeze(1), batch.z)
                    optimizer_z.zero_grad()
                    batch_z_loss.backward()
                    optimizer_z.step()
                    t_pred, y_pred, z_pred = self.model(
                        batch.x, batch.t, batch.z, batch.edge_index
                    )
                    batch_y_loss = criterion(
                        y_pred[:batch_size].squeeze(1), batch.y[:batch_size]
                    )
                    batch_t_loss = BCE_loss(
                        t_pred[:batch_size].squeeze(1), batch.t[:batch_size]
                    )
                    batch_z_loss = BCE_loss(z_pred[:batch_size].squeeze(1), batch.z)
                    total_loss = (
                        batch_y_loss - alpha * batch_t_loss - gamma * batch_z_loss
                    )
                    optimizer_p.zero_grad()
                    total_loss.backward()
                    optimizer_p.step()
                    wandb.log(
                        {
                            "batch_y_loss": batch_y_loss.cpu().item(),
                            "batch_t_loss": batch_t_loss.cpu().item(),
                            "batch_z_loss": batch_z_loss.cpu().item(),
                            "batch_total_loss": total_loss.cpu().item(),
                        }
                    )

                elif (
                    self.model_type == "HINet" or self.model_type == "HINet_no_net_conf"
                ):
                    total_loss = batch_y_loss_train + alpha * batch_t_loss_train
                    wandb.log(
                        {
                            "batch_y_loss": batch_y_loss.cpu().item(),
                            "batch_t_loss": batch_t_loss.cpu().item(),
                            "batch_total_loss": total_loss.cpu().item(),
                        }
                    )
                    total_loss.backward()
                    optimizer.step()
                elif self.model_type == "SPNet":
                    rep_t1, rep_t0 = (
                        representations[(batch.t > 0).nonzero()],
                        representations[(batch.t < 1).nonzero()],
                    )
                    dLoss, _ = utils.wasserstein(rep_t1, rep_t0)
                    total_loss = batch_y_loss + alpha * batch_t_loss + gamma * dLoss
                    total_loss.backward()
                    optimizer.step()
                    wandb.log(
                        {
                            "batch_y_loss": batch_y_loss.cpu().item(),
                            "batch_t_loss": batch_t_loss.cpu().item(),
                            "dLoss": dLoss.cpu().item(),
                            "batch_total_loss": total_loss.cpu().item(),
                        }
                    )
                elif (
                    self.model_type == "GINModel"
                    or self.model_type == "TARNet"
                    or self.model_type == "GCN_DECONF"
                ):
                    optimizer.zero_grad
                    total_loss = batch_y_loss
                    wandb.log(
                        {
                            "batch_y_loss": batch_y_loss.cpu().item(),
                            "batch_t_loss": batch_t_loss.cpu().item(),
                            "batch_total_loss": total_loss.cpu().item(),
                        }
                    )
                    total_loss.backward()
                    optimizer.step()

            # calculate val_loss:
            if self.config["track_loss"]:
                self.model.eval()
                with torch.no_grad():
                    if self.model_type == "NetEst" or self.model_type == "GINNetEst":
                        t_val_out, y_val_out, z_val_out = self.model(
                            val_data.x, val_data.t, val_data.z, val_data.edge_index
                        )
                    elif self.model_type == "SPNet":
                        t_val_out, y_val_out, representations_val = self.model(
                            val_data.x, val_data.t, val_data.z, val_data.edge_index
                        )
                    else:
                        t_val_out, y_val_out = self.model(
                            val_data.x, val_data.t, val_data.z, val_data.edge_index
                        )
                    # renormalize
                    y_val_out_squeeze = y_val_out.squeeze(1)
                    y_val_out_squeeze = Normalize_outcome_recover(
                        y_val_out_squeeze, mean_y_train, std_y_train
                    )
                    y_val_out = Normalize_outcome_recover(
                        y_val_out, mean_y_train, std_y_train
                    )
                    val_data.y = Normalize_outcome_recover(
                        val_data.y, mean_y_train, std_y_train
                    )
                    val_y_loss = (
                        criterion(y_val_out.squeeze(1), val_data.y).cpu().item()
                    )
                    val_t_loss = (
                        BCE_loss(t_val_out, val_data.t.unsqueeze(1)).cpu().item()
                    )

                    # cf loss
                    if self.model_type == "NetEst" or self.model_type == "GINNetEst":
                        cf_t_val_out, cf_y_val_out, cf_z_val_out = self.model(
                            val_data.x, val_data.cf_t, val_data.z, val_data.edge_index
                        )
                    elif self.model_type == "SPNet":
                        cf_t_val_out, cf_y_val_out, representations_val = self.model(
                            val_data.x, val_data.cf_t, val_data.z, val_data.edge_index
                        )
                    else:
                        cf_t_val_out, cf_y_val_out = self.model(
                            val_data.x, val_data.cf_t, val_data.z, val_data.edge_index
                        )
                    cf_y_val_out = Normalize_outcome_recover(
                        cf_y_val_out, mean_y_train, std_y_train
                    )
                    val_data.cf_y = Normalize_outcome_recover(
                        val_data.cf_y, mean_y_train, std_y_train
                    )
                    val_cf_y_loss = (
                        criterion(cf_y_val_out, val_data.cf_y.unsqueeze(1)).cpu().item()
                    )
                    val_cf_t_loss = (
                        BCE_loss(cf_t_val_out, val_data.cf_t.unsqueeze(1)).cpu().item()
                    )
                    # print losses
                    print(
                        "epoch",
                        epoch,
                        "batch_num",
                        batch_num,
                        "val_y_loss",
                        val_y_loss,
                        "val_t_loss",
                        val_t_loss,
                        "val_cf_y_loss",
                        val_cf_y_loss,
                        "val_cf_t_loss",
                        val_cf_t_loss,
                    )

                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "val_y_loss": val_y_loss,
                            "train_y_loss": train_y_loss,
                            "val_t_loss": val_t_loss,
                            "train_t_loss": train_t_loss,
                            "val_cf_y_loss": val_cf_y_loss,
                            "val_cf_t_loss": val_cf_t_loss,
                            "batch_num": batch_num,
                        }
                    )

        self.model.eval()
        print("Training done")
        with torch.no_grad():
            if self.model_type == "NetEst" or self.model_type == "GINNetEst":
                t_val_out, y_val_out, z_val_out = self.model(
                    val_data.x, val_data.t, val_data.z, val_data.edge_index
                )
            elif self.model_type == "SPNet":
                t_val_out, y_val_out, representations_val = self.model(
                    val_data.x, val_data.t, val_data.z, val_data.edge_index
                )
            else:
                t_val_out, y_val_out = self.model(
                    val_data.x, val_data.t, val_data.z, val_data.edge_index
                )
            # renormalize
            print("shape", y_val_out.shape)
            print("mean", y_val_out.mean(), "std", y_val_out.std())
            y_val_out_squeeze = y_val_out.squeeze(1)
            y_val_out_squeeze = Normalize_outcome_recover(
                y_val_out_squeeze, mean_y_train, std_y_train
            )
            y_val_out = Normalize_outcome_recover(y_val_out, mean_y_train, std_y_train)
            print("mean", y_val_out.mean(), "std", y_val_out.std())
            print("shape", y_val_out.shape)
            print(
                "squeezed mean",
                y_val_out_squeeze.mean(),
                "squeezed std",
                y_val_out_squeeze.std(),
            )
            print("shape", y_val_out_squeeze.shape)
            val_data.y = Normalize_outcome_recover(
                val_data.y, mean_y_train, std_y_train
            )
            print("real data", val_data.y)
            print("shape", val_data.y.shape)

            val_y_loss = criterion(y_val_out.squeeze(1), val_data.y).cpu().item()
            val_t_loss = BCE_loss(t_val_out, val_data.t.unsqueeze(1)).cpu().item()
            # neighbor loss

            # calculate CF losses:
            if self.model_type == "NetEst" or self.model_type == "GINNetEst":
                cf_t_val_out, cf_y_val_out, cf_z_val_out = self.model(
                    val_data.x, val_data.cf_t, val_data.z, val_data.edge_index
                )
            elif self.model_type == "SPNet":
                cf_t_val_out, cf_y_val_out, representations_val = self.model(
                    val_data.x, val_data.cf_t, val_data.z, val_data.edge_index
                )
            else:
                cf_t_val_out, cf_y_val_out = self.model(
                    val_data.x, val_data.cf_t, val_data.z, val_data.edge_index
                )
            cf_y_val_out = Normalize_outcome_recover(
                cf_y_val_out, mean_y_train, std_y_train
            )
            val_data.cf_y = Normalize_outcome_recover(
                val_data.cf_y, mean_y_train, std_y_train
            )
            val_cf_y_loss = (
                criterion(cf_y_val_out, val_data.cf_y.unsqueeze(1)).cpu().item()
            )
            val_cf_t_loss = (
                BCE_loss(cf_t_val_out, val_data.cf_t.unsqueeze(1)).cpu().item()
            )

            # neighbor loss

            if self.model_type == "NetEst" or self.model_type == "GINNetEst":
                train_t_out, train_y_out, train_z_out = self.model(
                    train_data.x, train_data.t, train_data.z, train_data.edge_index
                )
                train_cf_t_out, train_cf_y_out, train_cf_z_out = self.model(
                    train_data.x, train_data.cf_t, train_data.z, train_data.edge_index
                )
            elif self.model_type == "SPNet":
                train_t_out, train_y_out, representations_train = self.model(
                    train_data.x, train_data.t, train_data.z, train_data.edge_index
                )
                train_cf_t_out, train_cf_y_out, representations_train = self.model(
                    train_data.x, train_data.cf_t, train_data.z, train_data.edge_index
                )
            else:
                train_t_out, train_y_out = self.model(
                    train_data.x, train_data.t, train_data.z, train_data.edge_index
                )
                train_cf_t_out, train_cf_y_out = self.model(
                    train_data.x, train_data.cf_t, train_data.z, train_data.edge_index
                )
            train_y_out = Normalize_outcome_recover(
                train_y_out, mean_y_train, std_y_train
            )
            train_data.y = Normalize_outcome_recover(
                train_data.y, mean_y_train, std_y_train
            )
            train_cf_y_out = Normalize_outcome_recover(
                train_cf_y_out, mean_y_train, std_y_train
            )
            train_data.cf_y = Normalize_outcome_recover(
                train_data.cf_y, mean_y_train, std_y_train
            )
            train_y_loss = criterion(train_y_out.squeeze(1), train_data.y).cpu().item()
            train_t_loss = BCE_loss(train_t_out, train_data.t.unsqueeze(1)).cpu().item()

            cf_y_train_loss = (
                criterion(train_cf_y_out, train_data.cf_y.unsqueeze(1)).cpu().item()
            )
            cf_t_train_loss = (
                BCE_loss(train_cf_t_out, train_data.cf_t.unsqueeze(1)).cpu().item()
            )

            val_ITTE = val_data.ITTE

            t_0 = torch.zeros_like(val_data.t)

            val_ITTE_pred = cf_y_val_out.reshape(-1) - Normalize_outcome_recover(
                self.model(val_data.x, t_0, val_data.z, val_data.edge_index)[1].squeeze(
                    1
                ),
                mean_y_train,
                std_y_train,
            )

            train_ITTE = train_data.ITTE

            t_0 = torch.zeros_like(train_data.t)
            train_ITTE_pred = train_cf_y_out.reshape(-1) - Normalize_outcome_recover(
                self.model(train_data.x, t_0, train_data.z, train_data.edge_index)[
                    1
                ].squeeze(1),
                mean_y_train,
                std_y_train,
            )

            # calculate MSE loss
            ITTE_loss_val = criterion(val_ITTE_pred, val_ITTE).cpu().item()
            ITTE_loss_train = criterion(train_ITTE_pred, train_ITTE).cpu().item()

            print("train loss y", train_y_loss, "train loss t", train_t_loss)
            print(
                "train_cf loss y", cf_y_train_loss, "train_cf loss t", cf_t_train_loss
            )

            X_randomTrain = train_data.X_random
            PO_randomTrain = train_data.PO_random
            X_randomVal = val_data.X_random
            PO_randomVal = val_data.PO_random

            # First check whether X has an effect on Y according to the model
            train_y_out = self.model(
                train_data.x, train_data.t, train_data.z, train_data.edge_index
            )[1].squeeze(1)
            train_y_out = Normalize_outcome_recover(
                train_y_out, mean_y_train, std_y_train
            )
            train_random_y_out = self.model(
                X_randomTrain, train_data.t, train_data.z, train_data.edge_index
            )[1].squeeze(1)
            train_random_y_out = Normalize_outcome_recover(
                train_random_y_out, mean_y_train, std_y_train
            )
            # Check difference between the two
            y_diff = criterion(train_y_out, train_random_y_out).cpu().item()
            # isolate the effect for treated and untreated
            train_y_out_treated = train_y_out[train_data.t == 1]
            train_random_y_out_treated = train_random_y_out[train_data.t == 1]
            train_y_out_untreated = train_y_out[train_data.t == 0]
            train_random_y_out_untreated = train_random_y_out[train_data.t == 0]
            # Check difference between the two

            y_diff_actual = train_data.PO_random - train_data.y
            y_diff_actual_treated = (
                train_data.PO_random[train_data.t == 1]
                - train_data.y[train_data.t == 1]
            )
            y_diff_actual_untreated = (
                train_data.PO_random[train_data.t == 0]
                - train_data.y[train_data.t == 0]
            )
            # Check difference between the two
            y_diff_out = train_random_y_out - train_y_out
            y_diff_out_treated = train_random_y_out_treated - train_y_out_treated
            y_diff_out_untreated = train_random_y_out_untreated - train_y_out_untreated
            # Check difference between the two
            y_diff_normalvs_conf = criterion(y_diff_actual, y_diff_out).cpu().item()
            y_diff_normalvs_conf_treated = (
                criterion(y_diff_actual_treated, y_diff_out_treated).cpu().item()
            )
            y_diff_normalvs_conf_untreated = (
                criterion(y_diff_actual_untreated, y_diff_out_untreated).cpu().item()
            )

            if test:
                y_diff_treated = (
                    criterion(train_y_out_treated, train_random_y_out_treated)
                    .cpu()
                    .item()
                )

                y_diff_untreated = (
                    criterion(train_y_out_untreated, train_random_y_out_untreated)
                    .cpu()
                    .item()
                )

                val_y_out = self.model(
                    val_data.x, val_data.t, val_data.z, val_data.edge_index
                )[1].squeeze(1)
                val_y_out = Normalize_outcome_recover(
                    val_y_out, mean_y_train, std_y_train
                )
                val_random_y_out = self.model(
                    X_randomVal, val_data.t, val_data.z, val_data.edge_index
                )[1].squeeze(1)
                val_random_y_out = Normalize_outcome_recover(
                    val_random_y_out, mean_y_train, std_y_train
                )

                val_y_out_treated = val_y_out[val_data.t == 1]
                val_random_y_out_treated = val_random_y_out[val_data.t == 1]
                val_y_out_untreated = val_y_out[val_data.t == 0]
                val_random_y_out_untreated = val_random_y_out[val_data.t == 0]

                # Check difference between the two
                val_y_diff = criterion(val_y_out, val_random_y_out).cpu().item()
                val_y_diff_treated = (
                    criterion(val_y_out_treated, val_random_y_out_treated).cpu().item()
                )
                val_y_diff_untreated = (
                    criterion(val_y_out_untreated, val_random_y_out_untreated)
                    .cpu()
                    .item()
                )

                test_pehne = PEHNE(
                    self.config,
                    val_data,
                    self.model,
                    self.config["num_networks"],
                    mean_y_train,
                    std_y_train,
                )
                train_pehne = PEHNE(
                    self.config,
                    train_data,
                    self.model,
                    self.config["num_networks"],
                    mean_y_train,
                    std_y_train,
                )
                print("test_pehne", test_pehne)
                print("train_pehne", train_pehne)
                test_cnee = CNEE(
                    self.config,
                    val_data,
                    self.model,
                    self.config["num_networks"],
                    mean_y_train,
                    std_y_train,
                )
                train_cnee = CNEE(
                    self.config,
                    train_data,
                    self.model,
                    self.config["num_networks"],
                    mean_y_train,
                    std_y_train,
                )
                print("test_cnee", test_cnee)
                print("train_cnee", train_cnee)
                wandb.log(
                    {
                        "test_pehne": test_pehne,
                        "train_pehne": train_pehne,
                        "test_cnee": test_cnee,
                        "train_cnee": train_cnee,
                    }
                )

                print("alpha", alpha)
                print("gamma", gamma)

                wandb.log(
                    {
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
                    }
                )
                return val_y_loss, test_pehne, val_cf_y_loss, ITTE_loss_val, test_cnee

            return val_y_loss, val_t_loss, val_cf_y_loss, val_cf_t_loss

    def hyperparameter_tuning(self, epoch_range, lr_range, alpha_range, hidden_range):
        # We tune using the outcome loss
        best_val_loss = 100
        best_epoch = 0
        best_lr = 0
        best_alpha = 0
        for epoch in epoch_range:
            for lr in lr_range:
                for alpha in alpha_range:
                    for hidden in hidden_range:
                        self.config["hidden"] = hidden
                        self.config["num_epochs"] = epoch
                        self.config["learning_rate"] = lr
                        self.config["alpha"] = alpha
                        if self.model_type == "HINet":
                            torch.manual_seed(self.config["seed"])
                            np.random.seed(self.config["seed"])
                            self.model = HINet(
                                Xshape=self.config["covariate_dim"], hidden=hidden
                            )
                        elif self.model_type == "NetEst":
                            torch.manual_seed(self.config["seed"])
                            np.random.seed(self.config["seed"])
                            self.model = NetEst(
                                Xshape=self.config["covariate_dim"], hidden=hidden
                            )
                        elif self.model_type == "HINet_no_net_conf":
                            torch.manual_seed(self.config["seed"])
                            np.random.seed(self.config["seed"])
                            self.model = HINet_no_net_conf(
                                Xshape=self.config["covariate_dim"], hidden=hidden
                            )
                        elif self.model_type == "GINModel":
                            torch.manual_seed(self.config["seed"])
                            np.random.seed(self.config["seed"])
                            self.model = GINModel(
                                Xshape=self.config["covariate_dim"], hidden=hidden
                            )
                        elif self.model_type == "GINNetEst":
                            torch.manual_seed(self.config["seed"])
                            np.random.seed(self.config["seed"])
                            self.model = GINNetEst(
                                Xshape=self.config["covariate_dim"], hidden=hidden
                            )
                        elif self.model_type == "TARNet":
                            torch.manual_seed(self.config["seed"])
                            np.random.seed(self.config["seed"])
                            self.model = TARNet(
                                Xshape=self.config["covariate_dim"], hidden=hidden
                            )
                        elif self.model_type == "SPNet":
                            torch.manual_seed(self.config["seed"])
                            np.random.seed(self.config["seed"])
                            self.model = SPNet(
                                Xshape=self.config["covariate_dim"], hidden=hidden
                            )
                        val_y_loss = self.train_model(
                            self.train_data, self.val_data, test=False
                        )[0]
                        print(
                            "configuration",
                            "epoch",
                            epoch,
                            "lr",
                            lr,
                            "alpha",
                            alpha,
                            "hidden",
                            hidden,
                        )
                        print("val_y_loss", val_y_loss)
                        print("best_val_loss", best_val_loss)

                        if val_y_loss < best_val_loss:
                            best_val_loss = val_y_loss
                            best_epoch = epoch
                            best_lr = lr
                            best_alpha = alpha
                            best_hidden = hidden
        print("best_val_loss", best_val_loss)
        print("best_epoch", best_epoch)
        print("best_lr", best_lr)
        print("best_alpha", best_alpha)
        print("best_hidden", best_hidden)
        return best_val_loss, best_epoch, best_lr, best_alpha, best_hidden
