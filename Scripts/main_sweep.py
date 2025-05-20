import os
import sys

DIR = r"" #change to your directory
os.chdir(DIR)
sys.path.append(DIR)
import random
import numpy as np
import src.data.data_generator as data_generator
from src.training import loadData
from methods.Causal_models import GNNModel,GINModel, NetEst,GINNetEst,my_full_model,my_model_no_net_conf,TARNet,GCN_DECONF,SPNet
from src.utils.utils import *
from training import Trainer
import os
import yaml
import torch
import torch_geometric
import wandb
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

#Set parameters for the simulation
num_nodes = 5000 #does nothing when dataset is BC or Flickr
dataset = "full_sim" #BC, Flickr, full_sim
T = int(0.05*num_nodes) #number of treated nodes
num_epochs = 1500
batch_size = -1 #-1
learning_rate = 0.005


random_seed = 2000 #2000
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
#Hyperparameters for the simulation
flipRate = 0.5
covariate_dim = 10
w_c = 2 * np.random.random_sample((covariate_dim)) - 1 #effect of X to T
w_c_n = 2 * np.random.random_sample((covariate_dim)) - 1 #effect of XN to T

w = 2 * np.random.random_sample((covariate_dim)) - 1 

w_n = 2 * np.random.random_sample((covariate_dim)) - 1 #effect of XN to Y
print("w",w)
#effect of X to Y
w_beta_T2Y = 2 * np.random.random_sample((covariate_dim)) - 1 #effect of T to Y
w_exposure = 2 * np.random.random_sample((covariate_dim)) - 1 #effect of X to exposure

betaConfounding = 3 # effect of features X to T (confounding1)
betaNeighborConfounding = 0# effect of Neighbor features to T (confounding2)
betaTreat2Outcome = 2 #3

bias_T2Y = 0# effect o treatment to potential outcome
bias_NT2Y = 0
#betaCovariate2Outcome =1 #effect of features to potential outcome (confounding1)

betaCovariate2Outcome =1.5
betaNeighborCovariate2Outcome =1.5 #effect of Neighbor features to potential outcome
   
    
#effect of interference
betaNeighborTreatment2Outcome = 2
betaNoise = 0.2 #noise
beta0 = 0#-3 #intercept
random_seed= 2000
alpha = 0.05
cuda = True
hyperparameter_defaults = {
    "dataset": dataset,
    "num_nodes": num_nodes,
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "alpha": alpha,
    "gamma": 0.5,
    "betaNeighborConfounding": betaNeighborConfounding,
    "betaNeighborTreatment2Outcome": betaNeighborTreatment2Outcome,
    "betaConfounding": betaConfounding,
    "betaTreat2Outcome": betaTreat2Outcome,
    "seed": random_seed,
    "hidden" : 16,
    "run": 1,
    "w_c": w_c,
    "w": w,
    "w_beta_T2Y": w_beta_T2Y,
    "w_c_n": w_c_n,
    "w_n": w_n,
    "w_exposure": w_exposure,
    "betaNoise": betaNoise,
    "beta0": beta0,
    "bias_T2Y": bias_T2Y,
    "bias_NT2Y":bias_NT2Y,
    "betaCovariate2Outcome": betaCovariate2Outcome,
    "betaNeighborCovariate2Outcome": betaNeighborCovariate2Outcome,
    "homophily": False,
    "node2vec": False,
    "edges_new_node":2,
    "exposure_type": "weight",
    "percent_treated": 0.25,
    "flipRate": flipRate,
    "covariate_dim": covariate_dim,
    "num_networks": 50,
    "model_type":"GINModel", #my_full_model, NetEst, no_net_conf
    "num_seeds": 5,
    "epochs_range": [1500,2000,3000],
    "hidden_range": [16,32],
    "alpha_range": [0],
    "gamma_range": [0],
    "lr_range": [0.001,0.0005,0.0001],
    "track_loss": False,

}

sweep_config = {
    "name": "sweep",
    "method": "grid",
    "parameters": {

        "bias_NT2Y": {
            "values": [0]
        },
        "betaNeighborTreatment2Outcome": {
            "values": [2]
        },
        "betaConfounding": {
            "values": [3]
        },
        "betaTreat2Outcome": {
            "values": [2]
        },
        "betaCovariate2Outcome": {
            "values": [1.5]
        },
        "betaNeighborCovariate2Outcome": {
            "values": [1.5]
        },
        "homophily": {
            "values": [False]
        },
        "model_type": {
            "values": ["my_full_model"]
        },
        "alpha_range": {
            "values": [[0]]
        }
}}

wandb_project_name = "" #put your wandb project name here
def sweep_function():
    wandb.init(config=hyperparameter_defaults,
               project = wandb_project_name,
               job_type="sweep")
    config = wandb.config
    setting = config["dataset"] + "_num_nodes" + str(config["num_nodes"]) + "_T2O_" + str(config["betaTreat2Outcome"]) + "_NT2O_" + str(config["betaNeighborTreatment2Outcome"]) + "_seed_" + str(config["seed"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    print("start sim")
    data_generator.simulate_data(config,setting = setting)
    print("load_data")
    train_data, val_data, test_data = loadData(setting)
    print(len(train_data.edge_index))
    print(len(train_data.edge_index[0]))
    print(len(val_data.edge_index[0]))
    print(len(test_data.edge_index[0]))
    print(len(train_data.t))
    print("z",torch.mean(train_data.z))
    print(torch.std(train_data.z))
    print("y")
    print(torch.mean(train_data.y))
    print(torch.std(train_data.y))
    # print(ze)
    print("loading done")
    if config["model_type"] == "my_full_model":
        model = my_full_model(Xshape=config["covariate_dim"],hidden=config["hidden"])
    elif config["model_type"] == "NetEst":
        model = NetEst(Xshape=config["covariate_dim"],hidden=config["hidden"])
    elif config["model_type"] == "GINNetEst":
        model = GINNetEst(Xshape=config["covariate_dim"],hidden=config["hidden"])
    elif config["model_type"] == "no_net_conf":
        model= my_model_no_net_conf(Xshape=config["covariate_dim"],hidden=config["hidden"])
    elif config["model_type"] == "GINModel":
        model = GINModel(Xshape=config["covariate_dim"],hidden=config["hidden"])
    elif config["model_type"] == "TARNet":
        model = TARNet(Xshape=config["covariate_dim"],hidden=config["hidden"])
    elif config["model_type"] == "GCN_DECONF":
        model = GCN_DECONF(Xshape=config["covariate_dim"],hidden=config["hidden"])
    elif config["model_type"] == "SPNet":
        model = SPNet(Xshape=config["covariate_dim"],hidden=config["hidden"])
    trainer = Trainer(config=config,model=model,train_data=train_data,val_data=val_data,test_data=test_data,device=True)

    trainer.train_test_best_model(epochs_range=config["epochs_range"],hidden_range=config["hidden_range"],alpha_range=config["alpha_range"],lr_range=config["lr_range"],num_seeds=config["num_seeds"])
    # train_seed = config["seed"] + config["run"]
    # torch.manual_seed(train_seed)
    # np.random.seed(train_seed)
    # random.seed(train_seed)

    # print("start training")
    # val_y_loss, val_t_loss, val_t_neighbor_loss, cf_val_y_loss, cf_val_t_loss, cf_val_t_neighbor_loss = train_my_full_model(train_data,val_data,model,config)


sweep_id = wandb.sweep(sweep_config, project = wandb_project_name)
wandb.agent(sweep_id, function = sweep_function)

