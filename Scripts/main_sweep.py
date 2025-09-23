import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.dirname(DIR)
os.chdir(DIR)
sys.path.append(DIR)
import numpy as np
import torch
import torch_geometric
import yaml
import random

import src.data.data_generator as data_generator
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
    TargetedModel_DoubleBSpline,
)
from src.training import Trainer
from src.utils.utils import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

#Set parameters for the simulation
num_nodes = 5000 #does nothing when dataset is BC or Flickr
dataset = "full_sim" #BC, Flickr, full_sim --> homophily = True or False (BA Sim) below
T = int(0.05*num_nodes) #number of treated nodes
num_epochs = 1500
batch_size = -1 #-1
learning_rate = 0.005


random_seed = 2000 
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
    "dataset": dataset,  #BC, Flickr, full_sim (for full_sim, homophily = True or False below)
    "num_nodes": num_nodes,
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "alpha": 1, #these are alpha and gamma used by TNet --> if you want to set alpha for HINet, change alpha_range
    "gamma": 1,
    "betaNeighborConfounding": betaNeighborConfounding,  #Xn --> T
    "betaNeighborTreatment2Outcome": betaNeighborTreatment2Outcome, #Tn --> Y
    "betaConfounding": betaConfounding, #X --> T
    "betaTreat2Outcome": betaTreat2Outcome, #T --> Y
    "seed": random_seed,
    "hidden" : 16, #does nothing, use hidden_range below
    "run": 1,
    "w_c": w_c, #these are randomly generated weights used in the data generation
    "w": w,
    "w_beta_T2Y": w_beta_T2Y,
    "w_c_n": w_c_n,
    "w_n": w_n,
    "w_exposure": w_exposure, #only does something for weight and weight_squared
    "betaNoise": betaNoise,
    "beta0": beta0, #intercept
    "bias_T2Y": bias_T2Y, #intercept for T --> Y
    "bias_NT2Y":bias_NT2Y, #intercept for NT --> Y
    "betaCovariate2Outcome": betaCovariate2Outcome, #X --> Y
    "betaNeighborCovariate2Outcome": betaNeighborCovariate2Outcome, #Xn --> Y
    "homophily": False,
    "node2vec": False,
    "edges_new_node":2, #m hyperparameter for BA network
    "exposure_type": "weight", #average, sum, weight, weight_squared, entropy
    "percent_treated": 0.25,
    "flipRate": flipRate,
    "covariate_dim": covariate_dim,
    "num_networks": 50, #changes m in the CNEE and PEHNE calculation
    "model_type":"HINet", #HINet, NetEst, TargetedModel_DoubleBSpline, GINModel, TARNet, GCN_DECONF, SPNet
    "num_seeds": 5, #number of seeds used in the test set evaluation
    "epochs_range": [500,1000,2000],
    "hidden_range": [16,32],
    "alpha_range": [0,0.025,0.05,0.1,0.2,0.3],
    "gamma_range": [0], #does not get used
    "lr_range": [0.001,0.0005,0.0001], 
    "dropout_range": [0, 0.1, 0.2],
    "track_loss": False, #training loss is tracked automatically even if False
    "weight_decay":0.001,
    "p_alpha":0.1, #p to select alpha in hyperparameter tuning
    #TNet parameters --> same parameters used in their paper
    "beta" : 20*(num_nodes**-0.5), #this gets transformed in the training object
    "num_grid" :20,
    "tr_knots" : 0.1,
    "lr_1_step" : 1e-3, #learning rate for the first step of the TNet
    "lr_2_step" : 1e-3, #learning rate for the second optimizer of TNet,
    "pre_train_epochs": 0, #number of epochs to pre-train the model
    "loss_2step_with_ly": 0, #'loss in 2 step contains loss of y, 0 means no, 1 means yes'
    "loss_2step_with_ltz":0, #'loss in 2 step contains loss of tz, 0 means no, 1 means yes'
    "fluctuation_train_epochs" : 50, #number of epochs to train before fluctuation optimization
}
#these parameter overwrite the ones defined above
sweep_config = {
    "name": "sweep",
    "method": "grid",
    "parameters": {
        "betaNeighborTreatment2Outcome": {
            "values": [2]
        },
        "betaConfounding": {  #change this to play with BetaXT
            "values": [6]
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
            "values": [False] #True
        },
        "exposure_type": {
            "values": ["weight"] #average, sum, weight, weight_squared, entropy
        },
        "dataset": {
            "values": ["BC"] #BC, Flickr, full_sim
        },
        "alpha_range": {
            "values": [[0,0.025,0.05,0.1,0.2,0.3]]
        },
        "model_type": {
            "values": ["HINet","NetEst","TargetedModel_DoubleBSpline","GINModel","TARNet","GCN_DECONF","SPNet"]
        },
}}

wandb_project_name = "your_project_name" #put your wandb project name here
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
    if config["model_type"] == "HINet":
        model = HINet(Xshape=config["covariate_dim"],hidden=config["hidden"])
    elif config["model_type"] == "NetEst":
        model = NetEst(Xshape=config["covariate_dim"],hidden=config["hidden"])
    elif config["model_type"] == "GINNetEst":
        model = GINNetEst(Xshape=config["covariate_dim"],hidden=config["hidden"])
    elif config["model_type"] == "HINet_no_net_conf":
        model= HINet_no_net_conf(Xshape=config["covariate_dim"],hidden=config["hidden"])
    elif config["model_type"] == "GINModel":
        model = GINModel(Xshape=config["covariate_dim"],hidden=config["hidden"])
    elif config["model_type"] == "TARNet":
        model = TARNet(Xshape=config["covariate_dim"],hidden=config["hidden"])
    elif config["model_type"] == "GCN_DECONF":
        model = GCN_DECONF(Xshape=config["covariate_dim"],hidden=config["hidden"])
    elif config["model_type"] == "SPNet":
        model = SPNet(Xshape=config["covariate_dim"],hidden=config["hidden"])
    elif config["model_type"] == "TargetedModel_DoubleBSpline":
        model = TargetedModel_DoubleBSpline(Xshape=config["covariate_dim"],hidden=config["hidden"],tr_knots=config["tr_knots"])
    trainer = Trainer(config=config,model=model,train_data=train_data,val_data=val_data,test_data=test_data,device=True)

    trainer.train_test_best_model(epochs_range=config["epochs_range"],hidden_range=config["hidden_range"],alpha_range=config["alpha_range"],lr_range=config["lr_range"],num_seeds=config["num_seeds"])

sweep_id = wandb.sweep(sweep_config, project = wandb_project_name)
wandb.agent(sweep_id, function = sweep_function)

