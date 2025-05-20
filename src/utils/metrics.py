from src.data.data_generator import potentialOutcomeSimulation,calculate_ITTE,calculate_ITTE_torch,potentialOutcomeSimulation_torch
import numpy as np
import math
import torch
from torch_geometric.utils import to_dense_adj
from src.utils.utils import Normalize_outcome,Normalize_outcome_recover


def PEHNE(config,data,model,num_networks,train_mean_y,train_std_y):
    """
    A: adjacency matrix
    X: features
    T: treatment
    num_networks: number of networks that are sampled
    """
    MSE_loss = torch.nn.MSELoss()
    X = data.x
    T = data.t
    edge_index = data.edge_index

    #get the adjacency matrix using pytorch geometric
    A = to_dense_adj(edge_index)[0]


    do_steps = True
    metric = 0
    length = T.shape[0]
    #create a list of T's from 0 to length with step size such that num_networks=len(list)
    if do_steps:
        T_list = np.linspace(0,length,num_networks)
        print("T_list",T_list)
    for T in T_list:
        #create a random tensor of size length with values 0 and 1 with the ones equal to T
        #round T to closest int
        T = int(T)
        T_tensor = torch.zeros(length).cuda()
        indices = torch.randperm(length)[:T]
        T_tensor[indices] = 1
        zero_tensor = torch.zeros(length).cuda()
        #calculate the ITTE
        #ITTE according to the model:
        if config["model_type"] == "NetEst" or config["model_type"] == "GINNetEst":
            #get z for each node:
            neighbors = torch.sum(A, 1)
            z = torch.div(torch.matmul(A, T_tensor.reshape(-1)), neighbors)
        else: 
            z= data.z
        y_pred_T =  model(data.x, T_tensor, z,data.edge_index)[1].squeeze(1)
        y_pred_T = Normalize_outcome_recover(y_pred_T,train_mean_y,train_std_y)

        y_pred_0 =  model(data.x, zero_tensor,zero_tensor,data.edge_index)[1].squeeze(1)
        y_pred_0 = Normalize_outcome_recover(y_pred_0,train_mean_y,train_std_y)
        ITTE_pred = y_pred_T - y_pred_0
        #ITTE according to the data:
        
        # print("ITTE_pred",ITTE_pred.shape)
        
        ITTE_data = calculate_ITTE_torch(config,X,A,T_tensor)
        # print("ITTE_data",ITTE_data.shape)
        #calculate the MSE loss between the two ITTEs
        metric += MSE_loss(ITTE_pred.cpu(),ITTE_data)
    
    return metric/num_networks
        
def CNEE(config,data,model,num_networks,train_mean_y,train_std_y):
    """
    A: adjacency matrix
    X: features
    T: treatment
    num_networks: number of networks that are sampled
    """
    MSE_loss = torch.nn.MSELoss()
    X = data.x
    T = data.t
    edge_index = data.edge_index

    #get the adjacency matrix using pytorch geometric
    A = to_dense_adj(edge_index)[0]


    do_steps = True
    metric = 0
    length = T.shape[0]
    #create a list of T's from 0 to length with step size such that num_networks=len(list)
    if do_steps:
        T_list = np.linspace(0,length,num_networks)
        print("T_list",T_list)
    for T in T_list:
        #create a random tensor of size length with values 0 and 1 with the ones equal to T
        #round T to closest int
        T = int(T)
        T_tensor = torch.zeros(length).cuda()
        indices = torch.randperm(length)[:T]
        T_tensor[indices] = 1
        
        if config["model_type"] == "NetEst" or config["model_type"] == "GINNetEst":
            #get z for each node:
            neighbors = torch.sum(A, 1)
            z = torch.div(torch.matmul(A, T_tensor.reshape(-1)), neighbors)
        else: 
            z= data.z
        y_pred_T =  model(data.x, T_tensor, z,data.edge_index)[1].squeeze(1)
        y_pred_T = Normalize_outcome_recover(y_pred_T,train_mean_y,train_std_y)

        y_T = potentialOutcomeSimulation_torch(config,X,A,T_tensor)
        
        loss = MSE_loss(y_pred_T.cpu(),y_T)
        metric += loss
    return metric/num_networks
        
       #get the counterfactual outcome for T
