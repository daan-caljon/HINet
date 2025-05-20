import numpy as np
import networkx as nx
import os
from sympy import beta
import torch
import pickle as pkl
from sklearn.metrics.pairwise import cosine_similarity
from src.data.utils import node2vec
from src.data.datatools import *
# import matplotlib.pyplot as plts
from scipy.spatial import distance






def treatmentSimulation(config,X,A):
    do_squared = False
    do_log = False
    do_sigmoid = False
    if config["betaConfounding"] == 0 and config["betaNeighborConfounding"] == 0:
        propensityT= sigmoid(np.zeros(len(X)))
        random_values = np.ones(len(propensityT))
        indices = np.random.choice(len(propensityT), int(len(propensityT)*config["percent_treated"]), replace=False)
        # random_values = np.random.rand(len(propensityT)) 
        random_values[indices] = 0
        T = (random_values < np.array(propensityT)).astype(int)
        return T, np.mean(T)

    if do_squared:
         #select 2 variables and square them (replace in X)
        X_squared = np.square(X)
    if do_log:
        X_log = np.log(X+1)
    if do_sigmoid:
        X_sigmoid = sigmoid(X)
    #for now we do linear
    X_extended = np.concatenate((X[:,:5],X[:,-5:]),axis=1)
    # X_extended = np.concatenate((X,X_sigmoid),axis=1)
    covariate2TreatmentMechanism = np.matmul(config["w_c"],X_extended.T)
    covariate2NeighborTreatmentMechanism = np.matmul(config["w_c_n"],X_extended.T)
    neighbors = np.sum(A,1)
    # print("covariate2T",np.mean(covariate2TreatmentMechanism),np.std(covariate2TreatmentMechanism))
    neighborAverage = np.divide(np.matmul(A, covariate2NeighborTreatmentMechanism.reshape(-1)), neighbors)
    neighborSum = neighborAverage
    # print("neighborSum",np.mean(neighborSum),np.std(neighborSum))
    # print ("confounding",np.mean(config["betaConfounding"]*covariate2TreatmentMechanism+betaNeighborConfounding*neighborSum),np.std(config["betaConfounding"]*covariate2TreatmentMechanism+betaNeighborConfounding*neighborSum))
    # propensityTwithout = config["betaConfounding"]*covariate2TreatmentMechanism+betaNeighborConfounding*neighborSum
    # propensityT = sigmoid(propensityTwithout - np.mean(propensityTwithout))
    propensityT= sigmoid(config["betaConfounding"]*covariate2TreatmentMechanism+config["betaNeighborConfounding"]*neighborSum)
    propensityT = config["betaConfounding"]*covariate2TreatmentMechanism+config["betaNeighborConfounding"]*neighborSum
    #get percentile
    percentile = np.percentile(propensityT, 100*(1 -config["percent_treated"]))
    propensityT = (propensityT - percentile)
    propensityT = sigmoid(propensityT)
    #find 25% of the nodes with the highest propensity score
    # if config["percent_treated"] != 0.5:


    #     value = np.percentile(propensityT, 100*(1 -config["percent_treated"]))
    #     print("value",value)
    #     random_values = (np.random.rand(len(propensityT))-0.5)/4
    #     random_values= np.clip(random_values+value,0,1)

    # else:
    #     random_values= np.random.rand(len(propensityT))
    # random_values = np.random.rand(len(propensityT)) 
    random_values = np.random.rand(len(propensityT))
    T = (random_values < propensityT).astype(int)
    # print("propsensityT np",np.array(propensityT)[100:120])
    print("T",np.mean(T),np.std(T))


    
    
    # #plot nodes with nx and color them according to treatment
    # plt.style.use("science")
    # G = nx.from_numpy_array(A)
    # default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # default_colors = ["black",default_colors[1]]
    # # Map node colors based on T or any custom attribute
    # node_colors = [default_colors[t % len(default_colors)] for t in T]
    

    # # Add node colors to G based on T
    # # node_colors = ['b' if t == 1 else 'r' for t in T]
    # for i, color in enumerate(node_colors):
    #     G.nodes[i]['color'] = color

    # # Extract the largest connected component
    # largest_cc = max(nx.connected_components(G), key=len)
    # G_lcc = G.subgraph(largest_cc).copy()

    # # Get node colors for the largest connected component
    # lcc_node_colors = [G_lcc.nodes[n]['color'] for n in G_lcc.nodes]

    # #Plot the largest connected component
    # plt.figure(figsize=(10, 10))
    # pos = nx.spring_layout(G_lcc, iterations=500, k=0.4)
    # # pos = nx.kamada_kawai_layout(G_lcc)
    # nx.draw(
    #     G_lcc, pos, with_labels=False, node_size=60, 
    #     node_color=lcc_node_colors, edge_color='grey', alpha=0.6
    # )
    # #add legend
    # plt.scatter([],[],color=default_colors[0],label='Control')
    # plt.scatter([],[],color=default_colors[1],label='Treatment')
    # #change boldness of legend
    # #put legend top right
    # plt.legend(
    #     loc='upper left',
    #     fontsize=40,
    #     frameon=False,
    #     handletextpad=0.1,                     # Reduce space between marker and text
    #     borderaxespad=0.0,                     # Reduce space from the edge of the plot
    #     bbox_to_anchor=(0, 1),                 # Align exactly to top right corner              # Make legend text bold
    # )
    # # plt.legend(fontsize=40)
    # num = np.random.randint(0,10000)
    # if config["homophily"]:
    #     # num = np.random.randint(0,10000)
    #     #put the title under the figure
    #     # plt.title('Homophilous network', fontsize=24, y=1.02)
    #     path = "homophilous_network" + str(num) + ".pdf"
    # else:
    #     # plt.title('Non-homophilous network', fontsize=24, y=0)
    #     path = "non_network" +str(num) + ".pdf"
    # plt.savefig(path, bbox_inches='tight', dpi=1000,format='pdf')
    
    # plt.close()
    #calculate ratio of treated neighbors and plot
    treated_neighbors = np.matmul(A,T.reshape(-1))
    ratio_treated_neighbors = np.divide(treated_neighbors,neighbors)
    # # # print("ratio_treated_neighbors",np.mean(ratio_treated_neighbors),np.std(ratio_treated_neighbors))
    # plt.hist(ratio_treated_neighbors,bins=100)
    # plt.title('Distribution of ratio of treated neighbors')
    # plt.xlabel('Ratio of treated neighbors')
    # plt.ylabel('Frequency')
    # plt.show()
    # # print("numT",np.sum(T))
    # # print(de)

    # # print(de)
    mean_T = np.mean(T)
    print("mean_T",mean_T)
    return T, mean_T

def masked_softmax(x):
    mask = x != 0
    
    
    # Subtract the maximum value for numerical stability
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    
    # Apply the mask to the exponentiated values (elements where x was zero will be zero)
    exp_x = np.where(mask, exp_x, 0)
    
    # Compute the softmax
    softmax = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    return softmax

def exposure_mapping(X,A,T,exposure_type="average",w_exposure=None,biasNT2Y=3):
    neighbors = np.sum(A,1) # number of neighbors
    #Four options
    #1 sum of treated neighbors
    if exposure_type == "sum":
        exposure = np.matmul(A,T.reshape(-1))
        
    #2 average exposure
    elif exposure_type == "average":
        sum_exposure = np.matmul(A,T.reshape(-1))
        exposure = np.divide(sum_exposure,neighbors)
    #3 weight through exposure mechanism
    elif exposure_type == "weight":
        exposure_mech = np.matmul(w_exposure,X.T) + biasNT2Y
        #mask A according to T
        A_masked = A * T.reshape(-1)
        exposure = np.matmul(A_masked,exposure_mech) 
        
        exposure = np.divide(exposure,neighbors)
       
    elif exposure_type == "similarity":
        #cosine similarity between the features of the nodes
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        cosine_sim = np.dot(X_norm, X_norm.T)
        
        neighbor_sim = A * cosine_sim  # Ensures that only adjacent nodes are considered

        softmax_sim = masked_softmax(neighbor_sim)
        
        exposure = np.matmul(softmax_sim, T.reshape(-1))
    
    return exposure




    

def flipTreatment(T,rate):
    
    numToFlip = int(len(T)*rate)
    nodesToFlip = set(np.random.choice(len(T), numToFlip, replace=False))
    cfT = np.array([1-T[i] if i in nodesToFlip else T[i] for i in range(len(T))])
    
    return cfT,nodesToFlip

def calculate_ITTE(config,X,A,T):
    T_treat_0 = np.zeros(len(T))     
    PO_0_treat = potentialOutcomeSimulation(config,X,A,T_treat_0)
    PO_T = potentialOutcomeSimulation(config,X,A,T)
    ITTE = PO_T - PO_0_treat
    return ITTE

def calculate_ITTE_torch(config,X,A,T):
    #make everything numpy
    X = X.cpu().numpy()
    A = A.cpu().numpy()
    T = T.cpu().numpy()
    ITTE = calculate_ITTE(config,X,A,T)
    return torch.tensor(ITTE, dtype=torch.float32)
def potentialOutcomeSimulation_torch(config,X,A,T,epsilon=0):
    X = X.cpu().numpy()
    A = A.cpu().numpy()
    T = T.cpu().numpy()
    po= potentialOutcomeSimulation(config,X,A,T,epsilon)
    return torch.tensor(po, dtype=torch.float32)
    

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def potentialOutcomeSimulation(config,X,A,T,epsilon=0):
    do_squared = True
    do_log = False
    do_sigmoid = True
    w= np.array(config["w"])
    w_beta_T2Y = np.array(config["w_beta_T2Y"])
    T = np.array(T,dtype=np.float32)
    if do_squared:
         #select 2 variables and square them (replace in X)
        X_squared = np.square(X)
    if do_log:
        X_log = np.log(X+1)
    if do_sigmoid:
        X_sigmoid = sigmoid(X)
    
    # X_extended = np.concatenate((X_squared,X_sigmoid),axis=1)
    X_extended = np.concatenate((X[:,:5],X_sigmoid[:,-5:]),axis=1)
    # X_extended = np.concatenate((X,X_sigmoid),axis=1)
    
    covariate2OutcomeMechanism = np.matmul(w,X_extended.T) #X.T is transpose 
    covariate2NeighborOutcomeMechanism = np.matmul(config["w_n"],X_extended.T)
   
    neighbors = np.sum(A,1)
    neighborAverage = np.divide(np.matmul(A, covariate2NeighborOutcomeMechanism.reshape(-1)), neighbors)

    beta_T2Y = np.matmul(w_beta_T2Y,X_extended.T) + config["bias_T2Y"] 
   
    total_Treat2Outcome = config["betaTreat2Outcome"]*beta_T2Y
    #distribution of total_Treat2Outcome
    # plt.hist(total_Treat2Outcome*T,bins=100)
    # plt.title('Distribution of total_Treat2Outcome')
    # plt.xlabel('total_Treat2Outcome')
    # plt.ylabel('Frequency')
    # plt.show()
    # total_network2Outcome = betaNeighborTreatment2Outcome*beta_T2Y
    exposure = exposure_mapping(X_extended,A,T,exposure_type=config["exposure_type"],w_exposure=config["w_exposure"],biasNT2Y=config["bias_NT2Y"])
    # print("exposure",exposure.mean(),exposure.std())

    
    #show exposure distribution
    # plt.hist(exposure,bins=100)
    # plt.title('Distribution of exposure')
    # plt.xlabel('Exposure')
    # plt.ylabel('Frequency')
    # plt.show()
    # print("neighborT2Y",config["betaNeighborTreatment2Outcome"])
    # print("exposure",exposure[0:20])
    # print("exposure",exposure[0:20]*config["betaNeighborTreatment2Outcome"])
    T = np.array(T)
    potentialOutcome = config["beta0"]+ total_Treat2Outcome*T + config["betaCovariate2Outcome"]*covariate2OutcomeMechanism + config["betaNeighborCovariate2Outcome"]*neighborAverage+config["betaNeighborTreatment2Outcome"]*exposure+config["betaNoise"]*epsilon

    PO_without_treatment = config["beta0"] + config["betaCovariate2Outcome"]*covariate2OutcomeMechanism + config["betaNeighborCovariate2Outcome"]*neighborAverage
    # print("potentialOutcoem",potentialOutcome[0:20])
    # print("potentialOutcome",potentialOutcome.mean(),potentialOutcome.std())
   
    return potentialOutcome




def generate_data(config,gen_type,nx_seed,watts_strogatz = False):
    do_node2vec = config["node2vec"]
    do_homophily = config["homophily"]
    # if gen_type == "test":
    #     do_homophily = False
    print("gen data")
    z_11 = 0.7
    z_22 = 0.2

        
    if config["dataset"] == "full_sim" and not do_homophily:
        if watts_strogatz:
            G = nx.connected_watts_strogatz_graph(config["num_nodes"], 4, 0.1,seed=config["seed"])
        else:
            G = nx.barabasi_albert_graph(config["num_nodes"], config["edges_new_node"],seed=config["seed"])
        adj_matrix = nx.adjacency_matrix(G)
    elif config["dataset"] == "enron":
        trainA, valA, testA =load_enron_network()
    elif config["dataset"] == "Flickr" or config["dataset"] == "BC" :
        
        data,parts = readData(config["dataset"])
        trainIndex,valIndex,testIndex = dataSplit(parts)
        trainX, valX, testX = covariateTransform(data,config["covariate_dim"],trainIndex,valIndex,testIndex)
        #We will normalize the covariates over the columns:
        mean_trainX = np.mean(trainX,axis=(0))
        std_trainX = np.std(trainX,axis=(0))
        trainX = (trainX-mean_trainX)/std_trainX
        valX = (valX-np.mean(valX,axis=0))/np.std(valX,axis=0)
        testX = (testX-np.mean(testX,axis=0))/np.std(testX,axis=0)
        trainA, valA, testA = adjMatrixSplit(data,trainIndex,valIndex,testIndex,config["dataset"])
        do_homophily=False

    # Convert the sparse matrix to a dense NumPy array
    if do_homophily:
        path = "data/simulated/" + "num_nodes_" + str(config["num_nodes"]) + "_homophilous_" + gen_type + ".pkl"
        if os.path.exists(path):
            with open(path,"rb") as f:
                A,X = pkl.load(f)
        else:
             
            A,X = create_homophilous_network(config["num_nodes"],config["covariate_dim"])
            with open(path,"wb") as f:
                pkl.dump((A,X),f)
        
        avg_deg = np.mean(np.sum(A,1))
        # print(stop)
        #plot degree distribution
        # plt.hist(np.sum(A,1),bins=100)
        # plt.title('Distribution of node degrees')

        # plt.xlabel('Degree')
        # plt.ylabel('Frequency')
        # plt.show()
        dense_adj_matrix = A
    elif config["dataset"] == "full_sim":
        dense_adj_matrix = adj_matrix.toarray()
        A = dense_adj_matrix
    else:
        if gen_type == "train":
            dense_adj_matrix = trainA
            A = dense_adj_matrix
        elif gen_type == "val":
            dense_adj_matrix = valA
            A = dense_adj_matrix
        elif gen_type == "test":
            dense_adj_matrix = testA
            A = dense_adj_matrix
    # Convert the NumPy array to a PyTorch tensor
        tensor_adj_matrix = torch.tensor(dense_adj_matrix, dtype=torch.float32)

    if do_node2vec:
        if config["dataset"] == "full_sim":
            file = "data/simulated/" +config["dataset"]+"_num_nodes_" + str(config["num_nodes"])+ "_" + gen_type +"X.pkl"

        else:

            file = "data/semi_synthetic/"+config["dataset"]+"/" +config["dataset"]+"_" + gen_type +"X.pkl"
        if os.path.exists(file):
            with open(file,"rb") as f:
                X = pkl.load(f)
            
        else:
            X = node2vec.generate_node_embeddings(dense_adj_matrix,embedding_dim=config["covariate_dim"])

            X = (X-np.mean(X,axis=0))/np.std(X,axis=0)
            #if a value is larger than 5, set it to 5, if it is smaller than -5 set to -5
            X = np.clip(X, -5, 5)
            with open(file,"wb") as f:
                pkl.dump(X,f)
        trainX = X
        valX = X
        testX = X
    elif do_homophily:
        print("X",X)
    elif config["dataset"] == "full_sim" and not do_homophily:

        X = np.random.randn(dense_adj_matrix.shape[1], config["covariate_dim"])
    else:
        if gen_type == "train":
            #column normalization --> calculate sd and mean for train and then use this also for test?! 
            X = trainX
            
        elif gen_type == "val":
            X = valX
        elif gen_type == "test":
            X = testX
    
    # print("X",X)
    

    epsilon = np.random.normal(0,1,X.shape[0])
 
    T, meanT= treatmentSimulation(config,X=np.array(X),A=np.array(A)) #effect of X to T
    T = torch.tensor(T, dtype=torch.float32)

    PO = potentialOutcomeSimulation(config,X,A,T,epsilon)
   
    #this randomly flips the treatment according to a fliprate (which can make it an RCT)
    cfT,nodesToFlip = flipTreatment(T,config["flipRate"])
    cfT = torch.from_numpy(cfT)
    epsiloncf = 0
    #no noise in the CF
    cfPOTrain = potentialOutcomeSimulation(config,X,A,cfT)
    

    ITTE = calculate_ITTE(config,X,A,cfT)

    num = X.shape[0]

    print("T",sum(T))
    #additional check to see whether a relation between X and Y is learned
    #Generate X randomly
    X_random = np.random.randn(num,config["covariate_dim"])
    PO_random = potentialOutcomeSimulation(config,X_random,A,T)
    PO_random_cf = potentialOutcomeSimulation(config,X_random,A,cfT)
    print("A",A)
    print("X",X)
    print("T",T)
    print("PO",PO)
    A_array = np.array(A)
    X_array = np.array(X)
    T_array = np.array(T)
    PO_array = np.array(PO)
    #standardize PO_array
    PO_array = (PO_array - np.mean(PO_array))/np.std(PO_array)
    G = nx.from_numpy_array(A_array)
    
    for i in range(G.number_of_nodes()):
    
        G.nodes[i]['X'] = X_array[i]
        G.nodes[i]['T'] = T_array[i]
        G.nodes[i]['Y'] = PO_array[i]
    #draw the graph with outcomes
    treatment_assortativity = nx.attribute_assortativity_coefficient(G,'T')
    print("treatment_assortativity",treatment_assortativity)
    outcome_assortativity = nx.numeric_assortativity_coefficient(G,'Y')
    print("outcome_assortativity",outcome_assortativity)
    # stop
    # feature_assortativity = nx.attribute_assortativity_coefficient(G,'X') 
    # print("feature_assortativity",feature_assortativity)
    print("PO",np.mean(PO),np.std(PO))
    # print(de)
    

    if gen_type == "train":

        my_data = {'T':np.array(T),
            'cfT':np.array(cfT),
            'features': np.array(X), 
            'PO':np.array(PO),
            'cfPO':np.array(cfPOTrain),
            'nodesToFlip':nodesToFlip,
            'network':A,
            "meanT":meanT,
            "ITTE":ITTE,
            "X_random":X_random,
            "PO_random":PO_random,


            # "train_t1z1":np.array(cfPOTrain_t1z1),
            # "train_t1z0":np.array(cfPOTrain_t1z0),
            # "train_t0z0":np.array(cfPOTrain_t0z0),
            # "train_t0z7":np.array(cfPOTrain_t0z7),
            # "train_t0z2":np.array(cfPOTrain_t0z2),
        }
    if gen_type == "val":
        my_data = {'T':np.array(T),
            'cfT':np.array(cfT),
            'features': np.array(X), 
            'PO':np.array(PO),
            'cfPO':np.array(cfPOTrain),
            'nodesToFlip':nodesToFlip,
            'network':A,
            "meanT":meanT,
            "ITTE":ITTE,
            "X_random":X_random,
            "PO_random":PO_random,

            # "val_t1z1":np.array(cfPOTrain_t1z1),
            # "val_t1z0":np.array(cfPOTrain_t1z0),
            # "val_t0z0":np.array(cfPOTrain_t0z0),
            # "val_t0z7":np.array(cfPOTrain_t0z7),
            # "val_t0z2":np.array(cfPOTrain_t0z2),
        }
    if gen_type == "test":
        my_data = {'T':np.array(T),
            'cfT':np.array(cfT),
            'features': np.array(X), 
            'PO':np.array(PO),
            'cfPO':np.array(cfPOTrain),
            'nodesToFlip':nodesToFlip,
            'network':A,
            "meanT":meanT,
            "ITTE":ITTE,
            "X_random":X_random,
            "PO_random":PO_random,

            # "test_t1z1":np.array(cfPOTrain_t1z1),
            # "test_t1z0":np.array(cfPOTrain_t1z0),
            # "test_t0z0":np.array(cfPOTrain_t0z0),
            # "test_t0z7":np.array(cfPOTrain_t0z7),
            # "test_t0z2":np.array(cfPOTrain_t0z2),
        }
    return my_data

def simulate_data(config,setting,watts_strogatz = False):
    train =  generate_data(config,gen_type = "train",nx_seed= config["seed"],watts_strogatz=watts_strogatz)
    
    nx_seed = config["seed"]+1
    val = generate_data(config,gen_type = "val",nx_seed=nx_seed,watts_strogatz=watts_strogatz)
    nx_seed +=nx_seed+1
    test = generate_data(config,gen_type = "test",nx_seed=nx_seed,watts_strogatz=watts_strogatz)
    data = {"train":train,"val":val,"test":test}

   
    

    file = "data/simulated/" + setting +".pkl"

    with open(file,"wb") as f:
        pkl.dump(data,f)    
    

def create_homophilous_network(num_nodes,num_features):
    my_cosine = True
    X = np.random.randn(num_nodes,num_features)
    #get euclidian distances between nodes
    dist_matrix = distance.cdist(X, X, metric='euclidean')  # Shape: (num_nodes, num_nodes)
    print("dist_matrix",dist_matrix)
    print("dist_matrix",dist_matrix.mean(),dist_matrix.std())
    print("max min",dist_matrix.max(),dist_matrix.min())
    euclidian_distances = dist_matrix
    similarity_matrix = 1/(1+euclidian_distances)
    print("similarity_matrix",similarity_matrix)
    print("similarity_matrix",similarity_matrix.mean(),similarity_matrix.std())

        


    if my_cosine:
        similarity_matrix = cosine_similarity(X)
    np.fill_diagonal(similarity_matrix,-1)
    if my_cosine:
        loc = 0.80
        scale = 0.025
    else: 
        loc= 0.3
        scale =0.025

    tolerance = 0.1
    goal = 4

    #fill similarity matrix with -1 on diagonal
    np.fill_diagonal(similarity_matrix,-1)
    for i in range(100):
        #make it a symmetric matrix
        
        threshold_num_matrix = np.random.normal(loc=loc,scale= scale,size=(num_nodes,num_nodes))
        
        A = (similarity_matrix > threshold_num_matrix).astype(int)
        # np.fill_diagonal(A,0)
        upper = np.triu(A)
        A = upper + upper.T
        np.fill_diagonal(A,0)
        
        avg_deg = np.mean(np.sum(A,1))
        # print("avg_degree",avg_deg)
        print("avg_deg",avg_deg)
        print(loc)
        if abs( avg_deg - goal) < tolerance:
            #make sure that the network is connected by connecting a node to the most similar node
            
            max_sim = np.argmax(similarity_matrix,1)
            # print(max_sim)
            for i in range(num_nodes):
                # print(i,max_sim[i])
                A[i,max_sim[i]] = 1
                A[max_sim[i],i] = 1
            #check if network is connected
            print(A)
            G = nx.from_numpy_array(A)
            if nx.is_connected(G):
                print("connected")
            else:
                print("not connected")
                #largest component
                largest_cc = max(nx.connected_components(G), key=len)
                print("largest_cc",len(largest_cc))
            break
                # # print(de)
            
        if avg_deg < goal-tolerance:
            loc -= 0.002
        elif avg_deg > goal+tolerance:
            loc += 0.002        
        
    # print("iter",i)
    # # print(de)
    
    return A,X