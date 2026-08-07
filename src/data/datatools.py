import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import pickle as pkl
from sklearn.decomposition import LatentDirichletAllocation

"""
    This is code from Song Jiang: https://github.com/songjiang0909/Causal-Inference-on-Networked-Data
    This code was used in "Estimating causal effects on networked observational data":
    @inproceedings{netest2022,
    title={Estimating Causal Effects on Networked Observational Data via Representation Learning},
    author={Song Jiang, Yizhou Sun},
    booktitle={Proceedings of the 31st ACM International Conference on Information & Knowledge Management},
    year={2022}
    }
    MIT License
    Copyright (c) 2022 Song Jiang

    The Coauthor-CS loader (_load_cs_npz and the "CS" branches in readData and
    adjMatrixSplit) is an addition for this paper and is NOT part of the code
    above. The CS dataset comes from the gnn-benchmark repository of Shchur et
    al. (2019), MIT license.
"""


def _load_cs_npz(path=r"data/semi_synthetic/CS/ms_academic_cs.npz"):
    """Coauthor-CS: the MS Academic co-authorship graph of Shchur et al. (2019).

    The npz file is downloaded verbatim from the gnn-benchmark repository
    (MIT license): https://github.com/shchur/gnn-benchmark/raw/master/data/npz/ms_academic_cs.npz
    Returns a dict with the same interface as the BC/Flickr .mat files
    ("Network", "Attributes"). The npz stores a directed adjacency; it is
    symmetrised, binarised, and stripped of self-loops here so every consumer of
    readData sees the same undirected graph.

    Reference: Shchur, Mumme, Bojchevski, Guennemann. Pitfalls of Graph Neural
    Network Evaluation. arXiv:1811.05868, 2019.
    """
    with np.load(path, allow_pickle=True) as f:
        adj = sp.csr_matrix(
            (f["adj_data"], f["adj_indices"], f["adj_indptr"]), shape=f["adj_shape"]
        )
        attr = sp.csr_matrix(
            (f["attr_data"], f["attr_indices"], f["attr_indptr"]), shape=f["attr_shape"]
        )
    adj = adj + adj.T
    adj.data[:] = 1.0
    adj.setdiag(0)
    adj.eliminate_zeros()
    return {"Network": adj, "Attributes": attr}


def readData(dataset):
    if dataset == "BC":
        data = sio.loadmat(r"data/semi_synthetic/BC/BC0.mat")
        with open(r'data/semi_synthetic/BC/BC_parts.pkl', 'rb') as f:
            parts = pkl.load(f)
    if dataset == "Flickr":
        data = sio.loadmat(r"data/semi_synthetic/Flickr/Flickr01.mat")
        with open(r'data/semi_synthetic/Flickr/Flickr_parts.pkl', 'rb') as f:
            parts = pkl.load(f)
    if dataset == "CS":
        data = _load_cs_npz()
        with open(r'data/semi_synthetic/CS/CS_parts.pkl', 'rb') as f:
            parts = pkl.load(f)
    return data, parts


def dataSplit(parts):
    trainIndex = []
    valIndex = []
    testIndex = []
    for i in range(len(parts["parts"])):
        if parts["parts"][i] == 0:
            trainIndex.append(i)
        elif parts["parts"][i] == 1:
            valIndex.append(i)
        else:
            testIndex.append(i)
    print("Size of train graph:{}, val graph:{}, test graph:{}".format(
        len(trainIndex), len(valIndex), len(testIndex)))
    return trainIndex, valIndex, testIndex


def covariateTransform(data, dimension, trainIndex, valIndex, testIndex, random_state=None):
    X = data["Attributes"]
    print("features shape:{}".format(X.shape))
    lda = LatentDirichletAllocation(n_components=dimension, random_state=random_state)
    lda.fit(X)
    X = lda.transform(X)
    trainX = X[trainIndex]
    valX = X[valIndex]
    testX = X[testIndex]
    print("Shape of graph covariate train:{}, val:{}, test:{}".format(
        trainX.shape, valX.shape, testX.shape))
    return trainX, valX, testX


def adjMatrixSplit(data, trainIndex, valIndex, testIndex, dataset):
    if dataset == "CS":
        # 18k x 18k dense would be ~2.7 GB; csr fancy indexing keeps the splits
        # sparse, which is what the full_sim path feeds downstream anyway.
        A = sp.csr_matrix(data["Network"])
        trainA = A[trainIndex][:, trainIndex]
        valA = A[valIndex][:, valIndex]
        testA = A[testIndex][:, testIndex]
        print("Shape of adj matrix train:{}, val:{}, test:{}".format(
            trainA.shape, valA.shape, testA.shape))
        return trainA, valA, testA
    if dataset == "Flickr":
        A = data["Network"]
    else:
        A = data["Network"].toarray()
    trainA = np.array([a[trainIndex] for a in A[trainIndex]])
    valA = np.array([a[valIndex] for a in A[valIndex]])
    testA = np.array([a[testIndex] for a in A[testIndex]])
    print("Shape of adj matrix train:{}, val:{}, test:{}".format(
        trainA.shape, valA.shape, testA.shape))
    return trainA, valA, testA
