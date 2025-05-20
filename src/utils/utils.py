import torch
import torch.nn.functional as F
import numpy as np

def Normalize_outcome(y,mean_y,std_y):
    y = (y-mean_y)/std_y
    return y

def Normalize_outcome_recover(y,mean_y,std_y):
    y = y*std_y + mean_y
    return y

"""
The following codes are originally by Ruocheng Guo for the WSDM'20 paper
@inproceedings{guo2020learning,
  title={Learning Individual Causal Effects from Networked Observational Data},
  author={Guo, Ruocheng and Li, Jundong and Liu, Huan},
  booktitle={Proceedings of the 13th International Conference on Web Search and Data Mining},
  pages={232--240},
  year={2020}
}
https://github.com/rguo12/network-deconfounder-wsdm20
"""

def wasserstein(x,y,p=0.5,lam=10,its=10,sq=False,backpropT=False,cuda=True):
    """return W dist between x and y"""
    '''distance matrix M'''
    nx = x.shape[0]
    ny = y.shape[0]
    
    x = x.squeeze()
    y = y.squeeze()
    
#    pdist = torch.nn.PairwiseDistance(p=2)

    M = pdist(x,y) #distance_matrix(x,y,p=2)
    
    '''estimate lambda and delta'''
    M_mean = torch.mean(M)
    M_drop = F.dropout(M,10.0/(nx*ny))
    delta = torch.max(M_drop).detach()
    eff_lam = (lam/M_mean).detach()

    '''compute new distance matrix'''
    Mt = M
    row = delta*(torch.ones(M[0:1,:].shape).cuda())
    col = torch.cat([delta*(torch.ones(M[:,0:1].shape)).cuda(),(torch.zeros((1,1))).cuda()],0)
    if cuda:
        row = row.cuda()
        col = col.cuda()
    Mt = torch.cat([M,row],0)
    Mt = torch.cat([Mt,col],1)

    '''compute marginal'''
    a = torch.cat([p*torch.ones((nx,1))/nx,(1-p)*torch.ones((1,1))],0)
    b = torch.cat([(1-p)*torch.ones((ny,1))/ny, p*torch.ones((1,1))],0)

    '''compute kernel'''
    Mlam = eff_lam * Mt
    temp_term = torch.ones(1)*1e-6
    if cuda:
        temp_term = temp_term.cuda()
        a = a.cuda()
        b = b.cuda()
    K = torch.exp(-Mlam) + temp_term
    U = K * Mt
    ainvK = K/a

    u = a

    for i in range(its):
        u = 1.0/(ainvK.matmul(b/torch.t(torch.t(u).matmul(K))))
        if cuda:
            u = u.cuda()
    v = b/(torch.t(torch.t(u).matmul(K)))
    if cuda:
        v = v.cuda()

    upper_t = u*(torch.t(v)*K).detach()

    E = upper_t*Mt
    D = 2*torch.sum(E)

    if cuda:
        D = D.cuda()

    return D, Mlam

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)




def MI(x,y,z,N):
    if x == 0:
        return torch.FloatTensor([0])
    else:
        return (x/N)*torch.log2((N*x)/(y*z))

def NMI(set1,set2,threshold=0.5):
    set1 =  torch.FloatTensor([(set1 >= threshold).sum(),(set1 < threshold).sum()])
    set2 =  torch.FloatTensor([(set2 >= threshold).sum(),(set2 < threshold).sum()])
    set1 = set1.reshape(1,-1)
    set2 = set2.reshape(1,-1)
    res = torch.cat((set1,set2),0).T
    N = torch.sum(torch.sum(res))
    NW = torch.sum(res,1)
    NC  = torch.sum(res,0)
    HC = -((NC[0]/N)*torch.log2(NC[0]/N)+(NC[1]/N)*torch.log2(NC[1]/N))
    HW = -((NW[0]/N)*torch.log2(NW[0]/N)+(NW[1]/N)*torch.log2(NW[1]/N))
    IF = MI(res[0][0],NW[0],NC[0],N)+MI(res[0][1],NW[0],NC[1],N)+MI(res[1][0],NW[1],NC[0],N)+MI(res[1][1],NW[1],NC[1],N)
    return (IF/torch.sqrt(HC*HW)).cuda()

def pearsonr(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val**2