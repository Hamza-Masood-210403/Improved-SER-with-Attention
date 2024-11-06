import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from numpy.linalg import eig
import sys
from torch.nn import init
from scipy.linalg import fractional_matrix_power
sys.path.append("models/")
from mlp import MLP

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

def Comp_degree(A):
    """ compute degree matrix of a graph """
    out_degree = torch.sum(A, dim=0)
    in_degree = torch.sum(A, dim=1)

    diag = torch.eye(A.size()[0]).cuda()

    degree_matrix = diag*in_degree + diag*out_degree - torch.diagflat(torch.diagonal(A))

    return degree_matrix

class GraphConv_Ortega(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2, hidden_dim=128):
        super(GraphConv_Ortega, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.MLP = MLP(num_layers, in_dim, hidden_dim, out_dim)

        for i in range(num_layers):
            init.xavier_uniform_(self.MLP.linears[i].weight)
            init.constant_(self.MLP.linears[i].bias, 0)

        #### Adding MLP to GCN
        # self.MLP = MLP(num_layers, in_dim, hidden_dim, out_dim)
        # self.batchnorm = nn.BatchNorm1d(out_dim)

        # for i in range(num_layers):
        #     init.xavier_uniform_(self.MLP.linears[i].weight)
        #     init.constant_(self.MLP.linears[i].bias, 0)

    def forward(self, features, A):
        b, n, d = features.shape
        assert (d == self.in_dim)
        if(len(A.shape) == 2):
            # A_norm = A + torch.eye(n).cuda()
            A_norm = A
            deg_mat = Comp_degree(A_norm)
            frac_degree = torch.FloatTensor(fractional_matrix_power(deg_mat.detach().cpu(),
                                                                    -0.5)).cuda()
            Laplacian = deg_mat - A_norm
            Laplacian_norm = frac_degree.matmul(Laplacian.matmul(frac_degree))
            
            L_complex, U = torch.linalg.eig(Laplacian_norm)
    
            repeated_U_t = U.t().repeat(b, 1, 1)
            repeated_U = U.repeat(b, 1, 1)
        else:
            repeated_U_t = []
            repeated_U = []
            for i in range(A.shape[0]):
                # A_norm = A[i] + torch.eye(n).cuda()
                A_norm = A[i]
                deg_mat = Comp_degree(A_norm)
                frac_degree = torch.FloatTensor(fractional_matrix_power(deg_mat.detach().cpu(),
                                                                        -0.5)).cuda()
                Laplacian = deg_mat - A_norm
                Laplacian_norm = frac_degree.matmul(Laplacian.matmul(frac_degree))

                landa, U = torch.eig(Laplacian_norm, eigenvectors=True)

                repeated_U_t.append(U.t().view(1, U.shape[0], U.shape[1]))
                repeated_U.append(U.view(1, U.shape[0], U.shape[1]))
            repeated_U_t = torch.cat(repeated_U_t)
            repeated_U = torch.cat(repeated_U)
                
        agg_feats = torch.bmm(repeated_U_t.real, features)

        #### Adding MLP to GCN
        out = self.MLP(agg_feats.view(-1, d)).view(b, -1, self.out_dim)
        out = torch.bmm(repeated_U.real, out)
        # out = self.batchnorm(out).view(b, -1, self.out_dim)

        return out

class SelfAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=0.2):
        super(SelfAttentionLayer, self).__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.FloatTensor(out_dim,1))
        self.leakyrelu = nn.LeakyReLU(alpha)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.view(-1, 1))
        
    def forward(self, H):
        Wh = self.W(H)
        Wh1 = Wh.unsqueeze(2)
        Wh2 = Wh.unsqueeze(1)
        
        e = self.leakyrelu(Wh1 + Wh2)

        e = torch.matmul(e, self.a).squeeze(-1)

        attention = F.softmax(e, dim=-1)

        H_att = torch.bmm(attention, Wh)
        return H_att


class KAF_random(nn.Module):
    def __init__(self, D=20, gamma=1.0):
        super(KAF_random, self).__init__()
        self.d = nn.Parameter(torch.linspace(-2, 2, D), requires_grad=False)
        self.alpha = nn.Parameter(torch.randn(D))
        self.gamma = gamma

    def forward(self, x):
        gauss_kernels = torch.exp(-self.gamma * (x.unsqueeze(-1) - self.d)**2)
        return torch.matmul(gauss_kernels, self.alpha)

class KAF(nn.Module):
    def __init__(self, D=20, gamma=1.0, epsilon=1e-5):
        super(KAF, self).__init__()
        self.d = nn.Parameter(torch.linspace(-2, 2, D), requires_grad=False)
        self.alpha = nn.Parameter(torch.randn(D))
        self.gamma = gamma

        with torch.no_grad():
            t = torch.nn.functional.elu(self.d)
            K = torch.exp(-self.gamma * (self.d.unsqueeze(-1) - self.d.unsqueeze(0))**2)
            K += epsilon * torch.eye(D)
            self.alpha.copy_(torch.linalg.solve(K, t.unsqueeze(1))[0].squeeze())

    def forward(self, x):
        gauss_kernels = torch.exp(-self.gamma * (x.unsqueeze(-1) - self.d)**2)
        return torch.matmul(gauss_kernels, self.alpha)

class Graph_CNN_ortega(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, final_dropout,
                 graph_pooling_type, activation_fn, use_attention, device, adj, D=10, gamma=1.0):
        super(Graph_CNN_ortega, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation_fn = activation_fn
        self.use_attention = use_attention

        ### Adj matrix
        self.Adj = adj

        ### List of GCN layers
        self.GCNs = torch.nn.ModuleList()
        self.GCNs.append(GraphConv_Ortega(self.input_dim, self.hidden_dim))
        for i in range(self.num_layers - 1):
            self.GCNs.append(GraphConv_Ortega(self.hidden_dim, self.hidden_dim))

        # Initialize KAF for activation
        if(self.activation_fn == "kaf"):
            self.kaf = KAF(D=D, gamma=gamma)
        else:
            self.kaf = KAF_random(D=D, gamma=gamma)

        # Self-attention layer
        self.attention_layer = SelfAttentionLayer(hidden_dim, hidden_dim)

        # Linear classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.Dropout(p=self.final_dropout),
            nn.PReLU(128),  # Can replace with KAF if needed
            nn.Linear(128, output_dim))

    def forward(self, batch_graph):
        X_concat = torch.cat([graph.node_features.view(1, -1, self.input_dim) for graph in batch_graph], 0).to(self.device)
        A = F.relu(self.Adj)  # Could use KAF here too, but let's keep ReLU for the adjacency

        h = X_concat
        for layer in self.GCNs:
            if(self.activation_fn == "relu"):
                h = F.relu(layer(h,A))
            else:
                h = self.kaf(layer(h, A))  # Apply KAF instead of ReLU
                
        # for i, layer in enumerate(self.GCNs):
        #     if i < self.num_layers // 2:
        #         h = F.relu(layer(h, A))  # Use ReLU in the first half
        #     else:
        #         h = self.kaf(layer(h, A))  # Use KAF in the second half

        if(self.use_attention):
            h = self.attention_layer(h)

        if self.graph_pooling_type == 'mean':
            pooled = torch.mean(h, dim=1)
        elif self.graph_pooling_type == 'max':
            pooled = torch.max(h, dim=1)[0]
        elif self.graph_pooling_type == 'sum':
            pooled = torch.sum(h, dim=1)

        score = self.classifier(pooled)
        return score
