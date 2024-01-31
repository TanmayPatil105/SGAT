"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import torch.nn.functional as F

sig = nn.Sigmoid()
hardtanh = nn.Hardtanh(0,1)
gamma = -0.1
zeta = 1.1
beta = 0.66
eps = 1e-20
const1 = beta*np.log(-gamma/zeta + eps)

# @goal: to introduce a stochastic element to regularization
#        -> explore differen configurations
#
# @docs: Read page no. 5
#
# @equation: f (log Alpha, u) = sigma ( (log u - log ( 1 - u) + log alpha) / beta)  * (zeta - gamma) + gamma
#
# Hardtanh: clips the value in the given range to mask
# @min = 0
# @max = 1
#
# Randomness: stochastic, different configurations
#
def l0_train(logAlpha, min, max):
    U = torch.rand(logAlpha.size()).type_as(logAlpha) + eps
    s = sig((torch.log(U / (1 - U)) + logAlpha) / beta)
    s_bar = s * (zeta - gamma) + gamma

    # @values : [ 0 - 1]
    mask = F.hardtanh(s_bar, min, max)
    return mask

#
# @goal: no randomisation here
#
def l0_test(logAlpha, min, max):
    s = sig(logAlpha/beta)
    s_bar = s * (zeta - gamma) + gamma
    mask = F.hardtanh(s_bar, min, max)
    return mask

#
# @docs: Refer page no. 5
#
# @equation:  Σ σ (log alpha - const1)
#
# @goals: uses sigmoid to range between [0, 1]
#
def get_loss2(logAlpha):
    return sig(logAlpha - const1)


class GraphAttention(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 out_dim,
                 num_heads,
                 feat_drop,
                 attn_drop,
                 alpha,
                 bias_l0,
                 residual=False,l0=0, min=0):
        super(GraphAttention, self).__init__()
        self.g = g
        self.num_heads = num_heads
        # fc -> fully connected
        self.fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x : x
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x : x

        # FIXME: API updates
        # @dimension: 1 * 1 * 64 
        self.attn_l = nn.Parameter(torch.Tensor(size=(1, 1, out_dim)))
        # @dimension: 1 * 1 * 64 
        self.attn_r = nn.Parameter(torch.Tensor(size=(1, 1, out_dim)))
        self.bias_l0 = nn.Parameter(torch.FloatTensor([bias_l0]))

        # Initialise weight tensors
        nn.init.xavier_normal_(self.fc.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_l.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_r.data, gain=1.414)

        # parameters for forward pass
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        self.residual = residual
        self.num = 0
        self.l0 = l0
        self.loss = 0
        self.dis = []
        self.min=min
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
                nn.init.xavier_normal_(self.res_fc.weight.data, gain=1.414)
            else:
                self.res_fc = None

    def forward(self, inputs, edges="__ALL__", skip=0):
        self.loss = 0
        
        # For cora dataset:
        #               N -> 2708; D -> 1433; num_head -> 2; num_classes -> 7;
        # N -> number of nodes
        # D -> number of features
        # H -> number of attention heads
        # @dimension h -> 2708 * 1433
        #
        h = self.feat_drop(inputs)  # N x D

        # ft -> reshaped feature matrix with multihead attention
        # @dimension fc -> 1433 * 128 (2 * 64)  ;
        #                             (2 * 64) -> num_heads * out_dim
        #
        ft = self.fc(h).reshape((h.shape[0], self.num_heads, -1))  # N x H x D'
        
        # @result: ft * self.attn_l = [ 2708 * 2 * 64 ]
        # @dimension: a1 -> 2708 * 2 * 1
        #
        a1 = (ft * self.attn_l).sum(dim=-1).unsqueeze(-1) # N x H x 1
        # @result: ft * self.attn_l = [ 2708 * 2 * 64 ]
        # @dimension: a1 -> 2708 * 2 * 1
        #
        a2 = (ft * self.attn_r).sum(dim=-1).unsqueeze(-1) # N x H x 1
        self.g = self.g.to("cuda:1");
        self.g.ndata.update({'ft' : ft, 'a1' : a1, 'a2' : a2})

        if skip == 0:
            # 1. compute edge attention
            self.g.apply_edges(self.edge_attention, edges)

            # 2. compute softmax
            if self.l0 == 1:
                ind = self.g.nodes()
                self.g.apply_edges(self.loop, edges=(ind, ind))

            # FIXME: please
            self.edge_softmax()

            if self.l0 == 1:
                self.g.apply_edges(self.norm)

        # 2. compute the aggregated node features scaled by the dropped,
            edges = self.g.edata['a'].squeeze().nonzero().squeeze()

        self.g.edata['a_drop'] = self.attn_drop(self.g.edata['a'])
        self.num = (self.g.edata['a'] > 0).sum()
        self.g.update_all(fn.u_mul_e('ft', 'a_drop', 'ft'), fn.sum('ft', 'ft'))
        ret = self.g.ndata['ft']

        # 4. residual
        if self.residual:
            if self.res_fc is not None:
                resval = self.res_fc(h).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
            else:
                resval = torch.unsqueeze(h, 1)  # Nx1xD'
            ret = resval + ret
        return ret, edges

    def edge_attention(self, edges):
        # an edge UDF to compute unnormalized attention values from src and dst
        if self.l0 == 0:
            m = self.leaky_relu(edges.src['a1'] + edges.dst['a2'])
        else:
            tmp = edges.src['a1'] + edges.dst['a2']
            logits = tmp + self.bias_l0

            if self.training:
                m = l0_train(logits, 0, 1)
            else:
                m = l0_test(logits, 0, 1)
            self.loss = get_loss2(logits[:,0,:]).sum()

        return {'a': m}
    
    def norm(self, edges):
        # normalize attention
        a = edges.data['a'] / edges.dst['z']
        return {'a' : a}

    def loop(self, edges):
        # set attention to itself as 1
        return {'a': torch.pow(edges.data['a'], 0)}

    def normalize(self, logits):
        self._logits_name = "_logits"
        self._normalizer_name = "_norm"
        self.g.edata[self._logits_name] = logits

        self.g.update_all(fn.copy_e(self._logits_name, self._logits_name),
                          fn.sum(self._logits_name, self._normalizer_name))
        return self.g.edata.pop(self._logits_name), self.g.ndata.pop(self._normalizer_name)

    def edge_softmax(self):

        if self.l0 == 0:
            scores = self.softmax(self.g, self.g.edata.pop('a'))
        else:
            scores, normalizer = self.normalize(self.g.edata.pop('a'))
            self.g.ndata['z'] = normalizer[:,0,:].unsqueeze(1)

        self.g.edata['a'] = scores[:,0,:].unsqueeze(1)

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 alpha,
                 bias_l0,
                 residual, l0=0):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GraphAttention(
            g, in_dim, num_hidden, heads[0], feat_drop, attn_drop, alpha,bias_l0, False, l0=l0, min=0))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GraphAttention(
                g, num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, alpha,bias_l0, residual, l0=l0, min=0))
        # output projection
        self.gat_layers.append(GraphAttention(
            g, num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, alpha,bias_l0, residual, l0=l0))


    def forward(self, inputs):
        h = inputs
        edges = "__ALL__"
        h, edges = self.gat_layers[0](h, edges)
        h = self.activation(h.flatten(1))
        for l in range(1, self.num_layers):
            # This line calls forward method of the GraphAttention object
            h, _= self.gat_layers[l](h, edges, skip=1)
            h = self.activation(h.flatten(1))

        # output projection
        logits,_ = self.gat_layers[-1](h, edges, skip=1)
        logits = logits.mean(1)
        return logits
