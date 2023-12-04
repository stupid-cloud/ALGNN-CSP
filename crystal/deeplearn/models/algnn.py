import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
import torch.nn as nn
from torch import Tensor
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.nn
from torch_geometric.nn import CGConv
from typing import Tuple, Union
from torch import Tensor
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from typing import Optional
from crystal.deeplearn.models.utils import global_first_pool

class GatedGCN(MessagePassing):
    """Gated GCN, also known as edge-gated convolution.
    Reference: https://arxiv.org/abs/2003.00982
    
    Different from the original version, in this version, the activation function is SiLU,
    and the normalization is LayerNorm.

    This implementation concatenates the `x_i`, `x_j`, and `e_ij` feature vectors during the edge update.
    """
    def __init__(self, dim=60, epsilon=1e-5):
        super().__init__(aggr='add')

        self.W_src  = nn.Linear(dim, dim)
        self.W_dst  = nn.Linear(dim, dim)
        self.W_e    = nn.Linear(dim*2 + dim, dim)
        self.act    = nn.SiLU()
        self.sigma  = nn.Sigmoid()
        self.norm_x = nn.LayerNorm([dim])
        self.norm_e = nn.LayerNorm([dim])
        self.eps    = epsilon

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W_src.weight); self.W_src.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.W_dst.weight); self.W_dst.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.W_e.weight);   self.W_e.bias.data.fill_(0)


    def forward(self, x, edge_index, edge_attr):
        edge_index = edge_index.type(torch.int64)
        i, j = edge_index  
        # Calculate gated edges
        sigma_e = self.sigma(edge_attr)
        # print(i.shape)
        e_sum = scatter(src=sigma_e, index=i, dim=0)
        e_gated = sigma_e / (e_sum[i] + self.eps)
        # Update the nodes (this utilizes the gated edges)
        out = self.propagate(edge_index, x=x, e_gated=e_gated)
        out = self.W_src(x) + out
        out = x + self.act(self.norm_x(out))

        # Update the edges
        z = torch.cat([x[i], x[j], edge_attr], dim=-1)
        edge_attr = edge_attr + self.act(self.norm_e(self.W_e(z)))
        return out, edge_attr

    def message(self, x_j, e_gated):
        return e_gated * self.W_dst(x_j)

class ALGNN(nn.Module):
    def __init__(
            self, dataset, conv_num, pool, dropout_rate, act, batch_norm, 
            pre_fc_num, pre_out_channel, post_fc_num, post_out_channel, epsilon, **kwargs
            ):
        super().__init__()
        # self.linear = nn.LazyLinear(out_channels)
        num_node_features_atom = dataset.num_node_features
        num_edge_features_atom = dataset.num_edge_features
        num_edge_features_line = dataset.edge_attr_line.shape[1]
        self.pre_lin_list1 = self.init_pre_fc(pre_fc_num, num_node_features_atom, pre_out_channel)
        self.pre_lin_list2 = self.init_pre_fc(pre_fc_num, num_edge_features_atom, pre_out_channel)
        self.pre_lin_list3 = self.init_pre_fc(pre_fc_num, num_edge_features_line, pre_out_channel)
        self.conv_list_atom = self.init_conv(conv_num, pre_out_channel, epsilon)
        self.conv_list_line = self.init_conv(conv_num, pre_out_channel, epsilon)
        self.post_lin_list = self.init_post_fc(post_fc_num, pre_out_channel, post_out_channel)
        self.lin_out = nn.Linear(post_out_channel, 1)
        self.act = getattr(nn.functional, act)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = BatchNorm1d(pre_out_channel)
        self.dropout = nn.Dropout(dropout_rate)

        if pool == 'global_first_pool':
            self.pool = global_first_pool
        else:
            self.pool = getattr(torch_geometric.nn, pool)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Resets all learnable parameters of the module"""
        for i in self.pre_lin_list1:
            i.reset_parameters()
        for i in self.pre_lin_list2:
            i.reset_parameters()
        for i in self.pre_lin_list3:
            i.reset_parameters()
        for i in self.post_lin_list:
            i.reset_parameters()
        self.lin_out.reset_parameters()
   
    def init_pre_fc(self, pre_fc_num,in_channels, pre_out_channel):
       
        pre_lin_list = torch.nn.ModuleList()

        if pre_fc_num <=0:
            pre_fc_num = 1
            
        for i in range(pre_fc_num):
            if i == 0:
                    lin = torch.nn.Linear(in_channels, pre_out_channel)
                    pre_lin_list.append(lin)
            else:
                lin = torch.nn.Linear(pre_out_channel, pre_out_channel)
                pre_lin_list.append(lin)
        return pre_lin_list

    def init_post_fc(self, post_fc_num, pre_out_channel, post_out_channel):
        
        post_lin_list = torch.nn.ModuleList()

        if post_fc_num <=0:
            post_fc_num = 1
            
        for i in range(post_fc_num):
            if i == 0:
                lin = torch.nn.Linear(pre_out_channel, post_out_channel)
                post_lin_list.append(lin)
            else:
                lin = torch.nn.Linear(post_out_channel, post_out_channel)
                post_lin_list.append(lin)

        return post_lin_list

    def init_conv(self, conv_num, pre_out_channel, epsilon):
        conv_list = torch.nn.ModuleList()
        bn_list = torch.nn.ModuleList()
        if conv_num <=0:
            conv_num = 1

        for _ in range(conv_num):           
            conv = GatedGCN(pre_out_channel, epsilon)
            conv_list.append(conv) 
        return conv_list
    
  


    def forward(self, data) -> Tensor:
        for id in range(len(self.pre_lin_list1)):
            if id == 0:
                node_attr_atom = self.pre_lin_list1[id](data.x)
                edge_attr_atom = self.pre_lin_list2[id](data.edge_attr)
                edge_attr_line = self.pre_lin_list3[id](data.edge_attr_line)  
            else:
                node_attr_atom = self.pre_lin_list1[id]( node_attr_atom)
                edge_attr_atom = self.pre_lin_list2[id](edge_attr_atom)
                edge_attr_line = self.pre_lin_list3[id](edge_attr_line) 
            node_attr_atom = self.act(node_attr_atom)
            edge_attr_atom = self.act(edge_attr_atom)
            edge_attr_line = self.act(edge_attr_line) 
        
        for index in range(len(self.conv_list_atom)):
            edge_attr_atom, edge_attr_line = self.conv_list_line[index](edge_attr_atom, data.edge_index_line, edge_attr_line)
            node_attr_atom, edge_attr_atom = self.conv_list_atom[index](node_attr_atom, data.edge_index, edge_attr_atom)
            node_attr_atom = self.dropout(node_attr_atom)
        out = self.bn(node_attr_atom)
        out = self.act(out)
        
        for id, post_lin in enumerate(self.post_lin_list):
            out = post_lin(out)
            out = self.act(out)
        
        out = self.pool(out, data.batch)
        
        out = self.lin_out(out).reshape(-1)
        
        return out
if __name__ == "__main__":
    from torch_geometric.data import Data
    data = torch.load("/home/gengzi/python/my_work/crystal_prediction/crystal/data/exp_icsd/data_0.pt")
    print(data)
    # gated_gcn = GatedGCN(data.num_node_features, data.num_edge_features)
    # out = gated_gcn(data.x, data.edge_index, data.edge_attr)
    # print(out)
    print(data.num_edge_features, data.edge_attr_line.shape)
    gated_gcn = GatedGCN(data.num_edge_features, data.edge_attr_line.shape[1])
    out = gated_gcn(data.edge_attr, data.edge_index_line, data.edge_attr_line)
    print(out)


