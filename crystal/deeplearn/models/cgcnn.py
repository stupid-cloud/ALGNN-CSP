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

class CGCNN(nn.Module):
    def __init__(
            self, dataset, conv_num, pool, dropout_rate, act, batch_norm, 
            pre_fc_num, pre_out_channel, post_fc_num, post_out_channel , **kwargs
            ):
        super().__init__()
        # self.linear = nn.LazyLinear(out_channels)
        num_node_features = dataset.num_node_features
        num_edge_features = dataset.num_edge_features
        self.pre_lin_list = self.init_pre_fc(pre_fc_num, num_node_features, pre_out_channel)
        self.conv_list, self.bn_list = self.init_conv(conv_num, pre_out_channel, num_edge_features, batch_norm, kwargs)
        self.post_lin_list = self.init_post_fc(post_fc_num, pre_out_channel, post_out_channel)
        self.lin_out = nn.Linear(post_out_channel, 1)
        self.act = getattr(nn.functional, act)
        self.dropout = nn.Dropout(dropout_rate)

        if pool == 'global_first_pool':
            self.pool = global_first_pool
        else:
            self.pool = getattr(torch_geometric.nn, pool)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Resets all learnable parameters of the module"""
        for i in self.pre_lin_list:
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

    def init_conv(self, conv_num, pre_out_channel, num_edge_features, batch_norm, kwargs):
        conv_list = torch.nn.ModuleList()
        bn_list = torch.nn.ModuleList()
        if conv_num <=0:
            conv_num = 1

        for _ in range(conv_num):           
            conv = CGConv(channels=pre_out_channel, dim=num_edge_features, batch_norm=False, **kwargs)
            conv_list.append(conv)
            bn_list.append(BatchNorm1d(pre_out_channel) if batch_norm else None) 
        return conv_list, bn_list

    def forward(self, data) -> Tensor:
        for id, pre_lin in enumerate(self.pre_lin_list):
            if id == 0:
                out = pre_lin(data.x)
            else:
                out = pre_lin(out)
            out = self.act(out)

        for id, conv in enumerate(self.conv_list):

            if id == 0:
                out = conv(out, data.edge_index, data.edge_attr)
            else:
                out = conv(out, data.edge_index, data.edge_attr)

            if self.bn_list[id] != None:
                out = self.bn_list[id](out)

            out = self.dropout(out)
            out = self.act(out)

        for id, post_lin in enumerate(self.post_lin_list):
            if id == 0:
                out = post_lin(out)
            else:
                out = post_lin(out)
            out = self.act(out)
        out = self.pool(out, data.batch)
        out = self.lin_out(out).reshape(-1)

        return out


if __name__ == "__main__":
    edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1],], dtype=torch.long)
    x = torch.tensor([[-1, 0], [0, 3], [1, 4]], dtype=torch.float)
    edge_attr = torch.tensor([[0, 1, 2],
                            [1, 0, 6],
                            [1, 2, 8],
                            [2, 1, 9],], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)

    model  = CGCNN(data.num_node_features, 6, 1, dim=data.num_edge_features)
    batch = torch.tensor([0, 0, 0])
    print(model(data.x, data.edge_index, data.edge_attr, batch))