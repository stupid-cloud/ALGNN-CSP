import os
import numpy as np
import ase 
from ase import io
import pandas as pd
from tqdm import tqdm
from scipy.stats import rankdata
import torch
from torch_geometric.data import  Data, InMemoryDataset, Dataset
from torch_geometric.utils import dense_to_sparse, add_self_loops
from torch_geometric.transforms import OneHotDegree
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.periodic_table import Element
import pickle
import itertools


def get_lb(processing_args, target_data):
    if processing_args['use_lb']:
        with open(os.path.join(processing_args['lb_path'], 'lb.pickle'), 'rb') as file:
            lb = pickle.load(file)  
    else:
        ## Process structure files and create structure graphs
        elements_list = []  # 晶体去重后的元素种类
        
        # 晶体结构处理
        for index in tqdm(range(0, len(target_data)), desc='Getting the all elements'):

            structure_id = target_data.iloc[index]

            ase_crystal = ase.io.read(os.path.join(processing_args['crystal_path'], structure_id)) # 读取晶体结构
        
            ## Compile structure sizes (# of atoms) and elemental compositions
            elements = ase_crystal.get_atomic_numbers().tolist()
            elements_list += elements
            elements_list = list(set(elements_list))
        lb = LabelBinarizer()
        lb.fit(elements_list)
        with open(os.path.join(processing_args['data_path'], 'lb.pickle'), 'wb') as file:
            pickle.dump(lb, file) 
    
    return lb
    
 

def process_data(processing_args, structure_id, targets, lb, ase_crystal):
    ## Begin processing data
    # 晶体结构处理
    data = Data()
    data.structure_id = structure_id  
    graph_max_radius = processing_args["graph_max_radius"]
    graph_max_neighbors = processing_args["graph_max_neighbors"]
    while True:
        distance_matrix = ase_crystal.get_all_distances(mic=True)  # mic最小镜像
        ## Create sparse graph from distance matrix
        distance_matrix_trimmed = distance_cutoff(
            distance_matrix,
            graph_max_radius,
            graph_max_neighbors
        )

        distance_matrix_trimmed = torch.Tensor(distance_matrix_trimmed)  
        edge_index, edge_weight = dense_to_sparse(distance_matrix_trimmed)  # 距离稀疏矩阵以及edge距离属性
        # adjacenct_matrix = (distance_matrix_trimmed != 0).int()  # 邻接矩阵
        
        position = ase_crystal.get_positions()
        df = pd.Series(edge_index[1])
        duplicate_df = df[df.duplicated(keep=False)]
        graph_max_radius += 2
        if df.duplicated(keep=False).sum() != 0:
            break

    edge_index_line = []
    edge_weight_line = []
    for i in duplicate_df.unique():
        for j in itertools.permutations(duplicate_df[duplicate_df==i].index.to_list(), r=2):
            edge_index_line.append(j)
            temp_list = set(edge_index[:, j[0]].tolist() + edge_index[:, j[1]].tolist())
            temp_list.remove(i) 
            temp_list = list(temp_list)
            p1 = position[i] - position[temp_list[0]]
            p2 = position[i] - position[temp_list[1]]
            cos_angle = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2)) 
            if cos_angle >= 1:
                cos_angle = 1
            if cos_angle <=0:
                cos_angle = 0

            angle = np.rad2deg(np.arccos(cos_angle))
            edge_weight_line.append(angle)
   
    # 自循环
    self_loops = True
    if self_loops == True:
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=0)
        # 距离矩阵添加自循环-->邻接矩阵dense
        # adjacenct_matrix = (adjacenct_matrix.fill_diagonal_(1))
    data.edge_index = edge_index
    data.edge_index_line = torch.LongTensor(edge_index_line).T
    edge_weight_line1 = edge_weight_line
    edge_weight = edge_weight / (graph_max_radius) # minmax缩放
    edge_weight_line = torch.Tensor(edge_weight_line) / 180
    if edge_weight_line.isnan().any():
        print(structure_id)
        print(pd.Series(edge_weight_line).iloc[199])
        print(edge_weight_line1[199])
        input()
    # 图的属性
    if processing_args['run_mode'] == "Training":
        data.y = torch.Tensor([targets])
    
    # Generate edge features
    if processing_args["edge_features"]: 
        # Distance descriptor using a Gaussian basis
        gaussian_smearing = GaussianSmearing(0, 1, processing_args["graph_edge_length"], 0.2)
        data.edge_attr = gaussian_smearing(edge_weight) # node_num * graph_edge_length
        
        data.edge_attr_line = gaussian_smearing(edge_weight_line)
    
    data.x = torch.Tensor(lb.transform(ase_crystal.get_atomic_numbers().tolist()))
    one_hot_degree = OneHotDegree(graph_max_neighbors + 1)
    data = one_hot_degree(data)
    return data


## 
def distance_cutoff(distance_matrix, max_distance, max_neighbors):
    """Selects edges with distance threshold and limited number of neighbors"""
    mask = distance_matrix > max_distance  # 距离截断

    # distance_matrix_rank = np.ma.array(distance_matrix, mask=mask)
    distance_matrix_rank = rankdata(distance_matrix, method="ordinal", axis=1) # 距离排名
    # 大于截断值的赋0
    # distance_matrix_trimmed = np.nan_to_num(np.where(mask, np.nan, distance_matrix_trimmed))
    distance_matrix_mask = np.where(mask, 0, distance_matrix_rank)
    distance_matrix_mask[distance_matrix_rank > max_neighbors + 1] = 0  # 最大邻居数截断 
    distance_matrix_mask = np.where(distance_matrix_rank==2, 2, distance_matrix_mask)
    # 还原成距离
    distance_matrix = np.where(distance_matrix_mask == 0, distance_matrix_mask, distance_matrix)
    return distance_matrix


##Slightly edited version from pytorch geometric to create edge from gaussian basis
class GaussianSmearing(torch.nn.Module):
    """单个距离属性扩展成高斯分布"""
    def __init__(self, start=0.0, stop=5.0, resolution=50, width=0.05, **kwargs):
        super(GaussianSmearing, self).__init__()
        self.offset = torch.linspace(start, stop, resolution)   # 不同的μ
        # self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.coeff = -0.5 / ((stop - start) * width) ** 2  # -γ or -1/2σ^2     

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset.view(1, -1)  # x-μ
        return torch.exp(self.coeff * torch.pow(dist, 2))

