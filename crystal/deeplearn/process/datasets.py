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
from .utils import process_data, get_lb
from pymatgen.io.ase import AseAtomsAdaptor
##Dataset class from pytorch/pytorch geometric; inmemory case



class CrystalDataset(Dataset):
    def __init__(self, processing_args, transform=None, pre_transform=None, pre_filter=None):
        self.processing_args = processing_args
        
        self.data_info = self._get_data_info()
        super().__init__(processing_args['data_path'], transform, pre_transform, pre_filter) 
    
    def _get_data_info(self):
        file_name = os.path.join(self.processing_args['data_path'], self.processing_args['targets_file_name'])
        assert os.path.exists(file_name), ("targets not found in " + file_name) 
        data_info = pd.read_csv(file_name)
        return data_info

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.processing_args['targets_file_name'].split(".")[0])

    @property
    def processed_file_names(self):
        return ["{}.pt".format(self.data_info.iloc[idx, 0].split('.')[0]) for idx in range(self.data_info.shape[0])]

    def process(self):
        lb = get_lb(self.processing_args, self.data_info.iloc[:, 0])
        for idx in tqdm(range(self.data_info.shape[0]), desc="Creating graph"):
            structure_id = self.data_info.iloc[idx, 0]
            targets = self.data_info.iloc[idx, 1]
            
            # Read data from `raw_path`.
            ase_crystal = ase.io.read(os.path.join(self.processing_args['crystal_path'], structure_id)) # 读取晶体结构 
            
            data = process_data(self.processing_args, structure_id, targets, lb, ase_crystal)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, os.path.join(self.processed_dir, f'{structure_id.split(".")[0]}.pt'))

    def get(self, idx):
        filename = self.data_info.iloc[idx, 0].split(".")[0]
        data = torch.load(os.path.join(self.processed_dir, f'{filename}.pt'))
        return data
    
    def len(self):
        return len(self.processed_file_names)


class SmallDataset(InMemoryDataset):
    def __init__(self, processing_args, transform=None, pre_transform=None, pre_filter=None):
        self.processing_args = processing_args
        super().__init__(processing_args['data_path'], transform, pre_transform, pre_filter) 
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        data = pd.read_csv(os.path.join(self.root, self.processing_args['targets_file_name']))
        return data.iloc[:, 0].to_list()

    @property
    def processed_file_names(self):
        model_name = self.processing_args["targets_file_name"].split(".")[0]+".pt"
        return [model_name]

    def process(self):
        data_list = [...]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data_list = process_data(self.processing_args)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def get_graph_data(processing_args, structure):
    structure = AseAtomsAdaptor().get_atoms(structure)     
    lb = get_lb(processing_args, None) 
    
    # Read data from `raw_path`.
    data = process_data(processing_args, None, None, lb, structure)
    
    return data
    

if __name__ == "__main__":
    
    import yaml
    with open('config.yml', "r") as ymlfile:
            config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    dataset = StructureDataset(
                    "/home/gengzi/python/GNN/test/data/Ef<0_data",
                    config["Processing"],
                    )
    print(dataset)
    print(dataset[0])