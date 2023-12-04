import os
import numpy as np 
import torch
from torch_geometric.data import DataLoader
from deeplearn.train.train import testing, bootstrap_aggregating

def predictor(device, dataset):

    dataloader = DataLoader(dataset, batch_size=len(dataset), pin_memory=True)
    model_list = os.listdir('./saved_model')
        
    for i, model in enumerate(model_list):
        model = torch.load(os.path.join('./saved_model', model))['full_model']
        testing(device, dataloader, model, i)
    bootstrap_aggregating(len(model_list))