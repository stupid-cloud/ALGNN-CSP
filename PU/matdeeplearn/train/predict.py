# 加载已保存的模型
import os
import numpy as np 
import torch
from torch_geometric.data import DataLoader
from matdeeplearn.train.train import testing, bootstrap_aggregating

def predictor(device, dataset):

    dataloader = DataLoader(dataset, batch_size=len(dataset), pin_memory=True)
    model_list = os.listdir('./saved_model')
        
    for i, model in enumerate(model_list):
        model = torch.load(os.path.join('./saved_model', model))['full_model']
        testing(device, dataloader, model, i)
    bootstrap_aggregating(len(model_list))


 
   
# assert os.path.exists(train_params['model_path']), "Saved model not found"
# checkpoint = torch.load(train_params['model_path'])
# model.load_state_dict(checkpoint["model_state_dict"])
# # optimizer.load_state_dict(saved['optimizer_state_dict'])

# my_print('Model') 
# model_summary(model)  # 模型的信息打印