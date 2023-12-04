##General imports
import csv
import os
import time
from datetime import datetime
import shutil
import pandas as pd
import copy
import numpy as np
from functools import partial
from tqdm import tqdm
##Torch imports
import torch.nn.functional as F
import torch
from torch_geometric.data import DataLoader, Dataset
from torch_geometric.nn import DataParallel
import torch_geometric.transforms as T
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
from crystal.tools import my_print, print1
##Matdeeplearn imports
from sklearn import metrics
import crystal.deeplearn.models as models
import crystal.deeplearn.process as process
from crystal.deeplearn.models.utils import model_summary
import yaml
import json
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.nn import global_mean_pool
################################################################################
#  Training functions
################################################################################

##Train step, runs model in train mode
def training(device, model, optimizer, loader, loss_method):
    
    loss_all = 0
    count = 0
    model.eval()
    # model.train()
    model.to(device)
    for data in loader:
        data.to(device)
        
        pred = model(data)  
        optimizer.zero_grad() 
        loss = getattr(F, loss_method)(pred, data.y)
        loss_all += float(loss.detach().cpu()) * pred.size(0)
        count = count + pred.size(0)    
        loss.backward()
        optimizer.step()
    torch.cuda.empty_cache()
    mean_loss = loss_all / count  # 平均损失
    return mean_loss


##Evaluation step, runs model in eval mode 
def evaluating(device, loader, model, loss_method):
    model.eval()
    loss_all = 0
    count = 0
    model.to(device)
    # model.to(device)
    data_list = [[], []]
    for data in loader:
        data.to(device)
        with torch.no_grad():
            pred = model(data)  
            loss = getattr(F, loss_method)(pred, data.y)
            loss_all += float(loss.detach().cpu()) * pred.size(0)
            count = count + pred.size(0)
            data_list[0] += pred.cpu().numpy().tolist()
            data_list[1] += data.y.cpu().numpy().tolist()
    
    data1 = pd.DataFrame(data_list).T
    # print(data1)
    R2 = metrics.r2_score(data1.iloc[:, 0], data1.iloc[:, 1])
    torch.cuda.empty_cache()
    data1.to_csv('pred.csv', index=False)
    
    mean_loss = loss_all / count
    return mean_loss, R2


##Model trainer
def train(
        device,
        model,
        optimizer,
        scheduler,
        train_params,
        train_loader, 
        val_loader,
        ):

    best_val_error = np.inf
    model_best = model
    ##Start training over epochs loop
    pbar = tqdm(range(1, train_params['epochs'] + 1))
    for epoch in pbar:
        
        lr = scheduler.optimizer.param_groups[0]["lr"]

        train_start = time.time()
        # Train model
        train_error = training(device, model, optimizer, train_loader, train_params['loss'])
        
        ##Train loop timings
        epoch_time = time.time() - train_start

        # Get validation performance
        val_error, r2 = evaluating(device, val_loader, model, train_params['loss'])
  
        ##remember the best val error and save model and checkpoint        
        if val_error < best_val_error:
            model_best = copy.deepcopy(model.cpu())
             # 存储模型
            torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model.cpu(),
                    },
                    train_params['model_path']
                )


        ##scheduler on train error
        scheduler.step(train_error)

        ##Print performance
        # print()
        pbar.set_description("Processing:")
        
        print("Learning Rate: {:.6f}, Training Error: {:.5f}, Val Error: {:.5f}, Time per epoch (s): {:.5f}\n".format(
                        lr, train_error, val_error, epoch_time))   

 
    return model_best


##Pytorch model setup
def model_setup(dataset, model_params, train_params, model_name):
    # 建立模型
    model = getattr(models, model_name)(
        dataset, **(model_params if model_params is not None else {})
        )
   
    # 加载已保存的模型
    if train_params['transfer']:
        assert os.path.exists(train_params['model_path']), "Saved model not found"
        saved = torch.load(train_params['model_path'])
        model = saved["full_model"]
        # optimizer.load_state_dict(saved['optimizer_state_dict'])
        
        for param in model.parameters():
            param.requires_grad_(False)
        model.lin_out.requires_grad_(True)
        model.post_lin_list[3].requires_grad_(True)
        model.post_lin_list[2].requires_grad_(True)
     
    my_print('Model') 
    model_summary(model)  # 模型的信息打印
    
    return model


##Pytorch loader setup
def split_dataset(train_params, dataset):
    """划分数据集"""
    # 划分数据集
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_params['train_ratio'])
    val_size = int(dataset_size * train_params['val_ratio'])
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split( 
                                            dataset, [train_size, val_size, test_size]
                                            )
    
    print("train length:", train_size,"val length:", val_size,"test length:",test_size)
    ##Load data
    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True, pin_memory=True, num_workers=24, persistent_workers=True, prefetch_factor=24) 
    val_loader = DataLoader(val_dataset, batch_size=train_params['batch_size'], shuffle=True, pin_memory=True, num_workers=24, persistent_workers=True, prefetch_factor=24)
    test_loader = DataLoader(test_dataset, batch_size=train_params['batch_size'], shuffle=False, pin_memory=True, num_workers=24, persistent_workers=True, prefetch_factor=24)
    return train_loader, val_loader, test_loader


def predictor(device, dataset, pred_params):
    # ref_loader = DataLoader(ref_dataset, batch_size=len(ref_dataset), shuffle=False, pin_memory=True)
    pred_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, pin_memory=True)

    saved = torch.load(pred_params["model_path"])
    model = saved["full_model"]
    model_summary(model)

    ##Get predictions
    data = next(iter(pred_loader))
    model.eval()
    model.to(device)
    data.to(device)
    with torch.no_grad():
        pred = model(data).cpu().numpy().tolist()
    test_error, r2 = evaluating(device, pred_loader, model, "l1_loss")
    print("Test Error: {:.5f}, r2: {:.5f}\n".format(test_error, r2))

    print("results.csv has written in:", os.path.abspath(pred_params['save_path']))

    result = pd.DataFrame([data.structure_id, pred]).T
    result.columns = ['id', 'pred_result']
    result.to_csv(os.path.join(pred_params['save_path'], "results.csv"), index=False)
    # [data.structure_id, pred].to_csv(pred_params['save_path'])

###Regular training with train, val, test split
def trainer(      
        device,
        dataset,
        train_params=None
        ):
    
    ##Set up loader
    model_name = train_params['model_name']
    dataset = dataset[:]
    train_loader, val_loader, test_loader = split_dataset(
        train_params,
        dataset
        )  
    
    # 加载模型参数
    with open('./crystal/model.yml', 'r') as f:
        model_params = yaml.load(f, Loader=yaml.FullLoader) 
    model_params = model_params[model_name]
 
    ##Set up model
    model = model_setup(               
                dataset[0],
                model_params,
                train_params,
                model_name
            )
    
    ##Set-up optimizer & scheduler
    optimizer = getattr(torch.optim, train_params["optimizer"])(
        model.parameters(),
        lr=train_params["lr"],
        **train_params["optimizer_args"]
        )
    
    scheduler = getattr(torch.optim.lr_scheduler, train_params["scheduler"])(
        optimizer, 
        **train_params["scheduler_args"]
        )

    ##Start training
    model = train(
                device,
                model,
                optimizer,
                scheduler,
                train_params,
                train_loader, 
                val_loader,
            )
    ## Get test error
    
    test_error, r2 = evaluating(device, test_loader, model, train_params["loss"])
    print("Test Error: {:.5f}, r2: {:.5f}\n".format(test_error, r2))

    print1()
    my_print('ending')

