##General imports
import csv
import os
import pandas as pd
import time
from datetime import datetime
import shutil
import copy
from sklearn import metrics
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
from PU.tools import my_print, print1
##Matdeeplearn imports
import PU.matdeeplearn.models as models
import PU.matdeeplearn.process as process
from PU.matdeeplearn.models.utils import model_summary
import yaml
from torch_geometric.nn import global_mean_pool
################################################################################
#  Training functions
################################################################################

##Train step, runs model in train mode
def training(device, model, optimizer, loader, loss_method):
    
    model.train()
    count = 0
    loss_all = 0
    # loss, accuracy, precision, recall, fscore, auc_score
    metrics_list = [0, 0, 0, 0, 0]
    model.to(device)
    for data in loader:

        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = getattr(F, loss_method)(pred, data.y.long())
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            loss_all += float(loss.cpu()) 
          
            props_list = class_eval(pred.cpu(), data.y.cpu())

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            for i in range(len(metrics_list)):
                metrics_list[i] += props_list[i]
            count += 1

    for i in range(len(metrics_list)):
        metrics_list[i] /= count
    torch.cuda.empty_cache()
    return loss_all / count , metrics_list


##Evaluation step, runs model in eval mode 
def evaluating(device, loader, model, loss_method, save=False):
    model.eval()
    count = 0
    loss_all = 0
    # accuracy, precision, recall, fscore, auc_score
    metrics_list = [0, 0, 0, 0, 0]
    model.to(device)
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            # readout layer
            loss = getattr(F, loss_method)(pred, data.y.long())
            loss_all += float(loss.cpu()) 
            props_list = class_eval(pred.cpu(), data.y.cpu())

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            for i in range(len(metrics_list)):
                metrics_list[i] += props_list[i]
            count += 1
    
    for i in range(len(metrics_list)):
        metrics_list[i] /= count
    torch.cuda.empty_cache()
    return loss_all / count, metrics_list


##Model trainer
def train(
        device, 
        model, 
        optimizer, 
        scheduler, 
        loss, 
        train_loader, 
        val_loader, 
        epochs, 
        ):

    best_val_error = np.inf
    model_best = model
    metrics_list = []
    ##Start training over epochs loop
    pbar = tqdm(range(1, epochs + 1))
    for epoch, _ in enumerate(pbar):
        
        lr = scheduler.optimizer.param_groups[0]["lr"]

        train_start = time.time()
        # Train model
        # loss, accuracy, precision, recall, fscore, auc_score

        train_error, metrics_list1 = training(device, model, optimizer, train_loader, loss)
        
        ##Train loop timings
        epoch_time = time.time() - train_start

        # Get validation performance
        val_error, metrics_list2 = evaluating(device, val_loader, model, loss)
  
        ##remember the best val error and save model and checkpoint        
        if val_error < best_val_error:
            model_best = copy.deepcopy(model.cpu())

        ##scheduler on train error
        scheduler.step(train_error)

        ##Print performance
        # print()
        pbar.set_description("Processing:")
        
        print("Learning Rate: {:.6f}, Training Error: {:.5f}, "
              "Val Error: {:.5f}, Time per epoch (s): {:.5f}\n".format(
                        lr, train_error, val_error, epoch_time))

        print("Training, Accuracy: {:.6f}, Precision: {:.5f}, "
              "Recall: {:.5f}, F1: {:.5f}, Auc: {:.5f}\n".format(
                        *metrics_list1))
        print("Validing, Accuracy: {:.6f}, Precision: {:.5f}, "
              "Recall: {:.5f}, F1: {:.5f}, Auc: {:.5f}\n".format(
                        *metrics_list2))
        # metrics_list2.append(val_error)
        # metrics_list.append(metrics_list2)
    # pd.DataFrame(metrics_list, columns=["accuracy", "precision", "recall", "fscore", "auc_score", "val_error"]).to_csv('./metrics.csv', header=True)
    
    return model_best


##Pytorch model setup
def model_setup(dataset, model_params, train_params, model_name):
    # 建立模型
    model = getattr(models, model_name)(
        dataset, **(model_params if model_params is not None else {})
        )
    my_print('Model') 
    model_summary(model)  # 模型的信息打印
    return model


##Pytorch loader setup



def split_dataset(positive, unlabeled, train_params, dataset):
    """划分数据集"""
    # Sample positive data for validation and training
    
    positive_num = len(positive)
    
    train_num = int(train_params['train_ratio']*positive_num)
    val_num = int(train_params['val_ratio']*positive_num)

    train_positive = positive.sample(n=train_num)
    temp = positive.drop(train_positive.index)
    valid_positive = temp.sample(n=val_num)
    test_positive = temp.drop(valid_positive.index)
    
    # Sample negative data for training
    
    # Randomly labeling to negative
    negative = unlabeled.sample(n=positive_num)
    # global out_bag
    # out_bag += negative.index.to_list()
    # out_bag = list(set(out_bag))
    train_negative = negative.sample(n=train_num)
    temp = negative.drop(train_negative.index)
    valid_negative = temp.sample(n=val_num)
    test_negative = temp.drop(valid_negative.index)

    train_dataset_index = pd.concat([train_positive, train_negative]).values.tolist()
    valid_dataset_index = pd.concat([valid_positive, valid_negative]).values.tolist()
    test_dataset_index = pd.concat([test_positive, test_negative]).values.tolist()
    print("train length:", len(train_dataset_index),"valid length:", len(valid_dataset_index), "test length:", len(test_dataset_index))
    ##Load data
    train_loader = DataLoader(dataset[train_dataset_index], batch_size=train_params['batch_size'], shuffle=True, pin_memory=True, num_workers=24) 
    val_loader = DataLoader(dataset[valid_dataset_index], batch_size=train_params['batch_size'], shuffle=True, pin_memory=True, num_workers=24)
    test_loader = DataLoader(dataset[test_dataset_index], batch_size=train_params['batch_size'], shuffle=False, pin_memory=True, num_workers=24)

    return train_loader, val_loader, test_loader

def class_eval(prediction, target, auc=True):

    prediction = F.softmax(prediction, dim=1).numpy()
    target = target.numpy().astype('int') 
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        if auc:
            try:
                auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
            except:
                auc_score = 0.8
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    if auc:
        return [accuracy, precision, recall, fscore, auc_score]
    else:
        return [accuracy, precision, recall, fscore]


    
def predictor(device, dataset, pred_params):
    dataset = dataset[:]
    dataloader = DataLoader(dataset, batch_size=pred_params['batch_size'], shuffle=False, pin_memory=True, num_workers=24)
    model_list = os.listdir(pred_params['model_path'])
    for i, model in enumerate(model_list):
        model = torch.load(os.path.join(pred_params['model_path'], model))['full_model']
        model.eval()
        model_summary(model)
        model.to(device)
        structure_id_list = []
        prediction_list = []
        for data in dataloader:   
            # mean_file
            data.to(device)
            with torch.no_grad():
                prediction = model(data)
                prediction = F.softmax(prediction.cpu(), dim=1).numpy()
                file_name = os.path.join(pred_params['save_path'], f"bagging{i}.csv")
                # structure_id = np.array(data.structure_id).reshape(-1, 1)
                structure_id_list += data.structure_id
                prediction_list += prediction[:, 1].tolist()

            file = pd.DataFrame([structure_id_list, prediction_list]).T
            file.columns =['id', 'positive_prob']
            if i == 0:
                mean_file = file
            else:
                mean_file['positive_prob'] = (mean_file['positive_prob'] + file['positive_prob']) / 2
            file = file.sort_values(by='positive_prob', ascending=False)
            file.to_csv(file_name, index=False)
    mean_file = mean_file.sort_values(by='positive_prob', ascending=False)
    mean_file.to_csv(os.path.join(pred_params['save_path'], "mean.csv"), index=False)
    print("results have written in:", os.path.abspath(pred_params['save_path']))



###Regular training with train, val, test split
def trainer(      
        device,
        dataset,
        train_params=None,
        positive=None, 
        unlabeled=None
        ):
    
    model_name =  train_params["model_name"]
     # 加载模型参数
    with open('./PU/model.yml', 'r') as f:
        model_params = yaml.load(f, Loader=yaml.FullLoader) 
    model_params = model_params[model_name]

    bagging_size = train_params['bagging_size']

   
    # global out_bag
    # out_bag = []

    for i in range(bagging_size):
        print("\nbagging number: {}".format(i))
        ##Set up loader
        train_loader, val_loader, test_loader = split_dataset(
            positive, 
            unlabeled,
            train_params,
            dataset
            )  

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
                    train_params["loss"],
                    train_loader,
                    val_loader,
                    train_params["epochs"],
                )
       
        # 存储模型
        torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "full_model": model,
                        },
                        os.path.join('./PU/saved_model', f"model{i}.pt")
                    )
        ## Get test error
        test_error, metrics_list = evaluating(device, test_loader, model, train_params['loss'])
        # pd.DataFrame(out_bag).to_csv('./out.csv', index=False)
        print("Testing error, Accuracy: {:.6f}, Precision: {:.5f}, "
              "Recall: {:.5f}, F1: {:.5f}, Auc: {:.5f}\n".format(test_error, 
                        *metrics_list))
    # bootstrap_aggregating(bagging_size)

    print1()
    my_print('ending')


# def bootstrap_aggregating(bagging_size):

#     predval_dict = {}

#     print("Do bootstrap aggregating for %d models.............." % (bagging_size))
#     for i in range(bagging_size):
   
#         df = pd.read_csv(os.path.join('./pred_results', f"bagging{i}.csv"))
#         id_list = df.iloc[:, 0].tolist()
#         pred_list = df.iloc[:, 1].tolist()
#         for idx, mat_id in enumerate(id_list):
#             if mat_id in predval_dict:
#                 predval_dict[mat_id].append(float(pred_list[idx]))
#             else:
#                 predval_dict[mat_id] = [float(pred_list[idx])]

#     # print("Writing CLscore file....")
#     # with open('test_results_ensemble_'+str(bagging_size)+'models.csv', "w") as g:
#     #     g.write("id,CLscore,bagging")                                       # mp-id, CLscore, # of bagging size
#     key_value = []
#     for key, values in predval_dict.items(): 
#         key_value.append([key, np.mean(np.array(values))])
#     key_value = pd.DataFrame(key_value, columns=['id', 'pred_prob']).sort_values(by='pred_prob', ascending=False)
#     key_value.to_csv('./pred_results/mean.csv', index=False, header=True)
#     print("Done")



