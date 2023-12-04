import os
import yaml
from crystal.tools import set_random_seed
import torch
import warnings
from crystal.tools import my_print, print1, print_data_info
from crystal.deeplearn.train import train
from crystal.deeplearn.process import datasets
import numpy as np
import pandas as pd
from crystal.deeplearn.process.utils import get_lb

def job_setup():
    warnings.filterwarnings('ignore')
    print1()
    my_print('Starting')
    # parser = argparse.ArgumentParser(description="MatDeepLearn inputs")
   
    ##Open provided config file
    assert os.path.exists('./crystal/config.yml'), "Config file was not found!"
    with open('./crystal/config.yml', "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader) 
    
    my_print("Settings")
    print(yaml.dump(config, sort_keys=False, default_flow_style=False, indent=4))
    # args_dict.update(vars(args))
    config['Processing']['run_mode'] = config['Global']["run_mode"]

    run_mode = config['Global']['run_mode']
    ################################################################################
    #  Begin processing
    ################################################################################
    # 设置随机种子方便重复实验
    set_random_seed(config["Global"]['seed'])
    
    # 读取数据
    my_print('dataset')
    dataset = datasets.CrystalDataset(config['Processing'])
    
    # print_data_info(dataset_list)
    # 获得设备
    my_print('device')
    print("GPU is available:"+str(torch.cuda.is_available())+", Quantity: "+str(torch.cuda.device_count())+'\n')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    return run_mode, device, dataset, config

def main():
    # 任务初始化
    run_mode, device, dataset, config = job_setup()
    ##Regular training
    if run_mode == "Training":
        my_print("training")
        train.trainer(device, dataset, config["Training"])
    ##Predicting from a trained model
    if run_mode == "Predicting":
        # 读取数据
        my_print("Predicting")
        train.predictor(device, dataset, config['Predicting'])
        

if __name__ == "__main__":
    main()
